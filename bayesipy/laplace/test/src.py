import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call


class TestLaplace:
    """This class approximates the posterior distribution over model parameters with
    a Laplace approximation, and leverages a secondary model (model2) to learn
    features that mimic the (generalized) Gauss-Newton (GGN) structure of the
    primary model for scalable continual learning.

    Parameters
    ----------
    model : torch.nn.Module
        The primary predictive model (e.g., a neural network).
    model2 : torch.nn.Module
        The secondary model used for feature approximation (e.g., to approximate
        the GGN structure of the primary model).
    likelihood : {'regression', 'classification'}
        The type of likelihood for the primary model outputs. Must be either
        "regression" or "classification".
    prior_precision : float, optional
        The initial prior precision (assumed diagonal if scalar). Default is 1.
    sigma_noise : float, optional
        The initial observation noise standard deviation for regression tasks.
        Ignored for classification tasks. Default is 1.
    y_mean : float, optional
        Mean of the target variable for output denormalization in regression.
        Default is 0.
    y_std : float, optional
        Standard deviation of the target variable for output denormalization in
        regression. Default is 1.
    seed : int, optional
        Random seed for reproducibility. Default is 1234.
    """

    def __init__(
        self,
        model,
        model2,
        likelihood,
        prior_precision=1,
        sigma_noise=1,
        y_mean=0,
        y_std=1,
        seed=1234,
    ) -> None:
        super().__init__()

        # -- Primary model configuration --
        # We keep the primary model in evaluation mode because we are dealing
        # with a Laplace approximation around fixed parameters (no further training).
        self.model = model
        self.model.eval()

        # -- Secondary model configuration --
        # This model is trained (via kernel norm minimization) to match the approximate
        # Hessian structure of self.model. We keep it in train mode for that purpose.
        self.model2 = model2
        self.model2.train()

        # -- Extract parameters from primary model --
        # Create a vector of parameters and store it. This will serve as theta_MAP for
        # the Laplace approximation.
        self.params = [p for p in self.model.parameters()]
        self.mean = parameters_to_vector(self.params)  # The MAP estimate is stored here.

        # -- Determine device and dtype from the primary model parameters --
        self.device = next(iter(self.params)).device
        self.dtype = next(iter(self.params)).dtype
        # Move secondary model to the same device/dtype.
        self.model2 = self.model2.to(self.device).to(self.dtype)

        # -- Likelihood configuration --
        self.likelihood = likelihood
        assert self.likelihood in ["regression", "classification"], (
            "likelihood must be either 'regression' or 'classification'."
        )

        # -- Prior & noise parameters --
        self._prior_precision = prior_precision
        self._sigma_noise = sigma_noise
        self.prior_mean: float | torch.Tensor = 0  # The prior mean is assumed 0 by default.

        # -- Output normalization parameters (for regression) --
        self.y_mean = torch.tensor(y_mean).to(self.device).to(self.dtype)
        self.y_std = torch.tensor(y_std).to(self.device).to(self.dtype)

        # -- Random generator for any needed sampling --
        self.seed = seed
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # -- Posterior quantities (to be computed after fitting) --
        self._posterior_scale = None  # Cholesky factor of posterior covariance
        self.H = None                 # Approximate Hessian (GGN) placeholder
        self.loss = 0                 # Training loss placeholder for use in the log-likelihood

        # -- Additional attributes --
        self.n_outputs = None
        self.n_features = None
        self.num_train = None

    def fit(
        self,
        iterations: int,
        train_loader: DataLoader,
        context_points_loader: DataLoader = None,
        weight_decay: float = None,
        lr: float = None,
        optimize_hyper_parameters: bool = True,
        verbose: bool = False,
    ):
        """
        Fit the Laplace approximation to the training data.

        Steps:
        1. Kernel norm minimization:
           - Train the secondary model (model2) by matching its induced kernel
             (via the GGN proxy) to that of the primary model (model).
           - This uses random Jacobian-vector products (JVPs) from the primary model
             and backpropagation into model2.

        2. Build approximate Hessian (GGN):
           - Accumulate the final approximate GGN for all batches in the training set
             (and context set, if provided) to produce self.H.

        3. Hyperparameter optimization (optional):
           - If optimize_hyper_parameters=True, optimize prior precision and noise
             standard deviation (for regression) by maximizing the approximate marginal
             likelihood under the Laplace approximation.

        Parameters
        ----------
        iterations : int
            Number of mini-batch iterations to train the secondary model (model2).
        train_loader : torch.utils.data.DataLoader
            DataLoader providing the main training data batches.
        context_points_loader : torch.utils.data.DataLoader, optional
            DataLoader providing additional context data in a continual learning scenario.
        weight_decay : float, optional
            L2 regularization coefficient applied to model2's parameters.
        lr : float, optional
            Learning rate for model2's optimizer (Adam). Defaults to 1e-4 if not provided.
        optimize_hyper_parameters : bool, optional
            Whether to optimize the prior precision (and sigma_noise if regression)
            after building the Hessian. Default is True.
        verbose : bool, optional
            If True, display progress bars and losses. Default is False.

        Returns
        -------
        losses : list of float
            History of the kernel norm minimization loss over the specified iterations.
        losses2 : list of float
            History of the negative marginal likelihood (negated) while optimizing
            hyperparameters. Empty if optimize_hyper_parameters=False.
        """
        # Prepare for kernel norm minimization by ensuring shapes/dimensions are known.
        data = next(iter(train_loader))
        X = data[0]
        batch_size = X.shape[0]
        total_size = len(train_loader.dataset)
        self.num_train = total_size

        # Attempt a forward pass for shape initialization.
        try:
            out = self.model(X[:1].to(self.device).to(self.dtype))
            self.n_features = self.model2(X[:1].to(self.device).to(self.dtype)).shape[-2]
        except (TypeError, AttributeError):
            # For some models that may not allow slicing with [:1].
            out = self.model(X.to(self.device).to(self.dtype))
            self.n_features = self.model2(X.to(self.device).to(self.dtype)).shape[-2]

        self.n_outputs = out.shape[-1]

        # Reset loss and Hessian buffer.
        self.loss = 0
        self.H = torch.zeros(self.n_features, self.n_features, 
                             device=self.device,
                             dtype=self.dtype)

        # Optimizer for the secondary model.
        optimizer = torch.optim.Adam(self.model2.parameters(), 
                                     lr=1e-4 if lr is None else lr)
        # optimizer = torch.optim.SGD(self.model2.parameters(),
        #                             lr=1e-4 if lr is None else lr,
        #                             momentum=0.9)
        # Create iterators for the training loader and optional context loader.
        train_iter = iter(train_loader)
        if context_points_loader is not None:
            context_points_iter = iter(context_points_loader)

        losses = []

        # Make sure the primary model's parameters require grad during JVP calculations.
        for p in self.model.parameters():
            p.requires_grad = True

        # -- Step 1: Kernel norm minimization --
        tq = tqdm(range(iterations), desc="Minimizing kernel norm", disable=not verbose)
        for _ in tq:
            # Fetch the next batch from train data; re-init iterator if exhausted.
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(self.device).to(self.dtype)
            y = y.to(self.device)

            # If context loader is available, combine that batch as well.
            if context_points_loader is not None:
                try:
                    x_context, y_context = next(context_points_iter)
                except StopIteration:
                    context_points_iter = iter(context_points_loader)
                    x_context, y_context = next(context_points_iter)

                x_context = x_context.to(self.device).to(self.dtype)
                y_context = y_context.to(self.device)

                x = torch.cat([x, x_context], dim=0)
                y = torch.cat([y, y_context], dim=0)
                batch_size = x.shape[0]

            # Compute a random JVP for the primary model (approx. GGN row).
            J = self._true_jacobian(x, y)

            # Compute features from the secondary model (model2).
            phi = self.model2(x)

            # K ~ J*J^T in batched form => shape: (batch_size, batch_size, output_dim, output_dim)
            K = torch.einsum("ia,jb->ijab", J, J)
            # Q ~ phi*phi^T in batched form => shape: (batch_size, batch_size, feature_dim, feature_dim)
            Q = torch.einsum("ika,jkb->ijab", phi, phi)

            # Construct scaling matrix for the loss to approximate matching
            # of the two kernel representations (K and Q) across the batch.
            scale_loss_matrix1 = torch.eye(batch_size) * (batch_size / total_size)
            scale_loss_matrix2 = (torch.ones_like(scale_loss_matrix1) - torch.eye(batch_size)) * (
                batch_size * (batch_size - 1) / (total_size * (total_size - 1))
            )
            scale_loss_matrix = (scale_loss_matrix1 + scale_loss_matrix2).to(self.device).to(self.dtype)
            # Expand to align with the extra dimensions in K and Q.
            scale_loss_matrix = scale_loss_matrix.unsqueeze(-1).unsqueeze(-1)

            # Core loss: kernel norm difference between primary model GGN proxy and model2 features.
            loss = torch.norm(scale_loss_matrix * (K - Q))

            # Optional weight decay on model2 parameters.
            if weight_decay is not None:
                loss += weight_decay * torch.norm(
                    torch.cat([p.view(-1) for p in self.model2.parameters()])
                )

            if verbose:
                tq.set_postfix({"Loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # -- Step 2: Build approximate Hessian (GGN) using the final model2 --
        with torch.no_grad():
            for x, y in tqdm(train_loader, desc="Building approximate GGN", disable=not verbose):
                x = x.to(self.device).to(self.dtype)
                y = y.to(self.device)

                phi = self.model2(x)
                f = self.model(x)

                if self.likelihood == "classification":
                    # Standard cross-entropy-based GGN for classification.
                    loss_batch = torch.nn.CrossEntropyLoss()(f, y)
                    ps = torch.softmax(f, dim=-1)
                    G = torch.diag_embed(ps) - torch.einsum("mk,mc->mck", ps, ps)
                    H_batch = torch.einsum("bpc,bkc,bqk->pq", phi, G, phi)
                else:
                    # MSE-based GGN for regression.
                    loss_batch = 0.5 * torch.nn.MSELoss(reduction="sum")(f, y)
                    H_batch = torch.einsum("bpc,bqc->pq", phi, phi)

                self.loss += loss_batch.detach()
                self.H += H_batch.detach()

        # -- Step 3: Hyperparameter optimization (optional) --
        losses2 = []
        if optimize_hyper_parameters:
            log_sigma = torch.zeros(1, requires_grad=True, device=self.device)
            log_prior = torch.zeros(1, requires_grad=True, device=self.device)
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-3)

            hyper_iter = tqdm(range(10000), desc="Optimizing Hyper-parameters", disable=not verbose)
            for _ in hyper_iter:
                hyper_optimizer.zero_grad()
                neg_marglik = -self.log_marginal_likelihood(
                    log_prior.exp(),
                    log_sigma.exp() if self.likelihood == "regression" else None,
                )
                neg_marglik.backward()
                hyper_optimizer.step()
                losses2.append(neg_marglik.item())

            # Update the internal hyperparameters after optimization.
            self.prior_precision = log_prior.exp().item()
            self.sigma_noise = log_sigma.exp().item()

        return losses, losses2

    def _true_jacobian(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute a single random Jacobian-vector product (JVP) for the primary model.

        Uses forward-mode AD to get a JVP that, when accumulated over many batches,
        approximates the (generalized) Gauss-Newton matrix structure. This is used
        to train the secondary model's features to align with the primary model's
        curvature information.

        Parameters
        ----------
        X : torch.Tensor
            Input batch of shape (batch_size, ...).
        Y : torch.Tensor
            Target batch (unused here but retained for consistency).

        Returns
        -------
        jvp : torch.Tensor
            Random JVP of shape (batch_size, output_dim). Each row corresponds
            to one sample's JVP for the current random tangent.
        """
        dual_params = {}
        params = {name: p for name, p in self.model.named_parameters()}
        # Generate random tangents for each parameter.
        tangents = {name: torch.randn_like(p) for name, p in params.items()}

        # Use forward-mode AD to generate the JVP.
        with fwAD.dual_level():
            for name, p in params.items():
                dual_params[name] = fwAD.make_dual(p, tangents[name])
            out = functional_call(self.model, dual_params, X)
            _, jvp = fwAD.unpack_dual(out)

        return jvp.detach()

    def log_marginal_likelihood(
        self,
        prior_precision: torch.Tensor | None = None,
        sigma_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the approximate log marginal likelihood under the Laplace approximation.

        The approximation is based on:
            log p(D) ≈ log p(D | θ_MAP) - 0.5 * [ (log det P) - (log det P0) + scatter ],
        where:
            - P is the posterior precision (approx. Hessian + prior diagonal).
            - P0 is the prior precision (diagonal).
            - scatter = (θ_MAP - μ0)^T * P0 * (θ_MAP - μ0), with μ0=0 by default.

        Parameters
        ----------
        prior_precision : torch.Tensor, optional
            Overrides the current prior precision if provided.
        sigma_noise : torch.Tensor, optional
            Overrides the current noise standard deviation (relevant only for regression).

        Returns
        -------
        log_marglik : torch.Tensor
            Scalar tensor of the approximate log marginal likelihood.
        """
        if prior_precision is not None:
            self.prior_precision = prior_precision
        if sigma_noise is not None:
            if self.likelihood != "regression":
                raise ValueError("sigma_noise is only applicable for regression.")
            self.sigma_noise = sigma_noise

        # log p(D | θ_MAP) - 0.5 * [log det ratio + scatter].
        return self.log_likelihood - 0.5 * (self.log_det_ratio + self.scatter)

    def _compute_scale(self) -> None:
        """
        Update the cached posterior scale (Cholesky factor of covariance).

        Uses `_precision_to_scale_tril` from PyTorch to invert the posterior
        precision matrix and produce a lower-triangular factor.
        """
        self._posterior_scale = _precision_to_scale_tril(self.posterior_precision).to(self.dtype)

    @property
    def scatter(self) -> torch.Tensor:
        r"""
        Scatter term: (θ_MAP - μ0)^T * P0 * (θ_MAP - μ0).

        With μ0 = 0 (the default prior mean), this simplifies to:
        θ_MAP^T * P0 * θ_MAP.

        Returns
        -------
        torch.Tensor
            Scalar scatter term.
        """
        delta = self.mean - self.prior_mean
        return (delta * self.prior_precision) @ delta

    @property
    def prior_precision_diag(self) -> torch.Tensor:
        """
        Return the diagonal of the prior precision matrix.

        If `self.prior_precision` is scalar, broadcast it to length n_features.
        If it is a tensor of length n_features, return it as-is.

        Returns
        -------
        torch.Tensor
            1D tensor of shape (n_features,).
        """
        prior_prec = (
            self.prior_precision
            if isinstance(self.prior_precision, torch.Tensor)
            else torch.tensor(self.prior_precision, device=self.device)
        )
        if prior_prec.ndim == 0 or len(prior_prec) == 1:
            return prior_prec * torch.ones(self.n_features, device=self.device)
        elif len(prior_prec) == self.n_features:
            return prior_prec
        else:
            raise ValueError("Mismatch between prior_precision and n_features.")

    @property
    def posterior_scale(self) -> torch.Tensor:
        """
        Lower-triangular factor (Cholesky) of the posterior covariance.

        Returns
        -------
        torch.Tensor
            2D lower-triangular tensor such that
            posterior_covariance = posterior_scale @ posterior_scale.T.
        """
        if self._posterior_scale is None:
            self._compute_scale()
        return self._posterior_scale

    @property
    def posterior_covariance(self) -> torch.Tensor:
        r"""
        Full posterior covariance matrix P^{-1}.

        Computed as:
            P^{-1} = posterior_scale @ posterior_scale^T.
        """
        return self.posterior_scale @ self.posterior_scale.T

    @property
    def posterior_precision(self) -> torch.Tensor:
        r"""
        Posterior precision matrix P.

        Defined as:
            P = α * H + diag(P0),
        where:
            H is the approximate Hessian (GGN),
            α = 1 / σ^2 (for regression) or a user-defined scale for classification,
            diag(P0) is the diagonal prior precision.

        Returns
        -------
        torch.Tensor
            2D tensor of shape (n_features, n_features).
        """
        if self.H is None:
            raise RuntimeError("Hessian (self.H) is not computed. Call `fit` first.")
        if self.prior_precision_diag is None:
            raise RuntimeError("Prior precision is not set properly.")
        if self._H_factor is None:
            raise RuntimeError("Hessian scaling factor (_H_factor) is not set.")
        return self._H_factor * self.H + torch.diag(self.prior_precision_diag)

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        """
        Log-determinant of the posterior precision matrix.

        Returns
        -------
        torch.Tensor
            Scalar tensor of the log-determinant.
        """
        return self.posterior_precision.logdet()

    @property
    def log_det_prior_precision(self) -> torch.Tensor:
        """
        Log-determinant of the diagonal prior precision matrix.

        Returns
        -------
        torch.Tensor
            Scalar tensor (sum of log of each diagonal element).
        """
        return self.prior_precision_diag.log().sum()

    @property
    def log_det_ratio(self) -> torch.Tensor:
        r"""
        \(\log \det(P) - \log \det(P_0)\).

        Represents the difference in log-determinants between the posterior
        precision and the prior precision.
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision

    @property
    def _H_factor(self) -> torch.Tensor:
        """
        Scaling factor for the Hessian term H.

        By default, for regression:
            _H_factor = 1 / (σ_noise^2).
        For classification, this is used similarly to scale the approximate GGN.

        Returns
        -------
        torch.Tensor
            Scalar factor for self.H.
        """
        sigma2 = self.sigma_noise**2
        return 1 / sigma2

    @property
    def log_likelihood(self) -> torch.Tensor:
        r"""
        Log likelihood of the training data under the current parameters.

        For regression:
            - (1 / (2σ^2)) * loss - (n_train * n_outputs / 2) * log(2πσ^2).

        For classification:
            Uses the same σ^2 factor for consistency, i.e.:
            - (1 / σ^2) * loss.

        Returns
        -------
        torch.Tensor
            Scalar log-likelihood.
        """
        factor = -self._H_factor
        if self.likelihood == "regression":
            c = self.num_train * self.n_outputs * torch.log(self.sigma_noise * np.sqrt(2 * np.pi))
            return factor * self.loss - c
        else:
            return factor * self.loss

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Forward pass that computes predictive mean and covariance factor for each input.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, input_dim).

        Returns
        -------
        (F_mean, F_var)
            F_mean : torch.Tensor
                Primary model output (predictive mean).
            F_var : torch.Tensor
                Predictive variance term computed as φ * Cov * φ^T, where φ are
                features from model2 and Cov is the posterior covariance. The shape
                may be higher-dimensional depending on the batch size and output dimension.
        """
        x = x.to(self.device).to(self.dtype)
        F_mean = self.model(x)
        phi = self.model2(x)

        # Einsum for φ * Cov * φ^T in a batched sense. This can produce a higher-rank
        # output if x has multiple dimensions. Adjust usage based on your final needs.
        F_var = torch.einsum("npa,pq,mqb->nmab", phi, self.posterior_covariance, phi)
        return F_mean, F_var

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Return denormalized predictions and their variances.

        For regression, adds σ_noise^2 to the diagonal of the predictive variance.
        Also denormalizes using (y_mean, y_std).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, input_dim).

        Returns
        -------
        (mean_out, var_out)
            mean_out : torch.Tensor
                Denormalized predictive mean.
            var_out : torch.Tensor
                Denormalized predictive variance. For regression, includes the
                noise variance σ_noise^2 added to the model’s predictive variance.
        """
        F_mean, F_var = self.forward(x)
        if self.likelihood == "regression":
            # For scalar output, the shape might be simpler. Adjust if you have multi-dimensional outputs.
            F_var = F_var.squeeze(-1) + self.sigma_noise**2
        return F_mean * self.y_std + self.y_mean, F_var * self.y_std**2

    @property
    def prior_precision(self) -> torch.Tensor:
        """
        The prior precision vector/tensor.

        If originally provided as a float, it is stored as a 1D tensor with length 1.
        Otherwise, it is a 1D tensor of shape (n_features,).

        Returns
        -------
        torch.Tensor
            The prior precision.
        """
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision: float | torch.Tensor):
        # Invalidate the posterior scale to force re-computation when prior changes.
        self._posterior_scale = None
        if isinstance(prior_precision, (float, int)):
            self._prior_precision = torch.tensor([prior_precision], device=self.device)
        elif isinstance(prior_precision, torch.Tensor):
            self._prior_precision = prior_precision.reshape(-1).to(self.device)
        else:
            raise ValueError("prior_precision must be a scalar or a torch.Tensor.")

    @property
    def sigma_noise(self) -> torch.Tensor:
        """
        Observation noise standard deviation for regression tasks.

        Stored as a 1D tensor with shape (1,) if originally a scalar.

        Returns
        -------
        torch.Tensor
            Noise standard deviation (regression only).
        """
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise: float | torch.Tensor):
        # Invalidate the posterior scale to force re-computation.
        self._posterior_scale = None
        if isinstance(sigma_noise, (float, int)):
            self._sigma_noise = torch.tensor([sigma_noise], device=self.device)
        elif isinstance(sigma_noise, torch.Tensor):
            self._sigma_noise = sigma_noise.reshape(-1).to(self.device)
        else:
            raise ValueError("sigma_noise must be a scalar or a torch.Tensor.")
