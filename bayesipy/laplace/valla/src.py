from typing import Callable

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from tqdm import tqdm

from bayesipy.utils import gaussian_logdensity, safe_cholesky
from bayesipy.utils.metrics import score

from .backpack_interface import BackPackInterface


class VaLLA(torch.nn.Module):
    """Base class for the Variational Linearized Laplace Approximation.

    Parameters
    ----------
    net : torch.nn.Module
        Contains the neural network model
    kernel : variational_uncertainty_estimation.kernels.base.Kernel
        Contains the kernel function
    output_dim : int
        Contains the number of output dimensions
    inducing_locations : np.ndarray of shape (num_inducing, input_dim)
        Contains the inducing locations
    track_inducing_locations : bool
        If True, the inducing locations are stored at each iteration
    num_data : int
        Contains the number of data points
    alpha : float
        Contains the alpha parameter
    device : torch.device
        Contains the device where the model is stored
    dtype : torch.dtype
        Contains the precision of the model
    """

    def __init__(
        self,
        model: Callable,
        likelihood: str,
        noise_variance: float = None,
        prior_precision: float = 1,
        inducing_locations=None,
        inducing_classes=None,
        num_inducing=None,
        y_mean: float = 0,
        y_std: float = 1,
        seed=0,
    ):
        super().__init__()

        self.model = [model]
        auxiliar_param = list(model.parameters())[0]
        self.device = auxiliar_param.device
        self.dtype = auxiliar_param.dtype

        self.log_prior_precision = (
            torch.log(torch.tensor(prior_precision)).to(self.device).to(self.dtype)
        )
        self.log_prior_precision = torch.nn.Parameter(self.log_prior_precision)
        assert likelihood in ["regression", "classification"]
        self.likelihood = likelihood
        if self.likelihood == "regression":
            if noise_variance is None:
                print("Initial noise variance set to 1")
                noise_variance = 1
            self.log_noise_variance = (
                torch.log(torch.tensor(noise_variance)).to(self.device).to(self.dtype)
            )
            self.log_noise_variance = torch.nn.Parameter(self.log_noise_variance)

        self.y_mean = torch.tensor(y_mean).to(self.device).to(self.dtype)
        self.y_std = torch.tensor(y_std).to(self.device).to(self.dtype)

        if isinstance(inducing_locations, str):
            if num_inducing is None:
                raise ValueError(
                    "If no inducing locations are provided \
                                 `num_inducing` must be provided"
                )
            else:
                self.initialize_inducing_locations = inducing_locations
                self.num_inducing = num_inducing

        else:
            self.initialize_inducing_locations = False
            self.num_inducing = inducing_locations.shape[0]

            self.inducing_locations = torch.tensor(
                inducing_locations, device=self.device, dtype=self.dtype
            )
            self.inducing_locations = torch.nn.Parameter(self.inducing_locations)

            if self.likelihood == "classification":
                # If inducing classes are provided, use them
                self.inducing_classes = torch.tensor(
                    inducing_classes, device=self.device, dtype=torch.long
                )
            else:
                self.inducing_classes = torch.zeros(
                    self.num_inducing, device=self.device, dtype=torch.long
                )

        # Initialize cholesky decomposition of identity
        eye = np.eye(self.num_inducing)

        # Initialize cholesky decomposition of identity
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        triangular_q_sqrt = eye[li, lj]
        # Shape (num_inducing, num_inducing)
        self.L = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )

        self.L = torch.nn.Parameter(self.L)
        self.seed = 2147483647 - seed
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

    def train_step(self, optimizer, X, y):
        """Performs a training step on the model.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Contains the optimizer to use
        X : torch Tensor of shape (batch_size, input_dim)
            Contains the input features
        y : torch Tensor of shape (batch_size, output_dim)
            Contains the target values

        Returns
        -------
        torch Tensor of shape ()
            Contains the loss of the model.
        """
        # If targets are unidimensional,
        # ensure there is a second dimension (N, 1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        X = X.to(self.device).to(self.dtype)
        y = y.to(self.device)

        loss = self.loss(X, y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        # print gradients

        return loss

    def compute_inducing_term(self, Kz):
        """Compute the auxiliar matrices H and A for the model.

        Parameters
        ----------
        Kz : torch Tensor of shape (num_inducing, num_inducing)
            Contains the kernel matrix of the inducing locations.
        """
        # Transform flattened cholesky decomposition parameter into matrix
        L = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        L[li, lj] = self.L

        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        # H = I + L^T @ self.Kz @ L
        eye = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        self.H = eye + torch.einsum("mn, ml, lk -> nk", L, Kz, L)
        # Shape [num_inducing, num_inducing]
        # A = L @ H^{-1} @ L^T
        # self.A = torch.einsum("nm, ml, kl -> nk", L, torch.inverse(self.H), L)
        self.A = L @ torch.linalg.solve(self.H, L.T)

    def forward(self, X):
        """Computes the predictive mean and variance of the model.

        Parameters
        ----------
        X : torch Tensor of shape (batch_size, input_dim)
            Contains the input features

        Returns
        -------
        F_mean : torch Tensor of shape (batch_size, output_dim)
            Contains the predictive mean of the model
        F_var : torch Tensor of shape (batch_size, output_dim, output_dim)
            Contains the predictive variance of the model
        """
        with torch.no_grad():
            F_mean = self.model[0](X)

        # Shape (batch_size)
        Jx = self.backend.jacobians(X, enable_back_prop=False)
        Jz = self.backend.jacobians_on_outputs(
            self.inducing_locations,
            self.inducing_classes.unsqueeze(-1),
            enable_back_prop=self.training,
        ).squeeze(1)
        var = 1 / torch.exp(self.log_prior_precision)
        Kx_diag = var * torch.einsum("nai, nbi -> nab", Jx, Jx)
        # Shape (batch_size, num_inducing)
        Kxz = var * torch.einsum("nai, mi -> nma", Jx, Jz)
        # Shape (num_inducing, num_inducing)
        Kzz = var * torch.einsum("mi, ni -> mn", Jz, Jz)

        # Pre-compute these values to avoid repeated computation
        self.Kz = Kzz
        self.compute_inducing_term(Kzz)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        diag = torch.einsum("nma, ml, nlb -> nab", Kxz, self.A, Kxz)
        # Shape [batch_size, output_dim, output_dim]
        Fvar = Kx_diag - diag

        return F_mean, Fvar

    def predict(self, X):
        """Computes the predictive mean and variance of the model.

        Parameters
        ----------
        X : torch Tensor of shape (batch_size, input_dim)
            Contains the input features

        Returns
        -------
        F_mean : torch Tensor of shape (batch_size, output_dim)
            Contains the predictive mean of the model
        F_var : torch Tensor of shape (batch_size, output_dim, output_dim)
            Contains the predictive variance of the model
        """
        # Compute predictive mean and variance
        F_mean, F_var = self(X)
        # Add Likelihood noise
        if self.likelihood == "regression":
            noise_variance = torch.exp(self.log_noise_variance)

            return F_mean * self.y_std + self.y_mean, (
                F_var.squeeze(-1) + noise_variance
            ) * self.y_std**2

        else:
            return F_mean, F_var

    def _compute_variance_term_KL(self):
        """Compute the KL divergence of the model.

        Returns
        -------
        torch Tensor of shape ()
            Contains the KL divergence of the model.

        """
        log_det = torch.logdet(self.H)
        trace = torch.sum(torch.diagonal(self.Kz @ self.A))
        KL = 0.5 * log_det - 0.5 * trace
        return torch.sum(KL)

    def loss(self, X, y):
        """Compute the loss of the model.

        Parameters
        ----------
        X : torch Tensor of shape (batch_size, input_dim)
            Contains the input features
        y : torch Tensor of shape (batch_size, output_dim)
            Contains the target values

        Returns
        -------
        torch Tensor of shape ()
            Contains the loss of the model.
        """
        F_mean, F_var = self(X)

        # Compute divergence term
        divergence = self.alpha_divergence(F_mean, F_var, y)

        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= y.shape[0]

        # Compute KL term
        KL_var = self._compute_variance_term_KL()
        loss = -scale * divergence + KL_var

        return loss

    def alpha_divergence(self, F_mean, F_var, y, alpha=1):
        """Compute the divergence of the model.

        Parameters
        ----------
        F_mean : torch Tensor of shape (batch_size, output_dim)
            Contains the predictive mean of the model
        F_var : torch Tensor of shape (batch_size, output_dim, output_dim)
            Contains the predictive variance of the model
        y : torch Tensor of shape (batch_size, output_dim)
            Contains the target values.

        Returns
        -------
        torch Tensor of shape ()
            Contains the divergence of the model.
        """
        if self.likelihood == "regression":
            # Black-box alpha-energy
            variance = torch.exp(self.log_noise_variance)
            # Proportionality constant
            C = (
                torch.sqrt(2 * torch.pi * variance / alpha)
                / torch.sqrt(2 * torch.pi * variance) ** alpha
            )

            logpdf = gaussian_logdensity(
                F_mean, F_var.squeeze(-1) + variance / alpha, y
            )
            logpdf = logpdf + torch.log(C)
            logpdf = logpdf / alpha
        else:
            # Compute scaled logits
            F = F_mean / torch.sqrt(
                1 + torch.pi / 8 * torch.diagonal(F_var, dim1=1, dim2=2)
            )
            # Compute probabilities
            probs = F.softmax(-1) ** alpha

            # Compute divergence
            logpdf = (
                -1
                / alpha
                * torch.nn.functional.cross_entropy(
                    probs.log(), y.to(torch.long).squeeze(-1), reduction="none"
                )
            )

        # # Aggregate on data dimension
        return torch.sum(logpdf)

    @torch.no_grad()
    def _initialize_random_inducing_locations(self, loader):
        b = loader.batch_size
        iterator = iter(loader)
        inducing_locations = []
        if self.likelihood == "classification":
            classes = []

        while len(inducing_locations) * b < self.num_inducing:
            inputs, targets = next(iterator)
            if (len(inducing_locations) + 1) * b > self.num_inducing:
                inputs = inputs[: self.num_inducing - len(inducing_locations) * b]
                targets = targets[: self.num_inducing - len(inducing_locations) * b]
            inducing_locations.append(inputs)
            if self.likelihood == "classification":
                classes.append(targets)
        self.inducing_locations = (
            torch.concatenate(inducing_locations, axis=0).to(self.device).to(self.dtype)
        )
        self.inducing_locations = torch.nn.Parameter(inducing_locations)
        if self.likelihood == "classification":
            self.inducing_class = (
                torch.concatenate(classes, axis=0)
                .flatten()
                .to(self.device)
                .to(torch.long)
            )
        else:
            self.inducing_classes = torch.zeros(
                self.num_inducing, device=self.device, dtype=torch.long
            )

    @torch.no_grad()
    def _initialize_kmeans_inducing_locations(self, loader):
        training_data = []
        training_targets = []
        for inputs, targets in loader:
            training_data.append(inputs)
            training_targets.append(targets)
        training_data = torch.cat(training_data, axis=0)
        training_targets = torch.cat(training_targets, axis=0)

        if self.likelihood == "regression":
            self.inducing_locations = kmeans2(
                training_data.detach().cpu().numpy(),
                self.num_inducing,
                minit="points",
                seed=self.seed,
            )[0]

            self.inducing_locations = (
                torch.tensor(self.inducing_locations).to(self.device).to(self.dtype)
            )
            self.inducing_locations = torch.nn.Parameter(self.inducing_locations)
            self.inducing_classes = torch.zeros(
                self.num_inducing, device=self.device, dtype=torch.long
            )
        else:
            if self.num_inducing < self.output_dim:
                raise ValueError(
                    "The number of inducing locations must be greater than the output dimension for KMEANS initialization"
                )
            self.inducing_locations = []
            self.inducing_classes = []
            for c in range(self.output_dim):
                s = training_data[training_targets.flatten() == c]
                s = s.detach().cpu().numpy()
                z = kmeans2(
                    s.reshape(s.shape[0], -1),
                    self.num_inducing // self.output_dim,
                    minit="points",
                    seed=self.seed,
                )[0]
                z = z.reshape(
                    self.num_inducing // self.output_dim, *training_data.shape[1:]
                )
                z = torch.tensor(z).to(self.device).to(self.dtype)

                self.inducing_locations.append(z)
                cls = torch.ones(self.num_inducing // self.output_dim) * c
                self.inducing_classes.append(cls)

            self.inducing_locations = torch.concatenate(self.inducing_locations)
            self.inducing_locations = torch.nn.Parameter(self.inducing_locations)
            self.inducing_classes = (
                torch.concatenate(self.inducing_classes).to(self.device).to(torch.long)
            )

    def fit(
        self,
        iterations,
        lr,
        train_loader,
        val_loader=None,
        val_steps=100,
        metrics_cls=None,
        verbose=True,
        override=True,
    ):
        losses = []

        if override:
            data = next(iter(train_loader))
            X = data[0].to(self.device).to(self.dtype)
            try:
                model_output = self.model[0](X[0])
            except (TypeError, AttributeError, ValueError):
                model_output = self.model[0](X[:1])

            self.output_dim = model_output.shape[-1]
            self.num_data = len(train_loader.dataset)

            if self.initialize_inducing_locations == "random":
                print("Initializing inducing locations...", end=" ")
                self._initialize_random_inducing_locations(train_loader)
                print("done")

            elif self.initialize_inducing_locations == "kmeans":
                print("Initializing inducing locations...", end=" ")
                self._initialize_kmeans_inducing_locations(train_loader)
                print("done")

        self.backend = BackPackInterface(
            model=self.model[0], output_dim=self.output_dim
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if val_loader is not None:
            best_score = torch.tensor(np.inf, device=self.device, dtype=self.dtype)

        if metrics_cls is not None:
            stored_metrics = []

        if verbose:
            # Initialize TQDM bar
            iters = tqdm(range(iterations), unit=" iteration")
            iters.set_description("Training ")
        else:
            iters = range(iterations)
        data_iter = iter(train_loader)

        for i in iters:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)

            inputs = inputs.to(self.device).to(self.dtype)
            targets = targets.to(self.device).to(self.dtype)
            loss = self.train_step(optimizer, inputs, targets)

            losses.append(loss.detach().cpu().numpy())

            if val_loader is not None:
                if i % val_steps == 0:
                    cur_score = self._validate(val_loader)
                    if metrics_cls is not None:
                        stored_metrics.append(
                            score(self, val_loader, metrics_cls, verbose)
                        )

                    print(cur_score, best_score)
                    if cur_score < best_score:
                        best_score = cur_score
                    else:
                        break

        if metrics_cls is not None:
            return losses, stored_metrics
        return losses

    def _validate(self, val_loader):
        NLL = 0
        n = 0
        with torch.no_grad():
            # Batches evaluation
            for inputs, target in val_loader:
                target = target.to(self.device).to(self.dtype)
                inputs = inputs.to(self.device).to(self.dtype)

                Fmean, Fvar = self.predict(inputs)

                if self.likelihood == "regression":
                    nll = -gaussian_logdensity(
                        Fmean.squeeze(), Fvar.squeeze(), target.squeeze()
                    )
                else:
                    chol = safe_cholesky(Fvar)
                    z = torch.randn(
                        2048,
                        Fmean.shape[0],
                        Fvar.shape[-1],
                        generator=self.generator,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)

                    probs = samples.softmax(-1)
                    F = probs.mean(0).log()
                    nll = torch.nn.functional.cross_entropy(
                        F, target.to(torch.long).squeeze(-1), reduction="none"
                    )
                NLL += nll.sum()
                n += target.shape[0]

            return NLL / n
