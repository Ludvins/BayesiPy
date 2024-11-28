import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from bayesipy.utils.metrics import score
from bayesipy.utils.utils import gaussian_logdensity, safe_cholesky, safe_inverse

from .utils import SoDSampler, subsample_balanced, psd_safe_eigen


class ELLA(torch.nn.Module):
    def __init__(
        self,
        model,
        likelihood,
        subsample_size,
        n_eigenvalues,
        prior_precision=1,
        sigma_noise=0,
        y_mean=0,
        y_std=1,
        seed=1234,
    ) -> None:
        """Initialize the ELLA model.

        Parameters
        ----------
        model : torch.nn.Module
            The predictive model (e.g., a neural network).
        likelihood : torch.nn.Module
            The likelihood type; "regression" or "classification".
        subsample_size : int
            The size of the subsample to use for the SoD approximation.
        n_eigenvalues : int
            The number of eigenvalues to compute in the SoD approximation.
        prior_precision : float
            The precision of the prior distribution.
        sigma_noise : float
            The standard deviation of the gaussian likelihood.
        y_mean : float, optional
            The mean of the target variable for normalization.
        y_std : float, optional
            The standard deviation of the target variable for normalization.
        seed : int, optional
            The random seed for reproducibility. Default is 1234.
        """
        super(ELLA, self).__init__()

        # Save model and set to evaluation mode
        self.model = model
        self.model.eval()

        # Store the parameters of the model in a separate dictionary
        self.params = {name: p for name, p in self.model.named_parameters()}
        self.n_params = sum(p.numel() for p in self.params.values())

        # Store device and dtype
        self.device = next(iter(self.params.values())).device
        self.dtype = next(iter(self.params.values())).dtype

        # Save constants
        self.likelihood = likelihood
        assert self.likelihood in ["regression", "classification"]
        self.M = subsample_size
        self.K = n_eigenvalues

        # Initialice hyper.parameters
        self._prior_precision = prior_precision
        self._sigma_noise = sigma_noise

        # Store normalization parameters
        self.y_mean = torch.tensor(y_mean).to(self.device).to(self.dtype)
        self.y_std = torch.tensor(y_std).to(self.device).to(self.dtype)

        # Set random seed
        self.seed = seed
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # Set Hessian approximation to None
        self.G = None

    @torch.no_grad()
    def fit(
        self,
        train_loader,
        val_loader=None,
        val_steps=-1,
        balanced = False,
        update_prior_precision = False,
        weight_decay = None,
        metrics_cls=None,
        verbose=False,
    ):
        """Fit the ELLA model to the training data.

        This method initializes the model's output dimension based on the first batch of
        the training data and performs necessary setup steps.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader containing the training data.
        val_loader : torch.utils.data.DataLoader, optional
            DataLoader containing the validation data.
            If provided, the model early-stops when the NLL in this dataset worsens.
            Default is None.
        val_steps : int, optional
            Number of train loader batches computed between evaluation steps. Default is
            no evaluation.
        metrics_cls : bayesipy.utils.metrics class
            If provided, these metris are computed every val_steps and returned.
        verbose : bool, optional
            If True, prints verbose output during training. Default is False.
        """
        # Take an input from the iterator to obtain a sample output.
        data = next(iter(train_loader))
        X = data[0]
        batch_size = X.shape[0]
        try:
            out = self.model(X[:1].to(self.device).to(self.dtype))
        except (TypeError, AttributeError):
            out = self.model(X.to(self.device).to(self.dtype))

        # Store the output dimension of the model
        self.n_outputs = out.shape[-1]

        if balanced:
            xs, ys = subsample_balanced(train_loader, self.n_outputs, self.M, self.device, True)
            subsample_dataset = TensorDataset(xs, ys)
            sub_sample_loader = DataLoader(
                dataset=subsample_dataset,
                batch_size=train_loader.batch_size,
                shuffle=False,
            )
        else:
            # Get random subsample loader
            sub_sample_loader = DataLoader(
                dataset=train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=SoDSampler(N=len(train_loader.dataset), M=self.M, seed=self.seed),
                shuffle=False,
            )
        # Bould the dual parameters list
        self._build_dual_params_list(sub_sample_loader)
        
        #self._build_dual_params_list(self.params, xs, ys, num_classes=self.n_outputs, random=True, K=20, args=None, num_batches=100, verbose=True)

        # Initialize feature product matrix
        self.subset_feature_product = torch.zeros(self.K, self.K).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )

        train_loader = (
            tqdm(train_loader, desc="Iterating Training Data")
            if verbose
            else train_loader
        )

        if val_loader is not None:
            best_score = torch.tensor(np.inf, device=self.device, dtype=self.dtype)
        if metrics_cls is not None:
            stored_metrics = []

        for i, (x, y) in enumerate(train_loader):
            # Take batch of data
            x = x.to(self.device).to(self.dtype)
            y = y.to(self.device)
            # Compute the features and the output
            Psi_x, logits = self._features(x, return_output=True)

            # For softmax classification, the hessian of the likelihood can be comptuted
            # as the product of the softmax probabilities
            if self.likelihood == "classification":
                prob = logits.softmax(-1)
                Delta_x = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
            # For regression, the hessian of the likelihood is the identity matrix
            #  divided by the noise variance. Which can be taken out of the addition.
            #  This parameter is added later in build_approximate_GGN.
            else:
                Delta_x = torch.ones(1, 1, device=self.device, dtype=self.dtype)
                Delta_x = Delta_x[None, :, :].expand(x.shape[0], -1, -1)
            # Update the feature product matrix
            self.subset_feature_product += torch.einsum(
                "bok,boj,bjl->kl", Psi_x, Delta_x, Psi_x
            )

            if val_loader is not None and (i == 0 or (i + 1) % val_steps == 0):
                if update_prior_precision:
                    self._prior_precision = torch.tensor([weight_decay * (i + 1) * batch_size], device=self.device)
                    
                # Build matrix using initial hyper-parameters
                self.build_approximate_GGN()
                # Get validation NLL
                val_score = self._validate(val_loader)

                # Compute metrics
                if metrics_cls is not None:
                    stored_metrics.append(
                        score(self, val_loader, metrics_cls, verbose = False)
                    )
                # If val NLL improves store it, break otherwise
                if val_score < best_score:
                    best_score = val_score
                else:
                    break

        # Leave the matrix built.
        self.build_approximate_GGN()

        if metrics_cls is not None:
            return stored_metrics

        return i * batch_size

    @torch.no_grad()
    def _validate(self, val_loader):
        """Computes the NLL of the provided loader.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            DataLoader containing the data.

        Returns
        -------
        nll : float
            Contains the NLL of the data.
        """
        # Initialize nll and total points.
        nll = 0
        n = 0
        # Loop through the data
        for x, y in val_loader:
            x = x.to(self.device).to(self.dtype)
            y = y.to(self.device)
            # Compute predictive distribution
            F_mean, F_var = self.predict(x)

            # For regression compute the likelihood in closed form
            if self.likelihood == "regression":
                nll += -gaussian_logdensity(
                    F_mean.squeeze(), F_var.squeeze(), y.squeeze()
                ).sum()

            # For classification, use MonteCarlo estimation.
            else:
                # Cholesky Facotization
                chol = safe_cholesky(F_var)
                # Standard Gaussian Samples of shape (n_samples, batch_size, output_size)
                z = torch.randn(
                    512,
                    F_mean.shape[0],
                    F_var.shape[-1],
                    generator=self.generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                # Re-Parameterization Trick
                logit_samples = F_mean + torch.einsum("sna, nab -> snb", z, chol)

                # Apply softmax to logits
                prob_samples = logit_samples.softmax(-1)
                # Average MC
                mean = prob_samples.mean(0)
                # Compute CE loss (NLL)
                nll += torch.nn.functional.cross_entropy(
                    mean.log(), y.to(torch.long).squeeze(-1), reduction="sum"
                )
            n += x.shape[0]
        return nll / n

    @torch.no_grad()
    def optimize_hyperparameters(self, val_loader, grid, verbose=False):
        """Finds the best hyper-parameters that minimize the NLL on a validation
        set.

        Parameters
        ----------
        val_loader : torch.data.DataLoader
            Contains the validation set
        grid : list of float or list of tuple of floats
            Contains the list of prior precisions (and sigma noises) for validation.
        verbose : boolean


        """
        # Initialize best score and params
        best_score = torch.tensor(np.inf, device=self.device, dtype=self.dtype)
        best_params = None

        # Initialize TQDM
        grid = tqdm(grid, desc="Hyperparameter Grid") if verbose else grid

        for param in grid:
            if self.likelihood == "regression":
                # The first parameter is set without _ to avoid the double
                #  computation of the matrix. See the setter for more information
                self._prior_precision = param[0]
                self.sigma_noise = param[1]
            else:
                self.prior_precision = param

            # Compute score
            score = self._validate(val_loader)
            # Set TQDM postfix
            if verbose:
                grid.set_postfix(
                    {
                        "score": score.item(),
                        "param": param,
                        "best_score": best_score.item(),
                        "best_param": best_params,
                    }
                )
            # Update best configuration.
            if score < best_score:
                best_score = score
                best_params = param

        # Set the best params to the model
        if self.likelihood == "regression":
            self.prior_precision = best_params[0]
            self.sigma_noise = best_params[1]
        else:
            self.prior_precision = best_params

    @torch.no_grad()
    def _build_dual_params_list(self, subsample_loader):
        """Build the list of dual parameters for the Nystrom approximation.

        Parameters
        ----------
        subsample_loader : torch.utils.data.DataLoader
            DataLoader containing the subsample data.
        """
        # Initialize matrix to store the subset kernel
        mat = torch.zeros(self.M, self.M, device=self.device)
        s = subsample_loader.batch_size
        
        indices = torch.empty(self.M).random_(self.n_outputs).long()

        # Generate iterartor for the subsample loader
        iterator1 = iter(subsample_loader)
        # Iterate in batches of size s
        for i in tqdm(range(0, self.M, s), desc="Computing Subset Kernel"):
            # Take batch
            X, _ = next(iterator1)
            # Take end index of batch
            ii = i + X.shape[0]
            # Compute Jacobian
            Js = self._slice_jacobian(X, indices[i:ii])

            # Generate second iterator for the subsample loader
            iterator2 = iter(subsample_loader)
            for j in range(0, i, s):
                # Take batch
                X2, _ = next(iterator2)
                # Take end index of batch
                jj = j + X2.shape[0]
                # Compute Jacobian
                Js2 = self._slice_jacobian(X2, indices[j:jj])

                # Compute the subset kernel
                mat[i:ii, j:jj] = Js @ Js2.T

                # Fill the upper triangular part of the matrix
                if j < i:
                    mat[j:jj, i:ii] = mat[i:ii, j:jj].T

        # Delete the Jacobians as they are no longer needed
        del Js, Js2

        # Compute the eigenvalues and eigenvectors of the subset kernel
        eigvals, eigvecs = psd_safe_eigen(mat)
        # Get the K largest eigenvalues and eigenvectors
        eigvals = eigvals[range(-1, -(self.K + 1), -1)]
        eigvecs = eigvecs[:, range(-1, -(self.K + 1), -1)]

        # Compute scaled eigenvectors
        U = eigvecs.div(eigvals.sqrt()).to(self.device).to(self.dtype)
        # Compute scaled eigenvalues
        eigvals = eigvals / self.M
        
        # Initialize the dual parameters
        V = torch.zeros(self.n_params, self.K, device=U.device)
        
        # Generate iterator for the subsample loader
        iterator1 = iter(subsample_loader)
        for i in tqdm(range(0, self.M, s), desc="Computing Dual Parameters"):
            # Take batch
            X, _ = next(iterator1)
            # Take end index of batch
            ii = min(i + s, self.M)
            # Compute Jacobian
            Js = self._slice_jacobian(X, indices[i:ii])
            # Compute the dual parameters
            V += Js.T @ U[i:ii]
        del Js

        # Store the dual parameters as a list of dictionaries
        self.dual_params_list = []
        for item in V.T:
            dual_params = {}
            start = 0
            for name, param in self.params.items():
                dual_params[name] = item[start : start + param.numel()].view_as(param)
                start += param.numel()
            self.dual_params_list.append(dual_params)


    @torch.enable_grad()
    def _slice_jacobian(self, X, Y):
        """Compute the Jacobian of the model with respect to the input at the
        given outputs.

        Parameters
        ----------
        X : torch.Tensor
            The input data.
        Y : torch.Tensor
            The output data.

        Returns
        -------
        J : torch.Tensor
            Constains the Jacobians w.r.t. the parameters of the model
        """
        Js = []
        for x, y in zip(X, Y):
            x = x.to(self.device).to(self.dtype)
            y = y.to(self.device)
            o = self.model(x.unsqueeze(0))
            self.model.zero_grad()
            if self.likelihood == "classification":
                identity = torch.eye(o.shape[-1]).to(x.device)

                grad_in = identity[y.item()].view(1, -1)
                o.backward(grad_in)
            else:
                o.backward()

            g = torch.cat([p.grad.flatten() for p in self.model.parameters()])
            Js.append(g)
        return torch.stack(Js)

    def _find_module_by_name(self, name):
        names = name.split(".")
        module = self.model
        for n in names[:-1]:
            module = getattr(module, n)
        return module, names[-1]

    @torch.no_grad()
    def _features(self, x_batch, return_output=False):
        with fwAD.dual_level():
            Jvs = []
            for dual_params in self.dual_params_list:
                for name, param in self.params.items():
                    module, name_p = self._find_module_by_name(name)
                    delattr(module, name_p)
                    setattr(module, name_p, fwAD.make_dual(param, dual_params[name]))

                output, Jv = fwAD.unpack_dual(self.model(x_batch))
                Jvs.append(Jv)
        Jvs = torch.stack(Jvs, -1)
        if return_output:
            return Jvs, output
        else:
            return Jvs

    def build_approximate_GGN(self):
        """Build the approximate Generalized Gauss-Newton matrix."""
        # For regression, the likelihood hessian was not considered during training
        if self.likelihood == "regression":
            G = self.subset_feature_product / (self._sigma_noise**2)
        else:
            G = self.subset_feature_product
        # Add the prior precision to the diagonal
        G = self.subset_feature_product + self.prior_precision * torch.eye(
            self.K, device=self.device
        )
        # Invert the matrix
        self.G = safe_inverse(G)

    @torch.no_grad()
    def forward(self, x):
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        F_mean : torch.Tensor
            The mean of the output.
        F_var : torch.Tensor
            The variance of the output.
        """
        # If the GGN matrix has not been computed, build it
        if self.G is None:
            self.build_approximate_GGN()

        # Compute the features
        x = x.to(self.device).to(self.dtype)
        Psi_x, F_mean = self._features(x, return_output=True)
        # Compute the variance
        F_var = Psi_x @ self.G.unsqueeze(0) @ Psi_x.permute(0, 2, 1)

        return F_mean, F_var

    def predict(self, x):
        """Predict the output of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        F_mean : torch.Tensor
            The mean of the output.
        F_var : torch.Tensor
            The variance of the output.
        """
        F_mean, F_var = self(x)
        if self.likelihood == "regression":
            F_var = F_var.squeeze(-1)
            F_var += self._sigma_noise**2
        return F_mean * self.y_std + self.y_mean, F_var * self.y_std**2

    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision):
        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.tensor([prior_precision], device=self.device)
            self.build_approximate_GGN()
        else:
            raise ValueError("Prior precision must be scalar.")

    @property
    def sigma_noise(self):
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise):
        if self.likelihood == "regression":
            if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
                self._sigma_noise = torch.tensor([sigma_noise], device=self.device)
                self.build_approximate_GGN()
            else:
                raise ValueError("sigma_noise must be scalar.")
        else:
            raise ValueError("sigma_noise is only available for regression.")
