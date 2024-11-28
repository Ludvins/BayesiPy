from typing import Callable

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from tqdm import tqdm

from bayesipy.fmgp.kernels import LastLayerNTK_SquaredExponential, SquaredExponential
from bayesipy.utils import gaussian_logdensity, safe_inverse
from bayesipy.utils.metrics import score

from .utils import compute_length_scale_estimation


class FMGP_Base(torch.nn.Module):
    """Base class for the Uncertainty Estimation models.

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
        kernel: str | Callable,
        noise_variance: float = None,
        subrogate_regularizer: bool = True,
        inducing_locations=None,
        inducing_classes=None,
        num_inducing=None,
        y_mean: float = 0,
        y_std: float = 1,
        seed=0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.model = model
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

            auxiliar_param = list(model.parameters())[0]
            self.device = auxiliar_param.device
            self.dtype = auxiliar_param.dtype
        else:
            self.device = device
            self.dtype = dtype

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

        self.subrogate_regularizer = subrogate_regularizer

        self.y_mean = torch.tensor(y_mean).to(self.device).to(self.dtype)
        self.y_std = torch.tensor(y_std).to(self.device).to(self.dtype)

        self.construct_kernel = isinstance(kernel, str)
        self.kernel = kernel

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
            if isinstance(inducing_locations, list):
                self.num_inducing = inducing_locations[0].shape[0]
                self.inducing_locations = torch.nn.ParameterList(
                    [
                        torch.tensor(
                            inducing_locations[0], device=self.device, dtype=self.dtype
                        ),
                        torch.tensor(
                            inducing_locations[1], device=self.device, dtype=self.dtype
                        ),
                    ]
                )
            else:
                self.num_inducing = inducing_locations.shape[0]

                self.inducing_locations = torch.tensor(
                    inducing_locations, device=self.device, dtype=self.dtype
                )
                self.inducing_locations = torch.nn.Parameter(self.inducing_locations)

            if self.likelihood == "classification":
                self.inducing_classes = torch.tensor(
                    inducing_classes, device=self.device, dtype=torch.long
                )

        if subrogate_regularizer:
            self.q_mu = torch.zeros(
                (self.num_inducing, 1), device=self.device, dtype=self.dtype
            )
            self.q_mu = torch.nn.Parameter(self.q_mu)

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

        loss = self.loss(X, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print gradients

        return loss

    @torch.no_grad()
    def handle_input(self, X):
        if isinstance(X, list) and len(X) == 2:
            X, outputs = X
            X = X.to(self.device).to(self.dtype)
            outputs = outputs.to(self.device).to(self.dtype)
        elif self.model is not None:
            X = X.to(self.device).to(self.dtype)
            outputs = self.model(X)
        else:
            raise ValueError("No network or outputs provided")

        return X, outputs

    def set_input_shape(self, X):
        if isinstance(X, list):
            X = X[0]
        self.input_shape = X.shape[1:]

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

    def _subrogate_sgp_forward(self, X):
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

        # Shape (batch_size)
        Kx_diag = self.kernel(X, diag=True)
        batch_size = Kx_diag.shape[0]
        # Shape (batch_size, num_inducing)
        Kxz = self.kernel(X, self.inducing_locations)
        # Shape (num_inducing, num_inducing)
        Kzz = self.kernel(self.inducing_locations)

        if self.likelihood == "regression":
            # For regression problems add output dimensions to Kx and Kxz
            Kx_diag = Kx_diag.unsqueeze(-1).unsqueeze(-1)
            Kxz = Kxz.unsqueeze(-1)
        else:
            indices_expanded = self.inducing_classes.view(1, self.num_inducing, 1, 1)
            indices_expanded2 = indices_expanded.repeat(
                batch_size, 1, self.output_dim, 1
            )

            Kxz = torch.gather(Kxz, -1, indices_expanded2).squeeze(-1)

            indices_expanded = self.inducing_classes.view(1, self.num_inducing, 1, 1)
            indices_expanded = indices_expanded.repeat(
                self.num_inducing, 1, self.output_dim, 1
            )

            Kzz = torch.gather(Kzz, -1, indices_expanded).squeeze(-1)

            indices_expanded2 = self.inducing_classes.view(self.num_inducing, 1, 1)
            indices_expanded2 = indices_expanded2.repeat(1, self.num_inducing, 1)

            Kzz = torch.gather(Kzz, -1, indices_expanded2).squeeze(-1)

        # Pre-compute these values to avoid repeated computation
        self.Kz = Kzz
        self.compute_inducing_term(Kzz)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        diag = torch.einsum("nma, ml, nlb -> nab", Kxz, self.A, Kxz)
        # Shape [batch_size, output_dim, output_dim]
        Fvar = Kx_diag - diag

        if self.subrogate_regularizer:
            self.Kz_inv = torch.linalg.inv(
                self.Kz + 1e-3 * torch.eye(self.num_inducing).to(self.device)
            )
            aux = self.Kz_inv @ self.q_mu
            Q_mean = torch.einsum("nma, m -> na", Kxz, aux.squeeze(-1))
        else:
            Q_mean = torch.zeros((batch_size, self.output_dim), device=self.device)

        return Q_mean, Fvar

    def forward(self, X):
        # Compute predictive mean
        X, F_mean = self.handle_input(X)

        _, Fvar = self._subrogate_sgp_forward(X)

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
        X, F_mean = self.handle_input(X)
        _, F_var = self._subrogate_sgp_forward(X)
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

    def _compute_mean_term_KL(self):
        """Compute the KL divergence of the model.

        Returns
        -------
        torch Tensor of shape ()
            Contains the KL divergence of the model.

        """
        KL = 0.5 * self.q_mu.T @ self.Kz_inv @ self.q_mu
        KL = torch.sum(KL)
        return KL

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

        X, F_mean = self.handle_input(X)

        Q_mean, F_var = self._subrogate_sgp_forward(X)

        # Compute divergence term
        divergence = self.alpha_divergence(F_mean, F_var, y)

        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= y.shape[0]

        # Compute KL term
        KL_var = self._compute_variance_term_KL()
        loss = -scale * divergence + KL_var

        if self.subrogate_regularizer:
            _sgd_divergence = self.alpha_divergence(Q_mean, F_var, y)
            KL_mean = self._compute_mean_term_KL()

            loss += -scale * _sgd_divergence + KL_mean + KL_var

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
        self.inducing_locations = torch.nn.Parameter(self.inducing_locations)
        if self.likelihood == "classification":
            self.inducing_class = (
                torch.concatenate(classes, axis=0)
                .flatten()
                .to(self.device)
                .to(torch.long)
            )

    @torch.no_grad()
    def _initialize_kmeans_inducing_locations(self, loader):
        training_data = []
        training_targets = []
        for inputs, targets in loader:
            inputs = self.handle_input(inputs)[0]
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

    def _create_kernel(self, length_scale):
        if self.kernel == "RBF":
            self.kernel = SquaredExponential(
                initial_length_scale=np.log(length_scale),
                initial_amplitude=1,
                n_features=self.input_shape,
                n_outputs=self.output_dim,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError("Invalid Kernel")

    def fit(
        self,
        iterations,
        lr,
        train_loader,
        scheduler_gamma=None,
        scheduler_steps=None,
        val_loader=None,
        val_steps=100,
        metrics_cls=None,
        verbose=True,
        override=True,
    ):
        losses = []

        if override:
            data = next(iter(train_loader))
            X = data[0]
            try:
                model_output = self.handle_input(X[0])[1]
            except (TypeError, AttributeError, ValueError):
                model_output = self.handle_input(X[:1])[1]

            self.set_input_shape(X)

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

            if self.construct_kernel:
                print("Creating Kernel Function...", end=" ")
                length_scale = compute_length_scale_estimation(train_loader)
                self._create_kernel(length_scale)
                print("done")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if scheduler_gamma is not None:
            if scheduler_steps is None:
                scheduler_steps = len(train_loader)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=scheduler_gamma
            )
        elif scheduler_gamma is None:
            scheduler = None

        if verbose:
            # Initialize TQDM bar
            iters = tqdm(range(iterations), unit=" iteration")
            iters.set_description("Training ")
        else:
            iters = range(iterations)
        data_iter = iter(train_loader)

        if metrics_cls is not None:
            stored_metrics = []

        for i in iters:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)

            if scheduler is not None and i != 0 and i % scheduler_steps == 0:
                scheduler.step()

            # inputs = inputs.to(self.device).to(self.dtype)
            targets = targets.to(self.device).to(self.dtype)
            loss = self.train_step(optimizer, inputs, targets)

            losses.append(loss.detach().cpu().numpy())
            if val_loader is not None and (i == 0 or (i + 1) % val_steps == 0):
                stored_metrics.append(score(self, val_loader, metrics_cls))
                print(stored_metrics[-1])
            if verbose:
                iters.set_postfix(
                    {
                        "loss": np.around(losses[-1], 3),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    refresh=False,
                )

        if val_loader is not None:
            return losses, stored_metrics
        return losses

    def print_variables(self):
        """Prints the model variables in a prettier format."""

        print("\n---- MODEL PARAMETERS ----")
        np.set_printoptions(threshold=3, edgeitems=2)
        sections = []
        pad = "  "
        for name, param in self.named_parameters():
            if not param.requires_grad or "net" in name:
                continue
            name = name.split(".")
            for i in range(len(name) - 1):
                if name[i] not in sections:
                    print(pad * i, name[i].upper())
                    sections = name[: i + 1]

            padding = pad * (len(name) - 1)
            print(
                padding,
                "{}: ({})".format(name[-1], str(list(param.data.size()))[1:-1]),
            )
            print(
                padding + " " * (len(name[-1]) + 2),
                param.data.detach().cpu().numpy().flatten(),
            )

        print("\n---------------------------\n\n")


class FMGP_Embedding(FMGP_Base):
    def __init__(
        self,
        embedding: Callable,
        classifier: Callable,
        likelihood: str,
        kernel: str | Callable,
        noise_variance: float = None,
        subrogate_regularizer: bool = True,
        inducing_locations=None,
        inducing_classes=None,
        num_inducing=None,
        y_mean: float = 0,
        y_std: float = 1,
        seed=0,
        device=None,
        dtype=None,
    ):
        if not isinstance(inducing_locations, str):
            assert (
                isinstance(inducing_locations, list) and len(inducing_locations) == 2
            ), print("Inducing locations must be a list of two tensors")

        if classifier is not None:
            auxiliar_param = list(classifier.parameters())[0]
            device = auxiliar_param.device
            dtype = auxiliar_param.dtype

        super().__init__(
            model=None,
            likelihood=likelihood,
            kernel=kernel,
            noise_variance=noise_variance,
            subrogate_regularizer=subrogate_regularizer,
            inducing_locations=inducing_locations,
            inducing_classes=inducing_classes,
            num_inducing=num_inducing,
            y_mean=y_mean,
            y_std=y_std,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        if classifier is not None:
            if isinstance(classifier, torch.nn.Module):
                self.classifier = classifier.to(device).to(dtype)
                for param in self.classifier.parameters():
                    param.requires_grad = False
            else:
                self.classifier = classifier

        if embedding is not None:
            if isinstance(embedding, torch.nn.Module):
                self.embedding = embedding.to(device).to(dtype)
                for param in self.embedding.parameters():
                    param.requires_grad = False
            else:
                self.embedding = embedding

    @torch.no_grad()
    def handle_input(self, X):
        if isinstance(X, list) and len(X) == 3:
            X, embedding, outputs = X

            X = X.to(self.device).to(self.dtype)
            embedding = embedding.to(self.device).to(self.dtype)
            outputs = outputs.to(self.device).to(self.dtype)

        elif self.embedding is not None and self.classifier is not None:
            X = X.to(self.device).to(self.dtype)
            embedding = self.embedding(X)
            outputs = self.classifier(embedding)
        else:
            raise ValueError("No network or outputs provided")

        return [X, embedding], outputs

    @torch.no_grad()
    def _initialize_random_inducing_locations(self, loader):
        b = loader.batch_size
        iterator = iter(loader)
        inducing_locations = []
        classes = []
        while self.num_inducing > len(inducing_locations) * b:
            inputs, targets = next(iterator)
            if (len(inducing_locations) + 1) * b > self.num_inducing:
                inputs = inputs[: self.num_inducing - len(inducing_locations) * b]
                targets = targets[: self.num_inducing - len(inducing_locations) * b]
            inducing_locations.append(inputs)
            classes.append(targets)
        inducing_locations = (
            torch.concatenate(inducing_locations, axis=0).to(self.device).to(self.dtype)
        )

        inducing_locations_embedding = self.embedding(inducing_locations)
        self.inducing_locations = torch.nn.ParameterList(
            [
                torch.nn.Parameter(inducing_locations),
                torch.nn.Parameter(inducing_locations_embedding),
            ]
        )

        self.inducing_classes = (
            torch.concatenate(classes, axis=0).flatten().to(self.device).to(torch.long)
        )

    @torch.no_grad()
    def _initialize_kmeans_inducing_locations(self, loader):
        training_data = []
        training_targets = []
        for inputs, targets in iter(loader):
            inputs = self.handle_input(inputs)[0][0]
            training_data.append(inputs)
            training_targets.append(targets)
        training_data = torch.cat(training_data, axis=0)
        training_targets = torch.cat(training_targets, axis=0)

        if self.likelihood == "regression":
            inducing_locations = kmeans2(
                training_data, self.num_inducing, minit="points", seed=self.seed
            )[0]

            inducing_locations = (
                torch.tensor(self.inducing_locations).to(self.device).to(self.dtype)
            )
            inducing_locations_embedding = self.embedding(inducing_locations)
            self.inducing_locations = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(inducing_locations),
                    torch.nn.Parameter(inducing_locations_embedding),
                ]
            )
        else:
            if self.num_inducing < self.output_dim:
                raise ValueError(
                    "The number of inducing locations must be greater than the output dimension for KMEANS initialization"
                )
            inducing_locations = []
            inducing_locations_embedding = []
            self.inducing_classes = []
            for c in range(self.output_dim):
                s = training_data[training_targets.flatten() == c]
                s2 = self.embedding(s)
                s = s.detach().cpu().numpy()
                s2 = s2.detach().cpu().numpy()
                z = kmeans2(
                    s.reshape(s.shape[0], -1),
                    self.num_inducing // self.output_dim,
                    minit="points",
                    seed=self.seed,
                )[0]
                z2 = kmeans2(
                    s2.reshape(s2.shape[0], -1),
                    self.num_inducing // self.output_dim,
                    minit="points",
                    seed=self.seed,
                )[0]
                z = z.reshape(self.num_inducing // self.output_dim, *s.shape[1:])
                z = torch.tensor(z).to(self.device).to(self.dtype)
                z2 = z2.reshape(self.num_inducing // self.output_dim, *s2.shape[1:])
                z2 = torch.tensor(z2).to(self.device).to(self.dtype)

                inducing_locations.append(z)
                inducing_locations_embedding.append(z2)
                self.inducing_classes.append(
                    torch.ones(self.num_inducing // self.output_dim) * c
                )
            inducing_locations = torch.concatenate(inducing_locations)
            inducing_locations_embedding = torch.concatenate(
                inducing_locations_embedding
            )
            self.inducing_locations = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(inducing_locations),
                    torch.nn.Parameter(inducing_locations_embedding),
                ]
            )
            self.inducing_classes = (
                torch.concatenate(self.inducing_classes).to(self.device).to(torch.long)
            )

    def _create_kernel(self, length_scale):
        if self.kernel == "RBFxNTK":
            self.kernel = LastLayerNTK_SquaredExponential(
                initial_length_scale=np.log(length_scale),
                initial_amplitude=1,
                n_features=self.input_shape,
                n_outputs=self.output_dim,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            super()._create_kernel(length_scale)

    def _subrogate_sgp_forward(self, X):
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

        if isinstance(X, list) and len(X) == 2:
            X, embedding = X
        else:
            with torch.no_grad():
                embedding = self.embedding(X)

        return super()._subrogate_sgp_forward((X, embedding))

        # with record_function("Input Kernels"):
        # Shape (batch_size, output_dim, output_dim)
        Kx_diag = self.kernel((X, embedding), diag=True)
        # Shape (batch_size, num_inducing, output_dim, output_dim)
        Kxz = self.kernel((X, embedding), self.inducing_locations)
        # Shape (num_inducing, num_inducing, output_dim, output_dim)
        Kzz = self.kernel(self.inducing_locations)

        indices_expanded = self.inducing_classes.view(1, self.num_inducing, 1, 1)
        indices_expanded2 = indices_expanded.repeat(X.shape[0], 1, self.output_dim, 1)

        Kxz = torch.gather(Kxz, -1, indices_expanded2).squeeze(-1)

        indices_expanded = self.inducing_classes.view(self.num_inducing, 1, 1, 1)
        indices_expanded = indices_expanded.repeat(
            1, self.num_inducing, self.output_dim, 1
        )

        Kzz = torch.gather(Kzz, -1, indices_expanded).squeeze(-1)

        indices_expanded2 = self.inducing_classes.view(1, self.num_inducing, 1)
        indices_expanded2 = indices_expanded2.repeat(self.num_inducing, 1, 1)

        Kzz = torch.gather(Kzz, -1, indices_expanded2).squeeze(-1)

        # with record_function("Inducing terms"):
        self.Kz = Kzz
        self.compute_inducing_term(Kzz)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        diag = torch.einsum("nma, ml, nlb -> nab", Kxz, self.A, Kxz)
        # Shape [batch_size, output_dim, output_dim]

        Fvar = Kx_diag - diag

        if self.subrogate_regularizer:
            self.Kz_inv = safe_inverse(self.Kz)

            # Shape (num_inducing, output_dim)
            aux = self.Kz_inv @ self.q_mu

            Q_mean = torch.einsum("nma, m -> na", Kxz, aux.squeeze(-1))
        else:
            Q_mean = torch.zeros((X.shape[0], self.output_dim), device=self.device)

        return Q_mean, Fvar
