"""Fixed-mean SOLVE-GP covariance extension for regression experiments."""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2

from bayesipy.utils import safe_cholesky

from .src import FMGP_Base


class SOLVEFMGP(FMGP_Base):
    """FMGP with a zero-mean variational posterior for an orthogonal GP.

    The pretrained network remains the complete posterior mean.  The original
    FMGP inducing variables parameterize the projected-process covariance, and
    ``orthogonal_inducing_locations`` parameterize a second, zero-mean
    variational posterior under the residual kernel

    ``C(x, x') = k(x, x') - k(x, Z) K(Z, Z)^-1 k(Z, x')``.

    This experimental implementation deliberately supports scalar regression
    only.  It inherits FMGP's training objective and optimizer so comparisons
    differ only in the covariance family.

    Parameters
    ----------
    orthogonal_inducing_locations:
        ``"kmeans"``, ``"random"``, or an explicit array of locations.
    num_orthogonal_inducing:
        Number of residual-process inducing locations when an initialization
        strategy is supplied.
    """

    _CHOLESKY_EPS = 1e-8

    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: str,
        kernel: str | Callable,
        noise_variance: Optional[float] = -1,
        inducing_locations=None,
        num_inducing=None,
        orthogonal_inducing_locations="kmeans",
        num_orthogonal_inducing=None,
        y_mean: float = 0,
        y_std: float = 1,
        alpha: float = 1.0,
        seed: int = 0,
    ) -> None:
        if likelihood != "regression":
            raise NotImplementedError(
                "The isolated SOLVEFMGP experiment supports regression only."
            )
        if model is None:
            raise ValueError("SOLVEFMGP requires a pretrained regression model.")

        super().__init__(
            model=model,
            likelihood=likelihood,
            kernel=kernel,
            noise_variance=noise_variance,
            inducing_locations=inducing_locations,
            num_inducing=num_inducing,
            y_mean=y_mean,
            y_std=y_std,
            alpha=alpha,
            seed=seed,
        )

        if isinstance(orthogonal_inducing_locations, str):
            if orthogonal_inducing_locations not in {"kmeans", "random"}:
                raise ValueError(
                    "orthogonal_inducing_locations must be 'kmeans', 'random', "
                    "or an explicit array"
                )
            if num_orthogonal_inducing is None:
                raise ValueError(
                    "num_orthogonal_inducing is required for initialized "
                    "orthogonal inducing locations"
                )
            self.initialize_orthogonal_inducing_locations = (
                orthogonal_inducing_locations
            )
            self.num_orthogonal_inducing = int(num_orthogonal_inducing)
        elif isinstance(
            orthogonal_inducing_locations,
            (list, np.ndarray, torch.Tensor),
        ):
            locations = torch.as_tensor(
                orthogonal_inducing_locations,
                device=self.device,
                dtype=self.dtype,
            )
            if locations.ndim < 2:
                raise ValueError(
                    "Explicit orthogonal inducing locations must have a batch "
                    "dimension and at least one feature dimension."
                )
            self.initialize_orthogonal_inducing_locations = False
            self.num_orthogonal_inducing = int(locations.shape[0])
            self.orthogonal_inducing_locations = torch.nn.Parameter(locations)
        else:
            raise TypeError(
                "orthogonal_inducing_locations must be a strategy string or array"
            )

        if self.num_orthogonal_inducing <= 0:
            raise ValueError("num_orthogonal_inducing must be positive")

    def _initialize_parameters(self) -> None:
        """Initialize FMGP and whitened orthogonal covariance parameters."""
        super()._initialize_parameters()

        m = self.num_orthogonal_inducing
        li, lj = torch.tril_indices(m, m, device=self.device)
        diagonal = li == lj
        eps = self._cholesky_eps()
        inverse_softplus_one = math.log(math.expm1(1.0 - eps))

        raw = torch.zeros(len(li), device=self.device, dtype=self.dtype)
        raw[diagonal] = inverse_softplus_one
        self.orthogonal_raw_cholesky = torch.nn.Parameter(raw)

    def _cholesky_eps(self) -> float:
        if self.dtype == torch.float32:
            return 1e-6
        return self._CHOLESKY_EPS

    def _orthogonal_cholesky(self) -> torch.Tensor:
        """Return a lower triangular factor with strictly positive diagonal."""
        m = self.num_orthogonal_inducing
        li, lj = torch.tril_indices(m, m, device=self.device)
        diagonal = li == lj
        values = torch.where(
            diagonal,
            F.softplus(self.orthogonal_raw_cholesky) + self._cholesky_eps(),
            self.orthogonal_raw_cholesky,
        )
        chol = torch.zeros((m, m), device=self.device, dtype=self.dtype)
        chol[li, lj] = values
        return chol

    @torch.no_grad()
    def _collect_training_inputs(self, loader) -> torch.Tensor:
        inputs = []
        for batch_inputs, _ in iter(loader):
            handled_inputs = self.handle_input(batch_inputs)[0]
            inputs.append(handled_inputs)
        return torch.cat(inputs, dim=0)

    @torch.no_grad()
    def _select_locations(
        self,
        training_inputs: torch.Tensor,
        count: int,
        strategy: str,
        seed: int,
    ) -> torch.Tensor:
        if count > training_inputs.shape[0]:
            raise ValueError(
                f"Cannot select {count} inducing locations from "
                f"{training_inputs.shape[0]} training examples."
            )

        if strategy == "kmeans":
            centers = kmeans2(
                training_inputs.detach().cpu().numpy(),
                count,
                minit="points",
                seed=seed,
            )[0]
            return torch.as_tensor(centers, device=self.device, dtype=self.dtype)

        if strategy == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            indices = torch.randperm(
                training_inputs.shape[0], generator=generator
            )[:count]
            return training_inputs[indices.to(training_inputs.device)].clone()

        raise ValueError(f"Unknown inducing-location strategy: {strategy!r}")

    @torch.no_grad()
    def _initialize_both_inducing_sets(self, loader, primary_strategy: str) -> None:
        training_inputs = self._collect_training_inputs(loader)

        primary = self._select_locations(
            training_inputs,
            self.num_inducing,
            primary_strategy,
            self.seed,
        )
        self.inducing_locations = torch.nn.Parameter(primary)

        if self.initialize_orthogonal_inducing_locations:
            # Adjacent deterministic seeds can still select different initial
            # points while preserving reproducibility across experiment arms.
            orthogonal_seed = self.seed - 1 if self.seed > 0 else self.seed + 1
            orthogonal = self._select_locations(
                training_inputs,
                self.num_orthogonal_inducing,
                self.initialize_orthogonal_inducing_locations,
                orthogonal_seed,
            )
            self.orthogonal_inducing_locations = torch.nn.Parameter(orthogonal)

    @torch.no_grad()
    def _initialize_kmeans_inducing_locations(self, loader) -> None:
        self._initialize_both_inducing_sets(loader, "kmeans")

    @torch.no_grad()
    def _initialize_random_inducing_locations(self, loader) -> None:
        self._initialize_both_inducing_sets(loader, "random")

    def _orthogonal_features(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute whitened residual cross-covariances and ``C(O, O)``."""
        Z = self.inducing_locations
        orthogonal_locations = self.orthogonal_inducing_locations

        Kzz = self.kernel(Z)
        Kzz = 0.5 * (Kzz + Kzz.T)
        Kzo = self.kernel(Z, orthogonal_locations)
        Kxz = self.kernel(X, Z)
        Kxo = self.kernel(X, orthogonal_locations)
        Koo = self.kernel(orthogonal_locations)

        Kzz_cholesky = safe_cholesky(Kzz)
        Kzz_inverse_Kzo = torch.cholesky_solve(Kzo, Kzz_cholesky)

        Cxo = Kxo - Kxz @ Kzz_inverse_Kzo
        Coo = Koo - Kzo.T @ Kzz_inverse_Kzo
        Coo = 0.5 * (Coo + Coo.T)
        Coo_cholesky = safe_cholesky(Coo)

        # B = C(X, O) C(O, O)^(-1/2).  With Coo = R R^T,
        # B^T = R^-1 C(O, X).
        B = torch.linalg.solve_triangular(
            Coo_cholesky,
            Cxo.T,
            upper=False,
        ).T
        return B, Coo

    def _variational_variance(self, X: torch.Tensor) -> torch.Tensor:
        base_variance = super()._variational_variance(X)
        B, Coo = self._orthogonal_features(X)

        orthogonal_cholesky = self._orthogonal_cholesky()
        S_orthogonal = orthogonal_cholesky @ orthogonal_cholesky.T
        identity = torch.eye(
            self.num_orthogonal_inducing,
            device=self.device,
            dtype=self.dtype,
        )
        covariance_difference = S_orthogonal - identity
        correction = torch.einsum(
            "bi,ij,bj->b", B, covariance_difference, B
        )

        # Keep these terms available for diagnostics and mathematical tests.
        self.Coo = Coo
        self.B_orthogonal = B
        self.S_orthogonal = S_orthogonal
        self.orthogonal_variance_correction = correction

        return base_variance + correction[:, None, None]

    def _compute_variance_term_KL(self) -> torch.Tensor:
        base_kl = super()._compute_variance_term_KL()
        orthogonal_cholesky = self._orthogonal_cholesky()
        trace = torch.sum(orthogonal_cholesky.square())
        log_determinant = 2.0 * torch.sum(
            torch.log(torch.diagonal(orthogonal_cholesky))
        )
        orthogonal_kl = 0.5 * (
            trace - log_determinant - self.num_orthogonal_inducing
        )
        return base_kl + orthogonal_kl
