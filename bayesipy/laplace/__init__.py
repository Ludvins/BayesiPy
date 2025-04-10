from __future__ import annotations

import torch
from laplace.baselaplace import ParametricLaplace

from .ella import ELLA
from .valla import VaLLA
from .test import TestLaplace

__all__ = ["Laplace", "ELLA", "VaLLA", "TestLaplace"]


def Laplace(
    model: torch.nn.Module,
    likelihood: str,
    subset_of_weights: str = "last_layer",
    hessian_structure: str = "kron",
    y_mean=0,
    y_std=1,
    *args,
    **kwargs,
) -> ParametricLaplace:
    """Simplified Laplace access using strings instead of different classes.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : Likelihood or str in {'classification', 'regression'}
    subset_of_weights : SubsetofWeights or {'last_layer', 'subnetwork', 'all'}, default=SubsetOfWeights.LAST_LAYER
        subset of weights to consider for inference
    hessian_structure : HessianStructure or str in {'diag', 'kron', 'full', 'lowrank'}, default=HessianStructure.KRON
        structure of the Hessian approximation
    y_mean : float
        Contains the normalization mean of the targets.
    y_std : float
        Contains the normalization standard deviation of the tarets.

    Returns
    -------
    laplace : ParametricLaplace
        chosen subclass of ParametricLaplace instantiated with additional arguments
    """
    if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
        raise ValueError(
            "Subnetwork Laplace requires a full or diagonal Hessian approximation!"
        )

    laplace_map = {
        subclass._key: subclass
        for subclass in _all_subclasses(ParametricLaplace)
        if hasattr(subclass, "_key")
    }
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
    instance = laplace_class(model, likelihood, *args, **kwargs)

    # Set device and dtype attributes
    p = next(model.parameters())
    instance.device = p.device
    instance.dtype = p.dtype

    # Create predict method that computes glm_predictive and scales the output
    # using the provided normalization constants.
    def predict(self, X):
        F_mean, F_var = self._glm_predictive_distribution(X)
        if self.likelihood == "regression":
            noise_variance = self.sigma_noise**2
            F_var = F_var.squeeze(-1) + noise_variance
        y_std_t = torch.tensor(y_std).to(F_mean.device).to(F_mean.dtype)
        y_mean_t = torch.tensor(y_mean).to(F_mean.device).to(F_mean.dtype)
        return F_mean * y_std_t + y_mean_t, F_var * y_std_t**2

    # Set that method to the instance
    instance.predict = predict.__get__(instance)
    return instance


def _all_subclasses(cls) -> set:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)]
    )
