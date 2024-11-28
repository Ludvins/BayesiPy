import math
import warnings
import random
import torch
import numpy as np

def assert_reproducibility(seed: int):

    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gaussian_logdensity(mu, var, x):
    """Computes the log density of a one dimensional Gaussian distribution
    of mean mu and variance var, evaluated on x.

    Parameters
    ----------
    mu : torch Tensor of shape (batch_size, output_dim)
        Contains the mean of the Gaussian distribution
    var : torch Tensor of shape (batch_size, output_dim)
        Contains the variance of the Gaussian distribution
    x : torch Tensor of shape (batch_size, output_dim)
        Contains the values to evaluate the log density

    Returns
    -------
    torch Tensor of shape (batch_size, output_dim)
        Contains the log density of the Gaussian distribution
    """
    # Compute log density
    return -0.5 * (math.log(2 * math.pi) + torch.log(var) + (mu - x) ** 2 / var)


def safe_inverse(A, out=None, jitter=None):
    """Compute the inverse of A. If A is not invertible, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the inverse of
        :attr:`out` (Tensor, optional):
            See torch.inverse
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is not invertible. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        return torch.inverse(A, out=out)
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            warnings.warn(
                f"inverse_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN.",
                RuntimeWarning,
            )
            return torch.randn_like(A)
        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10**i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                return torch.inverse(Aprime, out=out)
            except RuntimeError:
                continue
        raise e


def safe_cholesky(A, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, out=out)
        return L
    except Exception as e:
        isnan = torch.isnan(A)
        if isnan.any():
            warnings.warn(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN.",
                RuntimeWarning,
            )
            return torch.randn_like(A).tril()

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10**i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e
