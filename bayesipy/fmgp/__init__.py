from .src.solve import SOLVEFMGP
from .src.src import FMGP_Base, FMGP_Embedding


def FMGP(
    embedding=None,
    *args,
    covariance="standard",
    **kwargs,
):
    """Construct an FMGP model with the requested covariance structure.

    Parameters
    ----------
    covariance:
        ``"standard"`` selects the existing FMGP covariance. ``"solve"``
        selects the fixed-mean orthogonal covariance extension. The SOLVE
        implementation currently supports ordinary scalar regression only.
    """
    if covariance == "solve":
        if embedding is not None:
            raise NotImplementedError(
                "The SOLVE covariance currently supports FMGP regression "
                "without an embedding only."
            )
        return SOLVEFMGP(*args, **kwargs)

    if covariance != "standard":
        raise ValueError(
            "covariance must be either 'standard' or 'solve', "
            f"got {covariance!r}"
        )

    if embedding is None:
        return FMGP_Base(*args, **kwargs)
    return FMGP_Embedding(embedding, *args, **kwargs)


__all__ = ["FMGP", "FMGP_Base", "FMGP_Embedding", "SOLVEFMGP"]
