from .src.src import FMGP_Base, FMGP_Embedding


def FMGP(
    embedding=None,
    *args,
    **kwargs,
):
    if embedding is None:
        return FMGP_Base(*args, **kwargs)
    else:
        return FMGP_Embedding(embedding, *args, **kwargs)
