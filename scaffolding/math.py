"""Helper maths methods"""
from math import copysign


# UTILS
def sign(x: float) -> float:
    """
    Given a number return its sign.

    Parameters
    ----------
    x: float
        Input number

    Returns
    -------
        float
    """
    return copysign(1, x)


def soft_threshold(x: float, lmbd: float) -> float:
    """
    Given a number apply soft thresholding and return.

    Notes
    -----
    >>> y = sign(x) * max(abs(x) - lmbd, 0)

    Parameters
    ----------
    x: float
        Input number.
    lmbd: float
        The threshold value.

    Returns
    -------
        float
    """
    return sign(x) * max(abs(x) - lmbd, 0)
