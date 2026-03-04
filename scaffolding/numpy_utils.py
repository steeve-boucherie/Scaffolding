"""Utility functions for easier handling of numpy arrays."""
import logging
from typing import List, Literal, Optional, Tuple, Union

import numpy as np


# PAKCAGE IMPORT
from scaffolding.io import is_allowed


# LOGGER
logger = logging.getLogger(__name__)


# FUNCTIONS
def get_nth_element(
    arr: np.ndarray,
    values: Optional[np.ndarray] = None,
    axis=-1,
    rank: Union[int, List[int]] = 0,
    order: Literal['ascending', 'descending'] = 'ascending'
) -> np.ndarray:
    """
    Get the index that sort the input array along the given axis,
    and return the values corresponding to the specified rank(s).

    Parameters
    ----------
    arr: np.array
        Array to be sorted
    values: np.ndarray
        Array of values to return. Uses the values in arr \
            if None is passed.
    axis: int
        Axis along which the data will be sorted.
    rank: int | List[int]
        Rank(s) of the values to be returned.
    order: 'ascending' | 'descending'
        Option to sort the data by ascending or descending order.
        Default is ascending.

    Returns
    -------
        np.ndarray
    """
    # Test inputs
    allowed = ['ascending', 'descending']
    if order not in allowed:
        msg = f'Invalid value for parameter "order". Must be one of {allowed}, ' \
              f'but received {order}. Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    # Assign default
    values = [values, arr][values is None]

    # Get the indices that sort arr
    fact = {'ascending': 1, 'descending': -1}[order]
    ind = np.argsort(fact * arr, axis=axis)

    # Get the corresponding element in values
    out = np.take(
        np.take_along_axis(values, ind, axis=axis),
        indices=rank,
        axis=axis
    )

    return out


def is1d(arr: np.ndarray) -> None:
    """
    Test if the input array is 1-dimensional, \
        raises ValueError if otherwise.

    Parameters
    ----------
    arr: np.ndarray
        Input array

    Returns
    -------
        None

    Raises
    ------
    ValueError:
        - if arr.ndim != 1
    """
    if arr.ndim != 1:
        msg = 'Input array must de 1-dimensional. ' \
              f'Received array with shape "{arr.shape}". ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def equal_shape(a1: np.ndarray, a2: np.ndarray) -> None:
    """
    Given two arrays, test if the have the same shape, \
        and raise ValueError if otherwise.

    Parameters
    ----------
    a1: np.ndarray
        Input array #1.
    a2: np.ndarray
        Input array #2.

    Raises
    ------
        ValueError
    """
    if a1.shape != a2.shape:
        msg = f'Shape mismatch between inputs array.\nArray #1 has shape {a1.shape},' \
              f'while array #2 has shape {a2.shape}.\n' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def cbrt(arr: np.ndarray) -> np.ndarray:
    """
    Return the cubic root of the input array.

    Parameters
    ----------
    arr: np.ndarray
        Input array.

    Returns
    -------
        np.ndarray
    """
    return np.sign(arr) * np.abs(arr)**(1 / 3)


def custom_ci(arr: np.ndarray, level: float = 95.) -> Tuple[float, float]:
    """
    Given the array of estimated statistic samples, \
        return the associated confidence interval.

    Notes
    -----
    Bounds of the  interval is computed as the percentile centered around \
        the median as described  below :
        - min: q = 50 - ci / 2
        - max: q = 50 + ci / 2

    Parameters
    ----------
    arr: np.ndarray
        Input array of estimated statistic samples.
    level: float
        Value defining the confidence level, must be between 0 and 100.
    """
    # Check input
    if not (0 <= level <= 100):
        msg = 'Invalid value for confidence interval. Must be between 0 and 100, ' \
              f'but received "{level}". Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    bounds = [50 - level / 2, 50 + level / 2]
    return (np.nanpercentile(arr, q) for q in bounds)


def peak_to_peak(
    arr: np.ndarray,
    axis: Optional[int] = None,
    nan_policy: Literal['omit', 'propagate'] = 'omit'
) -> np.ndarray:
    """
    Compute the peak to peak value along the given axis.

    Parameters
    ----------
    arr: np.ndarray
        Input array.
    axis: int | None
        Axis along which to compute statistics.
    nan_policy: 'omit' | 'propagate'
        Option to deal with NaN values. Ignore them is set to "omit" or \
        propagate them is set to "propagate".
    """
    # Compute corresponding interval
    if nan_policy == 'omit':
        res = np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)

    elif nan_policy == 'propagate':
        res = np.max(arr, axis=axis) - np.min(arr, axis=axis)

    else:
        allowed = ['omit', 'propagate']
        msg = 'Invalid value for "nan_policy" parameter. ' \
              f'Expected one of "{allowed}", but received "{nan_policy}". ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    return res


def round_to_n(
    arr: np.ndarray,
    n: int,
    option: Literal['right', 'center', 'left'] = 'left'
) -> np.ndarray:
    """
    Given an array of float, round the values to the nearest multiple of "n".

    Parameters
    ----------
    arr: np.ndarray
        Input array of float/int values.
    n: int
        Integer to round the values to.
    option: 'center' | 'left'
        Option whether the multiple of n should be the center or the left edge \
        of the bin:
        If set to "center" then all values within [N - n/2, N + n/2) \
            where N = m*n will be allocated to N.
        If set to "left"  then all values within [N, N + n) \
            where N = m*n will be allocated to N.
        If set to "right"  then all values within (N - n, N + n] \
            where N = m*n will be allocated to N.


    Returns
    -------
        np.ndarray
    """
    allowed = ['right', 'center', 'left']
    if option not in allowed:
        msg = f'Invalid value for parameter option. Expected one of "{allowed}", ' \
              f'but received "{option}". Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    offset = {'right': n - 1e6, 'center': n / 2, 'left': 0}[option]
    res = ((arr + offset) // n) * n

    return res


def np_ffill(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Given an array of numeric values, replace NaN elements using forward fill, \
        along the selected axis.

    Notes
    -----
    No idea how this works...
    See: https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array  # noqa: E501

    Parameters
    ----------
    arr: np.ndarray
        Input array
    axis: int
        The axis along which the ffill should be performed.

    Returns
    -------
        np.ndarray
    """

    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [
        np.arange(k)[
            tuple([
                slice(None) if dim == i else np.newaxis
                for dim in range(len(arr.shape))
            ])
        ]
        for i, k in enumerate(arr.shape)
    ]
    slc[axis] = idx

    return arr[tuple(slc)]


def abs_perc(
    arr: np.ndarray,
    q: float,
    axis: Optional[int] = None,
    nan_policy: Literal['omit', 'propagate'] = 'omit'
) -> Union[np.ndarray, float]:
    """
    Given an array of numeric values, compute the given percentile \
        of the absolutes of the array.

    Parameters
    ----------
    arr: np.ndarray
        Input array.
    q: float
        The selected percentile.
    axis: int
        The axis along which to perform the aggregation.
    nan_policy: 'omit' |'propagate'
        A string defining how to handle NaNs in the input array. If "omit" nans \
        are ingored (use nan_percentile). If "propagate" any NaN in the input \
        will cause the result to be NaN as well.

    Returns
    -------
        float | np.ndarray
    """
    # Verify
    is_allowed(nan_policy, ['omit', 'propagate'])

    agg_fun = {
        'omit': np.nanpercentile,
        'propagate': np.percentile,
    }[nan_policy]

    return agg_fun(np.abs(arr), q=q, axis=axis)


def soft_threshold_vec(x: np.ndarray, lmbd: float) -> np.ndarray:
    """
    Given an array, apply soft thresholding to all its elements.

    Parameters
    ----------
    x: np.ndarray
        Input array.
    lmbd: float
        Threshold value.

    Returns
    -------
        np.ndarray
    """
    diff = np.abs(x) - lmbd
    return np.sign(x) * np.where(diff > 0, diff, 0)
