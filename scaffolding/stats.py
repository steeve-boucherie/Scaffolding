"""Helper functions for statistics operations."""
import logging
# from functools import partial
from typing import Any, Dict, Literal, Mapping, Optional, Union

import numpy as np

from scipy import stats


# PACKAGE IMPORT
from scaffolding.io import is_allowed
from scaffolding.numpy_utils import is1d


# LOGGER
logger = logging.getLogger(__name__)


# METHODS
def pmean(
    arr: np.ndarray,
    axis: Optional[int] = -1,
    m: float = 2,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Perform power average of the input array.

    Notes
    -----
    pmean = (1/Nelem * sum(arr**m))**(1/m)

    Parameters
    ----------
    arr: np.ndarray
        Input array to aggregate
    axis: int
        Axis on the input array to aggregate.
    m: float | None
        Exponent of the power average
    kwargs: Dict[str, Any]
        Optional arguments to pass to np.mean.
    """
    # TODO: write tests

    out = np.nanmean(arr**m, axis=axis, **kwargs)**(1 / m)

    return out


def mode(
    arr: np.ndarray,
    axis: Optional[int] = -1,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Compute mode of the input array, along the given axis.

    Notes
    -----
    Wrapper around scipy.stats.mode

    Parameters
    ----------
    arr: np.ndarray
        Input array to aggregate.
    axis: int
        Axis along which aggregation will be performed.
    kwargs: Dict[str, Any]
        Optional arguments to pass to scipy.stats.mean.
    """
    # TODO: write tests

    # Set defaults
    kwargs.setdefault('nan_policy', 'omit')
    kwargs.setdefault('keepdims', False)

    out = stats.mode(arr, axis=axis, **kwargs)[0].data

    return out


def standardize(
    arr: np.ndarray,
    axis: Optional[int] = -1,
) -> np.ndarray:
    """
    Standardize input array.

    Notes
    -----
    Data standardization is done by substracting the mean sample \
        and dividing by standard deviation of the sample

    Parameters
    ----------
    arr: np.ndarray
        Input array to aggregate
    axis: int
        Axis on the input array to aggregate.
        Optional arguments to pass to np.mean.
    """
    out = (arr - arr.mean(axis)) / arr.std(axis)

    return out


def equally_spaced_bins(arr: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Given an array of values and a number of bins width, returns equally spaced bins \
        edges covering the whole data range.

    Parameters
    ----------
    arr: np.ndarray
        Input array of data
    n_bins: int
        Number of bins

    Returns
    -------
        np.ndarray
    """
    # Get bin edges
    edges = np.linspace(arr.min(), arr.max(), n_bins)

    # Expand left and right to ensure all data is captured
    delta = edges[1] - edges[0]
    edges[0] += -(delta / 100)
    edges[-1] += delta / 100

    return edges


def constant_width_bins(arr: np.ndarray, bin_width: float) -> np.ndarray:
    """
    Given an array of values and a fixed bin width, returns equally spaced bins \
        edges covering the whole data range.

    Parameters
    ----------
    arr: np.ndarray
        Input array of data
    n_bins: int
        Number of bins

    Returns
    -------
        np.ndarray
    """
    # Get bin edges
    edges = np.arange(arr.min(), arr.max() + bin_width, bin_width)

    # Expand left and right to ensure all data is captured
    delta = edges[1] - edges[0]
    edges[0] += -(delta / 100)
    edges[-1] += delta / 100

    return edges


def equally_populated_bins(
    arr: np.ndarray,
    n_bins: int,
    return_centers: bool = False,
    nan_policy: Literal['include', 'omit'] = 'omit',
    p_thresh: float = 1
) -> np.ndarray:
    """
    Given an array of values, returns the bins edges, covering the whole \
        data range, that will return equally populated bins.

    Parameters
    ----------
    arr: np.ndarray
        Input array of data.
    n_bins: int
        Number of bins.
    return_centers: bool
        Option whether to return the bin centers in addition to the bin edges.

    Returns
    -------
        np.ndarray
    """
    # Test inputs
    allowed_values = ['include', 'omit']
    if nan_policy not in allowed_values:
        msg = 'Invalid value for parameter "nan_policy". ' \
              f'Expected one of "{allowed_values}", but received "{nan_policy}". ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    agg_fun = {
        'include': np.percentile,
        'omit': np.nanpercentile,
    }[nan_policy]

    # Get bin edges
    p_edges = np.linspace(0 + p_thresh, 100 - p_thresh, n_bins)
    edges = agg_fun(arr, q=p_edges)

    # Expand left and right to ensure all data is captured
    delta = edges[1] - edges[0]
    edges[0] += -(delta / 100)
    edges[-1] += delta / 100

    if return_centers:
        centers = agg_fun(arr, q=(.5 * (p_edges[1:] + p_edges[:-1])))

        return edges, centers
    else:
        return edges


def edges2labels(edges: np.ndarray) -> np.ndarray:
    """
    Given the input array of bin edges, return the corresponding labels \
        taken as the bin centers.

    Notes
    -----
    Input array must be 1-dimensional.

    Parameters
    ----------
    edges: np.ndarray
        Input array of bin edges (must be 1d)

    Returns
    -------
        np.ndarray
    """
    # Verify inputs
    is1d(edges)

    return .5 * (edges[1:] + edges[:-1])


def bin_mean(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Given the input array and bin edges, return the corresponding bin \
        means to be used as labels.

    Notes
    -----
    Input array must be 1-dimensional.

    Parameters
    ----------
    x: np.ndarray
        Input array of values (must be 1d)
    edges: np.ndarray
        Input array of bin edges (must be 1d)

    Returns
    -------
        np.ndarray
    """
    # Verify inputs
    is1d(x)
    is1d(edges)

    # Drop out of bounds data
    x = x[~np.isnan(x) & (edges[0] <= x) & (x <= edges[-1])]

    # Get the indices and aggregate
    ind = np.digitize(x, edges)
    labels = [x[ind == (n + 1)].mean() for n in range(len(edges) - 1)]

    return labels


def standard_error(std: np.ndarray, count: np.ndarray) -> np.ndarray:
    """
    Given the arrays of estimated standard deviations and data counts, \
        compute the corresponding standard error.

    Notes
    -----
    The standard error on the mean of an estimated quantity is defined by \
    the formula:

    >>> se = std / np.sqrt(count)

    Parameters
    ----------
    std: np.darray
        An array of float, containing the estimated standard deviations.
    count: np.darray
        An array of float, containing the corresponding data counts.

    Returns
    -------
        np.ndarray
    """
    return std / np.sqrt(count)


def bias(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    **kwargs: Mapping[str, Any],
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding bias.

    Notes
    -----
    bias = mean(y_pred - y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int
        Axis to use for aggregation.

    Returns
    -------
        float
    """
    # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    return np.nanmean(y_pred - y_actual, axis=axis, **kwargs)


def bias_perc(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding bias normalized by the actual mean.

    Notes
    -----
    bias_perc = mean(y_pred - y_actual) / mean(y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int
        Axis to use for aggregation.

    Returns
    -------
        float
    """
    # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    num = np.nanmean(y_pred - y_actual, axis=axis)
    den = np.nanmean(y_actual, axis)

    return 100 * num / den


def mae(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Absolute Error (MAE).

    Notes
    -----
    mae = mean(abs(y_pred - y_actual))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]

    return np.nanmean(np.abs(err - offset), axis)


def mae_perc(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Absolute Error (MAE) normalized by the actual mean.

    Notes
    -----
    mae = mean(abs(y_pred - y_actual))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    center: bool
        An option to center the data (remove the bias). \
        Default is False

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]
    num = np.nanmean(np.abs(err - offset), axis)

    den = np.nanmean(y_actual, axis)

    return 100 * num / den


def mape(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Absolute Percentage Error (MAPE).

    Notes
    -----
    mape = mean(abs(y_pred / y_actual - 1))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    return 100 * np.nanmean(np.abs(y_pred / y_actual - 1), axis=axis)


def mpe(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Percentage Error (MPE).

    Notes
    -----
    mpe = mean(y_pred / y_actual - 1)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    return 100 * np.nanmean((y_pred / y_actual - 1), axis=axis)


def rmse(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Root Mean Square Error (RMSE).

    Notes
    -----
    rmse = sqrt(mean((y_pred - y_actual)**2))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]

    return np.nanmean((err - offset)**2, axis)**.5


def cov(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
    unit: Literal['fraction', 'percent'] = 'percent',
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Coefficient of variations (CoV).

    Notes
    -----
    >>> cov = sqrt(mean((y_pred - y_actual)**2)) / mean(y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int | None
        (Optional) Axis along which to compute the aggregation \
        aggrgate all axes if None is passed.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.
    unit: 'fraction' | 'percent'
        The unit in which the result should be displayed.

    Returns
    -------
        float
    """
    # Check inputs
    is_allowed(unit, ['percent', 'fraction'])
    fact = {'percent': 100, 'fraction': 1}[unit]
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]
    num = np.nanmean((err - offset)**2, axis)**.5
    den = np.nanmean(y_actual, axis)

    return fact * num / den


def nrmse(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
    unit: Literal['fraction', 'percent'] = 'percent',
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute the corresponding Normalize Root Mean Square Error (NRMSE).

    Notes
    -----
    >>> nrmse = sqrt(mean((y_pred - y_actual)**2)) / std(y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int | None
        (Optional) Axis along which to compute the aggregation \
        aggrgate all axes if None is passed.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.
    unit: 'fraction' | 'percent'
        The unit in which the result should be displayed.

    Returns
    -------
        float
    """
    # Check inputs
    is_allowed(unit, ['percent', 'fraction'])
    fact = {'percent': 100, 'fraction': 1}[unit]
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]
    num = np.nanmean((err - offset)**2, axis)**.5
    den = np.nanstd(y_actual, axis)

    return fact * num / den


def rsquare(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> Union[float, np.ndarray]:
    """
    Given arrays of predicted and actual values, \
        compute corresponding R-square (R2).

    Notes
    -----
    r2 = 1 - mean(y_pred - y_actual)**2 / var(y_pred)**2

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    # Get the terms
    ss_res = ((y_pred - y_actual)**2).sum(axis=axis)
    ss_tot = ((y_actual - y_actual.mean())**2).sum(axis=axis)

    return 1 - ss_res / ss_tot


def add_noise(
    arr: np.ndarray,
    level: float = 1,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Given an array apply random noise to the data.

    Parameters
    ----------
    arr: np.ndarray
        Input array
    level: float
        The magnitude of the noise.
    randim_seed: int
        The seed number to use for reproducibility of the results.

    Returns
    -------
        np.ndarray
    """
    rng = np.random.RandomState(random_seed)

    return arr + level * (rng.random(arr.shape) - .5)


def kde_eval_1d(
    arr: np.ndarray,
    x: np.ndarray,
    nan_policy: Literal['omit', 'propagate'] = 'omit',
    **kwargs: Mapping[str, Any]
) -> np.ndarray:
    """
    Given a one-dimensional array of numerical values, compute the kernel \
        density estimate (KDE) and estimate the pdf the given points.

    Notes
    -----
    Wrapper around scipy.stats.gaussian_kde method with the option to drop NaN \
    if the input array count any.
    Only works for 1d array.

    Parameters
    ----------
    arr: np.ndarray
        An array of float, defining the samples to be used for the KD estimation. \
        The array must be 1-dimensional.
    x: np.ndarray
        An array of float, defining the point at which the pdf will be estimated. \
        The array must be 1-dimensional.
    nan_policy: 'omit' | 'propagate'
        An option for whether to drop the NaN from the input array to avoid \
        ValueError during KD estimation.
    kwargs: Mapping[str, Any]
        A mapper of the form param_name -> param_value of optional parameters \
        to be passed to the `scipy.stats.gaussian_kde` method. Look function \
        doc for more info:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Returns
    -------
        np.ndarray
    """
    # Check inputs
    is1d(arr)
    is1d(x)
    is_allowed(nan_policy, ['omit', 'propagate'])

    # Get data
    arr = {
        'omit': arr[np.isfinite(arr)],
        'propagate': arr
    }[nan_policy]

    return stats.gaussian_kde(arr, **kwargs).pdf(x)
