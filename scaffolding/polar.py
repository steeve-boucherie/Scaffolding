"""Helper fonctions for circular/polar operations"""
from typing import Any, Dict, Union

import numpy as np

import pandas as pd

from scipy import stats

import xarray as xr


# UTILITIES FUNCTIONS
def circdiff(
    x: Union[float, np.ndarray, pd.Series, xr.DataArray],
    y: Union[float, np.ndarray, pd.Series, xr.DataArray],
) -> Union[float, np.ndarray, pd.Series, xr.DataArray]:
    """
    Find the smallest angular distance between the two inputs angles.

    Notes
    -----
    Angles must be in DEGREES

    Parameters
    ----------
    x: float | np.ndarray | pd.Series | xr.DataArray
        first input angle.
    b: float | np.ndarray | pd.Series | xr.DataArray
        second input angle.

    Returns
    -------
        float | np.ndarray | pd.Series | xr.DataArray
    """
    # a = (x - y) % 360
    # b = (y - x) % 360

    # res = -a if a < b else b
    # return res
    return ((x - y) % 360 + 180) % 360 - 180


def circsum(
    a: Union[float, np.ndarray, pd.Series, xr.DataArray],
    b: Union[float, np.ndarray, pd.Series, xr.DataArray],
) -> Union[float, np.ndarray, pd.Series, xr.DataArray]:
    """
    Sum the two angles and return result within [0, 360]

    Notes
    -----
    Angles must be in DEGREES

    Parameters
    ----------
    a: float | np.ndarray | pd.Series | xr.DataArray
        first input angle.
    b: float | np.ndarray | pd.Series | xr.DataArray
        second input angle.

    Returns
    -------
        float | np.ndarray | pd.Series | xr.DataArray
    """
    return (a + b) % 360


def circmean_deg(
    arr: np.ndarray,
    axis: int = -1,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Perform circular average of the input array in degrees.

    Notes
    -----
    This is a wrapper around scipy.circmean which perform deg2rad and \
    rad2deg conversion before and after.

    Parameters
    ----------
    arr: np.ndarray
        Input array to aggregate
    axis: int
        Axis on the input array to aggregate.
    kwargs: Dict[str, Any]
        Optional arguments to pass to scipy.stats.circmean.
    """
    # TODO: write tests
    # Axis should not de higher than the number of dims

    # Deg to rad conversion
    out = np.deg2rad(arr)

    # Aggregation
    kwargs.setdefault('nan_policy', 'omit')
    out = stats.circmean(out, axis=axis, **kwargs)

    # Rad to deg conversion
    out = np.rad2deg(out)

    return out


def circstd_deg(
    arr: np.ndarray,
    axis: int = -1,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Perform circular stddev of the input array in degrees.

    Notes
    -----
    This is a wrapper around scipy.circstd which perform deg2rad and \
    rad2deg conversion before and after.

    Parameters
    ----------
    arr: np.ndarray
        Input array to aggregate
    axis: int
        Axis on the input array to aggregate.
    kwargs: Dict[str, Any]
        Optional arguments to pass to scipy.stats.circmean.
    """
    # TODO: write tests
    # Axis should not de higher than the number of dims

    # Deg to rad conversion
    out = np.deg2rad(arr)

    # Aggregation
    kwargs.setdefault('nan_policy', 'omit')
    out = stats.circstd(out, axis=axis, **kwargs)

    # Rad to deg conversion
    out = np.rad2deg(out)

    return out


def circvar_deg(
    arr: np.ndarray,
    axis: int = -1,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Perform circular variance of the input array in degrees.

    Notes
    -----
    This is a wrapper around scipy.circmean which perform deg2rad and \
    rad2deg conversion before and after.

    Parameters
    ----------
    arr: np.ndarray
        Input array to aggregate
    axis: int
        Axis on the input array to aggregate.
    kwargs: Dict[str, Any]
        Optional arguments to pass to scipy.stats.circmean.
    """
    # TODO: write tests
    # Axis should not de higher than the number of dims

    # Deg to rad conversion
    out = np.deg2rad(arr)

    # Aggregation
    kwargs.setdefault('nan_policy', 'omit')
    out = stats.circvar(out, axis=axis, **kwargs)

    # Rad to deg conversion
    out = np.rad2deg(out)

    return out
