"""Utility methods to handle time data."""
import logging
from typing import Literal

import numpy as np

import pandas as pd


# PAKCAGE IMPORT
from scaffolding.io import is_allowed
from scaffolding.numpy_utils import is1d


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def days_in_period(
    dates: np.ndarray,
    freq: Literal['yearly', 'quarterly', 'monthly', 'weekly'],
) -> np.ndarray:
    """
    Given an array of timestamps and a frequency, compute the number of  \
        days in the periods containing the timestamps.

    Parameters
    ----------
    dates: np.ndarray
        Input array of timestamps. Must de 1-dimensional
    freq: 'yearly' | 'quarterly' | 'monthly' | 'weekly'
        The frequency, defining the period length.

    Returns
    -------
        np.ndarray
    """
    # Check inputs
    is1d(dates)
    is_allowed(freq, ['yearly', 'quarterly', 'monthly', 'weekly'])

    # Get the frequency
    freq = {
        'yearly': 'Y',
        'quarterly': 'Q',
        'monthly': 'M',
        'weekly': 'W'
    }[freq]

    n_days = np.array([
        (ts.end_time - ts.start_time).days + 1
        for ts in pd.PeriodIndex(dates, freq=freq)
    ])

    return n_days
