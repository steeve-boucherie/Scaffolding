"""Utility functions for easier handling of pandas' objects."""
import logging
from typing import Any, Callable, Dict, List, Literal, Mapping, Union

import numpy as np

import pandas as pd
from pandas import DataFrame, Timedelta  # , Index, Series
from pandas.core.groupby import DataFrameGroupBy


# LOGGER
logger = logging.getLogger(__name__)


# HELPERS
def _check_allowed(value: Any, allowed: List[Any]) -> None:
    """
    Check whether the input value os part of the allowed values, \
        and raise ValueError if otherwise.

    Parameters
    ----------
    value: Any
        Value to be tested.
    allowed: List[Any]
        List of allowed values.

    Returns
    -------
        None

    Raises
    ------
        ValueError
    """
    if value not in allowed:
        msg = f'Invalid valid value "{value}". Must be one of "{allowed}". ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def verify_columns(df: DataFrame, req_cols: Union[str, List[str]]) -> None:
    """
    Given a list of columns keys, \
        verify if they are valid column of the dataframe.

    Parameters
    ----------
    df: DataFrame
        Input dataframe.
    req_cols: str | List[str]
        Column key(s) to be tested.

    Raises
    ------
        KeyError
    """
    # Force input type
    req_cols = set([[req_cols], req_cols][isinstance(req_cols, list)])

    # Test
    all_cols = set(df.columns)
    if not req_cols.issubset(all_cols):
        diff = list(req_cols - all_cols)
        msg = f'The following mandatory columns(s) are absent from input ' \
              f'dataframe: "{diff}".\nPlease check your inputs.'
        logger.error(msg)
        raise KeyError(msg)


# FUNCTIONS
def rename_index(
    df: DataFrame,
    name: str,
    axis: Literal['index', 'columns'] = 'columns'
) -> DataFrame:
    """
    Rename the index or columns of the input single-level indexed dataframe.

    Parameters
    ----------
    df: DataFrame
        Input dataframe.
    name: str
        New name of the index/column.
    axis: 'index' | 'column'
        Whether the index or columns should be renamed.

    Returns
    -------
        DataFrame
    """
    # Check inputs
    _check_allowed(axis, ['index', 'columns'])

    # Rename
    if axis == 'index':
        df.index = df.index.rename(name)

    else:
        df.columns = df.columns.rename(name)

    return df


def collapse_columns(
    df: pd.DataFrame,
    join_str: str = '-'
) -> pd.DataFrame:
    """
    Given a multi-indexed columns dataframe, collaspe all level into one \
        using the join string.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe to be collapsed.
    join_str: str
        String to use for joining the columns levels.

    Returns
    -------
        pd.Dataframe
    """
    df.columns = [
        join_str.join([str(lvl) for lvl in levels])
        for levels in df.columns
    ]

    return df


def collapse_index(
    df: pd.DataFrame,
    join_str: str = '-'
) -> pd.DataFrame:
    """
    Given a multi-indexed columns dataframe, collaspe all level into one \
        using the join string.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe to be collapsed.
    join_str: str
        String to use for joining the columns levels.

    Returns
    -------
        pd.Dataframe
    """
    df.index = [join_str.join(levels) for levels in df.index]

    return df


def resample_multi_index(
    df: pd.DataFrame,
    index: Any,
    freq: str,
    convention: Literal['start', 'end'] = 'start'
) -> DataFrameGroupBy:
    """
    Given a multi-indexed dataframe, resample the specified column \
        at the desired  frequency.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be resampled.
    index: Any
        Name of the index-level to resample along.
    freq: str
        Frequency at which to resample dataframe.
    resamp_kw: Mapping
        Optional argument to pass to the pd.Grouper for the resampled index.

    Returns
    -------
    groups: DataFrameGroupBy
        Dataframe groups to be aggregated.
    """
    # Get the indices names
    indices = list(df.index.names)
    if index not in indices:
        msg = f'Selected index "{index}" is not a valid index of the ' \
              f'dataframe. Should be one of "{indices}".' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    # Verify convention
    allowed_values = ['start', 'end']
    if convention not in allowed_values:
        msg = f'Invalid value "{convention}" for parameter "convention". ' \
              f'Must be one of "{allowed_values}".' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    resamp_kw = {
        'label': {'start': 'left', 'end': 'right'}[convention],
        'closed': {'start': 'left', 'end': 'right'}[convention],
        'convention': convention,
    }

    groups = df.groupby([
        pd.Grouper(
            level=ind,
            freq=[None, freq][ind == index],
            **[{}, resamp_kw][ind == index]
        )
        for ind in indices
    ])

    return groups


def rolling_multi_index(
    df: pd.DataFrame,
    index: Any,
    window: Union[int, str, Timedelta],
    agg_fun: Union[
        str,
        Callable[[np.ndarray], float],
        List[Union[str, Callable[[np.ndarray], float]]],
        Dict[str, Union[str, Callable[[np.ndarray], float]]]
    ] = 'mean',
    roll_kw: Mapping[str, Any] = {},
    agg_kw: Mapping[str, Any] = {},
) -> DataFrameGroupBy:
    """
    Given a multi-indexed dataframe, compute rolling statistic along the \
        given level.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be resampled.
    index: Any
        Name of the index-level to perform rolling along.
    agg_fun: str | Callable | Dict[str, str | Callable]
        Aggregation function(s) to estimate ambient condition. \
        if a single string or callable is passed, then all fields are aggregated \
        the same way. If a dictionnary is passed then each fields is \
        aggregated using the callable given by agg_fun[field].
    window: int | str | timedelta
        Size of the rolling window. If a `int` then use constant window size. \
        If a `str`, `timedelta` or `BaseOffset` then it defines the time period \
        of the window. 
        See: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html#pandas.DataFrame.rolling  # noqa E501 
    roll_kw: Mapping
        Optional arguments to pass to the pandas rolling function.

    Returns
    -------
    groups: DataFrameGroupBy
        Dataframe groups to be aggregated.
    """
    # Get the indices names
    indices = list(df.index.names)
    if index not in indices:
        msg = f'Selected index "{index}" is not a valid index of the ' \
              f'dataframe. Should be one of "{indices}".' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    # Get data groups
    others = [ind for ind in indices if ind != index]
    groups = (
        df
        .reset_index()
        .sort_values(others + [index])
        .groupby(others)
    )

    # Compute rolling stats
    out = (
        groups
        .apply(lambda df: (
            df
            .drop(others, axis=1)
            .set_index(index)
            .rolling(window, **roll_kw)
            .aggregate(agg_fun, **agg_kw)
        ))
    )

    return out
