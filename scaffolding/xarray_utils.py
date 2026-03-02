"""Utility functions for easier handling of  xarrays."""
import logging
import warnings
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np

import pandas as pd

import xarray as xr


# PACKAGE IMPORT
from pyward.utils.io import create_chunks, to_list
from pyward.utils.numpy_utils import get_nth_element
from pyward.utils.stats import edges2labels, equally_spaced_bins
from pyward import get_coord_mapper


# LOGGER
logger = logging.getLogger(__name__)


# DEFAULT
COORD_MAPPER = get_coord_mapper()


# FUNCTIONS
def nondim_coords(xr_obj: Union[xr.DataArray, xr.Dataset]) -> List[str]:
    """
    List all to coordinates that are not core dimension of the \
        input xarray object.

    Parameters
    ----------
    xr_obj: DataArray | Dataset
        The input xarray object to be processed.

    Returns
    -------
        List[str]
    """
    coords = [
        c for c in xr_obj.coords
        if c not in xr_obj.dims
    ]
    return coords


def drop_nondim_coords(
    xr_obj: Union[xr.DataArray, xr.Dataset],
    to_keep: Optional[Union[str, List[str]]] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Drop all the coordinations that are not dimension of the input xarray object.

    Parameters
    ----------
    xr_obj: DataArray | Dataset
        The input xarray object to be processed.
    to_keep: str | List[str] | None
        (Optional) element(s) to be kept. Leave to None to drop all.
    """
    # Verify inputs
    to_keep = [[], to_list(to_keep)][to_keep is not None]

    # Drop
    xr_obj = xr_obj.drop([
        c for c in nondim_coords(xr_obj)
        if c not in to_keep
    ])

    return xr_obj


def check_required(
    ds: xr.Dataset,
    req_elems: Union[str, List[str]],
    elem_type: Literal['key', 'dim', 'coord', 'all', 'attr'] = 'key'
) -> None:
    """
    Verify that the required elements are present within the input dataset,
        raise KeyError, otherwise.

    Parameters
    ----------
    ds: xr.Dataset
        Input dataset to be tested.
    req_keys: str | List[str]
        Mandatory key(s)
    elem_type: 'key' | 'dim' | 'coord' | 'all'
        Element type to be tested.
        Default is 'key'

    Returns
    -------
        None

    Raises
    ------
    - KeyError:
        - if any of req_elems is missing from the input dataset.
    - ValueError:
        - if elem_type is not one of 'key' | 'dim' | 'coord' | 'all'
    """
    # Check input
    allowed_type = ['key', 'dim', 'coord', 'all', 'attr']
    if elem_type not in allowed_type:
        msg = 'Invalid value for parameter "elem" type. ' \
              f'Expected one of "{allowed_type}", but received "{elem_type}". ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    # Force input type
    req_elems = [[req_elems], req_elems][isinstance(req_elems, list)]
    ds = [ds, xr.merge([ds])][isinstance(ds, xr.DataArray)]

    all_elems = {
        'key': ds.keys(),
        'dim': ds.dims,
        'coord': ds.coords,
        'all': ds.variables,
        'attr': ds.attrs.keys()
    }[elem_type]
    all_elems = set(all_elems)

    req_elems = set(req_elems)
    if not req_elems.issubset(all_elems):
        diff = list(req_elems - all_elems)
        msg = f'The following mandatory {elem_type}(s) are absent from input ' \
              f'dataset: "{diff}".\nPlease check your inputs.'
        logger.error(msg)
        raise KeyError(msg)


def get_nth_along_dim(
    da: xr.DataArray,
    dim: str,
    values: xr.DataArray | xr.Dataset | None = None,
    rank: Union[int, List[int]] = 0,
    order: Literal['ascending', 'descending'] = 'ascending'
) -> xr.DataArray:
    """
    Get the index that sort the input array along the given dimension,
    and return the values corresponding to the specified rank(s).

    Notes
    -----
    Wrapper around corresponding numpy method get_nth_element method.
    See pyward.utils.numpy_utils.get_nth_element for more details.

    Parameters
    ----------
    arr: np.array
        Array to be sorted
    values: xr.DataArray | xr.Dataset | None
        Array(s) of values to return. Uses the values in arr \
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
    # Assign default
    values = [values, da][values is None]

    # Rename values if it as the same name as the dimension
    try:
        # This raises an Attribute error if values is a Dataset
        name = values.name
    except AttributeError:
        is_dataarray = False
    else:
        is_dataarray = True
        if name == dim:
            values = values.rename('val')

    # Check inputs
    [check_required(da, dim, 'dim') for da in [da, values]]

    # Get elements
    dataarrays = xr.broadcast(da, values)
    elements: xr.DataArray = xr.apply_ufunc(
        get_nth_element,
        *dataarrays,
        input_core_dims=[[dim] for _ in range(len(dataarrays))],
        output_core_dims=[[[]], [['rank']]][isinstance(rank, list)],
        kwargs=dict(
            rank=rank,
            axis=-1,
            order=order
        ),
    )

    # Add rank dimension ?
    # TODO: Check this does not cause problem in yaw calibration.
    if isinstance(rank, list):
        elements = elements.assign_coords(rank=('rank', rank))

    # Rename
    if is_dataarray:
        elements = elements.rename(name)

    return elements


def safe_drop(ds: xr.Dataset, to_drop: List[str]) -> xr.Dataset:
    """
    Given a dataset and a list of keys, drop all keys from the list that \
        are valid key of the dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Input dataset to process.
    to_drop: List[str]
        List of keys to be dropped.

    Returns
    -------
        xr.Dataset
    """
    return ds.drop([key for key in to_drop if key in ds.keys()])


def single_key_ds(ds: xr.Dataset) -> Union[xr.DataArray, xr.Dataset]:
    """
    Given a dataset, convert to data array if it contains only a single key, \
        return the unmodified object otherwise.

    Parameters
    ----------
    ds: xr.Dataset
        Input dataset.

    Returns
    -------
        Dataset | DataArray
    """
    out_keys = list(ds.keys())
    out_keys = [out_keys, out_keys[0]][len(out_keys) == 1]

    return ds[out_keys]


def common_keys(*args: List[xr.Dataset]) -> List[str]:
    """
    Given the input datasets find the keys that are common to all and return \
        them in a list.

    Parameters
    ----------
    args: Tuple[xr.Dataset]
        The inputs datasets to concatenate.

    Returns
    -------
        List[str]
    """
    # Test inputs
    common_keys = set(args[0].keys())
    for ds in args[1:]:
        common_keys = common_keys.intersection(ds.keys())

    if common_keys == set():
        msg = 'The input datasets do not share any common key. Output will be empty.'
        logger.warning(msg)
        warnings.warn(msg, UserWarning)

    return list(common_keys)


def concat_on_common_keys(
    *args: List[xr.Dataset],
    dim: str,
    dim_val: Optional[List[Any]] = None
) -> xr.Dataset:
    """
    Concatenate the inputs datasets into a single object, using only the keys \
    that are common across all datasets.

    Parameters
    ----------
    args: List[xr.Dataset]
        The inputs datasets to concatenate.
    dim: str
        Dimension along which the dataset should be concatenated.
    dim_val: List[Any] | None
        Coordinate values to assign to the concat dimension.

    Returns
    -------
        xr.Dataset
    """
    # Force type to datasets (if needed)
    args = [
        [arg, xr.merge([arg])][isinstance(arg, xr.DataArray)]
        for arg in args
    ]

    # Test inputs
    common_keys = set(args[0].keys())
    for ds in args[1:]:
        common_keys = common_keys.intersection(ds.keys())

    if common_keys == set():
        msg = 'No common keys founds across al dataset. Cannot concatenate. ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    new_ds = xr.concat(
        [ds[common_keys] for ds in args],
        dim=dim
    )

    if dim_val is not None:
        if len(dim_val) != len(args):
            msg = 'When passed, parameter "dim_val" must be of the same length ' \
                f'as the datasets list to concatenate. Received "{len(args)}" ' \
                f'datasets and "{len(dim_val)}" coordinates values. ' \
                'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)

        new_ds = new_ds.assign_coords({dim: (dim, dim_val)})

    return new_ds


def concat_with_padding(
    *args: Tuple[xr.Dataset],
    dim: str,
    dim_val: Optional[List[Any]] = None,
    fill_val: Any = np.nan,
) -> xr.Dataset:
    """
    Concatenate the inputs datasets into a single object, for all individual
    dataset wiht missing keys, create variable and pad with NaN.

    Parameters
    ----------
    args: Tuple[xr.Dataset]
        The inputs datasets to concatenate.
    dim: str
        Dimension along which the dataset should be concatenated.
    dim_val: List[Any] | None
        Coordinate values to assign to the concat dimension.
    fill_val: Any
        Value to use to pad missing keys in datasets.

    Notes
    -----
    Dataset would have to be broadcasted against each other. This may \
    change dimension of some specific keys. Use with caution.

    Returns
    -------
        xr.Dataset
    """
    # Get all list of all variables
    all_keys = set()
    for ds in args:
        all_keys = all_keys.union(ds.keys())

    # Then pad missing keys
    datasets = []
    for ds in args:
        # Get the list of missing keys
        missing_keys = [key for key in all_keys if key not in ds.keys()]

        # Get one reference key
        ref_key = list(all_keys.difference(missing_keys))[0]

        # Get the padded dataarrays
        pad_ds = xr.merge([
            fill_val * xr.ones_like(ds[ref_key]).rename(key)
            for key in missing_keys
        ])

        # Merge and append to list
        datasets.append(xr.merge([ds, pad_ds]))

    return concat_on_common_keys(*datasets, dim=dim, dim_val=dim_val)


def get_dttm_elem(
    time_da: xr.DataArray,
    elements: Union[str, List[str]]
) -> xr.Dataset:
    """
    Given the input array of timestamps, get the corresponding datetime elements \
        and assemble them in a dataset.

    Parameters
    ----------
    time_da: xr.DataArray
        DataArray of time data.
    elements: str | List[str]
        Element to extract.
    """
    # Check inputs
    if isinstance(elements, str):
        elements = [elements]

    dataarrays = [
        xr.DataArray(
            data=(
                pd.to_datetime(time_da).map(
                    lambda dttm: dttm.__getattribute__(elem)
                )
                # .__getattribute__(elem)
            ),
            name=elem,
            dims=time_da.name,
            coords={time_da.name: time_da}
        )
        for elem in elements
    ]

    return xr.merge(dataarrays)


def select_n(
    ds: Union[xr.DataArray, xr.Dataset],
    dim: str,
    n: int,
    random_seed: int = 42
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Given a dataset, randomly select n element along the given dimension.

    Parameters
    ----------
    ds: DataArray | Dataset
        Input dataset/datarray
    dim: str
        Name of the dimension along to use for selection.
    n: int
        Number of element to select.
    random_seed: int
        Seed number for initializing numpy's RandomState

    Returns
    -------
        DataArray | Dataset
    """
    # Verify
    check_required(ds, dim, 'dim')

    # Get
    rng = np.random.RandomState(random_seed)
    subset = rng.choice(ds[dim].data, n, replace=False)

    return ds.sel({dim: subset})


def dim_chunks(
    ds: Union[xr.DataArray, xr.Dataset],
    dim: str,
    chunksize: int,
    sort: bool = True,
) -> List[Any]:
    """
    Given an xarray object, split the specified dimension in chunk of a given size.

    Parameters
    ----------
    ds: DataArray | Dataset
        Input xarray object.
    dim: str
        Name of the selected dimension.
    chunksize: int
        The chunksize.
    sort: bool
        Option whether to sort the data prior to do the split.

    Returns
    -------
        List[Any]
    """
    # Check input
    check_required(ds, dim, 'dim')

    # Get the data
    dim_val = list(ds[dim].data)
    if sort:
        dim_val.sort()

    chunks = create_chunks(dim_val, chunksize)

    return [chunk for chunk in chunks]


def find_axis(ds: Union[xr.DataArray, xr.Dataset], dim: str) -> int:
    """
    Given an xarray object, find the index corresponding to the selected dimension.

    Parameters
    ----------
    ds: DataArray | Dataset
        Input xarray object.
    dim: str
        Name of the selected dimension.

    Returns
    -------
        int
    """
    # First check the input has the selected dimension
    check_required(ds, dim, 'dim')

    return list(ds.dims).index(dim)


def force_dim_order(ds: xr.Dataset) -> xr.Dataset:
    """
    Given a dataset, transpose the variables such that they all have the \
        same dimension order as the dataset.

    Parameters
    ----------
    ds: Dataset
        Dataset to tranpose.

    Returns
    -------
        Dataset
    """
    dims = list(ds.dims)
    for dim in dims:
        ds = ds.transpose(..., dim)

    return ds


def stack_arrays(
    ds: xr.Dataset,
    name: str,
    dim: str,
    keys: Optional[List[str]] = None,
    dim_val: Optional[List[Any]] = None,
) -> xr.DataArray:
    """
    Given a dataset, stack the listed keys into a single data array \
        with a new dimension.

    Parameters
    ----------
    ds: Dataset
        Dataset to be stacked.
    keys: List[str]
        The list of keys to stack.
    name: str
        The name of the variable to rename the data array.
    dim: str
        Name of the new dimension.
    dim_val: List[Any] | None
        (Optional) the values of the new dimension. Use the list of keys \
        if None is passed.
    """
    # Check inputs
    keys = [keys, list(ds.keys())][keys is None]
    check_required(ds, keys, 'key')
    dim_val = [keys, dim_val][dim_val is not None]

    # Get the data arrays
    dataarrays = [
        ds[key].rename(name)
        for key in keys
    ]

    # Stack
    da = (
        xr.concat(xr.broadcast(*dataarrays), dim=dim, coords='minimal')
        .assign_coords({dim: (dim, dim_val)})
    )

    return da


def unstack_array(da: xr.DataArray, dim: str) -> xr.Dataset:
    """
    Given a dataarray, unstack along the given dimension by converting \
        each elemnt of the dim into a variable of new dataset.

    Parameters
    ----------
    da: DataArray
        Input dataarray to unstack.
    dim: dim
        Name of the dimension to unstack.

    Returns
    -------
        Dataset
    """
    # Check inputs
    check_required(da, dim, 'dim')

    # Get the arrays
    dataarrays = [
        da.sel({dim: d}).rename(d).drop(dim)
        for d in list(da[dim].data)
    ]

    return xr.merge(dataarrays)


def categorize(
    da: xr.DataArray,
    bins: Union[int, range, List[float], np.ndarray] = 21,
    labels: Optional[Union[range, List[float], np.ndarray]] = None,
    bin_kw: Mapping[str, Any] = {},
) -> xr.DataArray:
    """
    Given the datarray of continuous, transform it to categorical variable \
        of the given bin size.

    Parameters
    ----------
    da: DataArray
        The input data array to be turned to categorical.
    bins: int | Sequence[int] | Sequence[float]
        If int, defines the number of bins, otherwise the bin edges.
    labels: Sequence[int] | Sequence[float] | Sequence[str] | None
        (Optional) labels to use for the bins.
    bin_kw: Dict[str, Any]
        Optional argument to pass to the underlying pd.cut method.
    """
    # Convert to series
    s = da.to_series()

    # Get the bins and labels
    if isinstance(bins, int):
        bins = equally_spaced_bins(s, bins)

    bins = np.array(bins)

    labels = [labels, edges2labels(bins)][labels is None]

    # Categorize
    res = pd.Series(
        pd.cut(s, bins=bins, labels=labels, **bin_kw),
        index=s.index,
        name=da.name
    )
    res = res.astype(np.float64)

    return res.to_xarray()


def xr_percentile(
    ds: Union[xr.DataArray, xr.Dataset],
    dim: str,
    q: Union[float, List[float]],
    perc_kw: Mapping[str, Any] = {}
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Given an xarray object, compute the percentiles values \
        along the selected dimensions.

    Parameters
    ----------
    ds: DataArray | Dataset
        Input xarray object
    dim: str
        A string defining the dimension along which to compute \
        the percentiles.
    q: float | List[float]
        A number of list of numbers defininf the percentiles to \
        be computed.
    perc_kw: Mapping[str, Any]
        A mapper of the form param_name -> param_value, defining \
        optional argument to pass to the np.percentile method.

    Returns
    -------
        DataArray | Dataset
    """
    # Inner method
    def _perc(arr: np.ndarray, axis: int, **kwargs) -> np.ndarray:
        """Ad-hoc percentile method with automatic tranpose"""

        ndims = arr.ndim
        axis = [axis, ndims - 1][axis == -1]

        left = [k + 1 for k in range(axis)]
        right = [k for k in range(axis + 1, ndims)]
        new_shape = left + [0] + right

        res = np.nanpercentile(arr, axis=axis, **kwargs)
        res = np.transpose(res, new_shape)

        return res

    # Check
    check_required(ds, dim, 'dim')

    # Compute
    perc_kw = {**perc_kw, **{'axis': -1, 'q': q}}
    res_ds = (
        xr.apply_ufunc(
            _perc,
            drop_nondim_coords(ds),
            input_core_dims=[[dim]],
            output_core_dims=[['q']],
            kwargs=perc_kw
        )
        .assign_coords(q=('q', q))
    )

    return res_ds
