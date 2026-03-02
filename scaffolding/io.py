"""Functions and method for handling of I/O"""
import itertools
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Type,
    Union
)

from IPython.display import display, Markdown

# import pandas as pd

# from pydantic import BaseModel

# import xarray as xr

import yaml
from yaml.loader import SafeLoader


# LOGGER
logger = logging.getLogger(__name__)


# LOGGER AND DISPLAY UTILS
def setup_logger() -> None:
    """Setup logger basic config."""
    logging.basicConfig(
        format=(
            '[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s] '
            '%(levelname)s: %(message)s'
        ),
        level=logging.INFO,
        datefmt='%I:%M:%S'
    )


@contextmanager
def all_logging_disabled(highest_level: int = logging.CRITICAL):
    """
    A context manager that will prevent any logging messages \
        triggered during the body from being processed.

    Parameters
    ----------
    highest_level: int
        The maximum logging level in use. This would only need to be changed \
            if a custom level greater than CRITICAL is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def _format_doctstring(doc: str) -> str:
    """
    Given the file or method docstring, format it prior to print.

    Parameters
    ----------
    doc: str
        The docstring to be printed.

    Returns
    -------
        str
    """
    # Remove the leading line jump (if it's there)
    doc = doc[[0, 1][doc[0] == '\n']:]

    # Drop leading spaces corresponding to the tabulation of the first line
    lines = doc.split('\n')
    try:
        n_spaces = len(re.search(r'(?<=^)\W+(?=\w)', lines[0]).group(0))

    except AttributeError:
        n_spaces = 0

    lines = [line[n_spaces:] for line in lines]

    # Drop multiples except the remaining indentation
    lines = [
        re.sub(r'(?<=\w)[\s]{2,}(?=\w)', ' ', line)
        for line in lines
    ]

    # Join it again
    doc = '\n'.join(lines)

    return doc


def display_docstring(doc: str) -> None:
    """
    Given the file or method docstring, print it to stdout.

    Notes
    -----
    Removes spaces caused by line continuations.

    Parameters
    ----------
    doc: str
        The docstring to be printed.
    """
    print(_format_doctstring(doc))


def display_docstring_notebook(doc: str) -> None:
    """
    Given the file or method docstring, print it in notebook context.

    Parameters
    ----------
    doc: str
        The docstring to be printed.
    """
    # Get the doc string
    # doc = analyzer_cls.__doc__

    # Apply generic formatting
    doc = _format_doctstring(doc)

    # Drop the attribute section
    doc = doc.split('Attributes')[0]

    # Replace line breaks symbols
    # doc = doc.replace('\n', '<br>')

    display(Markdown(doc))


def callable_name(fun: Union[str, Callable]) -> str:
    """
    Given a string or function object, return the \
        corresponding name.

    Parameters
    ----------
    fun: str | Callable
        A string or callable. If a string, it is return as it. If a callable, \
        fun.__name__ is returned instead.

    Returns
    -------
        str
    """
    if isinstance(fun, Callable):
        name = fun.__name__

    elif isinstance(fun, str):
        name = fun

    else:
        msg = f'Invalid type "{type(fun)}". Input must be either a string ' \
              'or a callable of some form.'
        logger.error(msg)
        raise ValueError(msg)

    return name


# UTILITY FUNCTIONS
def path_exists(path: Union[str, Path]) -> None:
    """
    Check if the input path exists, return "FileNotFoundError" otherwise.

    Parameters
    ----------
    path: str | Path
        Input path to be tested.

    Returns
    -------
        None

    Raises
    ------
    FileNotFoundError:
        - if "path" does not exists.
    """
    # Force type to be Path
    path = Path(path)

    if not path.exists():
        msg = f'Input path "{path}" does not exists. ' \
              'Please check your inputs.'
        logger.error(msg)
        raise FileNotFoundError(msg)


def list_files(
    path: Union[str, Path],
    pattern: str,
    recursive: bool = False,
) -> List[Path]:
    """
    Given the input path, scan it content to list the files matchin the given \
        pattern.

    Parameters
    ----------
    path: str | Path
        The input folder to be scanned.
    pattern: str
        A string defining the pattern to search files.
    recurvise: bool
        An option for whether to search recursively or not. \
        Default is False.

    Returns
    -------
        List[Path]

    Raises
    ------
        ValueError
    """
    # Verify
    is_dir(path)

    path = Path(path)
    fun = [path.glob, path.rglob][recursive]
    all_files = list(fun(pattern=pattern))

    if all_files == []:
        msg = f'No find matchin the selected pattern "{pattern}" found in {str(path)}' \
              ' Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        f'Found {len(all_files)} file(s), '
        f'matching pattern "{pattern}" in "{str(path)}" '
        f'using {['non-', ''][recursive]}recursive search.'
    )

    return all_files


def is_dir(path: Union[str, Path]) -> None:
    """
    Test if the input path is a valid directory, \
        raises a ValueError if otherwise.

    Parameters
    ----------
    path: str | Path
        Path to be tested.

    Raises
    ------
        ValueError
    """
    # Force type to be Path
    path = Path(path)

    if not path.is_dir():
        msg = f'Input path "{path}" is not a directory. ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def is_file(path: Union[str, Path]) -> None:
    """
    Test if the input path is a valid file, \
        raises a ValueError if otherwise.

    Parameters
    ----------
    path: str | Path
        Path to be tested.

    Raises
    ------
        ValueError
    """
    # Force type to be Path
    path = Path(path).resolve()

    if not path.is_file():
        msg = f'Input path "{path}" is not a file. ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def validate_folder(path: Union[str, Path]) -> Path:
    """
    Verify the input path is a folder, and creates it if does not exist.

    Parameters
    ----------
    path: str | Path
        The path to validate.

    Returns
    -------
        Path
    """
    path = Path(path)
    if not path.exists():
        os.makedirs(path)

    is_dir(path)

    return path


def atleast_one(lst: List[Any]) -> None:
    """
    Given a list, test that it contains at least one element, \
        raise a ValueError if otherwise.

    Parameters
    ----------
    lst: List[Any]
        The list to be tested.

    Raises
    ------
        ValueError
    """
    if len(lst) < 1:
        msg = 'The list if empty, while it must contains at least one element. ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def only_one(lst: List[Any]) -> None:
    """
    Given a list, test that it contains one and only one element, \
        raise a ValueError if otherwise.

    Parameters
    ----------
    lst: List[Any]
        The list to be tested.

    Raises
    ------
        ValueError
    """
    if len(lst) != 1:
        msg = 'The list should contains only one element, ' \
               f'but received : {lst}. ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def find_pattern(pattern: str, s: str, out_type: Optional[Type] = None) -> str:
    """
    Given the pattern and string, using regexp to find the the substring \
        matchin the pattern and return.

    Parameters
    ----------
    pattern: str
        Pattern to look for.
    s: str
        The string to be scanned.
    out_type: Type | None
        (Optional) The type of the desired output (e.g., float, int or str).

    Returns
    -------
        str

    Raises
    ------
        ValueError
    """
    match = re.search(pattern, s)

    if match is None:
        msg = f'Unable to find substring matchin the pattern {pattern} in {s}. ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    out = match.group(0)

    if out_type:
        out = out_type(out)

    return out


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Given the input path, read a yaml file and returns its
        content into a dictionnary.

    Parameters
    ----------
    path: str | Path
        Path to the input yaml file.

    Returns
    -------
    data: Dict[str, Any]
        Dictionnary filled with the content of the yaml file.
    """
    # Check inputs
    path_exists(path)

    # Read file content
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=SafeLoader)

    return data


def is_allowed(val: Any, allowed: List[Any]) -> None:
    """
    Test if the input value is one of the allowed values, \
        raise ValueError if otherwise.

    Parameters
    ----------
    val: Any
        The value to be tested.
    allowed: List[Any]
        The list of allowed values.

    Raises
    ------
        ValueError
    """
    if val not in allowed:
        msg = f'Invalid value "{val}", must be one of "{allowed}". ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def is_subset(values: List[Any], allowed: List[Any]) -> None:
    """
    Given a list of elements verify it is a valid subset of the allowed values, \
        raise a ValueError if otherwise.

    Parmaters
    ---------
    values: List[Any]
        A list containing the values to be tested.
    allowed: List[Any]
        A list containing all the possible valid values.

    Raises
    ------
        ValueError
    """
    diff = set(values).difference(allowed)
    if diff != set():
        msg = f'The input contains forbidden values: "{diff}".\n' \
              f'Please check your inputs. Allowed values are: "{allowed}".'
        logger.error(msg)
        raise ValueError(msg)


def to_list(x: Union[Any, List[Any]]) -> List[Any]:
    """
    Convert the input variable to a list.

    Notes
    -----
    Without effect if x is already of list type.

    Parameters
    ----------
    x: Any | List[Any]
        Input variable
    allowed: List[Any]
        The list of allowed values.

    Raises
    ------
        ValueError
    """
    return [[x], x][isinstance(x, list)]


def create_chunks(iterable: Iterable, chunksize: int) -> Generator:
    """
    Given a interable, split in in several chunks of fixed size \
        and return in the form of a Generator.

    Notes
    -----
    Could be replaced by itertools.batched for python 3.12 of higher
    See: https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks/22045226#22045226  # noqa: E501

    Parameters
    ----------
    it: Iterable
        An iterable
    chunksize: int
        The chunksize.

    Returns
    -------
        Generator
    """
    it = iter(iterable)
    while item := list(itertools.islice(it, chunksize)):
        yield item
