"""Module for utility functions for file I/O that can be used outside of the context of JDFTx files."""

from pathlib import Path
from typing import Callable, Any
from functools import wraps
import numpy as np
import zlib


def format_dir_path(path: Path | str) -> Path:
    """Return Path object from path.

    Return Path object from path.

    Parameters
    ----------
    path : str | Path
        Path to directory.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory.")
    return path


def format_file_path(path: Path | str) -> Path:
    """Return Path object from path.

    Return Path object from path.

    Parameters
    ----------
    path : str | Path
        Path to directory.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file.")
    return path


def get_text_with_key_in_bounds(filepath: str | Path | list[str], key: str, start: int, end: int) -> str:
    """Return contents of file at line with key in bounds.

    Return the line with the key in the file between the start and end lines.

    Parameters
    ----------
    filepath : str | Path
        Path to file.
    key : str
        Key to search for.
    start : int
        Start line.
    end : int
        End line.
    """
    rval = None
    if isinstance(filepath, list):
        texts = filepath
    else:
        texts = read_file(filepath)
    filelength = len(texts)
    if start < 0:
        start = filelength + start
    if end < 0:
        end = filelength + end
    for line, text in enumerate(texts):
        if line > start and line < end:
            if key in text:
                rval = text
                break
    if rval is not None:
        return rval
    else:
        raise ValueError(f"Could not find {key} in {filepath} within bounds {start} and {end}")


def check_file_exists(func: Callable) -> Any:
    """Check if file exists.

    Check if file exists (and continue normally) or raise an exception if
    it does not.
    """

    @wraps(func)
    def wrapper(filename: str) -> Any:
        filepath = Path(filename)
        if not filepath.is_file():
            raise OSError(f"'{filename}' file doesn't exist!")
        return func(filename)

    return wrapper


@check_file_exists
def read_file(file_name: str) -> list[str]:
    """
    Read file into a list of str.

    Parameters
    ----------
    filename: Path or str
        name of file to read

    Returns
    -------
    texts: list[str]
        list of strings from file
    """
    with open(file_name, "r") as f:
        texts = f.readlines()
    f.close()
    return texts


def safe_load(filepath: Path, allow_pickle: bool = True) -> Any:
    """
    Safely load a numpy array from a file. If the file is corrupted, it will be deleted.
    """
    if not filepath.exists():
        return None
    try:
        return np.load(filepath, allow_pickle=allow_pickle)
    except (zlib.error) as e:
        print(f"Error loading {filepath}: {e}")
        print("Deleting corrupted file.")
        if filepath.exists():
            filepath.unlink()
        return None