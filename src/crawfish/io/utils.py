"""Module for utility functions for file I/O that can be used outside of the context of JDFTx files."""

from pathlib import Path
from typing import Callable, Any
from wrapper import wraps


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


def get_text_with_key_in_bounds(filepath: str | Path, key: str, start: int, end: int) -> str:
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
