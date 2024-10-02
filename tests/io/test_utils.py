import pytest
from pathlib import Path
from crawfish.utils.testing import EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR

exfdir = Path(EXAMPLE_FILES_DIR)
excddir = Path(EXAMPLE_CALC_DIRS_DIR)


def test_format_path():
    from crawfish.io.utils import format_dir_path, format_file_path

    outfile_str = str(exfdir / "out")
    example_dir_str = str(exfdir)
    with pytest.raises(ValueError, match=f"Path {Path(outfile_str)} is not a directory."):
        format_dir_path(outfile_str)
    assert isinstance(format_dir_path(example_dir_str), Path)

    with pytest.raises(ValueError, match=f"Path {Path(example_dir_str)} is not a file."):
        format_file_path(exfdir)
    assert isinstance(format_file_path(outfile_str), Path)


def test_read_file():
    from crawfish.io.utils import read_file

    with pytest.raises(OSError):
        read_file("nonexistent")
    assert isinstance(read_file(exfdir / "out"), list)


def test_get_text_with_key_in_bounds():
    from crawfish.io.utils import get_text_with_key_in_bounds

    nonoccuring_key = "nonoccuring_key"
    outfile_filepath = exfdir / "out"
    with pytest.raises(
        ValueError, match=f"Could not find {nonoccuring_key} in {outfile_filepath} within bounds {0} and {500}"
    ):
        get_text_with_key_in_bounds(outfile_filepath, nonoccuring_key, 0, 500)
    assert isinstance(get_text_with_key_in_bounds(exfdir / "out", "mu", 0, -1), str)
