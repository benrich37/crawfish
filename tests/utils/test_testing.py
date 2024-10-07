from pathlib import Path
import numpy as np


def test_global_paths():
    from crawfish.utils.testing import TEST_FILES_DIR, EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR

    for _path in [TEST_FILES_DIR, EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR]:
        path = Path(_path)
        assert path.exists()
        assert path.is_dir()


def test_idx_fetching():
    from crawfish.utils.testing import approx_idx, get_pocket_idx

    arr = np.array([1, 2, 3, 4, 5])
    val = 3.1
    assert approx_idx(arr, val) == 2
    e_sabcj = np.zeros([5, 5, 5])
    e_sabcj[2, 2, 2] = 10
    erange = np.arange(0, 10, 1)
    assert get_pocket_idx(e_sabcj, erange) == 5
