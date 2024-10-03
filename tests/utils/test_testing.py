from pathlib import Path


def test_global_paths():
    from crawfish.utils.testing import TEST_FILES_DIR, EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR

    for _path in [TEST_FILES_DIR, EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR]:
        path = Path(_path)
        assert path.exists()
        assert path.is_dir()
