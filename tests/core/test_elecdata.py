from crawfish.utils.testing import EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR
from pathlib import Path

exfdir = Path(EXAMPLE_FILES_DIR)
excddir = Path(EXAMPLE_CALC_DIRS_DIR)


def test_elecdata_initialization():
    from crawfish.core.elecdata import ElecData

    exdir_path = excddir / "N2_bare_min"
    ElecData(exdir_path)
