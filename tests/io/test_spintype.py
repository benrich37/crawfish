"""Tests for the spin types crawfish supports (notes/theory_audit.md, finding F9)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from pathlib import Path
import shutil
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
vectorspin_dir = calc_dirs / "N2_vectorspin"


@pytest.mark.parametrize(
    ("name", "spintype", "nspin"),
    [("N2_nospin", "no-spin", 1), ("N2_bare_min", "z-spin", 2), ("N2_vectorspin", "vector-spin", None)],
)
def test_spintype_and_nspin_from_outfile(name, spintype, nspin):
    from crawfish.io.data_parsing import get_spintype_from_outfile_filepath, get_nspin_from_outfile_filepath

    outfile = calc_dirs / name / "out"
    assert get_spintype_from_outfile_filepath(outfile) == spintype
    if nspin is None:
        # Used to return 2, although JDFTx keeps one spinor state per k-point
        with pytest.raises(NotImplementedError, match="spinor"):
            get_nspin_from_outfile_filepath(outfile)
    else:
        assert get_nspin_from_outfile_filepath(outfile) == nspin


def test_vectorspin_fixture_layout():
    # The reason crawfish refuses these runs: one state per k-point, and two projections per spatial orbital
    from crawfish.io.data_parsing import get_nstates_from_bandfile_filepath, get_norbsperatom_from_bandfile_filepath

    assert get_nstates_from_bandfile_filepath(vectorspin_dir / "bandProjections") == 1
    assert get_norbsperatom_from_bandfile_filepath(vectorspin_dir / "bandProjections") == [8, 8]  # N: (s + 3 p) x 2


@pytest.mark.parametrize("spintype", ["vector-spin", "spin-orbit"])
def test_elecdata_refuses_spinor_runs(spintype, tmp_path):
    from crawfish.core.elecdata import ElecData

    calc_dir = tmp_path / spintype
    shutil.copytree(vectorspin_dir, calc_dir)
    out = calc_dir / "out"
    out.write_text(out.read_text().replace("spintype vector-spin", f"spintype {spintype}"))
    for make in (ElecData, ElecData.from_calc_dir):
        with pytest.raises(NotImplementedError, match=f"'spintype {spintype}'"):
            make(calc_dir)
    # Refused before anything was parsed or cached
    assert not (calc_dir / ".crawfish_cache").exists()


def test_check_spintype_supported():
    from crawfish.io.data_parsing import check_spintype_supported

    for spintype in ["no-spin", "z-spin"]:
        check_spintype_supported(spintype)
    with pytest.raises(ValueError, match="Unrecognized"):
        check_spintype_supported("up-down")
