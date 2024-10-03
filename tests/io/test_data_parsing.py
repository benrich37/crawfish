import pytest
from crawfish.utils.testing import EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
from pathlib import Path

exfdir = Path(EXAMPLE_FILES_DIR)
excddir = Path(EXAMPLE_CALC_DIRS_DIR)


def test_outfile_parsing():
    from crawfish.io.data_parsing import (
        get_nspin_from_outfile_filepath,
        get_mu_from_outfile_filepath,
        get_kfolding_from_outfile_filepath,
    )

    outfile_filepath = exfdir / "out"
    assert get_nspin_from_outfile_filepath(outfile_filepath) == 2
    assert get_mu_from_outfile_filepath(outfile_filepath) == pytest.approx(-0.247047099)
    kfold_known = [3, 3, 3]
    for i in range(3):
        assert get_kfolding_from_outfile_filepath(outfile_filepath)[i] == kfold_known[i]


def test_bandprojections_parsing():
    from crawfish.io.data_parsing import (
        get_nbands_from_bandfile_filepath,
        get_nstates_from_bandfile_filepath,
        get_nspecies_from_bandfile_filepath,
        get_nproj_from_bandfile_filepath,
        get_norbsperatom_from_bandfile_filepath,
        is_complex_bandfile_filepath,
        _parse_bandfile_complex,
    )

    bandfile_filepath = exfdir / "bandprojections"
    assert get_nbands_from_bandfile_filepath(bandfile_filepath) == 15
    assert get_nstates_from_bandfile_filepath(bandfile_filepath) == 54
    assert get_nspecies_from_bandfile_filepath(bandfile_filepath) == 1
    assert get_nproj_from_bandfile_filepath(bandfile_filepath) == 8
    for i in range(2):
        assert get_norbsperatom_from_bandfile_filepath(bandfile_filepath)[i] == 4
    assert is_complex_bandfile_filepath(bandfile_filepath)
    # TODO: Get a non-complex bandfile to test
    with pytest.raises(RuntimeError):
        _parse_bandfile_complex(bandfile_filepath=exfdir / "bandprojections_corrupted")
        is_complex_bandfile_filepath(exfdir / "out")


def test_token_parsers():
    from crawfish.io.data_parsing import _complex_token_parser, _normalized_token_parser

    out1 = _complex_token_parser(["1", "1"])
    assert len(out1) == 1
    assert isinstance(out1[0], COMPLEX_DTYPE)
    assert out1[0] == pytest.approx(1 + 1j)
    out2 = _normalized_token_parser(["1", "1"])
    assert len(out2) == 2
    assert isinstance(out2[0], REAL_DTYPE)
    assert out2[0] == pytest.approx(1)
