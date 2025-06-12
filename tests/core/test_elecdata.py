from crawfish.utils.testing import EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR, TMP_FILES_DIR
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
from pathlib import Path
import numpy as np
from os import remove
from shutil import rmtree, copy
import pytest
import re

exfdir = Path(EXAMPLE_FILES_DIR)
excddir = Path(EXAMPLE_CALC_DIRS_DIR)


def test_elecdata_initialization():
    from crawfish.core.elecdata import ElecData

    exdir_path = excddir / "N2_bare_min"
    edata1 = ElecData(exdir_path)
    edata2 = ElecData.from_calc_dir(exdir_path)
    for edata in [edata1, edata2]:
        assert isinstance(edata, ElecData)


def test_elecdata_properties():
    from crawfish.core.elecdata import ElecData

    exdir_path = excddir / "N2_bare_min"
    edata = ElecData(exdir_path)
    for intprop in ["nspin", "nstates", "nbands", "nproj"]:
        setattr(edata, f"_{intprop}", None)
        assert hasattr(edata, intprop)
        val = getattr(edata, intprop)
        assert val is not None
        assert isinstance(val, int)
    for listintprop in ["norbsperatom"]:
        setattr(edata, f"_{listintprop}", None)
        assert hasattr(edata, listintprop)
        val = getattr(edata, listintprop)
        assert val is not None
        assert isinstance(val, list)
        assert all(isinstance(v, int) for v in val)
    for floatprop in ["mu"]:
        assert hasattr(edata, floatprop)
        val = getattr(edata, floatprop)
        assert val is not None
        ## Unsure if this should be converted from float to REAL_DTYPE for consistency
        # assert isinstance(val, REAL_DTYPE)
    for realarr in ["e_tj", "occ_tj", "wk_t", "ks_t"]:
        assert hasattr(edata, realarr)
        val = getattr(edata, realarr)
        assert val is not None
        assert isinstance(val, np.ndarray)
        assert val.dtype == REAL_DTYPE
    for boolval in ["lti_allowed", "bandprojfile_is_complex"]:
        assert hasattr(edata, boolval)
        val = getattr(edata, boolval)
        assert val is not None
        assert isinstance(val, bool)
    for filepath in [
        "bandfile_filepath",
        "kptsfile_filepath",
        "eigfile_filepath",
        "fillingsfile_filepath",
        "outfile_filepath",
    ]:
        assert hasattr(edata, filepath)
        val = getattr(edata, filepath)
        assert val is not None
        assert isinstance(val, Path)
    for proj_var in ["proj_tju"]:
        proj = getattr(edata, proj_var)
        assert proj is not None
        assert isinstance(proj, np.ndarray)
        if edata.bandprojfile_is_complex:
            assert proj.dtype == COMPLEX_DTYPE
        else:
            assert proj.dtype == REAL_DTYPE
        assert len(proj.shape) == len(proj_var.split("_")[-1])


def test_set_file_paths():
    from crawfish.core.elecdata import ElecData

    tmp_dir = TMP_FILES_DIR / "tmp_calc_dir"
    tmp_dir.mkdir(exist_ok=True)
    exdir_path = excddir / "N2_bare_min"
    ref_files = ["out", "fillings", "kPts", "eigenvals", "bandProjections"]
    for ref_file in ref_files:
        src_file = exdir_path / ref_file
        dst_file = tmp_dir / ref_file
        copy(src_file, dst_file)
    edata = ElecData(tmp_dir)
    edata.fprefix = None
    with pytest.raises(RuntimeError):
        edata._set_files_paths()
    for nonoptional in ["out", "eigenvals", "bandProjections"]:
        src_file = exdir_path / nonoptional
        dst_file = tmp_dir / nonoptional
        remove(dst_file)
        with pytest.raises(FileNotFoundError):
            ElecData(tmp_dir)
        copy(src_file, dst_file)
    # Values returning None
    for optional, opt_vars in [("fillings", ["fillingsfile_filepath"]), ("kPts", ["kptsfile_filepath"])]:
        src_file = exdir_path / optional
        dst_file = tmp_dir / optional
        if dst_file.exists():
            remove(dst_file)
        edata = ElecData(tmp_dir)
        for opt_var in opt_vars:
            assert getattr(edata, opt_var) is None
        if src_file.exists():
            copy(src_file, dst_file)
    rmtree(tmp_dir)


def test_get_fprefix():
    from crawfish.core.elecdata import ElecData

    exdir_path = excddir / "N2_bare_min"
    edata = ElecData(exdir_path)
    empty_prefixes = [None, "", "$VAR"]
    for ep in empty_prefixes:
        assert edata._get_fprefix(ep) == ""
    assert edata._get_fprefix("prefix") == "prefix."
    with pytest.raises(ValueError, match="prefix must be a string or None"):
        edata._get_fprefix(1)


def test__get_data_and_path():
    from crawfish.core.elecdata import ElecData, _get_data_and_path

    exdir_path = excddir / "N2_bare_min"
    edata = ElecData(exdir_path)
    for data, path in [(edata, exdir_path), (edata, None), (None, exdir_path)]:
        outdata, outpath = _get_data_and_path(data, path)
        assert outdata is not None
        assert outpath is not None
        assert isinstance(outdata, ElecData)
        assert outpath == exdir_path
        assert outdata.calc_dir == outpath
    with pytest.raises(ValueError, match="calc_dir and data.calc_dir must be the same"):
        outdata, outpath = _get_data_and_path(edata, exdir_path.parent)
    with pytest.raises(
        ValueError, match=re.escape("Must provide at least a calc_dir or a data=ElecData (both cannot be none)")
    ):
        outdata, outpath = _get_data_and_path(None, None)
