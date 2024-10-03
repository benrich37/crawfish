from crawfish.utils.testing import EXAMPLE_FILES_DIR, EXAMPLE_CALC_DIRS_DIR
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
from pathlib import Path
import numpy as np

exfdir = Path(EXAMPLE_FILES_DIR)
excddir = Path(EXAMPLE_CALC_DIRS_DIR)


def test_elecdata_initialization():
    from crawfish.core.elecdata import ElecData

    exdir_path = excddir / "N2_bare_min"
    ElecData(exdir_path)


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
        assert isinstance(val, REAL_DTYPE)
    for realarr in ["e_sabcj", "occ_sabcj", "wk_sabc", "ks_sabc"]:
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
    for proj_var in ["proj_sabcju", "proj_tju"]:
        proj = getattr(edata, proj_var)
        assert proj is not None
        assert isinstance(proj, np.ndarray)
        if edata.bandprojfile_is_complex:
            assert proj.dtype == COMPLEX_DTYPE
        else:
            assert proj.dtype == REAL_DTYPE
        assert len(proj.shape) == len(proj_var.split("_")[-1])
