"""Tests for |projection|^2 bandProjections files (band-projection-params <ortho> yes; finding F18)."""

from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR, write_norm_bandprojections
from pathlib import Path
import re
import shutil
import warnings
import numpy as np
import pytest

calc_dirs = Path(EXAMPLE_CALC_DIRS_DIR)
NOCACHE = dict(use_cached_spectrum=False, save_spectrum=False)


def _copy(name, dest: Path, norm: bool) -> Path:
    """Copy fixture `name` (without its cache) to dest, with its bandProjections as |T|^2 if norm."""
    shutil.copytree(calc_dirs / name, dest, ignore=shutil.ignore_patterns(".crawfish_cache"))
    if norm:
        write_norm_bandprojections(calc_dirs / name / "bandProjections", dest / "bandProjections")
    return dest


def _load(calc_dir, **settings):
    """Return the ElecData and the messages of the warnings raised while loading it."""
    from crawfish.core.elecdata import ElecData

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edata = ElecData(calc_dir)
        for key, val in settings.items():
            setattr(edata, key, val)
    return edata, [str(w.message) for w in caught if "Numba" not in str(w.message)]


@pytest.fixture(scope="module")
def n2(tmp_path_factory):
    """(|T|^2 ElecData, complex ElecData with los_orbs=False, load warnings of the former) for N2_nospin."""
    tmp = tmp_path_factory.mktemp("f18")
    norm, msgs = _load(_copy("N2_nospin", tmp / "norm", norm=True))
    cplx, _ = _load(_copy("N2_nospin", tmp / "complex", norm=False), los_orbs=False)
    return norm, cplx, msgs


def test_loads_as_amplitudes_without_lowdin(n2):
    from crawfish.core.elecdata import get_edata_settings_str

    norm, cplx, msgs = n2
    assert norm.projections_are_norms and not cplx.projections_are_norms
    assert len(msgs) == 1 and "|projection|^2" in msgs[0] and "los_orbs is switched off" in msgs[0]
    assert not norm.los_orbs and "lou_False" in get_edata_settings_str(norm)
    assert not np.iscomplexobj(norm.proj_tju)
    # |T| from 7 printed decimals of |T|^2
    assert np.allclose(norm.proj_tju, np.abs(cplx.proj_tju), atol=3e-4)


def test_pdos_matches_complex_file(n2):
    from crawfish.funcs import get_pdos

    norm, cplx, _ = n2
    erange = np.arange(-1.2, 0.6, 1e-3, dtype=np.float32)
    for idx in (0, 1):
        for orb in norm.atom_orb_labels_dict["N"]:
            a = get_pdos(norm, idcs=idx, orbs=orb, erange=erange, **NOCACHE)[1]
            b = get_pdos(cplx, idcs=idx, orbs=orb, erange=erange, **NOCACHE)[1]
            assert np.max(np.abs(a - b)) <= 1e-6 * np.max(np.abs(b)), (idx, orb)


def test_orbital_completeness_matches(n2):
    norm, cplx, _ = n2
    assert np.allclose(norm.orbital_completeness_u, cplx.orbital_completeness_u, atol=1e-5)


def test_phase_dependent_quantities_refuse(n2):
    norm, _, _ = n2
    with pytest.raises(ValueError, match="unavailable"):
        _ = norm.orbital_span_t
    with pytest.raises(ValueError, match=re.escape("|projection|^2")):
        norm.los_orbs = True
    assert not norm.los_orbs


@pytest.mark.parametrize("func_name", ["get_pcohp", "get_ipcohp", "get_pcoop", "get_ipcoop", "get_pcobi", "get_ipcobi",
                                       "get_pcomo", "get_ipcomo"])
def test_pair_functions_raise(n2, func_name):
    from crawfish import funcs

    norm, _, _ = n2
    with pytest.raises(ValueError, match="band-projection-params"):
        getattr(funcs, func_name)(norm, idcs1=0, idcs2=1)
    # Which image a pair would use doesn't need phases
    assert funcs.get_pair_images(norm, idcs1=0, idcs2=1)[0]["R"] == (0, 0, 0)


def test_stale_raw_cache(tmp_path, n2):
    # proj_tju.npy as written before F18: the raw file values (|T|^2). It must still give |T|.
    from crawfish.io.data_parsing import get_proj_tju_from_file

    norm, _, _ = n2
    calc_dir = _copy("N2_nospin", tmp_path / "norm", norm=True)
    first, _ = _load(calc_dir)
    raw = get_proj_tju_from_file(calc_dir / "bandProjections")
    assert np.array_equal(np.load(first.proj_tju_filepath), raw)  # the cache stays raw
    np.save(first.proj_tju_filepath, raw)
    second, _ = _load(calc_dir)
    assert np.array_equal(second.proj_tju, norm.proj_tju)
    assert np.allclose(second.orbital_completeness_u, norm.orbital_completeness_u)


def test_time_reversal_unfolding_of_amplitudes(tmp_path):
    # A k/-k reduced mesh is unfolded with T(-k) = T(k)*, so |T(-k)| = |T(k)|
    norm, _ = _load(_copy("graphene_zspin_tr", tmp_path / "norm", norm=True))
    cplx, _ = _load(_copy("graphene_zspin_tr", tmp_path / "complex", norm=False), los_orbs=False)
    assert norm.kmesh_unfolded and norm.time_reversal_unfolded
    assert norm.nstates == cplx.nstates
    assert np.allclose(norm.proj_tju, np.abs(cplx.proj_tju), atol=3e-4)


def test_point_group_reduced_mesh_stays_guarded(tmp_path):
    # Rotating orbitals needs phases, so a mesh reduced by the point group isn't unfolded and stays guarded
    from crawfish.funcs import get_pdos

    norm, _ = _load(_copy("graphene_zspin_pg", tmp_path / "norm", norm=True))
    assert not norm.kmesh_unfolded and norm.kmesh_is_reduced
    with pytest.warns(UserWarning, match="averaged over symmetry-equivalent atoms"):
        get_pdos(norm, idcs=0, **NOCACHE)
