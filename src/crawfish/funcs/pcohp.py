"""Module for user methods to get PCOHP spectrum.

Module for user methods to get PCOHP spectrum.
"""

from crawfish.utils.arg_correction import check_repeat
from crawfish.core.operations.matrix import (
    _get_gen_tj_coef_uu,
    _get_gen_tj_coef_tuu,
    _get_gen_tj_block,
    get_spin_idx_t,
)
from crawfish.utils.lattice import get_orb_atom_idx_u, get_pair_image_map, normalize_image, check_reduced_kmesh
from crawfish.core.elecdata import ElecData
from crawfish.utils.typing import REAL_DTYPE
from crawfish.utils.indexing import get_orb_idcs
from crawfish.utils.arg_correction import edata_input_to_edata, get_use_cache, _assert_complex_bandprojections
from crawfish.funcs.general import evaluate_or_retrieve_generic_spectrum, get_generic_integrate
from pathlib import Path
import numpy as np
import warnings
from crawfish.funcs.general import SIGMA_DEFAULT, RES_DEFAULT
from crawfish.utils.typing import COMPLEX_DTYPE


def _resolve_image(state_sep: bool, image) -> str | list[int]:
    """Return the image to use: state_sep=True is the older spelling of image="all" and takes precedence."""
    return "all" if state_sep else normalize_image(image)


def _image_cache_key(image: str | list[int]) -> str:
    """Return a normalized image as the string stored in the spectrum-cache metadata.

    Lists in that metadata are compared as unordered collections (utils/caching.is_matching_list), so lattice
    vectors that are permutations of each other, e.g. (1, 0, 0) and (0, 1, 0), used to share a cached spectrum.
    A string keeps the order.
    """
    return image if isinstance(image, str) else str(tuple(int(x) for x in image))


def _legacy_state_sep(image) -> bool:
    """Map `image` onto the legacy (pre-F2) state_sep switch. The legacy matrices always pair within one cell."""
    if isinstance(image, list):
        if image != [0, 0, 0]:
            raise ValueError("Choosing a lattice image needs legacy_matrices=False.")
        return False
    return image == "all"


def _check_mesh_for_images(edata: ElecData, pair_map: dict):
    """Real-space matrices for R != 0 need the full k-mesh (notes/theory_audit.md, findings F3 and F5)."""
    if not edata.jdftx or edata.lti_allowed:
        return
    off_cell = sorted({R for R, _ in pair_map.values() if R != (0, 0, 0)})
    if off_cell:
        raise ValueError(
            f"Lattice images {off_cell} need the full k-point mesh, but this calculation's k-points were reduced "
            "by symmetry. Rerun JDFTx with 'symmetries none', or use image='all' or image=(0, 0, 0)."
        )


def _compute_signed_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        mat_name: str | None,
        image: str | list[int],
        ):
    """Return pair weights w_t Re[conj(C_u) M_uv C_v] for a signed matrix (notes/theory_audit.md, findings F2, F3).

    mat_name is "h" (pCOHP), "s" (pCOOP), "d" (pCOBI), or None for M = 1 (pCOMO, a bond-order density).
    image="all": the state-resolved M_t, i.e. the interaction summed over all lattice images of the second set.
    image="nearest" or a lattice vector: for each atom pair (a, b), only the image of b at tau_b + R_ab, through
    M_ab(R_ab) exp(2 pi i k_t.R_ab). "nearest" picks R_ab by distance (R=0 for every pair when
    edata.legacy_lattice_image is True); see crawfish.utils.lattice.get_pair_image_map.
    An orbital paired with itself (u == v) is left out where it is the on-site term: image="all" and pairs at
    R_ab = 0. For R_ab != 0 it pairs orbital u in cell 0 with its copy in cell R_ab, an interaction between two
    atoms, and is included (finding F19; before, it was always left out, so an atom's interaction with its own
    image lacked its u == v part and the public functions refused it).
    """
    orbs_u = np.asarray(orbs_u, dtype=np.int64)
    orbs_v = np.asarray(orbs_v, dtype=np.int64)
    include_ab = orbs_u[:, None] != orbs_v[None, :]
    if image == "all":
        if mat_name is None:
            raise ValueError("pCOMO is defined for one lattice image at a time; use image='nearest' or a lattice vector.")
        mat_t_ab = getattr(edata, f"{mat_name}_t_uv")[:, orbs_u][:, :, orbs_v]
    else:
        pair_map = get_pair_image_map(edata, list(orbs_u), list(orbs_v), image)
        _check_mesh_for_images(edata, pair_map)
        atom_u = get_orb_atom_idx_u(edata)
        spin_t = get_spin_idx_t(edata.nstates, edata.nspin)
        all_t = np.arange(edata.nstates)
        mat_t_ab = np.zeros([edata.nstates, len(orbs_u), len(orbs_v)], dtype=COMPLEX_DTYPE)
        for R in sorted({R for R, _ in pair_map.values()}):
            if mat_name is None:
                mat_suv = np.ones([edata.nspin, edata.nproj, edata.nproj], dtype=COMPLEX_DTYPE)
            elif R == (0, 0, 0):
                mat_suv = edata.get_mat_r0_suv(mat_name)
            else:
                mat_suv = edata.get_mat_r_suv(mat_name, R)
            if R == (0, 0, 0):
                phase_t = np.ones(edata.nstates, dtype=COMPLEX_DTYPE)
            else:
                phase_t = np.exp(2j * np.pi * (np.asarray(edata.ks_t, dtype=np.float64) @ np.asarray(R, dtype=np.float64)))
            for (a, b), (R_ab, _) in pair_map.items():
                if R_ab != R:
                    continue
                ia = np.flatnonzero(atom_u[orbs_u] == a)
                ib = np.flatnonzero(atom_u[orbs_v] == b)
                block_sab = mat_suv[:, orbs_u[ia]][:, :, orbs_v[ib]]
                mat_t_ab[np.ix_(all_t, ia, ib)] = block_sab[spin_t] * phase_t[:, None, None]
                if R != (0, 0, 0):
                    include_ab[np.ix_(ia, ib)] = True  # u == v here is u with its image in cell R (F19)
    return _get_gen_tj_block(edata.coef_tju, mat_t_ab, edata.wk_t, orbs_u, orbs_v, include_ab)


def _retrieve_weights(edata: ElecData, name: str, orbs_u, orbs_v, compute_func, use_cache: bool, image):
    """Use the per-setting weight cache for image="nearest" or "all"; explicit lattice vectors are not cached."""
    cache_name = None
    if image == "nearest":
        cache_name = f"{name}_tj_cache"
    elif image == "all":
        cache_name = f"tsep_{name}_tj_cache"
    if use_cache and cache_name is not None and hasattr(edata, cache_name):
        return getattr(edata, cache_name).compute_or_retrieve(orbs_u, orbs_v, compute_func)
    return compute_func(orbs_u, orbs_v)


def get_pair_images(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    image: str | list[int] | tuple[int, int, int] = "nearest",
) -> list[dict]:
    """Return which lattice image each atom pair uses in a pair spectrum, and the bond length.

    Takes the same atom/orbital selection and `image` as the pair-spectrum functions (get_pcohp, ...).

    Returns
    -------
    list[dict]
        One entry per atom pair: {"atom1": "N #1", "atom2": "N #2", "R": (0, 0, 0), "distance": 1.0923}, with R
        in lattice vectors (the image of atom2 at tau_2 + R) and distance in Angstrom.
    """
    edata = edata_input_to_edata(edata_input)
    image = normalize_image(image)
    if image == "all":
        raise ValueError("image='all' sums over every lattice image; pass 'nearest' or a lattice vector.")
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    pair_map = get_pair_image_map(edata, orbs_u, orbs_v, image)
    return [
        {"atom1": edata.kmap[a], "atom2": edata.kmap[b], "R": R, "distance": dist}
        for (a, b), (R, dist) in pair_map.items()
    ]


# Below this, a non-orthogonal pCOOP is noticeably reduced by the overlap missing from the band space (F7, F11)
COOP_COMPLETENESS_WARN = 0.85


def _warn_if_coop_is_artifact(edata: ElecData, orbs: list[int] | None = None):
    """Warn when pCOOP between different orbitals is zero, a band-truncation artifact, or underestimated (F7).

    The band-space overlap S_t = T T^dag only approaches <phi|O|phi> as the bands span the orbitals, so with
    non-orthogonal projections pCOOP also needs enough bands: see ElecData.orbital_completeness_u.
    """
    if edata.legacy_matrices:
        return
    if edata.los_orbs:
        warnings.warn(
            "pCOOP between different orbitals is identically zero in an orthonormal (Löwdin) basis "
            "(los_orbs=True). Use los_orbs=False with projections from 'band-projection-params no no'.",
            stacklevel=3,
        )
    elif edata.jdftx and edata.bandprojfile_is_ortho:
        warnings.warn(
            "These projections are onto orthonormalized orbitals ('band-projection-params yes ...'), so the true "
            "orbital overlap is the identity and any pCOOP between different orbitals only reflects band "
            "truncation. Use projections from 'band-projection-params no no' for pCOOP.",
            stacklevel=3,
        )
    elif orbs is not None:
        completeness = edata.orbital_completeness_u[np.asarray(orbs, dtype=int)]
        if completeness.min() < COOP_COMPLETENESS_WARN:
            warnings.warn(
                f"Some selected orbitals are only {completeness.min():.2f} represented by the computed bands "
                f"(ElecData.orbital_completeness_u), so the band-space overlap behind pCOOP is underestimated. "
                "Increase elec-n-bands.",
                stacklevel=3,
            )

def get_pcohp(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    state_sep: bool = False,
    image: str | list[int] | tuple[int, int, int] = "nearest",
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the pCOHP spectrum for the system of interest.

    Get the energy array / pCOHP spectrum pair for the system of interest.

    Parameters
    ----------
    edata_input : ElecData | str | Path
        The ElecData object or path to the directory of the ElecData object.
    idcs(1/2) : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements(1/2) : list[str] | str | None
        The elements of interest.
    orbs(1/2) : list[str] | str | None
        The orbitals of interest.
    state_sep : bool
        Older spelling of image="all"; takes precedence over `image` when True.
    image : str | list[int] | tuple[int, int, int]
        Which periodic image of the second atom set pairs with the first (notes/theory_audit.md, finding F3).
        "nearest" (default): for each atom pair, the image of the second atom nearest to the first (check which
        with `get_pair_images`). "all": the interaction summed over every image. A lattice vector (n1, n2, n3):
        that image for every pair. Images other than the same cell need the full k-mesh (`symmetries none`).
        The two orbital sets may share orbitals only for a lattice vector other than (0, 0, 0): that is an
        atom's bond with its own periodic image, including each orbital with its own copy (finding F19;
        `get_nearest_images` lists the nearest ones).
    erange: np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    sig : REAL_DTYPE
        The width parameter of the Gaussian smearing (if lti is False), exp(-(E - e)^2 / sig): a standard
        deviation of sqrt(sig / 2), 2.2e-3 Ha by default. A grid step above 2 standard deviations warns (F17).
    res : REAL_DTYPE
        The step (Ha) of the automatic energy range (if erange is None): 1e-3 Ha by default (1e-2 Ha
        before finding F17; res=0.01 gives that grid). The range is padded beyond the extreme eigenvalues by
        max(10 res, 5 standard deviations of the Gaussian).
    spin_pol : bool
        If the spectrum should be returned with up/down intensities separated.
    lti : bool
        Use the linear tetrahedron integration method.
    rattle_eigenvals : bool
        Rattle the eigenvalues to up to twice erange resolution to avoid degeneracies.
        (only used if lti is True)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    use_cache : bool
        If True, cache the results of the calculation and retrieve them if they are already computed.
        This will be more time efficient for subsequent redundant pcohp evaluations
        (enumerating over the same orbitals) for calculations with more states and bands.
        Note: for each unique pair of orbitals, an array of size (nStates, nBands) is stored
        by a crawfish.utils.caching.CachedFunction object at ElecData._pcohp_tj_cache,
        where each value occupies 4 bytes. To free memory, the cache can be cleared
        via ElecData._pcohp_tj_cache.clear().
    """
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcohp_tj = _get_pcohp_tj(
        edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2,
        use_cache=use_cache, state_sep=state_sep, image=image
        )
    kwargs = {
        "erange": erange,
        "spin_pol": spin_pol,
        "sig": sig,
        "res": res,
        "lti": lti,
        "rattle_eigenvals": rattle_eigenvals,
        "norm_max": norm_max,
        "norm_intg": norm_intg,
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
            "state_sep": state_sep,
            "image": _image_cache_key(_resolve_image(state_sep, image)),
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcohp_tj, "pcohp", **kwargs)



def get_ipcohp(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    spin_pol: bool = False,
    use_fillings: bool = True,
    use_cache: bool | None = None,
    state_sep: bool = False,
    image: str | list[int] | tuple[int, int, int] = "nearest",
    ):
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcohp_tj = _get_pcohp_tj(
        edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2,
        use_cache=use_cache, state_sep=state_sep, image=image)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcohp_tj, **kwargs)
    return es, cs


def _get_pcohp_tj(
        edata: ElecData,
        idcs1: list[int] | int | None,
        elements1: list[str] | str | None,
        orbs1: list[str] | str | None,
        idcs2: list[int] | int | None,
        elements2: list[str] | str | None,
        orbs2: list[str] | str | None,
        use_cache: bool = False,
        state_sep: bool = False,
        image: str | list[int] | tuple[int, int, int] = "nearest",
        ):
    _assert_complex_bandprojections(edata)  # |projection|^2 files have no phases (F18)
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    image = _resolve_image(state_sep, image)
    check_repeat(orbs_u, orbs_v, image)
    check_reduced_kmesh(edata, list(orbs_u) + list(orbs_v), "pCOHP", image)
    compute_func = lambda u, v: _compute_pcohp_tj(edata, u, v, image)
    return _retrieve_weights(edata, "pcohp", orbs_u, orbs_v, compute_func, use_cache, image)

def _compute_pcohp_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        image: str | list[int],
        ):
    if not edata.legacy_matrices:
        return _compute_signed_tj(edata, orbs_u, orbs_v, "h", image)
    # Legacy (pre-F2) path
    if _legacy_state_sep(image):
        # h_t_uu already carries the k-point weight, so it is not applied again
        pcohp_tj = _get_gen_tj_coef_tuu(edata.proj_tju, edata.h_t_uu, orbs_u, orbs_v)
    else:
        h_uu = edata.h_uu
        pcohp_tj = _get_gen_tj_coef_uu(edata.proj_tju, edata.wk_t, h_uu, orbs_u, orbs_v)
    return pcohp_tj



# Cite (C. A. Coulson, Proc. Roy. Soc. (London) 169A, 419 (1939).)
def get_pcomo(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    image: str | list[int] | tuple[int, int, int] = "nearest",
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the pCOMO spectrum for the system of interest.

    Get the energy array / pCOMO spectrum pair for the system of interest.

    Parameters
    ----------
    edata_input : ElecData | str | Path
        The ElecData object or path to the directory of the ElecData object.
    idcs(1/2) : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements(1/2) : list[str] | str | None
        The elements of interest.
    orbs(1/2) : list[str] | str | None
        The orbitals of interest.
    image : str | list[int] | tuple[int, int, int]
        Which periodic image of the second atom set pairs with the first (notes/theory_audit.md, finding F3).
        "nearest" (default) or a lattice vector (n1, n2, n3); see get_pcohp. ("all" is not defined for pCOMO.)
        The orbital sets may share orbitals only for a lattice vector other than (0, 0, 0) (finding F19).
    erange: np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    sig : REAL_DTYPE
        The width parameter of the Gaussian smearing (if lti is False), exp(-(E - e)^2 / sig): a standard
        deviation of sqrt(sig / 2), 2.2e-3 Ha by default. A grid step above 2 standard deviations warns (F17).
    res : REAL_DTYPE
        The step (Ha) of the automatic energy range (if erange is None): 1e-3 Ha by default (1e-2 Ha
        before finding F17; res=0.01 gives that grid). The range is padded beyond the extreme eigenvalues by
        max(10 res, 5 standard deviations of the Gaussian).
    spin_pol : bool
        If the spectrum should be returned with up/down intensities separated.
    lti : bool
        Use the linear tetrahedron integration method.
    rattle_eigenvals : bool
        Rattle the eigenvalues to up to twice erange resolution to avoid degeneracies.
        (only used if lti is True)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    use_cache : bool
        If True, cache the results of the calculation and retrieve them if they are already computed.
        This will be more time efficient for subsequent redundant pcohp evaluations
        (enumerating over the same orbitals) for calculations with more states and bands.
        Note: for each unique pair of orbitals, an array of size (nStates, nBands) is stored
        by a crawfish.utils.caching.CachedFunction object at ElecData._pcohp_tj_cache,
        where each value occupies 4 bytes. To free memory, the cache can be cleared
        via ElecData._pcohp_tj_cache.clear().
    """
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcomo_tj = _get_pcomo_tj(
        edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2,
        use_cache=use_cache, image=image
        )
    kwargs = {
        "erange": erange,
        "spin_pol": spin_pol,
        "sig": sig,
        "res": res,
        "lti": lti,
        "rattle_eigenvals": rattle_eigenvals,
        "norm_max": norm_max,
        "norm_intg": norm_intg,
        "sep_channels": sep_channels,
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
            "image": _image_cache_key(normalize_image(image)),
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcomo_tj, "pcomo", **kwargs)

def get_ipcomo(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    spin_pol: bool = False,
    use_fillings: bool = True,
    use_cache: bool | None = None,
    image: str | list[int] | tuple[int, int, int] = "nearest",
):
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcobi_tj = _get_pcomo_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, use_cache=use_cache, image=image)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcobi_tj, **kwargs)
    return es, cs


def _get_pcomo_tj(
        edata: ElecData,
        idcs1: list[int] | int | None,
        elements1: list[str] | str | None,
        orbs1: list[str] | str | None,
        idcs2: list[int] | int | None,
        elements2: list[str] | str | None,
        orbs2: list[str] | str | None,
        use_cache: bool = False,
        image: str | list[int] | tuple[int, int, int] = "nearest",
        ):
    _assert_complex_bandprojections(edata)  # |projection|^2 files have no phases (F18)
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    image = normalize_image(image)
    check_repeat(orbs_u, orbs_v, image)
    check_reduced_kmesh(edata, list(orbs_u) + list(orbs_v), "pCOMO", image)
    compute_func = lambda u, v: _compute_pcomo_tj(edata, u, v, image)
    return _retrieve_weights(edata, "pcomo", orbs_u, orbs_v, compute_func, use_cache, image)

def _compute_pcomo_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        image: str | list[int],
        ):
    if not edata.legacy_matrices:
        # Uses the expansion coefficients, which equal the projections when los_orbs is True
        return _compute_signed_tj(edata, orbs_u, orbs_v, None, image)
    # Legacy (pre-F2) path
    _legacy_state_sep(image)  # the legacy kernel only pairs within one cell
    h_uu = np.ones([edata.nproj, edata.nproj], dtype=REAL_DTYPE)
    pcohp_tj = _get_gen_tj_coef_uu(edata.proj_tju, edata.wk_t, h_uu, orbs_u, orbs_v)
    return pcohp_tj


def get_pcoop(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    state_sep: bool = False,
    image: str | list[int] | tuple[int, int, int] = "nearest",
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the pCOOP spectrum for the system of interest.

    Get the energy array / pCOOP spectrum pair for the system of interest. pCOOP between different orbitals is
    zero in the default orthonormal basis; use los_orbs=False with non-orthogonal projections (see the README).

    Parameters
    ----------
    edata_input : ElecData | str | Path
        The ElecData object or path to the directory of the ElecData object.
    idcs(1/2) : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements(1/2) : list[str] | str | None
        The elements of interest.
    orbs(1/2) : list[str] | str | None
        The orbitals of interest.
    state_sep : bool
        Older spelling of image="all"; takes precedence over `image` when True.
    image : str | list[int] | tuple[int, int, int]
        Which periodic image of the second atom set pairs with the first (notes/theory_audit.md, finding F3).
        "nearest" (default): for each atom pair, the image of the second atom nearest to the first (check which
        with `get_pair_images`). "all": the interaction summed over every image. A lattice vector (n1, n2, n3):
        that image for every pair. Images other than the same cell need the full k-mesh (`symmetries none`).
        The two orbital sets may share orbitals only for a lattice vector other than (0, 0, 0): that is an
        atom's bond with its own periodic image, including each orbital with its own copy (finding F19;
        `get_nearest_images` lists the nearest ones).
    erange: np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    sig : REAL_DTYPE
        The width parameter of the Gaussian smearing (if lti is False), exp(-(E - e)^2 / sig): a standard
        deviation of sqrt(sig / 2), 2.2e-3 Ha by default. A grid step above 2 standard deviations warns (F17).
    res : REAL_DTYPE
        The step (Ha) of the automatic energy range (if erange is None): 1e-3 Ha by default (1e-2 Ha
        before finding F17; res=0.01 gives that grid). The range is padded beyond the extreme eigenvalues by
        max(10 res, 5 standard deviations of the Gaussian).
    spin_pol : bool
        If the spectrum should be returned with up/down intensities separated.
    rattle_eigenvals : bool
        Rattle the eigenvalues to up to twice erange resolution to avoid degeneracies.
        (only used if lti is True)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    """
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcoop_tj = _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, image=image, use_cache=use_cache)
    kwargs = {
        "erange": erange,
        "spin_pol": spin_pol,
        "sig": sig,
        "res": res,
        "lti": lti,
        "rattle_eigenvals": rattle_eigenvals,
        "norm_max": norm_max,
        "norm_intg": norm_intg,
        "sep_channels": sep_channels,
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
            "state_sep": state_sep,
            "image": _image_cache_key(_resolve_image(state_sep, image)),
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcoop_tj, "pcoop", **kwargs)

def get_ipcoop(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    spin_pol: bool = False,
    use_fillings: bool = True,
    use_cache: bool | None = None,
    state_sep: bool = False,
    image: str | list[int] | tuple[int, int, int] = "nearest",
):
    edata = edata_input_to_edata(edata_input)
    pcoop_tj = _get_pcoop_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, image=image, use_cache=use_cache)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcoop_tj, **kwargs)
    return es, cs


def _get_pcoop_tj(
        edata: ElecData,
        idcs1: list[int] | int | None,
        elements1: list[str] | str | None,
        orbs1: list[str] | str | None,
        idcs2: list[int] | int | None,
        elements2: list[str] | str | None,
        orbs2: list[str] | str | None,
        state_sep: bool = False,
        use_cache: bool = False,
        image: str | list[int] | tuple[int, int, int] = "nearest",
        ):
    _assert_complex_bandprojections(edata)  # |projection|^2 files have no phases (F18)
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    _warn_if_coop_is_artifact(edata, list(orbs_u) + list(orbs_v))
    image = _resolve_image(state_sep, image)
    check_repeat(orbs_u, orbs_v, image)
    check_reduced_kmesh(edata, list(orbs_u) + list(orbs_v), "pCOOP", image)
    compute_func = lambda u, v: _compute_pcoop_tj(edata, u, v, image)
    return _retrieve_weights(edata, "pcoop", orbs_u, orbs_v, compute_func, use_cache, image)

def _compute_pcoop_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        image: str | list[int],
        ):
    check_repeat(orbs_u, orbs_v, image)
    if not edata.legacy_matrices:
        return _compute_signed_tj(edata, orbs_u, orbs_v, "s", image)
    # Legacy (pre-F2) path
    if _legacy_state_sep(image):
        # s_t_uu already carries the k-point weight, so it is not applied again
        pcoop_tj = _get_gen_tj_coef_tuu(edata.proj_tju, edata.s_t_uu, orbs_u, orbs_v)
    else:
        s_uu = edata.s_uu
        pcoop_tj = _get_gen_tj_coef_uu(edata.proj_tju, edata.wk_t, s_uu, orbs_u, orbs_v)
    return pcoop_tj

##
def get_pcobi(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    state_sep: bool = False,
    image: str | list[int] | tuple[int, int, int] = "nearest",
    erange: np.ndarray[REAL_DTYPE] | None = None,
    sig: REAL_DTYPE = SIGMA_DEFAULT,
    res: REAL_DTYPE = RES_DEFAULT,
    spin_pol: bool = False,
    lti: bool = False,
    rattle_eigenvals: bool = False,
    norm_max: bool = False,
    norm_intg: bool = False,
    sep_channels: bool = False,
    use_cache: bool | None = None,
    use_cached_spectrum: bool = True,
    save_spectrum: bool = True,
) -> tuple[np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE]]:
    """Get the pCOBI spectrum for the system of interest.

    Get the energy array / pCOBI spectrum pair for the system of interest. The weights are
    2 w_t Re[conj(C_u) D_uv C_v], with D the density matrix of one spin channel, so the integrated pCOBI
    (get_ipcobi) of two atoms is the Wiberg/Mayer bond index that COBI is built on: a single bond gives about 1
    (Müller et al., J. Phys. Chem. C 125, 7959 (2021)). Before finding F16 it was half that; the
    edata.legacy_cobi_scale setting restores the old scale (see the README).

    Parameters
    ----------
    edata_input : ElecData | str | Path
        The ElecData object or path to the directory of the ElecData object.
    idcs(1/2) : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements(1/2) : list[str] | str | None
        The elements of interest.
    orbs(1/2) : list[str] | str | None
        The orbitals of interest.
    state_sep : bool
        Older spelling of image="all"; takes precedence over `image` when True.
    image : str | list[int] | tuple[int, int, int]
        Which periodic image of the second atom set pairs with the first (notes/theory_audit.md, finding F3).
        "nearest" (default): for each atom pair, the image of the second atom nearest to the first (check which
        with `get_pair_images`). "all": the interaction summed over every image. A lattice vector (n1, n2, n3):
        that image for every pair. Images other than the same cell need the full k-mesh (`symmetries none`).
        The two orbital sets may share orbitals only for a lattice vector other than (0, 0, 0): that is an
        atom's bond with its own periodic image, including each orbital with its own copy (finding F19;
        `get_nearest_images` lists the nearest ones).
    erange: np.ndarray[REAL_DTYPE] | None
        The energy range of interest.
    sig : REAL_DTYPE
        The width parameter of the Gaussian smearing (if lti is False), exp(-(E - e)^2 / sig): a standard
        deviation of sqrt(sig / 2), 2.2e-3 Ha by default. A grid step above 2 standard deviations warns (F17).
    res : REAL_DTYPE
        The step (Ha) of the automatic energy range (if erange is None): 1e-3 Ha by default (1e-2 Ha
        before finding F17; res=0.01 gives that grid). The range is padded beyond the extreme eigenvalues by
        max(10 res, 5 standard deviations of the Gaussian).
    spin_pol : bool
        If the spectrum should be returned with up/down intensities separated.
    rattle_eigenvals : bool
        Rattle the eigenvalues to up to twice erange resolution to avoid degeneracies.
        (only used if lti is True)
    norm_max : bool
        Normalize the spectrum to the maximum intensity to 1.
    norm_intg : bool
        Normalize the spectrum to the integral of the spectrum to 1.
    """
    edata = edata_input_to_edata(edata_input)
    use_cache = get_use_cache(use_cache, edata.use_cache_default)
    pcobi_tj = _get_pcobi_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, image=image, use_cache=use_cache)
    kwargs = {
        "erange": erange,
        "spin_pol": spin_pol,
        "sig": sig,
        "res": res,
        "lti": lti,
        "rattle_eigenvals": rattle_eigenvals,
        "norm_max": norm_max,
        "norm_intg": norm_intg,
        "sep_channels": sep_channels,
        "use_cached_spectrum": use_cached_spectrum,
        "save_spectrum": save_spectrum,
        "func_args_dict": {
            "idcs1": idcs1,
            "elements1": elements1,
            "orbs1": orbs1,
            "idcs2": idcs2,
            "elements2": elements2,
            "orbs2": orbs2,
            "state_sep": state_sep,
            "image": _image_cache_key(_resolve_image(state_sep, image)),
        },
    }
    return evaluate_or_retrieve_generic_spectrum(edata, pcobi_tj, "pcobi", **kwargs)

def get_ipcobi(
    edata_input: ElecData | str | Path,
    idcs1: list[int] | int | None = None,
    idcs2: list[int] | int | None = None,
    elements1: list[str] | str | None = None,
    elements2: list[str] | str | None = None,
    orbs1: list[str] | str | None = None,
    orbs2: list[str] | str | None = None,
    spin_pol: bool = False,
    use_fillings: bool = True,
    use_cache: bool | None = None,
    state_sep: bool = False,
    image: str | list[int] | tuple[int, int, int] = "nearest",
):
    """Get the integrated pCOBI up to each eigenvalue, (energies, running sum).

    With use_fillings=True (default), the final value summed over the orbitals of two atoms is their Wiberg/Mayer
    bond index (about 1 for a single bond; finding F16, edata.legacy_cobi_scale for the old half scale). Takes the
    same selection and `image` arguments as get_pcobi.
    """
    edata = edata_input_to_edata(edata_input)
    pcobi_tj = _get_pcobi_tj(edata, idcs1, elements1, orbs1, idcs2, elements2, orbs2, state_sep=state_sep, image=image, use_cache=use_cache)
    kwargs = {
        "spin_pol": spin_pol,
        "use_fillings": use_fillings,
    }
    es, cs = get_generic_integrate(edata, pcobi_tj, **kwargs)
    return es, cs



def _get_pcobi_tj(
        edata: ElecData,
        idcs1: list[int] | int | None,
        elements1: list[str] | str | None,
        orbs1: list[str] | str | None,
        idcs2: list[int] | int | None,
        elements2: list[str] | str | None,
        orbs2: list[str] | str | None,
        state_sep: bool = False,
        use_cache: bool = False,
        image: str | list[int] | tuple[int, int, int] = "nearest",
        ):
    _assert_complex_bandprojections(edata)  # |projection|^2 files have no phases (F18)
    orbs_u = get_orb_idcs(edata, idcs1, elements1, orbs1)
    orbs_v = get_orb_idcs(edata, idcs2, elements2, orbs2)
    image = _resolve_image(state_sep, image)
    check_repeat(orbs_u, orbs_v, image)
    check_reduced_kmesh(edata, list(orbs_u) + list(orbs_v), "pCOBI", image)
    compute_func = lambda u, v: _compute_pcobi_tj(edata, u, v, image)
    return _retrieve_weights(edata, "pcobi", orbs_u, orbs_v, compute_func, use_cache, image)

def _compute_pcobi_tj(
        edata: ElecData,
        orbs_u: list[int],
        orbs_v: list[int],
        image: str | list[int],
        ):
    check_repeat(orbs_u, orbs_v, image)
    if not edata.legacy_matrices:
        pcobi_tj = _compute_signed_tj(edata, orbs_u, orbs_v, "d", image)
        if not edata.legacy_cobi_scale:
            # Finding F16: d_t_uv is the density matrix of one spin channel (occupations in [0, 1]), so these
            # weights sum to W_s |D^s_uv|^2 over spin channels (W_s = sum_{t in s} w_t: 2 for no-spin, 1 per
            # channel for z-spin), half the Wiberg/Mayer bond index (sum P_uv^2 with P = 2D; 2 sum_s |D^s_uv|^2)
            # that COBI is built on. The factor 2 gives that index; edata.legacy_cobi_scale leaves it out.
            pcobi_tj = pcobi_tj * REAL_DTYPE(2)
        return pcobi_tj
    # Legacy (pre-F2) path
    if _legacy_state_sep(image):
        # p_t_uu already carries the k-point weight, so it is not applied again
        pcobi_tj = _get_gen_tj_coef_tuu(edata.proj_tju, edata.p_t_uu, orbs_u, orbs_v)
    else:
        p_uu = edata.p_uu
        pcobi_tj = _get_gen_tj_coef_uu(edata.proj_tju, edata.wk_t, p_uu, orbs_u, orbs_v)
    return pcobi_tj