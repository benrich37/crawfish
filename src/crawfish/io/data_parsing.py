"""Module for parsing data from JDFTx output files.

This module contains functions for parsing data from JDFTx output files.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
from pathlib import Path
from numba import jit
from crawfish.io.general import get_text_with_key_in_bounds, read_file
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
import warnings
from copy import copy

spintype_nspin = {"no-spin": 1, "spin-orbit": 2, "vector-spin": 2, "z-spin": 2}


def get_nspin_from_outfile_filepath(outfile_filepath: str | Path, slice_idx=-1) -> int:
    """Get number of spins from out file.

    Get the number of spins from the out file.

    Parameters
    ----------
    outfile_filepath : str | Path
        Path to output file.
    slice_idx : int
        Relevant slice of out file.

    Returns
    -------
    int
        Number of spins.
    """
    outfile = read_file(outfile_filepath)
    key = "spintype"
    rval = None
    start, end = get_outfile_slice_bounds(outfile, slice_idx=slice_idx)
    text = get_text_with_key_in_bounds(outfile, key, start, end)
    tokens = text.strip().split()
    val = tokens[1]
    if val in spintype_nspin:
        rval = spintype_nspin[val]
    if rval is not None:
        del outfile
        return rval
    else:
        raise ValueError(f"Unrecognized spintype {val}")


def get_mu_from_outfile_filepath(outfile_filepath: str | Path, slice_idx=-1) -> REAL_DTYPE:
    """Get the Fermi level from the output file.

    Get the Fermi level from the output file.

    Parameters
    ----------
    outfile_filepath : str | Path
        Path to output file.


    Returns
    -------
    mu: float
        Fermi level in Hartree.
    """
    mu = None
    lookkey = "FillingsUpdate:  mu:"
    outfile = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(outfile, slice_idx=slice_idx)
    text = get_text_with_key_in_bounds(outfile_filepath, lookkey, start, end)
    mu = REAL_DTYPE(text.split(lookkey)[1].strip().split()[0])
    return mu


def get_kfolding_from_outfile_filepath(outfile_filepath: str | Path, slice_idx=-1) -> np.ndarray[int]:
    """Return kpt folding from output file.

    Get the kpt folding from the output file.

    Parameters
    ----------
    outfile_filepath: str | Path
        Path to output file.
    start: int
        Start line.
    end: int
        End line.

    Returns
    -------
    np.ndarray[int]
    """
    key = "kpoint-folding "
    outfile = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(outfile, slice_idx=slice_idx)
    text = get_text_with_key_in_bounds(outfile_filepath, key, start, end)
    val = np.array(text.split(key)[1].strip().split(), dtype=int)
    return val


def get_nstates_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    """Get number of states from bandprojections file.

    Get the number of states from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    """
    return get__from_bandfile_filepath(bandfile_filepath, 0)


def get_nbands_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    """Get number of bands from bandprojections file.

    Get the number of bands from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    """
    return get__from_bandfile_filepath(bandfile_filepath, 2)


def get_nproj_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    """Get number of projections from bandprojections file.

    Get the number of projections from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    """
    return get__from_bandfile_filepath(bandfile_filepath, 4)


def get_nspecies_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    """Get number of species (ion types) from bandprojections file.

    Get the number of species (ion types) from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    """
    return get__from_bandfile_filepath(bandfile_filepath, 6)


def get_norbsperatom_from_bandfile_filepath(bandfile_filepath: Path | str) -> list[int]:
    """Get number of orbitals per atom from bandprojections file.

    Get the number of orbitals per atom from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    list[int]
    """
    nspecies = get_nspecies_from_bandfile_filepath(bandfile_filepath)
    norbsperatom = []
    bandfile = read_file(bandfile_filepath)
    for line, text in enumerate(bandfile):
        tokens = text.split()
        if line == 0:
            int(tokens[6])
        elif line >= 2:
            if line < nspecies + 2:
                natoms = int(tokens[1])
                norbsperatom.extend(
                    [
                        int(tokens[2]),
                    ]
                    * natoms
                )
            else:
                break
    return norbsperatom


def is_complex_bandfile_filepath(bandfile_filepath: str | Path) -> bool:
    """Determine if bandprojections file is complex.

    Determine if the bandprojections file is complex.
    Needed before attempting pCOHP analysis.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    bool
    """
    hash_lines = 0
    val = True
    with open(bandfile_filepath, "r") as f:
        for i, line in enumerate(f):
            if "#" in line:
                hash_lines += 1
                if hash_lines == 2:
                    if "|projection|^2" in line:
                        val = False
                    else:
                        val = True
                    break
    f.close()
    return val


def parse_kptsfile(kptsfile: str | Path) -> tuple[list[REAL_DTYPE], list[np.ndarray[REAL_DTYPE]], int]:
    """Parse kpts file.

    Parse the kpts file.

    Parameters
    ----------
    kptsfile : str | Path
        Path to kpts file.

    Returns
    -------
    wk_list: list[REAL_DTYPE]
        List of weights for each kpoint
    k_points_list: list[np.ndarray[REAL_DTYPE]]
        List of k-points.
    nStates: int
        Number of states.

    """
    wk_list: list[REAL_DTYPE] = []
    k_points_list: list[list[REAL_DTYPE]] = []
    with open(kptsfile, "r") as f:
        for line in f:
            k_points = line.split("[")[1].split("]")[0].strip().split()
            k_points_floats: list[REAL_DTYPE] = [REAL_DTYPE(v) for v in k_points]
            k_points_list.append(k_points_floats)
            wk = REAL_DTYPE(line.split("]")[1].strip().split()[0])
            wk_list.append(wk)
    nstates = len(wk_list)
    return wk_list, k_points_list, nstates


class _Kpts:
    def __init__(
        self,
        kpts: list[np.ndarray[REAL_DTYPE]],
        weights: list[REAL_DTYPE],
        floor_weight: REAL_DTYPE,
        ksteps: list[REAL_DTYPE],
    ):
        self.reduced_kpts = []
        for i in range(len(kpts)):
            kpt = _Kpt(kpts[i], weights[i], floor_weight, ksteps)
            self.reduced_kpts.append(kpt)
        self.floor_weight = floor_weight
        self.ksteps = ksteps
        self.kpt_map: list[list[_Kpt]] = []
        self.options: list[np.ndarray[REAL_DTYPE]] = [self._get_kpoint_entry_options(ksteps[i]) for i in range(3)]
        for _kpt in self.reduced_kpts:
            kpt = _Kpt(_kpt.kpt, _kpt.weight, _kpt.floor_weight, _kpt.ksteps)
            self.kpt_map.append(self._get_unreduced_kpts(kpt))

    def _get_unreduced_kpts(self, kpt: _Kpt) -> list[_Kpt]:
        kpt_list = [kpt]
        l1 = len(kpt_list)
        self._extend_kmap(kpt_list)
        l2 = len(kpt_list)
        while l2 > l1:
            l1 = len(kpt_list)
            self._extend_kmap(kpt_list)
            l2 = len(kpt_list)
        return kpt_list

    def _extend_kmap(self, klist: list[_Kpt]):
        for kpt in klist:
            avoid = [k.kpt for k in klist] + [k.kpt for k in self.reduced_kpts]
            klist += kpt._sign_split(avoid=avoid)
        for kpt in klist:
            avoid = [k.kpt for k in klist] + [k.kpt for k in self.reduced_kpts]
            klist += kpt._ax_split(avoid=avoid)

    def _has_kpt(self, kpt: np.ndarray[REAL_DTYPE]):
        return self._kpt_in_list(kpt, self.reduced_kpts)

    def _kpts_array_is_kpt_object(self, akpt: np.ndarray[REAL_DTYPE], okpt: _Kpt):
        return all([np.isclose(akpt[i], okpt.kpt[i]) for i in range(3)])

    def _kpt_in_list(self, kpt: np.ndarray[REAL_DTYPE], kpt_list: list[_Kpt]):
        return any([self._kpts_array_is_kpt_object(kpt, k) for k in kpt_list])

    def _is_valid_kpt(self, kpt: np.ndarray[REAL_DTYPE]):
        return all([np.isclose(0, min(self.options[i - kpt[i]])) for i in range(3)])

    def _get_kpoint_entry_options(self, kstep):
        kfold = int(1 / kstep)
        n_out = int(np.floor(kfold / 2))
        if kfold % 2 == 1:
            return np.linspace(-kstep * n_out, kstep * n_out, kfold)
        else:
            return np.linspace(-kstep * (n_out - 1), kstep * n_out, kfold)


class _Kpt:
    def __init__(
        self, kpt: np.ndarray[REAL_DTYPE], weight: REAL_DTYPE, floor_weight: REAL_DTYPE, ksteps: list[REAL_DTYPE]
    ):
        self.kpt = kpt
        self.weight = weight
        self.floor_weight = floor_weight
        self.ksteps = ksteps
        self.gamma = self._is_gamma()

    def _is_gamma(self) -> bool:
        return all([np.isclose(k, 0) for k in self.kpt])

    @property
    def _can_reduce(self) -> bool:
        return not np.isclose(self.weight, self.floor_weight)

    def _copy(self):
        return _Kpt(self.kpt.copy(), copy(self.weight), self.floor_weight, self.ksteps)

    def _sign_split(self, avoid: list[np.ndarray[REAL_DTYPE]] = []):
        kpt_arr = self.kpt.copy()
        kpt_arr *= -1
        if any(
            [
                self.gamma,
                self.weight == self.floor_weight,
                not np.isclose(self.weight / 2 % self.floor_weight, 0),
                self._kptarr_in_kptarr_list(kpt_arr, avoid),
            ]
        ):
            return []
        else:
            self.weight *= 0.5
            dup = self._copy()
            dup.kpt *= -1
            return [dup]

    def _ax_split(self, avoid: list[np.ndarray[REAL_DTYPE]] = []):
        if self.gamma or self.weight == self.floor_weight:
            return []
        else:
            kpts = [self.kpt]
            for i in range(3):
                for j in range(i + 1, 3):
                    kpt_copy = self.kpt.copy()
                    kpt_copy[[i, j]] = self.kpt[[j, i]]
                    if all(
                        [
                            self._is_valid_kpt(kpt_copy),
                            not self._kptarr_in_kptarr_list(kpt_copy, kpts),
                            not self._kptarr_in_kptarr_list(kpt_copy, avoid),
                        ]
                    ):
                        kpts.append(kpt_copy)
            if self.weight / len(kpts) < self.floor_weight:
                return []
            self.weight /= len(kpts)
            kpts_out = []
            if len(kpts) > 1:
                for kpt in kpts[1:]:
                    _kpt = self._copy()
                    _kpt.kpt = kpt
                    kpts_out.append(_kpt)
            return kpts_out

    def _is_same(self, other: _Kpt) -> bool:
        return all([np.isclose(self.kpt[i], other.kpt[i]) for i in range(3)])

    def _is_valid_kpt(self, kpt: np.ndarray[REAL_DTYPE]):
        return all([np.isclose(kpt[i] % self.ksteps[i], 0) for i in range(3)])

    def _kptarr_in_kptarr_list(self, kpt: np.ndarray[REAL_DTYPE], kpt_list: list[np.ndarray[REAL_DTYPE]]):
        return any([self._kpts_array_is_kpt_array(kpt, k) for k in kpt_list])

    def _kpts_array_is_kpt_array(
        self,
        akpt1: np.ndarray[REAL_DTYPE],
        akpt2: np.ndarray[REAL_DTYPE],
    ):
        return all([np.isclose(akpt1[i], akpt2[i]) for i in range(3)])


def _get_kpt_unfolding_map(kptsfile_filepath: str | Path, outfile_filepath: str | Path):
    kfolding = get_kfolding_from_outfile_filepath(outfile_filepath)
    nspin = get_nspin_from_outfile_filepath(outfile_filepath)
    wk_indiv_expected = 1 / (np.prod(kfolding) * nspin)
    wk, k_points_list, nstates = parse_kptsfile(kptsfile_filepath)
    if all(np.isclose(wk, wk_indiv_expected)):
        return k_points_list
    k_points_list_out = []
    for i, kpt in enumerate(k_points_list):
        if _kpt_can_reduce(kpt, wk[i], kfolding, wk_indiv_expected):
            n_possible_reductions = _get_n_possible_reductions(kpt, wk[i], kfolding, wk_indiv_expected)
            equiv_kpts = _get_equiv_kpts(kpt, kfolding)
            if len(equiv_kpts) != n_possible_reductions:
                raise ValueError("Number of equivalent kpts does not match expected number.")
            k_points_list_out.append(equiv_kpts)


def _unreduce_kpt(kpt, kfolding, weight, wk_indiv_expected):
    if _kpt_can_reduce(kpt, weight, kfolding, wk_indiv_expected):
        n_possible_reductions = _get_n_possible_reductions(kpt, weight, kfolding, wk_indiv_expected)
        equiv_kpts = _get_equiv_kpts(kpt, kfolding)
        if len(equiv_kpts) != n_possible_reductions:
            raise ValueError("Number of equivalent kpts does not match expected number.")
        return equiv_kpts


def _get_equiv_kpts(kpt, kfolding):
    equiv_kpts = _get_axswap_equiv_kpts(kpt, kfolding)
    _equiv_kpts = [kpt * (-1) for kpt in equiv_kpts]
    equiv_kpts.extend(_equiv_kpts)
    return equiv_kpts


def _get_axswap_equiv_kpts(kpt, kfolding):
    equiv_kpts = []
    for i in range(3):
        for j in range(3):
            if i <= j:
                if kfolding[i] > 1 and kfolding[j] > 1:
                    kpt_copy = kpt.copy()
                    kpt_copy = np.swapaxes(kpt_copy, i, j)
                    if _is_acceptable_kpt(kfolding, kpt_copy):
                        equiv_kpts.append(kpt_copy)
    return equiv_kpts


def _get_signswap_equiv_kpts(kpt, kfolding):
    equiv_kpts = [kpt * (-1)]
    return equiv_kpts


def _get_n_possible_reductions(kpt, weight, kfolding, wk_indiv_expected):
    possible_reductions = weight / wk_indiv_expected
    n_possible_reductions = int(possible_reductions)
    if not np.isclose(possible_reductions, n_possible_reductions):
        raise ValueError("Kpt weight is not a multiple of the expected weight.")
    return n_possible_reductions


def _kpt_can_reduce(kpt, weight, kfolding, wk_indiv_expected):
    n_possible_reductions = _get_n_possible_reductions(kpt, weight, kfolding, wk_indiv_expected)
    return n_possible_reductions > 1


def _is_acceptable_kpt(kfolding: list[int], kpt: list[REAL_DTYPE]) -> bool:
    """Check if kpt is acceptable.

    Check if kpt is acceptable.

    Parameters
    ----------
    kfolding : list[int]
        kpt folding.
    kpt : list[REAL_DTYPE]
        kpt.

    Returns
    -------
    bool
    """
    ksteps = [1 / kf for kf in kfolding]
    return all([np.isclose(kpt[i] % ksteps[i], 0) for i in range(3)])


def get_kfolding(lti: bool, outfile_filepath: str | Path, nspin: int, nstates: int) -> list[int]:
    """Get the k-point folding.

    Get the k-point folding.

    Parameters
    ----------
    lti : bool
        Whether or not the nSpin * nk-pts == nStates.
    outfile_filepath : str | Path
        Path to output file.
    nspin : int
        Number of spins.
    nstates : int
        Number of states.
    """
    if lti:
        kfolding = get_kfolding_from_outfile_filepath(outfile_filepath)
    else:
        nk = int(nstates / nspin)
        kfolding = _get_arbitrary_kfolding(nk)
    return kfolding


def _get_lti_allowed(bandfile_filepath: str | Path, outfile_filepath: str | Path) -> bool:
    nspin = get_nspin_from_outfile_filepath(outfile_filepath)
    nstates = get_nstates_from_bandfile_filepath(bandfile_filepath)
    kfolding = get_kfolding_from_outfile_filepath(outfile_filepath)
    _nk = int(np.prod(kfolding))
    if nspin != int(nstates / _nk):
        warning_str = (
            "WARNING: Internal inconsistency found with respect to input parameters (nSpin * nk-pts != nStates)."
        )
        "No safety net for this which allows for tetrahedral integration currently implemented."
        warnings.warn(warning_str, stacklevel=2)
        lti = False
    else:
        lti = True
    return lti


def _get_arbitrary_wk(nk: int, nspin: int) -> np.ndarray[REAL_DTYPE]:
    wk = np.ones(nk * nspin)
    wk *= 1 / nk
    return wk


def get_wk_t(
    kptsfile_filepath: Path | str | None, nspin: int, kfolding: list[int], lti: bool
) -> np.ndarray[REAL_DTYPE]:
    """Return weights for k-points.

    Return weights for k-points.

    Parameters
    ----------
    kfolding : list[int]
        kpt folding (after correction if k-point mesh was reduced)
    kptsfile_filepath : Path | str | None
        Path to kpts file. (None to signify non-existence)
    nspin : int
        Number of spins.
    lti : bool
        Whether or not the nSpin * nk-pts == nStates.
    """
    nk = np.prod(kfolding)
    if kptsfile_filepath is None:
        wk = _get_arbitrary_wk(nk, nspin)
    else:
        wk, ks, nstates = parse_kptsfile(kptsfile_filepath)
        wk = np.array(wk, dtype=REAL_DTYPE)
        if not lti:
            if not len(wk) == nk:
                warnings.warn(
                    "Kpts file does not match expected number of k-points. k-point weights will be meaningless.",
                    stacklevel=2,
                )
                wk = _get_arbitrary_wk(nk, nspin)
    return wk


# def _get_ks_sabc(kfolding: list[int], kptsfile_filepath: Path | str | None, nspin: int, lti: bool) -> np.ndarray[REAL_DTYPE]:
#     """ Return the k-point coordinates.

#     Return the k-point coordinates.

#     Parameters
#     ----------
#     kfolding : list[int]
#         kpt folding (after correction if k-point mesh was reduced)
#     kptsfile_filepath : Path | str | None
#         Path to kpts file. (None to signify non-existence)
#     nspin : int
#         Number of spins.
#     lti : bool
#         Whether or not the nSpin * nk-pts == nStates.
#     """
#     if kptsfile_filepath is None or not lti:
#         ks = np.ones([nk * nspin, 3]) * np.nan
#     else:
#         wk, ks, nstates = parse_kptsfile(kptsfile_filepath)
#         ks = np.array(ks, dtype=REAL_DTYPE)
#     ks_sabc = ks.reshape([nspin, kfolding[0], kfolding[1], kfolding[2], 3])
#     return ks_sabc


# def _get_ks_sabc(kfolding: list[int], kptsfile_filepath: Path | str | None, nspin: int, lti: bool) -> np.ndarray[REAL_DTYPE]:
#     """ Return the k-point coordinates.

#     Return the k-point coordinates.

#     Parameters
#     ----------
#     kfolding : list[int]
#         kpt folding (after correction if k-point mesh was reduced)
#     kptsfile_filepath : Path | str | None
#         Path to kpts file. (None to signify non-existence)
#     nspin : int
#         Number of spins.
#     lti : bool
#         Whether or not the nSpin * nk-pts == nStates.
#     """
#     if kptsfile_filepath is None or not lti:
#         ks = np.ones([nk * nspin, 3]) * np.nan
#     else:
#         wk, ks, nstates = parse_kptsfile(kptsfile_filepath)
#         ks = np.array(ks, dtype=REAL_DTYPE)
#     ks_sabc = ks.reshape([nspin, kfolding[0], kfolding[1], kfolding[2], 3])
#     return ks_sabc


def get_ks_t(kptsfile_filepath: Path | str) -> np.ndarray[REAL_DTYPE]:
    """Return the k-point coordinates.

    Return the k-point coordinates. Assumes k-point file exists and mesh is regular.

    Parameters
    ----------
    kptsfile_filepath : Path | str
        Path to kpts file.
    nspin : int
        Number of spins.
    kfolding : list[int]
        kpt folding
    """
    wk, _ks, nstates = parse_kptsfile(kptsfile_filepath)
    ks = np.array(_ks, dtype=REAL_DTYPE)
    return ks


# def _get_kpts_info_handler_astuple(
#     nspin: int, kfolding: list[int] | np.ndarray[int], kptsfile_filepath: Path | str | None, nstates: int
# ) -> tuple[list[int], np.ndarray[REAL_DTYPE], np.ndarray[REAL_DTYPE], bool]:
#     _nk = int(np.prod(kfolding))
#     nk = int(np.prod(kfolding))
#     if nspin != int(nstates / _nk):
#         print("WARNING: Internal inconsistency found with respect to input parameters (nSpin * nk-pts != nStates).")
#         print("No safety net for this which allows for tetrahedral integration currently implemented.")
#         if kptsfile_filepath is None:
#             print("k-folding will be changed to arbitrary length 3 array to satisfy shaping criteria.")
#         lti = False
#         nk = int(nstates / nspin)
#     else:
#         lti = True
#     if kptsfile_filepath is None:
#         if nk != _nk:
#             kfolding = _get_arbitrary_kfolding(nk)
#         ks = np.ones([nk * nspin, 3]) * np.nan
#         wk = np.ones(nk * nspin)
#         wk *= 1 / nk
#     else:
#         if isinstance(kptsfile_filepath, str):
#             kptsfile_filepath = Path(kptsfile_filepath)
#         if not kptsfile_filepath.exists():
#             raise ValueError("Kpts file provided does not exist.")
#         # TODO: Write a function that can un-reduce a reduced kpts mesh
#         wk, ks, nstates = parse_kptsfile(kptsfile_filepath)
#         wk = np.array(wk)
#         ks = np.array(ks)
#         if nk != _nk:
#             if len(ks) == nk:  # length of kpt data matches interpolated nk value
#                 kfolding = get_kfolding_from_kptsfile_filepath(kptsfile_filepath, nk)
#             else:
#                 kfolding = _get_arbitrary_kfolding(nk)
#                 ks = np.ones([nk * nspin, 3]) * np.nan
#                 wk = np.ones(nk * nspin)
#                 wk *= 1 / nk
#     wk_sabc = wk.reshape([nspin, kfolding[0], kfolding[1], kfolding[2]])
#     ks_sabc = ks.reshape([nspin, kfolding[0], kfolding[1], kfolding[2], 3])
#     return kfolding, ks_sabc, wk_sabc, lti


def get_e_tj_helper(
    eigfile_filepath: str | Path, nstates: int, nbands: int
) -> np.ndarray[REAL_DTYPE]:
    """Return eigenvalues from file.

    Return eigenvalues from file. Returns a numpy array of shape
    [nspin (s), kfolding[0] (a), kfolding[1] (b), kfolding[2] (c), nbands (j)].

    Parameters
    ----------
    eigfile_filepath : str | Path
        Path to eigenvalues file.
    nstates : int
        Number of spins.
    nbands : int
        Number of bands.

    Returns
    -------
    np.ndarray
        Eigenvalues array in shape (state, band).
    """
    eigfile_filepath = Path(eigfile_filepath)
    if not eigfile_filepath.exists():
        raise ValueError(f"Eigenvalues file {eigfile_filepath} does not exist.")
    e = np.fromfile(eigfile_filepath)
    e = np.array(e, dtype=REAL_DTYPE)
    eshape = [nstates, nbands]
    e_tj = e.reshape(eshape)
    return e_tj


def get_proj_sabcju_helper(
    bandfile_filepath: Path | str, nspin: int, kfolding: list[int] | np.ndarray[int], nbands: int, nproj: int
) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
    """Return projections from file in sabcju shape.

    Return projections from file in (spin, kpt_a, kpt_b, kpt_c, band, proj) shape.

    Parameters
    ----------
    bandfile_filepath : Path | str
        Path to bandprojections file.
    nspin : int
        Number of spins.
    kfolding : list[int] | np.ndarray[int]
        kpt folding.
    nbands : int
        Number of bands.
    nproj : int
        Number of projections.

    Returns
    -------
    np.ndarray
        Projections array in shape (spin, kpt_a, kpt_b, kpt_c, band, proj).
    """
    proj_shape = [nspin] + list(kfolding) + [nbands, nproj]
    proj_tju = get_proj_tju_from_file(bandfile_filepath)
    proj_sabcju = proj_tju.reshape(proj_shape)
    del proj_tju
    return proj_sabcju


def get_proj_tju_from_file(bandfile_filepath: Path | str) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
    """Return projections from file in tju shape.

    Return projections from file in (state, band, proj) shape. Collected in this shape
    before sabcju shape due to ready availability of this shape in the file.

    Parameters
    ----------
    bandfile_filepath : Path | str
        Path to bandprojections file.

    Returns
    -------
    np.ndarray
        Projections array in shape (state, band, proj).
    """
    is_complex = is_complex_bandfile_filepath(bandfile_filepath)
    if is_complex:
        proj = _parse_bandfile_complex(bandfile_filepath)
    else:
        proj = _parse_bandfile_normalized(bandfile_filepath)
    return proj


def _parse_bandfile_complex(bandfile_filepath: str | Path) -> np.ndarray[COMPLEX_DTYPE]:
    dtype = COMPLEX_DTYPE
    token_parser = _complex_token_parser
    return _parse_bandfile_reader(bandfile_filepath, dtype, token_parser)


def _parse_bandfile_normalized(bandfile_filepath: str | Path) -> np.ndarray[REAL_DTYPE]:
    dtype = REAL_DTYPE
    token_parser = _normalized_token_parser
    return _parse_bandfile_reader(bandfile_filepath, dtype, token_parser)


def _parse_bandfile_reader(
    bandfile_filepath: str | Path, dtype: type, token_parser: Callable
) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
    nstates = get_nstates_from_bandfile_filepath(bandfile_filepath)
    nbands = get_nbands_from_bandfile_filepath(bandfile_filepath)
    nproj = get_nproj_from_bandfile_filepath(bandfile_filepath)
    nspecies = get_nspecies_from_bandfile_filepath(bandfile_filepath)
    get_norbsperatom_from_bandfile_filepath(bandfile_filepath)
    # Header of length 3, and then each states occupies 1 (header) + nbands lineas
    bandfile = read_file(bandfile_filepath)
    expected_length = 3 + (nstates * (1 + nbands))
    if not expected_length == len(bandfile):
        raise RuntimeError("Bandprojections file does not match expected length - ensure no edits have been made.")
    proj_tju = np.zeros((nstates, nbands, nproj), dtype=dtype)
    for line, text in enumerate(bandfile):
        tokens = text.split()
        if line >= nspecies + 2:
            istate = (line - (nspecies + 2)) // (nbands + 1)
            iband = (line - (nspecies + 2)) - istate * (nbands + 1) - 1
            if iband >= 0 and istate < nstates:
                proj_tju[istate, iband] = np.array(token_parser(tokens))
    return proj_tju


def _complex_token_parser(tokens: list[str]) -> np.ndarray[COMPLEX_DTYPE]:
    out = np.zeros(int(len(tokens) / 2), dtype=COMPLEX_DTYPE)
    tokens = np.array(tokens, dtype=REAL_DTYPE)
    out = _complex_token_parser_jit(tokens, out)
    return out


def _normalized_token_parser(tokens: list[str]) -> np.ndarray[REAL_DTYPE]:
    out = np.array(tokens, dtype=REAL_DTYPE)
    return out


@jit(nopython=True)
def _complex_token_parser_jit(
    tokens: np.ndarray[REAL_DTYPE], out: np.ndarray[COMPLEX_DTYPE]
) -> np.ndarray[COMPLEX_DTYPE]:
    reals = tokens[::2]
    imags = tokens[1::2]
    out += reals + 1j * imags
    return out


def get_outfile_start_lines(
    texts: list[str],
    start_key: str = "*************** JDFTx",
    add_end: bool = False,
) -> list[int]:
    """Get start line numbers for JDFTx calculations.

    Get the line numbers corresponding to the beginning of separate JDFTx calculations
    (in case of multiple calculations appending the same out file).

    Parameters:
    -----------
    texts: list[str]
        output of read_file for out file

    """
    start_lines = []
    line = None
    for line, text in enumerate(texts):
        if start_key in text:
            start_lines.append(line)
    if add_end and line is not None:
        start_lines.append(line)
    if line is None:
        raise ValueError("Outfile parser fed an empty file.")
    if not len(start_lines):
        raise ValueError("No JDFTx calculations found in file.")
    return start_lines


def get_outfile_slice_bounds(
    outfile: list[str],
    slice_idx: int = -1,
) -> tuple[int, int]:
    """Get slice bounds for JDFTx calculation.

    Get the slice bounds for a JDFTx calculation in the output file.

    Parameters:
    -----------
    texts: list[str]
        output of read_file for out file
    slice_idx: int
        index of slice to get

    Returns:
    --------
    tuple[int, int]
    """
    start_lines = get_outfile_start_lines(outfile, add_end=True)
    outfile_bounds_list = [[start_lines[i], start_lines[i + 1]] for i in range(len(start_lines) - 1)]
    if slice_idx >= len(outfile_bounds_list):
        raise ValueError(f"Slice index {slice_idx} out of bounds.")
    start, end = outfile_bounds_list[slice_idx]
    return start, end


def get_outfile_start_line(
    outfile: list[str],
) -> int:
    """Get start line for JDFTx calculation.

    Get the line number corresponding to the beginning of a JDFTx calculation.

    Parameters:
    -----------
    outfile: list[str]
        output of read_file for out file

    Returns:
    --------
    int
    """
    return get_outfile_slice_bounds(outfile, slice_idx=-1)[0]


def get__from_bandfile_filepath(bandfile_filepath: Path | str, tok_idx: int) -> int:
    """Get arbitrary integer from header of bandprojections file.

    Get an arbitrary integer from the header of a bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.
    tok_idx : int
        Index of token to return.

    Returns
    -------
    int
        Integer from header of bandprojections file.
    """
    ret_data = None
    bandfile = read_file(bandfile_filepath)
    for iLine, line in enumerate(bandfile):
        tokens = line.split()
        if iLine == 0:
            ret_data = int(tokens[tok_idx])
            break
    if ret_data is None:
        raise ValueError("Provided an empty file")
    return ret_data


def _get_arbitrary_kfolding(nK: int) -> list[int]:
    kfolding = [1, 1, nK]
    return kfolding


def _get_input_coord_vars_from_outfile(
    outfile: list[str],
) -> tuple[list[str], list[np.ndarray[REAL_DTYPE]], np.ndarray[REAL_DTYPE]]:
    start_line = get_outfile_start_line(outfile)
    names = []
    posns = []
    R = np.zeros([3, 3])
    lat_row = 0
    active_lattice = False
    for i, line in enumerate(outfile):
        if i > start_line:
            tokens = line.split()
            if len(tokens) > 0:
                if tokens[0] == "ion":
                    names.append(tokens[1])
                    posns.append(np.array([REAL_DTYPE(tokens[2]), REAL_DTYPE(tokens[3]), REAL_DTYPE(tokens[4])]))
                elif tokens[0] == "lattice":
                    active_lattice = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [REAL_DTYPE(x) for x in tokens[:3]]
                        lat_row += 1
                    else:
                        active_lattice = False
                elif "Initializing the Grid" in line:
                    break
    return names, posns, R
