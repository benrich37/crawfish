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


# def get_kfolding_from_kpts_reader(kptsfile_filepath: str | Path) -> list[int] | None:
#     """Get kpt folding from kpts file.

#     Get the kpt folding from the kpts file.

#     Parameters
#     ----------
#     kptsfile_filepath : str | Path
#         Path to kpts file.
#     """
#     wk_list, k_points_list, nstates = parse_kptsfile(kptsfile_filepath)
#     unique_vals: list[list[REAL_DTYPE]] = [[], [], []]
#     for arr in k_points_list:
#         for i, val in enumerate(arr):
#             if not True in [np.isclose(val, uval) for uval in unique_vals[i]]:
#                 unique_vals[i].append(val)
#     est_kfold = [len(uvals) for uvals in unique_vals]
#     if not np.prod(est_kfold) == len(arrs):
#         warnings.warn("Kpts file does not have a consistent kpt mesh.", stacklevel=2)
#         est_kfold = None
#     return est_kfold


# def get_kfolding_from_kptsfile_filepath(kptsfile_filepath: str | Path, nk: int) -> list[int]:
#     """Get the kfolding from the kpts file.

#     Get the kpt folding from the kpts file.

#     Parameters
#     ----------
#     kptsfile_filepath : str | Path
#         Path to kpts file.
#     nk : int
#         Number of kpoints.
#     """
#     kfolding = get_kfolding_from_kpts_reader(kptsfile_filepath)
#     if kfolding is None:
#         kfolding = _get_arbitrary_kfolding(nk)
#     return kfolding


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


def get_wk_sabc(
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
    wk_sabc = wk.reshape([nspin, kfolding[0], kfolding[1], kfolding[2]])
    return wk_sabc


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


def get_ks_sabc(kptsfile_filepath: Path | str, nspin: int, kfolding: list[int]) -> np.ndarray[REAL_DTYPE]:
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
    ks_sabc = ks.reshape([nspin, kfolding[0], kfolding[1], kfolding[2], 3])
    return ks_sabc


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


def get_e_sabcj_helper(
    eigfile_filepath: str | Path, nspin: int, nbands: int, kfolding: list[int] | np.ndarray[int]
) -> np.ndarray[REAL_DTYPE]:
    """Return eigenvalues from file.

    Return eigenvalues from file. Returns a numpy array of shape
    [nspin (s), kfolding[0] (a), kfolding[1] (b), kfolding[2] (c), nbands (j)].

    Parameters
    ----------
    eigfile_filepath : str | Path
        Path to eigenvalues file.
    nspin : int
        Number of spins.
    nbands : int
        Number of bands.
    kfolding : list[int] | np.ndarray[int]
        kpt folding.

    Returns
    -------
    np.ndarray
        Eigenvalues array in shape (spin, kpt_a, kpt_b, kpt_c, band).
    """
    eigfile_filepath = Path(eigfile_filepath)
    if not eigfile_filepath.exists():
        raise ValueError(f"Eigenvalues file {eigfile_filepath} does not exist.")
    e = np.fromfile(eigfile_filepath)
    e = np.array(e, dtype=REAL_DTYPE)
    eshape = [nspin, kfolding[0], kfolding[1], kfolding[2], nbands]
    e_sabcj = e.reshape(eshape)
    return e_sabcj


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
