r"""Module for parsing data from JDFTx output files.

This module contains functions for parsing data from JDFTx output files.
r"""

from __future__ import annotations
from typing import Callable
import numpy as np
from pathlib import Path
from numba import jit
from crawfish.io.general import get_text_with_key_in_bounds, read_file
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
import warnings
from copy import copy

spintype_nspin = {"no-spin": 1, "z-spin": 2}
# Spinor modes: crawfish doesn't support them yet (notes/theory_audit.md, finding F9). They used to be mapped to
# nspin = 2 here, which is wrong: JDFTx keeps one two-component spinor state per k-point.
SPINOR_SPINTYPES = ("vector-spin", "spin-orbit")


def check_spintype_supported(spintype: str) -> None:
    """Raise NotImplementedError for JDFTx's spinor modes (vector-spin, spin-orbit), which crawfish can't analyse yet.

    In these modes JDFTx stores one two-component spinor per band and k-point (ElecInfo.cpp::kpointsReduce), so
    there is no spin index to split states by. Each spatial orbital comes as two projections: spin-up and spin-down
    copies (SpeciesInfo_atomic.cpp, nSpinCopies = 2), or, for relativistic pseudopotentials, j = l -/+ 1/2 orbitals
    (getYlmToSpinAngleMatrix). crawfish's orbital labels, state layout and pair matrices assume collinear spin, so
    results would be silently wrong. See notes/theory_audit.md, finding F9.
    """
    if spintype in SPINOR_SPINTYPES:
        raise NotImplementedError(
            f"crawfish doesn't support 'spintype {spintype}' calculations yet. JDFTx stores spinor wavefunctions "
            "there, with one state per k-point and two projections per spatial orbital (spin copies, or j-resolved "
            "orbitals with spin-orbit coupling), while crawfish assumes collinear spin. Rerun with 'spintype z-spin' "
            "or 'no-spin' if spin-orbit coupling and non-collinear magnetism aren't needed "
            "(notes/theory_audit.md, finding F9)."
        )
    if spintype not in spintype_nspin:
        raise ValueError(f"Unrecognized spintype {spintype}")


def get_spintype_from_outfile_filepath(outfile_filepath: str | Path, slice_idx=-1) -> str:
    r"""Return the spintype JDFTx echoed in the out file (e.g. "z-spin").

    Parameters
    ----------
    outfile_filepath : str | Path
        Path to output file.
    slice_idx : int
        Relevant slice of out file.
    r"""
    outfile = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(outfile, slice_idx=slice_idx)
    text = get_text_with_key_in_bounds(outfile, "spintype", start, end)
    return text.strip().split()[1]


def get_nspin_from_outfile_filepath(outfile_filepath: str | Path, slice_idx=-1) -> int:
    r"""Get number of spins from out file.

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

    Raises
    ------
    NotImplementedError
        For spinor calculations (vector-spin, spin-orbit), see check_spintype_supported.
    r"""
    spintype = get_spintype_from_outfile_filepath(outfile_filepath, slice_idx=slice_idx)
    check_spintype_supported(spintype)
    return spintype_nspin[spintype]


def get_mu_from_outfile_filepath(outfile_filepath: str | Path, slice_idx=-1) -> float | None:
    r"""Get the Fermi level from the output file.

    Get the chemical potential from the last `FillingsUpdate` line of the calculation (ElecInfo.cpp::smearReport,
    printed in Hartree), as JDFTx's own tutorials do. Lines printed during the LCAO initial guess are skipped: LCAO
    always reports a mu, even without `elec-smearing` (which then prints none afterwards), and it isn't the
    converged one. Read here rather than from pymatgen, whose fallback for mu (`_mu_backup`) is in Hartree while its
    main value is in eV (notes/theory_audit.md, finding F10).

    Parameters
    ----------
    outfile_filepath : str | Path
        Path to output file.
    slice_idx : int
        Which calculation in the output file (default: the last).

    Returns
    -------
    mu: float | None
        Chemical potential in Hartree (float64, as printed), or None if no mu was printed after LCAO.
    r"""
    lookkey = "FillingsUpdate:  mu:"
    outfile = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(outfile, slice_idx=slice_idx)
    mu = None
    for line in outfile[start:end]:
        if line.startswith("LCAOMinimize:"):
            mu = None
        elif lookkey in line:
            mu = float(line.split(lookkey)[1].split()[0])
    return mu


def get_kfolding_from_outfile_filepath(outfile_filepath: str | Path, slice_idx=-1) -> np.ndarray[int]:
    r"""Return kpt folding from output file.

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
    r"""
    key = "kpoint-folding "
    outfile = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(outfile, slice_idx=slice_idx)
    text = get_text_with_key_in_bounds(outfile_filepath, key, start, end)
    val = np.array(text.split(key)[1].strip().split(), dtype=int)
    return val


def get_nstates_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    r"""Get number of states from bandprojections file.

    Get the number of states from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    r"""
    return get__from_bandfile_filepath(bandfile_filepath, 0)


def get_nbands_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    r"""Get number of bands from bandprojections file.

    Get the number of bands from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    r"""
    return get__from_bandfile_filepath(bandfile_filepath, 2)


def get_nproj_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    r"""Get number of projections from bandprojections file.

    Get the number of projections from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    r"""
    return get__from_bandfile_filepath(bandfile_filepath, 4)


def get_nspecies_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    r"""Get number of species (ion types) from bandprojections file.

    Get the number of species (ion types) from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    int
    r"""
    return get__from_bandfile_filepath(bandfile_filepath, 6)


def get_norbsperatom_from_edata(edata: ElecData) -> list[int]:
    r"""Get number of orbitals per atom from ElecData object.

    Get the number of orbitals per atom from the ElecData object.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest.

    Returns
    -------
    list[int]
    r"""
    norbsperatom = []
    atom_orb_labels_dict = edata.atom_orb_labels_dict
    ion_names = edata.ion_names
    for el in ion_names:
        norbsperatom.append(len(atom_orb_labels_dict[el]))
    return norbsperatom


def get_norbsperatom_from_bandfile_filepath(bandfile_filepath: Path | str) -> list[int]:
    r"""Get number of orbitals per atom from bandprojections file.

    Get the number of orbitals per atom from the bandprojections file.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    list[int]
    r"""
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


def get_orb_l_u_from_bandfile_filepath(bandfile_filepath: Path | str) -> np.ndarray[int]:
    r"""Get the angular momentum l of each orbital projection from the bandprojections file header.

    JDFTx orders orbitals by species, then atom, then (within an atom) by l, then shell n, then m = -l..l
    (then spin copy for spinor calculations) - see SpeciesInfo::setAtomicOrbitals and
    SpeciesInfo::atomicOrbitalOffset in jdftx/electronic/SpeciesInfo_atomic.cpp. Each species header line
    reads "Symbol nAtoms nOrbitalsPerAtom lMax nShells(l=0) ... nShells(l=lMax)".

    Parameters
    ----------
    bandfile_filepath : Path | str
        Path to bandprojections file (or the bandprojections header file).

    Returns
    -------
    np.ndarray[int]
        Angular momentum of each orbital projection, in shape (nproj,).
    r"""
    nspecies = get_nspecies_from_bandfile_filepath(bandfile_filepath)
    bandfile = read_file(bandfile_filepath)
    l_u: list[int] = []
    for text in bandfile[2 : nspecies + 2]:
        tokens = text.split()
        natoms = int(tokens[1])
        norbs_per_atom = int(tokens[2])
        lmax = int(tokens[3])
        nshells = [int(tokens[4 + l]) for l in range(lmax + 1)]
        nspatial = sum((2 * l + 1) * nshells[l] for l in range(lmax + 1))
        # Spinor calculations hold 2 orbitals per spatial orbital (spin copies, or the two j's of relativistic
        # pseudopotentials), still grouped by l
        ncopies = norbs_per_atom // nspatial
        if ncopies * nspatial != norbs_per_atom or ncopies not in (1, 2):
            raise ValueError(
                f"Cannot infer orbital angular momenta for species {tokens[0]} "
                f"({norbs_per_atom} orbitals per atom, {nspatial} spatial orbitals from the shell counts)"
            )
        l_atom: list[int] = []
        for l in range(lmax + 1):
            l_atom += [l] * ((2 * l + 1) * nshells[l] * ncopies)
        l_u += l_atom * natoms
    return np.array(l_u, dtype=int)


def get_kpoint_offset_from_outfile_filepath(outfile_filepath: str | Path) -> np.ndarray[REAL_DTYPE] | None:
    r"""Return the single `kpoint` offset that JDFTx folded into the Monkhorst-Pack mesh, or None.

    Reads the input echo of the last JDFTx run in the out file. Returns None unless exactly one `kpoint` line is
    present (an explicit k-point list is not a folded mesh).
    r"""
    offsets = []
    text = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(text, slice_idx=-1)
    for line in text[start:end]:
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0] == "kpoint":
            offsets.append([float(x) for x in tokens[1:4]])
    if len(offsets) != 1:
        return None
    return np.array(offsets[0], dtype=np.float64)


def get_kmesh_unfolding(
    ks_raw: np.ndarray,
    w_raw: np.ndarray,
    nspin: int,
    kfolding,
    k0,
    rots: list[np.ndarray] | None = None,
    time_reversal: bool = True,
    tol: float = 1e-5,
) -> dict | None:
    r"""Map a k-mesh JDFTx reduced by symmetry back onto the full Monkhorst-Pack mesh.

    JDFTx folds the offset k0 into the mesh k = (k0 + i) / kfolding with i0 slowest and i2 fastest, each
    coordinate reduced to (-0.5, 0.5] (ElecInfo.cpp), and states are spin-major. It keeps one k-point per orbit
    of its space group, plus k <-> -k when `kpoint-reduce-inversion yes` (the default) and the group lacks
    inversion, and gives it the orbit's weight (`Symmetries::reduceKmesh`). This returns, for each full-mesh state
    in that order, the computed state it comes from, the operation that maps it there, and whether time reversal
    is needed (the projections are then complex-conjugated).

    An operation with lattice-coordinate rotation rot (x -> rot x + a) maps k to rot^-T k. JDFTx's own k-point map
    (the `kMap` dump) applies rot^T instead, i.e. the inverse operation; the group contains both. Operations
    without time reversal are preferred, then the earliest source and the earliest operation.

    Parameters
    ----------
    rots : list[np.ndarray] | None
        Integer rotation matrices of the operations (lattice coordinates). None means the identity only, i.e. a
        mesh reduced by time reversal alone.
    time_reversal : bool
        Whether k -> -k (complex conjugation) may be used.

    Returns
    -------
    dict | None
        "src_t": int array (nstates_full,), the computed state for each full-mesh state;
        "op_t": int array, the index into rots of the operation used;
        "conj_t": bool array, True where time reversal is used as well;
        "ks_t": float array (nstates_full, 3), the full-mesh k-points;
        "wfactor_t": float array, the factor that splits the source's weight over its images.
        None if the computed k-points are already the full mesh, or if these operations don't reproduce JDFTx's
        reduction with its weights (e.g. a point-group reduction without the operations, or an explicit k-point
        list).
    r"""
    ks_raw = np.asarray(ks_raw, dtype=np.float64)
    w_raw = np.asarray(w_raw, dtype=np.float64)
    if k0 is None or len(ks_raw) % nspin:
        return None
    nk_raw = len(ks_raw) // nspin
    kfolding = [int(x) for x in kfolding]
    i = np.stack(np.meshgrid(*[np.arange(n) for n in kfolding], indexing="ij"), -1).reshape(-1, 3)
    k0_mod = np.mod(np.asarray(k0, dtype=np.float64), 1.0)
    grid = (k0_mod[None, :] + i) / np.array(kfolding)[None, :]
    grid[grid > 0.5] -= 1.0
    nk_full = len(grid)
    if nk_raw >= nk_full:
        return None
    k_spin = ks_raw[:nk_raw]
    for s in range(1, nspin):  # JDFTx gives every spin channel the same k-points
        if not np.allclose(ks_raw[s * nk_raw : (s + 1) * nk_raw], k_spin, atol=tol):
            return None
    if rots is None:
        rots = [np.eye(3, dtype=np.int64)]
    # Best (conj, source, operation) per full-mesh point; lexicographic order gives the preference described above
    best = np.full((nk_full, 3), np.iinfo(np.int64).max, dtype=np.int64)
    for op, rot in enumerate(rots):
        rot_inv = np.round(np.linalg.inv(np.asarray(rot, dtype=np.float64))).astype(np.int64)
        for conj, sign in ((0, 1.0), (1, -1.0)) if time_reversal else ((0, 1.0),):
            img = sign * (k_spin @ rot_inv)  # rows: rot^-T k
            pos = img * np.array(kfolding)[None, :] - k0_mod[None, :]
            on_grid = np.all(np.abs(pos - np.round(pos)) < tol * max(kfolding), axis=1)
            idx3 = np.mod(np.round(pos).astype(np.int64), np.array(kfolding)[None, :])
            flat = (idx3[:, 0] * kfolding[1] + idx3[:, 1]) * kfolding[2] + idx3[:, 2]
            for src in np.flatnonzero(on_grid):
                cand = np.array([conj, src, op], dtype=np.int64)
                if tuple(cand) < tuple(best[flat[src]]):
                    best[flat[src]] = cand
    if np.any(best[:, 0] == np.iinfo(np.int64).max):
        return None
    conj_k, src_k, op_k = best[:, 0].astype(bool), best[:, 1], best[:, 2]
    mult = np.bincount(src_k, minlength=nk_raw)
    if np.any(mult == 0):
        return None
    # JDFTx weights must be (images represented) x (weight of one full-mesh k-point), in every spin channel
    for s in range(nspin):
        w_unit = w_raw[s * nk_raw : (s + 1) * nk_raw] / mult
        if not np.allclose(w_unit, w_unit[0], rtol=1e-4):
            return None
    src_t = np.concatenate([s * nk_raw + src_k for s in range(nspin)])
    return {
        "src_t": src_t,
        "op_t": np.tile(op_k, nspin),
        "conj_t": np.tile(conj_k, nspin),
        "ks_t": np.tile(grid, (nspin, 1)),
        "wfactor_t": 1.0 / mult[np.tile(src_k, nspin)],
    }


def get_time_reversal_unfolding(
    ks_raw: np.ndarray, w_raw: np.ndarray, nspin: int, kfolding, k0, tol: float = 1e-5
) -> dict | None:
    r"""Map a k-mesh JDFTx reduced only by time reversal (k <-> -k) back onto the full Monkhorst-Pack mesh.

    `get_kmesh_unfolding` with the identity as the only operation (finding F5). With `kpoint-reduce-inversion
    yes` (the default) and no other symmetry, JDFTx keeps one of each k / -k pair with twice the weight.
    Returns None if the computed k-points are already the full mesh, or if they aren't exactly a k <-> -k
    reduction of it with JDFTx's weights.
    r"""
    return get_kmesh_unfolding(ks_raw, w_raw, nspin, kfolding, k0, rots=None, time_reversal=True, tol=tol)


def parse_symfile(symfile_filepath: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""Return JDFTx's space-group operations from a `sym` file (`dump End Symmetries`), as (rot, a) pairs.

    Each operation is written by `Dump.cpp` as the three rows of the integer rotation matrix (lattice coordinates)
    and a line with the translation a (lattice coordinates), followed by a blank line: x -> rot x + a. The first
    operation is the identity.
    r"""
    rows = [line.split() for line in read_file(symfile_filepath) if line.strip()]
    if len(rows) % 4:
        raise ValueError(f"Unexpected format of the symmetries file {symfile_filepath}")
    ops = []
    for i in range(0, len(rows), 4):
        rot = np.array([[int(x) for x in row] for row in rows[i : i + 3]], dtype=np.int64)
        a = np.array([float(x) for x in rows[i + 3]], dtype=np.float64)
        ops.append((rot, a))
    return ops


def get_input_echo_commands(outfile_filepath: str | Path, slice_idx: int = -1) -> list[list[str]]:
    r"""Return the tokens of each command in the input echo of a JDFTx out file (continuation lines joined)."""
    text = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(text, slice_idx=slice_idx)
    commands, current = [], []
    in_echo = False
    for line in text[start:end]:
        if line.startswith("Input parsed successfully"):
            in_echo = True
            continue
        if not in_echo:
            continue
        if line.startswith("----------"):
            break
        stripped = line.strip()
        cont = stripped.endswith("\\")
        current += stripped.rstrip("\\").split()
        if not cont:
            if current:
                commands.append(current)
            current = []
    return commands


def get_symmetries_from_outfile_filepath(outfile_filepath: str | Path) -> tuple[str, list | None]:
    r"""Return JDFTx's `symmetries` mode and, for `symmetries manual`, its operations as (rot, a) pairs.

    The operations of `symmetries automatic` aren't printed in the out file; they need the `sym` file
    (`dump End Symmetries`, see parse_symfile). Returns (mode, None) in that case and for `symmetries none`.
    r"""
    mode, ops = "automatic", []
    for tokens in get_input_echo_commands(outfile_filepath):
        if tokens[0] == "symmetries" and len(tokens) > 1:
            mode = tokens[1]
        elif tokens[0] == "symmetry-matrix" and len(tokens) == 13:
            rot = np.array([int(float(x)) for x in tokens[1:10]], dtype=np.int64).reshape(3, 3)
            ops.append((rot, np.array([float(x) for x in tokens[10:13]], dtype=np.float64)))
    return mode, (ops if mode == "manual" and ops else None)


def get_orb_shell_u_from_bandfile_filepath(bandfile_filepath: Path | str) -> np.ndarray[int]:
    r"""Get a shell index for each orbital projection: orbitals share an index iff same atom, l and shell n.

    Uses the same header and orbital order as `get_orb_l_u_from_bandfile_filepath`. A shell holds all (2l+1) m
    (times the spinor copies), which the point group only mixes among themselves.
    r"""
    l_u = get_orb_l_u_from_bandfile_filepath(bandfile_filepath)
    nspecies = get_nspecies_from_bandfile_filepath(bandfile_filepath)
    bandfile = read_file(bandfile_filepath)
    shell_u: list[int] = []
    shell = 0
    for text in bandfile[2 : nspecies + 2]:
        tokens = text.split()
        natoms, norbs_per_atom, lmax = int(tokens[1]), int(tokens[2]), int(tokens[3])
        nshells = [int(tokens[4 + l]) for l in range(lmax + 1)]
        ncopies = norbs_per_atom // sum((2 * l + 1) * nshells[l] for l in range(lmax + 1))
        for _ in range(natoms):
            for l in range(lmax + 1):
                for _ in range(nshells[l]):
                    shell_u += [shell] * ((2 * l + 1) * ncopies)
                    shell += 1
    assert len(shell_u) == len(l_u)
    return np.array(shell_u, dtype=int)


def is_complex_bandfile_filepath(bandfile_filepath: str | Path) -> bool:
    r"""Determine if bandprojections file is complex.

    Determine if the bandprojections file is complex.
    Needed before attempting pCOHP analysis.

    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.

    Returns
    -------
    bool
    r"""
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
    r"""Parse kpts file.

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

    r"""
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


def parse_kptsfrom_bandfile_header(bandfile_header_filepath: str | Path) -> tuple[list[REAL_DTYPE], list[np.ndarray[REAL_DTYPE]], int]:
    r"""Parse kpts file.

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

    r"""
    wk_list: list[REAL_DTYPE] = []
    k_points_list: list[list[REAL_DTYPE]] = []
    kpt_lines = []
    with open(bandfile_header_filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                if ";" in line:
                    _line = line.split(";")[0].lstrip("#")
                    kpt_lines.append(_line)
    for line in kpt_lines:
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
    r"""Check if kpt is acceptable.

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
    r"""
    ksteps = [1 / kf for kf in kfolding]
    return all([np.isclose(kpt[i] % ksteps[i], 0) for i in range(3)])


def get_kfolding(lti: bool, outfile_filepath: str | Path, nspin: int, nstates: int) -> list[int]:
    r"""Get the k-point folding.

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
    r"""
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


def _get_arbitrary_wk(nk: int, nspin: int, legacy: bool = False) -> np.ndarray[REAL_DTYPE]:
    """Return uniform weights, one per state (spin and k-point), summing to 2 as JDFTx's do.

    For no-spin, each state's weight carries the spin factor 2. With legacy=True the weights are 1/nk, as before
    finding F10 (notes/theory_audit.md), which sums to nspin and halves every no-spin result.
    """
    wk = np.ones(nk * nspin)
    wk *= (1.0 if legacy else 2.0 / nspin) / nk
    return wk


def get_wk_t(
    kptsfile_filepath: Path | str | None, nspin: int, kfolding: list[int], lti: bool, legacy: bool = False
) -> np.ndarray[REAL_DTYPE]:
    r"""Return weights for k-points.

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
    legacy : bool
        Reproduce the pre-F5 check (notes/theory_audit.md), which compared the number of weights (one per state,
        nspin * nk) with nk. For spin-polarized symmetry-reduced meshes that always failed, so the file's
        non-uniform weights were replaced by uniform ones. Also keeps the uniform fallback weights at 1/nk (they
        now sum to 2, finding F10).
    r"""
    nk = np.prod(kfolding)
    if kptsfile_filepath is None:
        wk = _get_arbitrary_wk(nk, nspin, legacy=legacy)
    else:
        wk, ks, nstates = parse_kptsfile(kptsfile_filepath)
        wk = np.array(wk, dtype=REAL_DTYPE)
        if not lti:
            # The kPts file holds one weight per state (spin and k-point)
            n_expected = nk if legacy else nspin * nk
            if not len(wk) == n_expected:
                warnings.warn(
                    "Kpts file does not match expected number of k-points. k-point weights will be meaningless.",
                    stacklevel=2,
                )
                wk = _get_arbitrary_wk(nk, nspin, legacy=legacy)
    return wk


# def _get_ks_sabc(kfolding: list[int], kptsfile_filepath: Path | str | None, nspin: int, lti: bool) -> np.ndarray[REAL_DTYPE]:
#     r""" Return the k-point coordinates.

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
#     r"""
#     if kptsfile_filepath is None or not lti:
#         ks = np.ones([nk * nspin, 3]) * np.nan
#     else:
#         wk, ks, nstates = parse_kptsfile(kptsfile_filepath)
#         ks = np.array(ks, dtype=REAL_DTYPE)
#     ks_sabc = ks.reshape([nspin, kfolding[0], kfolding[1], kfolding[2], 3])
#     return ks_sabc


# def _get_ks_sabc(kfolding: list[int], kptsfile_filepath: Path | str | None, nspin: int, lti: bool) -> np.ndarray[REAL_DTYPE]:
#     r""" Return the k-point coordinates.

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
#     r"""
#     if kptsfile_filepath is None or not lti:
#         ks = np.ones([nk * nspin, 3]) * np.nan
#     else:
#         wk, ks, nstates = parse_kptsfile(kptsfile_filepath)
#         ks = np.array(ks, dtype=REAL_DTYPE)
#     ks_sabc = ks.reshape([nspin, kfolding[0], kfolding[1], kfolding[2], 3])
#     return ks_sabc


def get_ks_t(kptsfile_filepath: Path | str, bandfile_header_filepath: Path | str | None) -> np.ndarray[REAL_DTYPE]:
    r"""Return the k-point coordinates.

    Return the k-point coordinates. Assumes k-point file exists and mesh is regular.

    Parameters
    ----------
    kptsfile_filepath : Path | str
        Path to kpts file.
    nspin : int
        Number of spins.
    kfolding : list[int]
        kpt folding
    r"""
    _ks = None
    if ((not Path(kptsfile_filepath).exists()) and (not bandfile_header_filepath is None)):
        wk, _ks, nstates = parse_kptsfrom_bandfile_header(bandfile_header_filepath)
    else:
        wk, _ks, nstates = parse_kptsfile(kptsfile_filepath)
    if not _ks is None:
        ks = np.array(_ks, dtype=REAL_DTYPE)
        return ks
    else:
        return None


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


def get_e_tj_helper(eigfile_filepath: str | Path, nstates: int, nbands: int) -> np.ndarray[REAL_DTYPE]:
    r"""Return eigenvalues from file.

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
    r"""
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
    r"""Return projections from file in sabcju shape.

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
    r"""
    proj_shape = [nspin] + list(kfolding) + [nbands, nproj]
    proj_tju = get_proj_tju_from_file(bandfile_filepath)
    proj_sabcju = proj_tju.reshape(proj_shape)
    del proj_tju
    return proj_sabcju


def get_proj_tju_from_file(bandfile_filepath: Path | str) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
    r"""Return projections from file in tju shape.

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
    r"""
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
    expected_length = 2 + nspecies + (nstates * (1 + nbands))
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

def save_bandfile_header(bandfile_filepath: Path, bandfile_header_filepath: Path):
    """ Save the header of bandfile_filepath to bandfile_header_filepath.

    Save the header of the file at bandfile_filepath to bandfile_header_filepath.
    """
    bandfile = read_file(bandfile_filepath)
    header_text: list[str] = []
    nspecies = get_nspecies_from_bandfile_filepath(bandfile_filepath)
    for line, text in enumerate(bandfile):
        if line >= nspecies + 2:
            # Only include state headers after general header
            if text.startswith("#"):
                header_text.append(text)
        else:
            header_text.append(text)
    with open(bandfile_header_filepath, "w") as f:
        f.write("".join(header_text))


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
    out += imags
    out *= 1j
    out += reals
    # out += reals + 1j * imags
    return out


def get_outfile_start_lines(
    texts: list[str],
    start_key: str = "*************** JDFTx",
    add_end: bool = False,
) -> list[int]:
    r"""Get start line numbers for JDFTx calculations.

    Get the line numbers corresponding to the beginning of separate JDFTx calculations
    (in case of multiple calculations appending the same out file).

    Parameters:
    -----------
    texts: list[str]
        output of read_file for out file

    r"""
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
    r"""Get slice bounds for JDFTx calculation.

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
    r"""
    start_lines = get_outfile_start_lines(outfile, add_end=True)
    outfile_bounds_list = [[start_lines[i], start_lines[i + 1]] for i in range(len(start_lines) - 1)]
    if slice_idx >= len(outfile_bounds_list):
        raise ValueError(f"Slice index {slice_idx} out of bounds.")
    start, end = outfile_bounds_list[slice_idx]
    return start, end


def get_outfile_start_line(
    outfile: list[str],
) -> int:
    r"""Get start line for JDFTx calculation.

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
    r"""Get arbitrary integer from header of bandprojections file.

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


