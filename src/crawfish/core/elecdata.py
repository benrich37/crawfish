"""Module for handling electronic data from JDFTx calculations.

This module contains the ElecData class, which is used to store and manipulate
electronic data from JDFTx calculations.
"""

from __future__ import annotations
import numpy as np
from crawfish.io.data_parsing import (
    get_mu_from_outfile_filepath,
    get_nstates_from_bandfile_filepath,
    get_nbands_from_bandfile_filepath,
    get_nproj_from_bandfile_filepath,
    get_norbsperatom_from_bandfile_filepath,
    get_e_tj_helper,
    get_proj_sabcju_helper,
    get_proj_tju_from_file,
    is_complex_bandfile_filepath,
    _get_lti_allowed,
    get_kfolding,
    get_ks_sabc,
    get_wk_sabc,
)
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
from crawfish.utils.indexing import get_kmap_from_edata
from crawfish.core.operations.matrix import get_h_uu_p_uu
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.io.jdftx.jdftxinfile import JDFTXInfile
from pymatgen.io.jdftx.jdftxoutfile import JDFTXOutfile
from pymatgen.core.structure import Structure
from pathlib import Path
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


class ElecData:
    """Class for handling electronic data from JDFTx calculations.

    The ElecData class is used to store and manipulate electronic data from JDFTx
    calculations.
    """

    bandfile_suffix: str = "bandProjections"
    kptsfile_suffix: str = "kPts"
    eigfile_suffix: str = "eigenvals"
    fillingsfile_suffix: str = "fillings"
    outfile_suffix: str = "out"
    gvecfile_suffix: str = "Gvectors"
    wfnfile_suffix: str = "wfns"
    #
    bandfile_filepath: Path | None = None
    kptsfile_filepath: Path | None = None
    eigfile_filepath: Path | None = None
    fillingsfile_filepath: Path | None = None
    outfile_filepath: Path | None = None
    gvecfile_filepath: Path | None = None
    wfnfile_filepath: Path | None = None
    #
    fprefix: str = ""
    #
    _nstates: int | None = None
    _nbands: int | None = None
    _nproj: int | None = None
    _nspin: int | None = None
    _e_tj: np.ndarray[REAL_DTYPE] | None = None
    _proj_tju: np.ndarray[REAL_DTYPE] | np.ndarray[COMPLEX_DTYPE] | None = None

    _p_uu: np.ndarray[REAL_DTYPE] | None = None
    _h_uu: np.ndarray[REAL_DTYPE] | None = None
    _s_uu: np.ndarray[REAL_DTYPE] | None = None
    _occ_tj: np.ndarray[REAL_DTYPE] | None = None
    _mu: REAL_DTYPE | None = None
    _norbsperatom: list[int] | None = None
    _orbs_idx_dict: dict | None = None
    _kmap: list[str] | None = None
    _complex_bandprojs: bool | None = None
    _norm: int | None = None
    #
    _wk_t: np.ndarray[REAL_DTYPE] | None = None
    _ks_t: np.ndarray[REAL_DTYPE] | None = None
    _kfolding: list[int] | None = None
    _lti_allowed: bool | None = None
    #
    norm_idx: int | None = None
    trim_excess_bands: bool = True

    @property
    def infile(self) -> JDFTXInfile:
        """Return JDFTx input file object from calculation.

        Return JDFTx input file object from calculation.
        """
        return JDFTXInfile.from_file(self.calc_dir / f"{self.fprefix}in")

    @property
    def outfile(self) -> JDFTXOutfile:
        """Return JDFTx output file object from calculation.

        Return JDFTx output file object from calculation.
        """
        return JDFTXOutfile.from_file(self.outfile_filepath)

    @property
    def bandstructure(self) -> BandStructure:
        """Return band structure object from calculation.

        Return band structure object from calculation.
        """
        raise NotImplementedError("BandStructure object not yet implemented")

    @property
    def structure(self) -> Structure:
        """Return pymatgen structure object from calculation.

        Return pymatgen structure object from calculation.
        """
        return self.outfile.slices[-1].structure

    @property
    def nspin(self) -> int:
        """Return number of spins in calculation.

        Return number of spins in calculation.
        """
        if self._nspin is None:
            self._nspin = self.outfile.nspin
        return self._nspin

    @property
    def nstates(self) -> int:
        """Return number of states in calculation.

        Return number of states in calculation.
        """
        if self._nstates is None:
            self._nstates = get_nstates_from_bandfile_filepath(self.bandfile_filepath)
        return self._nstates

    @property
    def nbands(self) -> int:
        """Return number of bands in calculation.

        Return number of bands in calculation.
        """
        if self._nbands is None:
            self._nbands = get_nbands_from_bandfile_filepath(self.bandfile_filepath)
        return self._nbands

    @property
    def nproj(self) -> int:
        """Return number of projections in calculation.

        Return number of projections in calculation.
        """
        if self._nproj is None:
            self._nproj = get_nproj_from_bandfile_filepath(self.bandfile_filepath)
        return self._nproj

    @property
    def norbsperatom(self) -> list[int]:
        """Return number of orbitals per atom in calculation.

        Return number of orbitals per atom in calculation.
        """
        if self._norbsperatom is None:
            self._norbsperatom = get_norbsperatom_from_bandfile_filepath(self.bandfile_filepath)
        return self._norbsperatom

    @property
    def ion_names(self) -> list[str]:
        """Return list of atom labels in calculation.

        Return list of atom labels in calculation.
        """
        return [specie.name for specie in self.structure.species]

    @property
    def mu(self) -> REAL_DTYPE:
        """Return chemical potential of calculation.

        Return chemical potential (fermi level) of calculation.
        """
        if self._mu is None:
            self._mu = get_mu_from_outfile_filepath(self.outfile_filepath)
        return self._mu

    @property
    def e_tj(self) -> np.ndarray[REAL_DTYPE]:
        """Return eigenvalues of calculation.

        Return eigenvalues of calculation in shape (nstate, nbands).
        """
        if self._e_tj is None:
            self._e_tj = get_e_tj_helper(self.eigfile_filepath, self.nstates, self.nbands)
        return self._e_tj


    @property
    def proj_tju(self) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
        """Return projections of calculation.

        Return projections of calculation in shape (nstates, nbands, nproj).
        """
        if self._proj_tju is None:
            self._proj_tju = get_proj_tju_from_file(self.bandfile_filepath)
        return self._proj_tju
    
    @property
    def occ_tj(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return occupations of calculation.

        Return occupations of calculation in shape (nspin, kfolding[0], kfolding[1], kfolding[2], nbands).
        """
        if self.fillingsfile_filepath is None:
            warnings.warn("No fillings file found, thus no occupations can be read")
            # TODO: Write a function to generate occupations from smearing and fermi level
            return None
        if self._occ_tj is None:
            fillings = np.fromfile(self.fillingsfile_filepath)
            fillings = np.array(fillings, dtype=REAL_DTYPE)
            self._occ_tj = fillings
        return self._occ_tj

    def set_mat_uu(self) -> None:
        h_uu, p_uu, s_uu = get_h_uu_p_uu(self.proj_tju, self.e_tj, self.occ_tj, self.wk_t)
        self._h_uu = h_uu
        self._p_uu = p_uu
        self._s_uu = s_uu

    @property
    def h_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return hamiltonian matrix of calculation.

        Return hamiltonian matrix of calculation in shape (nstates, nstates).
        """
        if self._h_uu is None:
            self.set_mat_uu()
        return self._h_uu

    @property
    def p_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return projection matrix of calculation.

        Return projection matrix of calculation in shape (nstates, nstates).
        """
        if self._p_uu is None:
            self.set_mat_uu()
        return self._p_uu

    @property
    def s_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return overlap matrix of calculation.

        Return overlap matrix of calculation in shape (nstates, nstates).
        """
        if self._s_uu is None:
            self.set_mat_uu()
        return self._s_uu

    @property
    def ion_orb_u_dict(self) -> dict[str, int]:
        """Return dictionary mapping atom labels to orbital indices.

        Return dictionary mapping each atom (using key of format 'el #n' (str)) to indices
        (int) of all atomic orbital projections (in 0-based indexing) belonging to
        said atom.
        """
        return self.orbs_idx_dict

    @property
    def orbs_idx_dict(self) -> dict[str, int]:
        """Return dictionary mapping orbital indices to atom labels.

        Return dictionary mapping each atom (using key of format 'el #n' (str)) to indices
        (int) of all atomic orbital projections (in 0-based indexing) belonging to
        said atom.
        """
        if self._orbs_idx_dict is None:
            unique_ion_names, ion_counts = count_ions(self.ion_names)
            self._orbs_idx_dict = orbs_idx_dict_helper(unique_ion_names, ion_counts, self.norbsperatom)
        return self._orbs_idx_dict

    @property
    def kmap(self) -> list[str]:
        """Return mapping of atom indices to atom labels.

        Return mapping of atom indices to atom labels.
        """
        if self._kmap is None:
            self._kmap = get_kmap_from_edata(self)
        return self._kmap

    @property
    def wk_t(self) -> np.ndarray[REAL_DTYPE]:
        """Return kpoint weights.

        Return kpoint weights in shape (kfolding[0], kfolding[1], kfolding[2]).
        """
        if self._wk_t is None:
            self._wk_t = get_wk_t(self.kptsfile_filepath, self.lti_allowed)
        return self._wk_t

    @property
    def ks_t(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return kpoint coordinates.

        Return kpoint coordinates in shape (nspin, kfolding[0], kfolding[1], kfolding[2], 3).
        """
        if self._ks_t is None and (self.lti_allowed and self.kptsfile_filepath is not None):
            self._ks_t = get_ks_t(self.kptsfile_filepath)
        return self._ks_t

    @property
    def kfolding(self) -> list[int]:
        """Return kpoint folding.

        Return kpoint folding as a list of 3 integers.
        """
        if self._kfolding is None:
            self._kfolding = get_kfolding(self.lti_allowed, self.outfile_filepath, self.nspin, self.nstates)
        return self._kfolding

    @property
    def bandprojfile_is_complex(self) -> bool:
        """Return if band projections are complex.

        Return if band projections are complex. (required for coop/cohp analysis)
        """
        if self._complex_bandprojs is None:
            self._complex_bandprojs = is_complex_bandfile_filepath(self.bandfile_filepath)
        return self._complex_bandprojs

    @property
    def lti_allowed(self) -> bool:
        """Return if LTI is allowed.

        Return if LTI (linear tetrahedral integration) is allowed (dependent on whether jdftx calculation
        automatically reduced the k-point mesh to a non-MK pack grid, and the existence of a kpts file
        to refer to).
        """
        if self._lti_allowed is None:
            self._lti_allowed = _get_lti_allowed(self.bandfile_filepath, self.outfile_filepath)
        return self._lti_allowed

    @classmethod
    def from_calc_dir(cls, calc_dir: Path, prefix: str | None = None, alloc_elec_data=True):
        """Create ElecData instance from JDFTx calculation directory.

        Create ElecData instance from JDFTx calculation directory.

        Parameters
        ----------
        calc_dir : Path
            Path to calculation directory.
        prefix : str, optional
            Prefix of files in calculation directory, by default None
        """
        instance = ElecData(calc_dir=Path(calc_dir), prefix=prefix, alloc_elec_data=alloc_elec_data)
        return instance

    def __init__(self, calc_dir: Path, prefix: str | None = None, alloc_elec_data=True):
        """Initialize ElecData instance.

        Initialize ElecData instance.

        Parameters
        ----------
        calc_dir : Path
            Path to calculation directory.
        prefix : str, optional
            Prefix of files in calculation directory, by default None
        """
        self.fprefix = self._get_fprefix(prefix)
        self.calc_dir = Path(calc_dir)
        self._set_files_paths()
        # self._parse_bandfile_header()
        # self._alloc_kpt_data()
        if alloc_elec_data:
            self._alloc_elec_data()

    def _set_files_paths(self, optional_sufficies: list[str] = ["gvec", "wfn", "kpts", "fillings"]):
        if self.fprefix is None:
            raise RuntimeError("File prefix (fprefix) must be set before setting file paths")
        for filetype in ["band", "kpts", "eig", "fillings", "out", "gvec", "wfn"]:
            suffix = getattr(self, f"{filetype}file_suffix")
            filepath = self._get_filepath_generic(suffix)
            if filepath is None and filetype not in optional_sufficies:
                raise FileNotFoundError(
                    f"File not found for suffix {suffix} \n hint: Make sure prefix does not contain '.'"
                )
            setattr(self, f"{filetype}file_filepath", filepath)

    def _get_filepath_generic(self, suffix: str) -> Path | None:
        filepath: Path = self.calc_dir / f"{self.fprefix}{suffix}"
        if not filepath.exists():
            return None
        return filepath

    @staticmethod
    def _get_fprefix(prefix: str | None = None):
        if prefix is None:
            return ""
        elif isinstance(prefix, str):
            if not len(prefix) or prefix == "$VAR":
                return ""
            else:
                return f"{prefix}."
        else:
            raise ValueError("prefix must be a string or None")

    #######

    def _alloc_elec_data(self):
        _ = self.proj_sabcju
        if self.fillingsfile_filepath is not None:
            _ = self.occ_sabcj
        _ = self.e_sabcj

    #################

    def unnorm_projs(self) -> None:
        """Remove normalization from projections.

        Remove normalization from projections.
        """
        self._proj_sabcju = None
        _ = self.proj_sabcju
        self.norm_idx = None
        return None

    def norm_projs(self) -> None:
        """Normalize projections.

        Normalize projections.
        """
        self.norm_projs_t1()
        return None

    def norm_projs_t1(self, mute_excess_bands=False) -> None:
        """Normalize projections for bands.

        Normalize projections for bands. (np.sum(np.abs(proj_tju) ** 2[:,j,:]) = nstates)
        nStates is chosen as the normalization factor for each band as bands that are completely
        covered by the projections will have a sum of nStates after orthonormalization.

        Parameters
        ----------
        mute_excess_bands : bool, optional
            Set all projection values for bands beyond nProj to zero, by default False.
            (Helpful to set up 1:1 basis change between bands and orbitals)
        """
        if self.norm_idx is None:
            proj_tju = self.proj_tju
            proj_tju = _norm_projs_for_bands(
                proj_tju, self.nstates, self.nbands, self.nproj, restrict_band_norm_to_nproj=mute_excess_bands
            )
            proj_shape = list(np.shape(self.e_sabcj))
            proj_shape.append(self.nproj)
            self._proj_sabcju = proj_tju.reshape(proj_shape)
            self.norm_idx = 1
        elif self.norm_idx != 1 or not mute_excess_bands == self.xs_bands_muted:
            self.unnorm_projs()
            self.norm_projs_t1(mute_excess_bands=mute_excess_bands)
        return None

    def norm_projs_t2(self, mute_excess_bands=False) -> None:
        """Normalize projections for orbitals.

        Normalize projections for orbitals. (np.sum(np.abs(proj_tju) ** 2[:,:,u]) = nstates)
        nstates is chosen somewhat arbitrarily as the normalization factor for each orbital
        to make the orbitals equivalent to bands after a basis change.
        """
        if self.norm_idx is None:
            proj_tju = self.proj_tju
            proj_tju = _norm_projs_for_orbs(
                proj_tju, self.nstates, self.nbands, self.nproj, mute_excess_bands=mute_excess_bands
            )
            proj_shape = list(np.shape(self.e_sabcj))
            proj_shape.append(self.nproj)
            self._proj_sabcju = proj_tju.reshape(proj_shape)
            self.norm_idx = 2
        elif self.norm_idx != 2 or not mute_excess_bands == self.xs_bands_muted:
            self.unnorm_projs()
            self.norm_projs_t2(mute_excess_bands=mute_excess_bands)
        return None

    def norm_projs_t3(self) -> None:
        """Orthogonalize bands with Lowdin symmetric orthogonalization.

        Orthogonalize bands with Lowdin symmetric orthogonalization.
        (np.tensordot(proj_sabcju[:,:,:,:,j,:].conj().T, proj_sabcju[:,:,:,:,k,:],
                      axes = ([4,3,2,1,0], [0,1,2,3,4])) = delta(j,k))
        """
        if self.norm_idx is None:
            proj_tju = self.proj_tju
            # proj_sabcju = self.proj_sabcju
            proj_tju = _norm_projs_for_orbs(proj_tju, self.nstates, self.nbands, self.nproj, mute_excess_bands=False)
            proj_sabcju = proj_tju.reshape(list(np.shape(self.e_sabcj)) + [self.nproj])
            self._proj_sabcju = los_projs_for_bands(proj_sabcju)
            self.norm_idx = 3
        elif self.norm_idx != 3:
            self.unnorm_projs()
            self.norm_projs_t3()
        return None


def los_projs_for_orbs(proj_tju: np.ndarray[COMPLEX_DTYPE]) -> np.ndarray[COMPLEX_DTYPE]:
    """Perform LOS on projections for projection orthogonality.

    Perform Lowdin symmetric orthogonalization on projections for orbital projection orthogonality.

    Parameters
    ----------
    proj_sabcju : np.ndarray[COMPLEX_DTYPE]
        Projections in shape (nstates, nbands, nproj).
    """
    low_proj_tju = np.zeros_like(proj_tju)
    nstates = np.shape(proj_tju)[0]
    for t in range(nstates):
        s_uu = np.tensordot(proj_tju[t].conj().T, proj_tju[t], axes=([1], [0]))
        eigs, low_u = np.linalg.eigh(s_uu)
        nsqrt_ss_uu = np.eye(len(eigs)) * (eigs ** (-0.5))
        low_s_uu = np.dot(low_u, np.dot(nsqrt_ss_uu, low_u.T.conj()))
        low_proj_tju[t,:,:] += np.tensordot(proj_tju[t], low_s_uu, axes=([1], [0]))
    return low_proj_tju

def los_projs_for_bands(proj_tju: np.ndarray[COMPLEX_DTYPE]) -> np.ndarray[COMPLEX_DTYPE]:
    """Perform LOS on projections for band orthogonality.

    Perform Lowdin symmetric orthogonalization on projections for band orthogonality.

    Parameters
    ----------
    proj_sabcju : np.ndarray[COMPLEX_DTYPE]
        Projections in shape (nstates, nbands, nproj).
    """
    low_proj_tju = np.zeros_like(proj_tju)
    nstates = np.shape(proj_tju)[0]
    for t in range(nstates):
        s_uu = np.tensordot(proj_tju[t].conj().T, proj_tju[t], axes=([0], [1]))
        eigs, low_u = np.linalg.eigh(s_uu)
        nsqrt_ss_uu = np.eye(len(eigs)) * (eigs ** (-0.5))
        low_s_uu = np.dot(low_u, np.dot(nsqrt_ss_uu, low_u.T.conj()))
        low_proj_tju[t,:,:] += np.tensordot(proj_tju[t], low_s_uu, axes=([0], [1]))
    return low_proj_tju


# def get_t1_loss(proj_tju, nStates):
#     v = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=-1), axis=0) - nStates)
#     return v


# def get_t2_loss(proj_tju, nStates):
#     v = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=0), axis=1) - nStates)
#     return v


# def get_t3_loss(proj_tju, nStates):
#     v2 = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=0), axis=1) - nStates)
#     v1 = np.sum(np.sum(np.sum(np.abs(proj_tju) ** 2, axis=-1), axis=0) - nStates)
#     return abs(v1) + abs(v2)




def normalize_square_proj_tju(proj_tju: np.ndarray[COMPLEX_DTYPE], nloops=1000, conv=0.01):
    """ Normalize projection matrices proj_tju.
    
    Normalize projection matrices proj_tju. Requires nproj == nbands. Performs the following:
    1. For each state t:
        1.a. For each band j:
            1.a.i. Sums proj_tju[t,j,u]^*proj_tju[t,j,u] for a given j and all u to "asum"
            1.a.ii. Divides proj_tju[t,j,:] by 1/(asum**0.5)
            1.b.iii. Adds |1-asum| to loss metric for state t
        1.b. For each projection u:
            1.b.i. Sums proj_tju[t,:,u]^*proj_tju[t,:,u] for a given u and all j to "asum"
            1.b.ii. Divides proj_tju[t,:,u] by 1/(asum**0.5)
            1.b.iii. Adds |1-asum| to loss metric for state t
        1.c. If loss metric for state t exceeds the "conv" threshold and 1.a/1.b have been
                performed for less than nloops, reset the loss metric and repeat 1.a/1.b for state t.
                Otherwise, move to the next state.
    2. Return the normalized proj_tju and the losses at each loop for each state.

    Parameters
    ----------
    proj_tju : np.ndarray[COMPLEX_DTYPE]
        Projection matrices in shape (nstates, nbands, nproj).
    nloops : int, optional
        Maximum number of loops to perform normalization, by default 1000.
    conv : float, optional
        Convergence threshold for loss metric, by default 0.01.
    """
    proj_tju_norm = proj_tju.copy()
    nproj: np.int64 = np.shape(proj_tju)[2]
    nstates: np.int64 = np.shape(proj_tju)[0]
    nbands: np.int64 = np.shape(proj_tju)[1]
    losses = np.zeros([nstates, nloops], dtype=REAL_DTYPE)
    proj_tju_norm = proj_tju.copy()
    return _normalize_proj_tju(proj_tju_norm, nloops, conv, losses, nstates, nproj, nbands)

@jit(nopython=True)
def _normalize_square_proj_tju(
    proj_tju_norm: np.ndarray[COMPLEX_DTYPE], nloops: int, conv: REAL_DTYPE, losses:np.ndarray[REAL_DTYPE],
    nstates: int, nproj: int, nbands
    ) -> tuple[np.ndarray[COMPLEX_DTYPE], np.ndarray[REAL_DTYPE]]:
    for t in range(nstates):
        for i in range(nloops):
            for j in range(nbands):
                asum: REAL_DTYPE = 0
                for u in range(nproj):
                    asum += np.real(np.conj(proj_tju_norm[t,j,u])*proj_tju_norm[t,j,u])
                proj_tju_norm[t,j,:] *= 1/(asum**0.5)
                losses[t,i] += np.abs(1-asum)
            for u in range(nproj):
                asum: REAL_DTYPE = 0
                for j in range(nbands):
                    asum += np.real(np.conj(proj_tju_norm[t,j,u])*proj_tju_norm[t,j,u])
                proj_tju_norm[t,:,u] *= 1/(asum**0.5)
                losses[t,i] += np.abs(1-asum)
            if losses[t,i] < conv:
                break
    return proj_tju_norm, losses


@jit(nopython=True)
def _norm_projs_for_bands_jit_helper_1(nProj, nStates, nBands, proj_tju, j_sums):
    for u in range(nProj):
        for t in range(nStates):
            for j in range(nBands):
                j_sums[j] += abs(np.conj(proj_tju[t, j, u]) * proj_tju[t, j, u])
    for j in range(nBands):
        proj_tju[:, j, :] *= 1 / np.sqrt(j_sums[j])
    # proj_tju *= np.sqrt(nStates)
    return proj_tju


@jit(nopython=True)
def _mute_xs_bands(nProj, nBands, proj_tju):
    for j in range(nProj, nBands):
        proj_tju[:, j, :] *= 0
    return proj_tju


def _norm_projs_for_bands(proj_tju, nStates, nBands, nProj, restrict_band_norm_to_nproj=False):
    j_sums = np.zeros(nBands)
    proj_tju = _norm_projs_for_bands_jit_helper_1(nProj, nStates, nBands, proj_tju, j_sums)
    if restrict_band_norm_to_nproj:
        proj_tju = _mute_xs_bands(nProj, nBands, proj_tju)
    return proj_tju


@jit(nopython=True)
def _norm_projs_for_orbs_jit_helper(nProj, nStates, nBands, proj_tju, u_sums):
    for u in range(nProj):
        for t in range(nStates):
            for j in range(nBands):
                u_sums[u] += abs(np.conj(proj_tju[t, j, u]) * proj_tju[t, j, u])
    for u in range(nProj):
        proj_tju[:, :, u] *= 1 / np.sqrt(u_sums[u])
    # proj_tju *= np.sqrt(nStates)
    # proj_tju *= np.sqrt(2)
    # proj_tju *= np.sqrt(nStates*nBands/nProj)
    return proj_tju


def _norm_projs_for_orbs(proj_tju, nStates, nBands, nProj, mute_excess_bands=False):
    if mute_excess_bands:
        proj_tju = _mute_xs_bands(nProj, nBands, proj_tju)
    u_sums = np.zeros(nProj)
    # TODO: Identify the error types raised by division by zero within a jitted function
    # (if certain orbitals are only represented by bands above nProj, this error should
    # be clarified to the user)
    proj_tju = _norm_projs_for_orbs_jit_helper(nProj, nStates, nBands, proj_tju, u_sums)
    return proj_tju


def count_ions(ionnames: list[str]) -> tuple[list[str], list[int]]:
    """Count the number of each unique atom in a list of atom labels.

    Return a list of unique atom labels and a list of the number each atom type
    appears in the input list.

    Parameters
    ----------
    ionnames : list[str]
        List of atom labels.

    Returns
    -------
    ion_names : list[str]
        List of unique atom labels.
    ion_counts : list[int]
        List of number of each atom type in input list.
    """
    unique_ion_names = []
    unique_ion_counts = []
    for name in ionnames:
        if name not in unique_ion_names:
            unique_ion_names.append(name)
            unique_ion_counts.append(ionnames.count(name))
    return unique_ion_names, unique_ion_counts


def orbs_idx_dict_helper(ion_names: list[str], ion_counts: list[int], norbsperatom: list[int]) -> dict:
    """Return a dictionary mapping orbital indices to atom labels.

    Returns a dictionary mapping each atom (using key of format 'el #n' (str),
    where el is atom id, and n is number of specific atom as it appears in JDFTx
    out file using 1-based indexing) to indices (int) of all atomic orbital
    projections (in 0-based indexing) belonging to said atom.

    Parameters
    ----------
    ion_names : list[str]
        List of atom labels.
    ion_counts : list[int]
        List of atom counts.
    norbsperatom : list[int]
        List of number of orbitals per atom.

    Returns
    -------
    orbs_dict_out : dict
        Dictionary mapping atom labels to orbital indices.
    """
    orbs_dict_out = {}
    iOrb = 0
    atom = 0
    for i, count in enumerate(ion_counts):
        for atom_num in range(count):
            atom_label = ion_names[i] + " #" + str(atom_num + 1)
            norbs = norbsperatom[atom]
            orbs_dict_out[atom_label] = list(range(iOrb, iOrb + norbs))
            iOrb += norbs
            atom += 1
    return orbs_dict_out


def _get_data_and_path(data: ElecData | None, calc_dir: Path | str | None):
    if data is None and calc_dir is not None:
        if isinstance(calc_dir, str):
            calc_dir = Path(calc_dir)
        data = ElecData.from_calc_dir(calc_dir)
    elif calc_dir is None and data is not None:
        calc_dir = data.calc_dir
    elif data is not None and calc_dir is not None:
        if data.calc_dir != calc_dir:
            raise ValueError("calc_dir and data.calc_dir must be the same")
    else:
        raise ValueError("Must provide at least a calc_dir or a data=ElecData (both cannot be none)")
    return data, calc_dir
