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
    get_e_sabcj_helper,
    get_proj_sabcju_helper,
    is_complex_bandfile_filepath,
    _get_lti_allowed,
    get_kfolding,
    get_ks_sabc,
    get_wk_sabc,
)
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
from crawfish.utils.indexing import get_kmap_from_edata
from crawfish.core.operations.matrix import get_p_uvjsabc, get_h_uvsabc
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
    _e_sabcj: np.ndarray[REAL_DTYPE] | None = None
    _proj_sabcju: np.ndarray[REAL_DTYPE] | np.ndarray[COMPLEX_DTYPE] | None = None
    _p_uvjsabc: np.ndarray[REAL_DTYPE] | None = None
    _h_uvsabc: np.ndarray[REAL_DTYPE] | None = None
    _occ_sabcj: np.ndarray[REAL_DTYPE] | None = None
    _mu: REAL_DTYPE | None = None
    _norbsperatom: list[int] | None = None
    _orbs_idx_dict: dict | None = None
    _kmap: list[str] | None = None
    _complex_bandprojs: bool | None = None
    _norm: int | None = None
    #
    _wk_sabc: np.ndarray[REAL_DTYPE] | None = None
    _ks_sabc: np.ndarray[REAL_DTYPE] | None = None
    _kfolding: list[int] | None = None
    _lti_allowed: bool | None = None
    #
    norm_idx: int | None = None
    xs_bands_muted: bool = False

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
    def e_sabcj(self) -> np.ndarray[REAL_DTYPE]:
        """Return eigenvalues of calculation.

        Return eigenvalues of calculation in shape (nspin, kfolding[0], kfolding[1], kfolding[2], nbands).
        """
        if self._e_sabcj is None:
            self._e_sabcj = get_e_sabcj_helper(self.eigfile_filepath, self.nspin, self.nbands, self.kfolding)
        return self._e_sabcj


    @property
    def proj_sabcju(self) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
        """Return projections of calculation.

        Return projections of calculation in shape (nspin, kfolding[0], kfolding[1], kfolding[2], nbands, nproj).
        """
        if self._proj_sabcju is None:
            self._proj_sabcju = get_proj_sabcju_helper(
                self.bandfile_filepath, self.nspin, self.kfolding, self.nbands, self.nproj
            )
        return self._proj_sabcju

    @property
    def proj_tju(self) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
        """Return projections of calculation.

        Return projections of calculation in shape (nstates, nbands, nproj).
        """
        return self.proj_sabcju.reshape(self.nstates, self.nbands, self.nproj)

    @property
    def occ_sabcj(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return occupations of calculation.

        Return occupations of calculation in shape (nspin, kfolding[0], kfolding[1], kfolding[2], nbands).
        """
        if self.fillingsfile_filepath is None:
            warnings.warn("No fillings file found, thus no occupations can be read")
            # TODO: Write a function to generate occupations from smearing and fermi level
            return None
        if self._occ_sabcj is None:
            occ_shape = [self.nspin]
            occ_shape += list(self.kfolding)
            occ_shape += [self.nbands]
            fillings = np.fromfile(self.fillingsfile_filepath)
            fillings = np.array(fillings, dtype=REAL_DTYPE)
            self._occ_sabcj = fillings.reshape(occ_shape)
        return self._occ_sabcj

    @property
    def p_uvjsabc(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return atomic projection P matrix of calculation.

        Return atomic projection P matrix of calculation in shape (norbs, norbs, nbands, nspin, nka, nkb, nkc).
        Fully evaluated for all u/v pairs.
        """
        if self._p_uvjsabc is None:
            self._p_uvjsabc = get_p_uvjsabc(self.proj_sabcju)
        return self._p_uvjsabc

    @property
    def h_uvsabc(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return atomic hamiltonian matrix of calculation.

        Return atomic hamiltonian matrix of calculation in shape (norbs, norbs, nspin, nka, nkb, nkc).
        Fully evaluated for all u/v pairs.
        """
        if self._h_uvsabc is None:
            self._h_uvsabc = get_h_uvsabc(self.p_uvjsabc, self.e_sabcj)
        return self._h_uvsabc

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
    def wk_sabc(self) -> np.ndarray[REAL_DTYPE]:
        """Return kpoint weights.

        Return kpoint weights in shape (kfolding[0], kfolding[1], kfolding[2]).
        """
        if self._wk_sabc is None:
            self._wk_sabc = get_wk_sabc(self.kptsfile_filepath, self.nspin, self.kfolding, self.lti_allowed)
        return self._wk_sabc

    @property
    def ks_sabc(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return kpoint coordinates.

        Return kpoint coordinates in shape (nspin, kfolding[0], kfolding[1], kfolding[2], 3).
        """
        if self._ks_sabc is None and (self.lti_allowed and self.kptsfile_filepath is not None):
            self._ks_sabc = get_ks_sabc(self.kptsfile_filepath, self.nspin, self.kfolding)
        return self._ks_sabc

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
        """Create ElecData instance from calculation directory.

        Create ElecData instance from calculation directory.

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


def los_projs_for_bands(proj_sabcju: np.ndarray[COMPLEX_DTYPE]) -> np.ndarray[COMPLEX_DTYPE]:
    """Perform LOS on projections for band orthogonality.

    Perform Lowdin symmetric orthogonalization on projections for band orthogonality.

    Parameters
    ----------
    proj_sabcju : np.ndarray[COMPLEX_DTYPE]
        Projections in shape (nstates, nbands, nproj).
    """
    s_jj = np.tensordot(proj_sabcju.conj().T, proj_sabcju, axes=([5, 4, 3, 2, 0], [0, 1, 2, 3, 5]))
    eigs, low_u = np.linalg.eigh(s_jj)
    nsqrt_ss_jj = np.eye(len(eigs)) * (eigs ** (-0.5))
    low_s_jj = np.dot(low_u, np.dot(nsqrt_ss_jj, low_u.T.conj()))
    low_proj_sabcju = np.tensordot(proj_sabcju, low_s_jj, axes=([4], [0]))
    low_proj_sabcju = np.swapaxes(low_proj_sabcju, 5, 4)
    return low_proj_sabcju


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


@jit(nopython=True)
def _norm_projs_for_bands_jit_helper_1(nProj, nStates, nBands, proj_tju, j_sums):
    for u in range(nProj):
        for t in range(nStates):
            for j in range(nBands):
                j_sums[j] += abs(np.conj(proj_tju[t, j, u]) * proj_tju[t, j, u])
    for j in range(nBands):
        proj_tju[:, j, :] *= 1 / np.sqrt(j_sums[j])
    proj_tju *= np.sqrt(nStates)
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
    proj_tju *= np.sqrt(nStates)
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
