"""Module for handling electronic data from JDFTx calculations.

This module contains the ElecData class, which is used to store and manipulate
electronic data from JDFTx calculations.
"""

from __future__ import annotations
import numpy as np
from crawfish.core.operations.matrix import get_h_t_uu_p_t_uu_s_t_uu, get_h_uu_p_uu_s_uu, normalize_square_proj_tju, los_projs_for_orbs, get_weighted_overlap_tjuv
from crawfish.io.data_parsing import (
    get_mu_from_outfile_filepath,
    get_nstates_from_bandfile_filepath,
    get_nbands_from_bandfile_filepath,
    get_nproj_from_bandfile_filepath,
    get_norbsperatom_from_bandfile_filepath,
    get_e_tj_helper,
    get_proj_tju_from_file,
    is_complex_bandfile_filepath,
    _get_lti_allowed,
    get_kfolding,
    get_ks_t,
    get_wk_t,
    get_nspecies_from_bandfile_filepath,
    get_norbsperatom_from_edata,
    save_bandfile_header,
)
from crawfish.utils.caching import CachedFunction
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
from crawfish.utils.indexing import get_kmap_from_edata, get_atom_orb_labels_dict
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view
from pymatgen.core.structure import Structure
from pymatgen.core.units import Ha_to_eV
from pathlib import Path
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from math import erf
from crawfish.io.general import safe_load

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


cache_mats = ["pcohp_tj", "pcoop_tj", "pcobi_tj", "tsep_pcohp_tj", "tsep_pcoop_tj", "tsep_pcobi_tj", "pcomo_tj", "pdos_tj"]

edata_setting_key_shorthand_dict = {
    "trim_excess_bands": "trimj",
    "los_orbs": "lou",
    "p_uu_consistent": "p-c",
    "norm_eigs_for_h": "h-en"
}

def get_edata_settings_str(edata: ElecData) -> str:
    """Return the ElecData settings string.

    Return the ElecData settings string. Gives portion of file name that encodes
    parts of the ElecData settings that would effect the character of the spectra.
    """
    parts_dict = {
        "trim_excess_bands": edata.trim_excess_bands,
        "los_orbs": edata.los_orbs,
        "p_uu_consistent": edata.p_uu_consistent,
        "norm_eigs_for_h": edata.norm_eigs_for_h,
    }
    # Sort the keys by alphabetical order
    sorted_keys = sorted(parts_dict.keys())
    # Sort the values by the sorted keys
    sorted_vals = [str(parts_dict[key]) for key in sorted_keys]
    # Join the sorted keys and values into a string
    parts_str = "_".join([f"{edata_setting_key_shorthand_dict[key]}_{val}" for key, val in zip(sorted_keys, sorted_vals)])
    return parts_str

class ElecData:
    """Class for handling electronic data from JDFTx calculations.

    The ElecData class is used to store and manipulate electronic data from JDFTx
    calculations.
    """

    jdftx: bool = True
    fprefix: str = ""
    ##
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
    ##
    proj_tju_filename: str = "proj_tju.npy"
    bandfile_header_filename: str = "bandProjections_header"
    #
    proj_tju_filepath: Path | None = None
    bandfile_header_filepath: Path | None = None
    ##
    _outfile: JDFTXOutfile | None = None
    _infile: JDFTXInfile | None = None
    ##
    _nstates: int | None = None
    _nbands: int | None = None
    _nproj: int | None = None
    _nspin: int | None = None
    _e_tj: np.ndarray[REAL_DTYPE] | None = None
    _proj_tju: np.ndarray[REAL_DTYPE] | np.ndarray[COMPLEX_DTYPE] | None = None
    _weighted_overlap_tjuv: np.ndarray[REAL_DTYPE] | None = None
    _p_uu: np.ndarray[REAL_DTYPE] | None = None
    _h_uu: np.ndarray[REAL_DTYPE] | None = None
    _s_uu: np.ndarray[REAL_DTYPE] | None = None
    _p_t_uu: np.ndarray[REAL_DTYPE] | None = None
    _s_t_uu: np.ndarray[REAL_DTYPE] | None = None
    _h_t_uu: np.ndarray[REAL_DTYPE] | None = None
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
    #
    _atom_orb_labels_dict: dict | None = None
    _nspecies: int | None = None
    _user_proj_tju: np.ndarray[REAL_DTYPE] | np.ndarray[COMPLEX_DTYPE] | None = None
    _broadening: REAL_DTYPE = 0.01
    _broadening_type: str = "Fermi"
    #
    use_cache_default: bool | None = None
    pcohp_tj_cache: CachedFunction = CachedFunction()
    pcoop_tj_cache: CachedFunction = CachedFunction()
    pcobi_tj_cache: CachedFunction = CachedFunction()
    tsep_pcohp_tj_cache: CachedFunction = CachedFunction()
    tsep_pcoop_tj_cache: CachedFunction = CachedFunction()
    tsep_pcobi_tj_cache: CachedFunction = CachedFunction()
    pcomo_tj_cache: CachedFunction = CachedFunction()
    pdos_tj_cache: CachedFunction = CachedFunction()
    """
    Picture adjustments
    -------------------
    trim_excess_bands: If True, selects the "top" nproj bands with the highest projection sums
    (implying these bands are the most representative of a LCAO picture of the calculation).
    When True, the projection vector (proj_tju) can be normalized at each state (t)
    for both bands and orbitals.
    Making the projection vector at each state square also enables techniques requiring
    dual spaces of bands and orbitals (not yet implemented)
    """

    def view_structure(self, **kwargs):
        """View structure using ASE's view method.

        View structure using ASE's view method.
        """
        atoms = AseAtomsAdaptor.get_atoms(self.structure)
        view(atoms, **kwargs)

    @property
    def trim_excess_bands(self) -> bool:
        return self._trim_excess_bands

    @trim_excess_bands.setter
    def trim_excess_bands(self, value: bool):
        if value != self._trim_excess_bands:
            self._trim_excess_bands = value
            self.alloc_elec_data()
            self._init_caches()

    _trim_excess_bands: bool = False

    """
    los_orbs: If True, performs Lowdin symmetric orthogonalization on the projections at each state. This may
    seem conterintuitive in a framework centered around how orbitals interact with each other,
    but remember that this orthogonality (<u,v>=d_{u,v}) does not eliminate overlap between orbitals
    at individual bands (<u|j><j|v> != 0), only over the sum of all bands present (sum_j <u|j><j|v> = d_{u,v}).
    This is an incredibly useful technique when trying to reformulate our calculation in a LCAO picture, as it
    ensures that for each bond, there is an equal antibond.
    """

    @property
    def los_orbs(self) -> bool:
        return self._los_orbs

    @los_orbs.setter
    def los_orbs(self, value: bool):
        if value != self._los_orbs:
            self._los_orbs = value
            self.alloc_elec_data()

    _los_orbs: bool = True

    """
    p_uu_consistent: If True, p_uu is first built as p_tj_uu (a population matrix at each state and band), and
    each projection matrix is normalized such that (where v signifies an independent orbital index from u) sum_{u,v} p_tj_uv = f_tj
    (ie the sum of all orbital populations at each band/state is equal to the filling at that state/band).
    This also coincidentally makes the LCAO hamiltonian matrix preserve energy conservation, as it will ensure
    sum_{u,v} h_tj_uv = e_tj * f_tj (as h_tj_uu = p_tj_uu * e_tj).
    """

    @property
    def p_uu_consistent(self) -> bool:
        return self._p_uu_consistent

    @p_uu_consistent.setter
    def p_uu_consistent(self, value: bool):
        if value != self._p_uu_consistent:
            self._p_uu_consistent = value
            self._init_caches()
            self.alloc_elec_data()
            

    _p_uu_consistent: bool = True

    """
    norm_eigs_for_h: WRITE ME
    """

    @property
    def norm_eigs_for_h(self) -> bool:
        return self._norm_eigs_for_h

    @norm_eigs_for_h.setter
    def norm_eigs_for_h(self, value: bool):
        if value != self._norm_eigs_for_h:
            self._norm_eigs_for_h = value
            self._init_caches()
            self.alloc_elec_data()
            
    _norm_eigs_for_h: bool = True

    @property
    def infile(self) -> JDFTXInfile | None:
        """Return JDFTx input file object from calculation.

        Return JDFTx input file object from calculation.
        """
        if self.jdftx:
            if self._infile is None:
                self._infile = JDFTXInfile.from_file(self.calc_dir / f"{self.fprefix}in")
            return self._infile

    @property
    def outfile(self) -> JDFTXOutfile | None:
        """Return JDFTx output file object from calculation.

        Return JDFTx output file object from calculation.
        """
        if self.jdftx:
            if self._outfile is None:
                self._outfile = JDFTXOutfile.from_file(self.outfile_filepath)
            return self._outfile

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
        if self.jdftx:
            return self.outfile.structure
        else:
            return self._structure

    @structure.setter
    def structure(self, value: Structure):
        self._structure = value


    @property
    def nspin(self) -> int:
        """Return number of spins in calculation.

        Return number of spins in calculation.
        """
        if self._nspin is None:
            if self.jdftx:
                self._nspin = self.outfile.nspin
        return self._nspin

    @nspin.setter
    def nspin(self, value: int):
        if value > 0 and isinstance(value, int):
            self._nspin = value
        else:
            raise ValueError(f"Invalid number of spins {value} (must be positive and an integer)")

    @property
    def nstates(self) -> int:
        """Return number of states in calculation.

        Return number of states in calculation.
        """
        if self._nstates is None:
            if self.jdftx:
                self._nstates = get_nstates_from_bandfile_filepath(self.bandfile_header_filepath)
        return self._nstates

    @property
    def nbands(self) -> int:
        """Return number of bands in calculation.

        Return number of bands in calculation.
        """
        if self._nbands is None:
            if self.jdftx:
                self._nbands = get_nbands_from_bandfile_filepath(self.bandfile_header_filepath)
        return self._nbands

    @property
    def nproj(self) -> int:
        """Return number of projections in calculation.

        Return number of projections in calculation.
        """
        if self._nproj is None:
            if self.jdftx:
                self._nproj = get_nproj_from_bandfile_filepath(self.bandfile_header_filepath)
        return self._nproj

    @property
    def norbsperatom(self) -> list[int]:
        """Return number of orbitals per atom in calculation.

        Return number of orbitals per atom in calculation.
        """
        if self._norbsperatom is None:
            if self.jdftx:
                self._norbsperatom = get_norbsperatom_from_bandfile_filepath(self.bandfile_header_filepath)
            else:
                self._norbsperatom = get_norbsperatom_from_edata(self)
        return self._norbsperatom

    @property
    def nspecies(self) -> list[int]:
        """Return number of unique elements in calculation.

        Return number of unique elements in calculation.
        """
        if self._nspecies is None:
            if self.jdftx:
                self._nspecies = get_nspecies_from_bandfile_filepath(self.bandfile_header_filepath)
            else:
                unique_names, _ = count_ions(self.ion_names)
                self._nspecies = len(list(self.atom_orb_labels_dict.keys()))
        return self._nspecies

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
            if self.jdftx:
                self._mu = self.outfile.mu / Ha_to_eV
            else:
                raise ValueError("mu (Fermi level) must be set by user for non-JDFTx calculations")
        return self._mu

    @mu.setter
    def mu(self, value: REAL_DTYPE):
        if np.isreal(value):
            self._mu = value
        else:
            raise ValueError(f"Invalid chemical potential value {value} (must be real)")



    @property
    def e_tj(self) -> np.ndarray[REAL_DTYPE]:
        """Return eigenvalues of calculation.

        Return eigenvalues of calculation in shape (nstate, nbands).
        """
        if self._e_tj is None:
            if self.jdftx:
                self._e_tj = get_e_tj_helper(self.eigfile_filepath, self.nstates, self.nbands)
        return self._e_tj

    @e_tj.setter
    def e_tj(self, value: np.ndarray[REAL_DTYPE]):
        if None in [self.nstates, self.nbands]:
            nstates, nbands = np.shape(value)
            self._nstates = nstates
            self._nbands = nbands
            self._e_tj = value
        elif np.shape(value) == (self.nstates, self.nbands):
            self._e_tj = value
        else:
            raise ValueError(
                f"Invalid shape for eigenvalue array {np.shape(value)} (nstates set to {self.nstates}, nbands set to {self.nbands})"
            )

    @property
    def e_sabcj(self) -> np.ndarray[REAL_DTYPE]:
        """Return eigenvalues of calculation.

        Return eigenvalues of calculation in shape (nstate, nbands).
        """
        return self.e_tj.reshape([self.nspin] + list(self.kfolding) + [self.nbands])

    @property
    def proj_tju(self) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
        """Return projections of calculation.

        Return projections of calculation in shape (nstates, nbands, nproj).
        """
        if self._proj_tju is None:
            if self.jdftx:
                #self._proj_tju = get_proj_tju_from_file(self.bandfile_filepath)
                self._set_proj_tju()
            else:
                self._proj_tju = self.user_proj_tju.copy()
        return self._proj_tju
    
    def _set_proj_tju(self) -> None:
        """ Set the _proj_tju variable.

        Set the _proj_tju variable. Loads from a precompiled binary file if available.
        Else, loads from the dumped text file and backs it up as a binary file, along
        with a stripped text file only containing the header.
        """
        loaded_from_bandfile = False
        proj_tju = safe_load(self.proj_tju_filepath, allow_pickle=False)
        if proj_tju is None:
            proj_tju = get_proj_tju_from_file(self.bandfile_filepath)
            loaded_from_bandfile = True
        self._proj_tju = proj_tju
        if loaded_from_bandfile:
            self._cache_proj_tju()

    def _cache_proj_tju(self) -> None:
        """ Back up bandprojections file in easier to load pieces.

        Back up bandprojections file as a [nstates, nbands, nproj] complex numpy array,
        along with a truncated file only containing the bandprojections header. 
        """
        np.save(self.proj_tju_filepath, self._proj_tju)
        save_bandfile_header(self.bandfile_filepath, self.bandfile_header_filepath)
            


    @property
    def user_proj_tju(self) -> np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]:
        """Return user-defined projections of calculation.

        Return user-defined projections of calculation in shape (nstates, nbands, nproj).
        """
        return self._user_proj_tju

    @user_proj_tju.setter
    def user_proj_tju(self, value: np.ndarray[COMPLEX_DTYPE] | np.ndarray[REAL_DTYPE]):
        t, j, u = np.shape(value)
        self._nproj = u
        self._nbands = j
        self._nstates = t
        self._user_proj_tju = value



    @property
    def broadening(self) -> REAL_DTYPE:
        """Return broadening of calculation.

        Return broadening of calculation.
        """
        if self.jdftx:
            return self.outfile.broadening
        else:
            return self._broadening

    @broadening.setter
    def broadening(self, value: REAL_DTYPE):
        if np.isreal(value) and value >= 0:
            self._broadening = value
        else:
            raise ValueError(f"Invalid broadening value {value} (must be real and positive)")


    @property
    def broadening_type(self) -> str:
        """Return broadening type of calculation.

        Return broadening type of calculation.
        """
        if self.jdftx:
            return self.outfile.broadening_type
        else:
            return self._broadening_type

    @broadening_type.setter
    def broadening_type(self, value: str):
        if value not in ["Fermi", "Gauss", "MP1", "cold"]:
            raise ValueError(f"Unknown broadening type {value}")
        self._broadening_type = value

    @property
    def occ_tj(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return occupations of calculation.

        Return occupations of calculation in shape (nspin, kfolding[0], kfolding[1], kfolding[2], nbands).
        """
        if self._occ_tj is None:
            if self.fillingsfile_filepath is None:
                if self.jdftx:
                    btype = self.outfile.broadening_type
                    broad = self.outfile.broadening
                else:
                    btype = self._broadening_type
                    broad = self._broadening
                def bfunc(eig):
                    return calculate_filling_nobroad(0.0, eig, self.mu)
                if isinstance(btype, str):
                    if btype == "Fermi":

                        def bfunc(eig):
                            return calculate_filling_fermi(broad, eig, self.mu)
                    elif btype == "Gauss":

                        def bfunc(eig):
                            return calculate_filling_gauss(broad, eig, self.mu)
                    elif btype == "MP1":

                        def bfunc(eig):
                            return calculate_filling_mp1(broad, eig, self.mu)
                    elif btype == "cold":

                        def bfunc(eig):
                            return calculate_filling_cold(broad, eig, self.mu)
                    elif btype.lower() == "none":

                        def bfunc(eig):
                            return calculate_filling_nobroad(0.0, eig, self.mu)
                    else:
                        raise ValueError(f"Unknown broadening type {btype}")
                self._occ_tj = bfunc(self.e_tj)
            else:
                fillings = np.fromfile(self.fillingsfile_filepath)
                fillings = np.array(fillings, dtype=REAL_DTYPE)
                fillings = fillings.reshape(self.nstates, self.nbands)
                self._occ_tj = fillings
        return self._occ_tj

    @occ_tj.setter
    def occ_tj(self, value: np.ndarray[REAL_DTYPE]):
        if np.shape(value) == (self.nstates, self.nbands):
            self._occ_tj = value
        else:
            raise ValueError(f"Invalid shape for occupation array {np.shape(value)}")

    def set_weighted_overlap_tjuv(self) -> None:
        if self._weighted_overlap_tjuv is None:
            self._weighted_overlap_tjuv = get_weighted_overlap_tjuv(
                self.proj_tju,
                self.wk_t,
            )

    def reval_mat_uu(self) -> None:
        """Re-evaluate the matrices for the current settings.

        Re-evaluate the matrices for the current settings. Only re-evaluate if the
        the matrices were ever evaluated in the first place.
        """
        mat_names = ["h_t_uu", "p_t_uu", "s_t_uu", "h_uu", "p_uu", "s_uu"]
        for arr_name in mat_names:
            val = getattr(self, f"_{arr_name}")
            if val is not None:
                self.set_mat_uu()
                break

    def set_mat_uu(self) -> None:
        missing = False
        mat_names = ["h_t_uu", "p_t_uu", "s_t_uu", "h_uu", "p_uu", "s_uu"]
        for arr_name in mat_names:
            arr_cache_path = self.cache_sub_dir / f"{arr_name}.npy"
            arr = safe_load(arr_cache_path, allow_pickle=False)
            if arr is None:
                missing = True
                break
        if missing:
            h_t_uu, p_t_uu, s_t_uu = get_h_t_uu_p_t_uu_s_t_uu(
                self.proj_tju,
                self.e_tj,
                self.occ_tj,
                self.wk_t,
                p_sc=self.p_uu_consistent,
                norm_eigs_for_h=self.norm_eigs_for_h,
                mu=self.mu,
            )
            h_uu, p_uu, s_uu = get_h_uu_p_uu_s_uu(
                self.proj_tju,
                self.e_tj,
                self.occ_tj,
                self.wk_t,
                p_sc=self.p_uu_consistent,
                norm_eigs_for_h=self.norm_eigs_for_h,
                mu=self.mu,
            )
            print("saving to", self.cache_sub_dir)
            np.save(self.cache_sub_dir / f"h_t_uu.npy", h_t_uu)
            np.save(self.cache_sub_dir / f"p_t_uu.npy", p_t_uu)
            np.save(self.cache_sub_dir / f"s_t_uu.npy", s_t_uu)
            np.save(self.cache_sub_dir / f"h_uu.npy", h_uu)
            np.save(self.cache_sub_dir / f"p_uu.npy", p_uu)
            np.save(self.cache_sub_dir / f"s_uu.npy", s_uu)
            self._h_t_uu = h_t_uu
            self._p_t_uu = p_t_uu
            self._s_t_uu = s_t_uu
            self._h_uu = h_uu
            self._p_uu = p_uu
            self._s_uu = s_uu
        else:
            print("loading from", self.cache_sub_dir)
            for arr_name in mat_names:
                # Not using safe_load here - corrupted or missing files were already identified.
                # If that doesn't make this process safe, an error should be raised.
                setattr(self, f"_{arr_name}", np.load(self.cache_sub_dir / f"{arr_name}.npy"))


    @property
    def h_t_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return state-resolved hamiltonian matrix of calculation.

        Return state-resolved hamiltonian matrix of calculation in shape (nstate, nproj, nproj).
        """
        if self._h_t_uu is None:
            self.set_mat_uu()
        return self._h_t_uu

    @property
    def p_t_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return state-resolved projection matrix of calculation.

        Return state-resolved projection matrix of calculation in shape (nstate, nproj, nproj).
        """
        if self._p_t_uu is None:
            self.set_mat_uu()
        return self._p_t_uu

    @property
    def s_t_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return state-resolved overlap matrix of calculation.

        Return state-resolved overlap matrix of calculation in shape (nstate, nproj, nproj).
        """
        if self._s_t_uu is None:
            self.set_mat_uu()
        return self._s_t_uu

    @property
    def h_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return hamiltonian matrix of calculation.

        Return hamiltonian matrix of calculation in shape (nproj, nproj).
        """
        if self._h_uu is None:
            self.set_mat_uu()
        return self._h_uu

    @property
    def p_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return projection matrix of calculation.

        Return projection matrix of calculation in shape (nproj, nproj).
        """
        if self._p_uu is None:
            self.set_mat_uu()
        return self._p_uu

    @property
    def s_uu(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return overlap matrix of calculation.

        Return overlap matrix of calculation in shape (nproj, nproj).
        """
        if self._s_uu is None:
            self.set_mat_uu()
        return self._s_uu
    
    @property
    def weighted_overlap_tjuv(self) -> np.ndarray[COMPLEX_DTYPE] | None:
        """Return weighted_overlap_tjuv.

        Return weighted_overlap_tjuv in shape (nstates, nbands, nproj, nproj).
        """
        self.set_weighted_overlap_tjuv()
        return self._weighted_overlap_tjuv

    @property
    def ion_orb_u_dict(self) -> dict[str, int]:
        """Return dictionary mapping atom labels to orbital indices.

        Return dictionary mapping each atom (using key of format 'el #n' (str)) to indices
        (int) of all atomic orbital projections (in 0-based indexing) belonging to
        said atom.
        """
        return self.orbs_idx_dict

    @property
    def atom_orb_labels_dict(self) -> dict[str, int]:
        if self._atom_orb_labels_dict is None:
            if self.jdftx:
                self._atom_orb_labels_dict = get_atom_orb_labels_dict(self.bandfile_header_filepath)
            else:
                raise RuntimeError("atom_orb_labels_dict must be set by user for non-JDFTx calculations")
        return self._atom_orb_labels_dict

    @atom_orb_labels_dict.setter
    def atom_orb_labels_dict(self, value: dict[str, int]):
        self._atom_orb_labels_dict = value

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
            if self.jdftx:
                self._wk_t = get_wk_t(self.kptsfile_filepath, self.nspin, self.kfolding, self.lti_allowed)
        return self._wk_t

    @wk_t.setter
    def wk_t(self, value: np.ndarray[REAL_DTYPE]):
        if len(value) == self.nstates and len(value.shape) == 1:
            self._wk_t = value
        else:
            raise ValueError(
                f"Invalid shape for kpoint weights array {np.shape(value)} (must be nspin*nkpts long and 1-dimensional)"
            )

    

    @property
    def ks_t(self) -> np.ndarray[REAL_DTYPE] | None:
        """Return kpoint coordinates.

        Return kpoint coordinates in shape (nspin, kfolding[0], kfolding[1], kfolding[2], 3).
        """
        if self._ks_t is None:
            if self.jdftx:
                self._ks_t = get_ks_t(self.kptsfile_filepath, self.bandfile_header_filepath)
        return self._ks_t
    
    @ks_t.setter
    def ks_t(self, value: np.ndarray[REAL_DTYPE]):
        if value.shape == (self.nstates, 3):
            self._ks_t = value
        else:
            raise ValueError(f"Invalid shape for kpoint coordinates array {np.shape(value)} (must be nstates x 3)")


    @property
    def kfolding(self) -> list[int]:
        """Return kpoint folding.

        Return kpoint folding as a list of 3 integers.
        """
        if self._kfolding is None:
            self._kfolding = get_kfolding(self.lti_allowed, self.outfile_filepath, self.nspin, self.nstates)
        return self._kfolding

    @kfolding.setter
    def kfolding(self, value: list[int]):
        if len(value) == 3:
            self._kfolding = value
            if self.nspin is None and (self.nstates is not None):
                self._nspin = int(self.nstates / np.prod(value))
        else:
            raise ValueError(f"Invalid shape for kpoint folding array {np.shape(value)} (must be length 3)")


    @property
    def bandprojfile_is_complex(self) -> bool:
        """Return if band projections are complex.

        Return if band projections are complex. (required for coop/cohp analysis)
        """
        if self._complex_bandprojs is None:
            self._complex_bandprojs = is_complex_bandfile_filepath(self.bandfile_header_filepath)
        return self._complex_bandprojs

    @property
    def lti_allowed(self) -> bool:
        """Return if LTI is allowed.

        Return if LTI (linear tetrahedral integration) is allowed (dependent on whether jdftx calculation
        automatically reduced the k-point mesh to a non-MK pack grid, and the existence of a kpts file
        to refer to).
        """
        if self._lti_allowed is None:
            self._lti_allowed = _get_lti_allowed(self.bandfile_header_filepath, self.outfile_filepath)
        return self._lti_allowed

    @classmethod
    def from_calc_dir(
        cls, calc_dir: Path, prefix: str | None = None, use_cache_default: bool | None = None,
        autosave_cache: bool = True) -> ElecData:
        """Create ElecData instance from JDFTx calculation directory.

        Create ElecData instance from JDFTx calculation directory.

        Parameters
        ----------
        calc_dir : Path
            Path to calculation directory.
        prefix : str, optional
            Prefix of files in calculation directory, by default None
        """
        instance = ElecData(
            calc_dir=Path(calc_dir), prefix=prefix, use_cache_default=use_cache_default, autosave_cache=autosave_cache
            )
        return instance

    @classmethod
    def from_bandstructure(
        cls,
        bandstructure: BandStructure,
        orbitals: list[str] = None,
        trim_0_orbs: bool = True,
        ) -> ElecData:
        """ Create an ElecData instance from a pymatgen BandStructure object.

        Create an ElecData instance from a pymatgen BandStructure object.

        Parameters
        ----------
        bandstructure: BandStructure
            pymatgen BandStructure object
        orbitals: list[str], optional
            List of orbital names corresponding to the order they are kept in the BandStructure
            projections attribute. If None, orbitals are assumed to be either;
                - ["s"]
                - ["s", "py", "pz", "px"]
                - ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]
            following PROCAR conventions.
        trim_0_orbs: bool, optional
            If True, orbitals with all-zero projections are removed from the projection tensor.
            If False, all orbitals are kept. (False may cause a RuntimeError if the len(orbitals)*len(atoms) > nbands)
        """
        raise NotImplementedError("Construction from BandStructure not yet implemented - working on it!")
        structure = bandstructure.structure
        ion_nums = [specie.element.Z for specie in structure.species]
        ion_names = [specie.element.symbol for specie in structure.species]
        is_sorted = all(ion_names[i] <= ion_names[i + 1] for i in range(len(ion_names) - 1))
        cls._from_bandstructure_check_projections(bandstructure)
        if not is_sorted:
            raise NotImplementedError("Automatic ion sorting not yet implemented")
        norbsperatom = []
        atom_orb_labels_dict = {}
        for i, el in enumerate(ion_names):
            norbs = len()

    def _from_bandstructure_check_projections(self, bandstructure: BandStructure):
        """Check if projections of BandStructure object contain phase.

        Check if projections of BandStructure object contain phase.

        Parameters
        ----------
        bandstructure: BandStructure
            pymatgen BandStructure object
        """
        if bandstructure.projections is None:
            raise ValueError("BandStructure object must have projections attribute")
        if isinstance(bandstructure.projections, dict):
            if not len(bandstructure.projections.keys()):
                raise ValueError("BandStructure object must have non-empty projections attribute")
            else:
                imag_sum = 0
                neg_real_sum = 0
                pos_real_sum = 0
                for spin in bandstructure.projections.keys():
                    imag_sum += np.sum(np.flatten(np.abs(np.imag(bandstructure.projections[spin]))))
                    neg_real_sum += np.sum(np.flatten(np.real(bandstructure.projections[spin]) < 0))
                    pos_real_sum += np.sum(np.flatten(np.real(bandstructure.projections[spin]) > 0))
                if (imag_sum == 0) and (neg_real_sum == 0):
                    raise ValueError(
                        "BandStructure object must have projections attribute with phase (detected by"
                        "non-zero imaginary part or negative real part)"
                        )
                return True
        else:
            raise ValueError("BandStructure object must have projections attribute of type dict")

    @classmethod
    def as_empty(cls) -> ElecData:
        """Create an empty ElecData instance.

        Create an empty ElecData instance.
        """
        instance = ElecData(calc_dir=None)
        return instance

    def __init__(
            self, calc_dir: Path, prefix: str | None = None, jdftx: bool = True, use_cache_default: bool | None = None,
            autosave_cache: bool = True):
        """Initialize ElecData instance.

        Initialize ElecData instance.

        Parameters
        ----------
        calc_dir : Path
            Path to calculation directory.
        prefix : str, optional
            Prefix of files in calculation directory, by default None
        jdftx : bool, optional
            If True, calculation directory is a JDFTx calculation, by default True
        use_cache_default : bool, optional
            If True, cache of (state, band) matrices for computed pDOS/pCOHP plots is used by default, by default None
        autosave_cache : bool, optional
            If True, cache of (state, band) matrices for newly computed pDOS/pCOHP plots are saved automatically, by default True
        """
        if jdftx:
            self.fprefix = self._get_fprefix(prefix)
            self.calc_dir = Path(calc_dir)
            if calc_dir is not None:
                self._set_files_paths()
                self._set_backup_files_paths()
                self.alloc_elec_data()
        else:
            self.calc_dir = calc_dir
        self.autosave_cache = autosave_cache
        self._init_caches()
        self.use_cache_default = use_cache_default

    
        

    def _set_files_paths(self, optional_sufficies: list[str] = ["gvec", "wfn", "kpts", "fillings"]):
        if self.fprefix is None:
            raise RuntimeError("File prefix (fprefix) must be set before setting file paths")
        for filetype in ["band", "kpts", "eig", "fillings", "out", "gvec", "wfn"]:
            suffix = getattr(self, f"{filetype}file_suffix")
            filepath = self._get_filepath_generic(suffix)
            if filepath is None and filetype not in optional_sufficies:
                raise FileNotFoundError(
                    f"File not found for suffix {suffix} in calculation directory {self.calc_dir} \n hint: Make sure prefix does not contain '.'"
                )
            setattr(self, f"{filetype}file_filepath", filepath)

    def _get_filepath_generic(self, suffix: str) -> Path | None:
        filepath: Path = self.calc_dir / f"{self.fprefix}{suffix}"
        if not filepath.exists():
            return None
        return filepath
    
    def _set_backup_files_paths(self):
        self.cache_dir = self.calc_dir / ".crawfish_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._write_cache_dir_readme()
        for filetype in ["proj_tju", "bandfile_header"]:
            filename = getattr(self, f"{filetype}_filename")
            filepath = self._get_backup_filepath_generic(filename)
            setattr(self, f"{filetype}_filepath", filepath)

    def _write_cache_dir_readme(self):
        cache_dir_readme_filepath = self.cache_dir / "readme.txt"
        if not cache_dir_readme_filepath.exists():
            with open(cache_dir_readme_filepath, "w") as f:
                f.write("\n".join(cache_dir_readme_lines))
            f.close()
    

    def _get_backup_filepath_generic(self, filename: str) -> Path | None:
        # The file path does not need to exist, as elecdata will write them
        filepath: Path = self.cache_dir / f"{filename}"
        return filepath
    
    # def _get_cache_filepath_generic(self, filename: str) -> Path | None:
    #     filepath: Path = self.cache_sub_dir / f"{filename}"
    #     if not filepath.exists():
    #         return None
    #     return filepath

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

    def alloc_elec_data(self):
        self._set_proj_tju()
        #_ = self.proj_tju
        _ = self.occ_tj
        _ = self.e_tj
        if self.nbands < self.nproj:
            self.trim_excess_bands = False
            self.los_orbs = False
        if self.trim_excess_bands:
            bestj = get_best_bands(self.proj_tju)
            _e_tj = self.e_tj[:, bestj]
            _proj_tju = self.proj_tju[:, bestj, :]
            _occ_tj = self.occ_tj[:, bestj]
            _proj_tju = normalize_square_proj_tju(_proj_tju)
            self._e_tj = _e_tj
            self._proj_tju = _proj_tju
            self._occ_tj = _occ_tj
            self._nbands = self.nproj
        if self.los_orbs:
            self._proj_tju = los_projs_for_orbs(self.proj_tju)
        self.reval_mat_uu()

    #################

    def unnorm_projs(self) -> None:
        """Remove normalization from projections.

        Remove normalization from projections.
        """
        self._proj_tju = None
        _ = self.proj_tju
        self.norm_idx = None
        return None
    
    def _init_caches(self):
        # self.cache_dir = self.calc_dir / ".crawfish_cache"
        # self.cache_dir.mkdir(exist_ok=True)
        self.cache_sub_dir = self.cache_dir / get_edata_settings_str(self)
        self.cache_sub_dir.mkdir(exist_ok=True, parents=True)
        for mat in cache_mats:
            setattr(self, f"{mat}_cache", CachedFunction(self.cache_sub_dir / f"{mat}.npz", auto_save=self.autosave_cache))
        self.load_caches()

    def backup_cache(self) -> None:
        """Backup cache of "tj" matrices.

        Backup cache of "tj" matrices.
        """
        # self.cache_dir = self.calc_dir / ".crawfish_cache"
        # self.cache_dir.mkdir(exist_ok=True)
        for mat in cache_mats:
            cached_func: CachedFunction = getattr(self, f"{mat}_cache")
            cached_func.save_cache()
        

    def load_caches(self) -> None:
        """Load caches of "tj" matrices.

        Load caches of "tj" matrices.
        """
        self.cache_dir = self.calc_dir / ".crawfish_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_sub_dir = self.cache_dir / get_edata_settings_str(self)
        for mat in cache_mats:
            cached_func: CachedFunction = getattr(self, f"{mat}_cache")
            cached_func.load_cache()


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


def get_best_bands(unnorm_proj_tju):
    np.shape(unnorm_proj_tju)[0]
    nbands = np.shape(unnorm_proj_tju)[1]
    nproj = np.shape(unnorm_proj_tju)[2]
    band_proj_sums = np.zeros(nbands)
    for j in range(nbands):
        band_proj_sums[j] = np.sum(np.abs(unnorm_proj_tju[:, j, :]), axis=(0, 1))
    idcs = np.argsort(band_proj_sums)[::-1]
    idcs_select = idcs[:nproj]
    idcs_sort = np.argsort(idcs_select)
    idcs_return = [idcs_select[i] for i in idcs_sort]
    return idcs_return


# @jit(nopython=True)
def calculate_filling_fermi(broadening: float, eig: float, efermi: float) -> float:
    x = (eig - efermi) / (2.0 * broadening)
    filling = 0.5 * (1 - np.tanh(x))
    return filling


# @jit(nopython=True)
def calculate_filling_gauss(broadening: float, eig: float, efermi: float) -> float:
    x = (eig - efermi) / (2.0 * broadening)
    filling = 0.5 * (1 - erf(x))
    return filling


# @jit(nopython=True)
def calculate_filling_mp1(broadening: float, eig: float, efermi: float) -> float:
    x = (eig - efermi) / (2.0 * broadening)
    filling = 0.5 * (1 - erf(x)) - x * np.exp(-1 * x**2) / (2 * np.pi**0.5)
    return filling


# @jit(nopython=True)
def calculate_filling_cold(broadening: float, eig: float, efermi: float) -> float:
    x = (eig - efermi) / (2.0 * broadening)
    filling = 0.5 * (1 - erf(x + 0.5**0.5)) + np.exp(-1 * (x + 0.5**0.5) ** 2) / (2 * np.pi) ** 0.5
    return filling

# @jit(nopython=True)
def calculate_filling_nobroad(broadening: float, eig: float, efermi: float) -> float:
    filling = np.heaviside(eig, efermi)
    return filling


cache_dir_readme_lines = [
    "This is the crawfish cache directory for this calculation directory.",
    "",
    "The files within this directory are stored here for faster loading in subsequent analyses.",
    "",
    "The files `proj_tju.npy` and `bandProjections_header` contain all of the data contained in",
    "the original `bandProjections` file for <50% the size, and can be parsed at <1% the time.",
    "",
    "The directories in this directory are named after the settings used for the created ElecData",
    "object. Within each of these directories are the binary files for the evaluated atomic",
    "matrices (ie h_t_uu, h_uu, p_t_uu, ...) for those particular settings, as well as ",
    "pickled weight matrices for creating pdos/pcomo/pcoop/pcohp/pcobi spectra. Use of ",
    "the latter pickled matrices can be disabled when running the spectral functions with ",
    "`use_cache`.",
    "",
    "The sub-directories of these sub-directories are named after particular energy ranges (with",
    " the syntax `erange_emin_emax_estep`). Within these sub-sub-directories are  ",
    "sub-sub-sub-directories named after spectral functions (pdos/pcomo...) and hold ",
    "binary files of evaluated spectra, named after settings used to create the spectrum ",
    "(ie the ion indices, orbitals selected, spin separation, lti parameters, ...). ",
    "These spectra are saved and loaded by default when running the spectral functions, ",
    "but can be disabled by `use_cached_spectrum` (to disable loading) and ",
    "`save_spectrum` (to disable saving). This functionality provides the most utility ",
    "for lti spectra. Gaussian spectra still get a speed-up, but it is hardly noticable.",
]