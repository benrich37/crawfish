from typing import Path, Callable, Any

spintype_nspin = {
    "no-spin": 1,
    "spin-orbit": 2,
    "vector-spin": 2,
    "z-spin": 2
}
def get_nspin_from_outfile_filepath(
    outfile_filepath: str | Path,
    slice_idx = -1
    ) -> int:
    """ Get number of spins from out file.
    
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
    start = get_start_line(outfile)
    key = "spintype"
    rval = None
    texts = read_file(outfile)
    start, end = get_outfile_slice_bounds(texts, slice_idx)
    text = get_text_with_key_in_bounds(outfile, key, start, end)
    tokens = text.strip().split()
    val = tokens[1]
    if val in spintype_nspin:
        rval = spintype_nspin[val]
    if not rval is None:
        del texts
        return rval
    else:
        raise ValueError(f"Unrecognized spintype {val}")


def get_mu_from_outfile_filepath(
    outfile_filepath: str | Path,
    slice_idx = -1
    ) -> float:
    """ Get the Fermi level from the output file.
    
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
    text = get_text_with_key_in_bounds(outfile_filepath, lookkey, start, end)
    mu = float(text.split(lookkey)[1].strip().split()[0])
    return mu
        
def get_kfolding_from_outfile_filepath(
    outfile_filepath: str | Path,
    slice_idx = -1
    ) -> np.ndarray[int]:
    """ Return kpt folding from output file.

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
    text =  get_text_with_key_in_bounds(outfile_filepath, key, start, end)
    val = np.array(line.split(key)[1].strip().split(), dtype=int)
    return val

def get_nstates_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    return get__from_bandfile_filepath(bandfile, 0)

def get_nbands_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    return get__from_bandfile_filepath(bandfile, 2)

def get_nproj_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    return get__from_bandfile_filepath(bandfile, 4)

def get_nspecies_from_bandfile_filepath(bandfile_filepath: Path | str) -> int:
    return get__from_bandfile_filepath(bandfile, 6)

def get_norbsperatom_from_bandfile_filepath(
    bandfile_filepath: Path | str
    ) -> list[int]:
    """ Get number of orbitals per atom from bandprojections file.
    
    Get the number of orbitals per atom from the bandprojections file.
    
    Parameters
    ----------
    bandfile : Path | str
        Path to bandprojections file.
    
    Returns
    -------
    list[int]
    """
    with open(bandfile_filepath, 'r') as f:
        for line, text in enumerate(f):
            tokens = text.split()
            if line==0:
                nspecies = int(tokens[6])
                norbsperatom = []
            elif line>=2:
                if line<nSpecies+2:
                    natoms = int(tokens[1])
                    norbsperatom.extend( [int(tokens[2]),] * natoms)
                else:
                    break
    f.close()
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
    with open(bandfile_filepath, 'r') as f:
        for i, line in enumerate(f):
            if "#" in line:
                hash_lines += 1
                if hash_lines == 2:
                    if "|projection|^2" in line:
                        return False
                    else:
                        return True

def get_kpts_info_handler(
    nspin: int,
    kfolding: list[int] | np.ndarray[int],
    kptsfile_filepath: Path | str | None,
    nstates: int
    ) -> dict:
    kpts_info = {}
    _nk = int(np.prod(kfolding))
    nk = int(np.prod(kfolding))
    if nspin != int(nstates / _nk):
        print(
            "WARNING: Internal inconsistency found with respect to input parameters (nSpin * nk-pts != nStates).")
        print(
            "No safety net for this which allows for tetrahedral integration currently implemented.")
        if kptsfile_filepath is None:
            print(
                "k-folding will be changed to arbitrary length 3 array to satisfy shaping criteria.")
        kpts_info["lti"] = False
        nk = int(nstates / nspin)
    else:
        kpts_info["lti"] = True
    if kptsfile_filepath is None:
        if nk != _nk:
            kfolding = get_arbitrary_kfolding(nk)
        ks = np.ones([nk * nspin, 3]) * np.nan
        wk = np.ones(nk * nspin)
        wk *= (1 / nk)
    else:
        if isinstance(kptsfile_filepath, str):
            kptsfile_filepath = Path(kptsfile_filepath)
        if not kptsfile_filepath.exists():
            raise ValueError(f"Kpts file provided does not exist.")
        # TODO: Write a function that can un-reduce a reduced kpts mesh
        wk, ks, nStates = parse_kptsfile(kptsfile_filepath)
        wk = np.array(wk)
        ks = np.array(ks)
        if (nk != _nk):
            if len(ks) == nk:  # length of kpt data matches interpolated nk value
                kfolding = get_kfolding_from_kpts(kptsfile_filepath, nk)
            else:
                kfolding = get_arbitrary_kfolding(nk)
                ks = np.ones([nk * nspin, 3]) * np.nan
                wk = np.ones(nk * nspin)
                wk *= (1 / nk)
    wk_sabc = wk.reshape([nspin, kfolding[0], kfolding[1], kfolding[2]])
    ks_sabc = ks.reshape([nspin, kfolding[0], kfolding[1], kfolding[2], 3])
    kpts_info["wk_sabc"] = wk_sabc
    kpts_info["ks_sabc"] = ks_sabc
    kpts_info["kfolding"] = kfolding
    return kpts_info

def get_input_coord_vars_from_outfile_filepath(
    outfile_filepath: str | Path,
    slice_idx = -1
    ) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """ Get input coordinates from output file.
    
    Get the input coordinates from the output file.
    
    Parameters
    ----------
    outfile_filepath : str | Path
        Path to output file.
    
    Returns
    -------
    names: list[str]
        List of atom names.
    posns: np.ndarray
        Array of atomic positions.
    lattice_matrix: np.ndarray
        Lattice vectors.
    """
    outfile = read_file(outfile_filepath)
    start, end = get_outfile_slice_bounds(outfile, slice_idx=slice_idx)
    names = []
    posns = []
    lattice_matrix = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    for line, text in enumerate(outfile):
        if line > start_line:
            tokens = text.split()
            if len(tokens) > 0:
                if tokens[0] == "ion":
                    names.append(tokens[1])
                    posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                elif tokens[0] == "lattice":
                    active_lattice = True
                elif active_lattice:
                    if lat_row < 3:
                        lattice_matrix[lat_row, :] = [float(x) for x in tokens[:3]]
                        lat_row += 1
                    else:
                        active_lattice = False
                elif "Initializing the Grid" in text:
                    break
    posns = np.array(posns)
    return names, posns, lattice_matrix


def get_e_sabcj_helper(
    eigfile_filepath: str | Path,
    nspin: int, 
    nbands: int,
    kfolding: list[int] | np.ndarray[int]
    ) -> np.ndarray:
    """ Return eigenvalues from file.
    
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
    eshape = [nspin, kfolding[0], kfolding[1], kfolding[2], nbands]
    e_sabcj = e.reshape(eshape)
    return e_sabcj

def get_proj_sabcju_helper(
    bandfile_filepath: Path | str,
    nspin: int,
    kfolding: list[int] | np.ndarray[int],
    nbands: int,
    nproj: int
) -> np.ndarray:
    """ Return projections from file in sabcju shape.
    
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
    proj_tju = self.get_proj_tju_from_file(bandfile_filepath, allow_normalized=False)
    proj_sabcju = proj_tju.reshape(proj_shape)
    del proj_tju
    return proj_sabcju

def get_proj_tju_from_file(
    bandfile_filepath: Path | str
    ) -> np.ndarray:
    is_complex = is_complex_bandfile(bandfile_filepath)
    if is_complex:
        proj = _parse_bandfile_complex(bandfile_filepath)
    else:
        proj = _parse_bandfile_normalized(bandfile_filepath)
    return proj

def _parse_bandfile_complex(bandfile_filepath: str | Path) -> np.ndarray:
    dtype = complex
    token_parser = _complex_token_parser
    return _parse_bandfile_reader(bandfile_filepath, dtype, token_parser)

def _parse_bandfile_normalized(bandfile_filepath: str | Path) -> np.ndarray:
    dtype = float
    token_parser = _normalized_token_parser
    return _parse_bandfile_reader(bandfile_filepath, dtype, token_parser)

def _parse_bandfile_reader(
    bandfile_filepath: str | Path,
    dtype: type,
    token_parser: Callable
    ) -> np.ndarray:
    bandfile = read_file(bandfile_filepath)
    nstates = get_nstates_from_bandfile_filepath(bandfile_filepath)
    nbands = get_nbands_from_bandfile_filepath(bandfile_filepath)
    nproj = get_nproj_from_bandfile_filepath(bandfile_filepath)
    nspecies = get_nspecies_from_bandfile_filepath(bandfile_filepath)
    norbsperatom = get_norbsperatom_from_bandfile_filepath(bandfile_filepath)
    proj_tju = np.zeros((nstates, nbands, nproj),
                            dtype=dtype)
    for line, text in enumerate(f):
        if line >= nspecies + 2:
            istate = (line - (nspecies + 2)) // (nbands + 1)
            iband = (line - (nspecies + 2)) - istate * (
                    nbands + 1) - 1
            if iband >= 0 and istate < nstates:
                proj_tju[istate, iband] = np.array(
                    token_parser(tokens))
    return proj_tju

def _complex_token_parser(tokens: list[str]) -> np.ndarray:
    out = np.zeros(int(len(tokens) / 2), dtype=complex)
    tokens = np.array(tokens, dtype=float)
    out = complex_token_parser_jit(tokens, out)
    return out

def _normalized_token_parser(tokens: list[str]) -> np.ndarray:
    out = np.array(tokens, dtype=float)
    return out


def check_file_exists(func: Callable) -> Any:
    """Check if file exists.

    Check if file exists (and continue normally) or raise an exception if
    it does not.
    """

    @wraps(func)
    def wrapper(filename: str) -> Any:
        filepath = Path(filename)
        if not filepath.is_file():
            raise OSError(f"'{filename}' file doesn't exist!")
        return func(filename)
    return wrapper


@check_file_exists
def read_file(file_name: str) -> list[str]:
    """
    Read file into a list of str.

    Parameters
    ----------
    filename: Path or str
        name of file to read

    Returns
    -------
    texts: list[str]
        list of strings from file
    """
    with zopen(file_name, "r") as f:
        texts = f.readlines()
    f.close()
    return texts


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
    slice_idx: int,
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
    start_lines = get_outfile_start_lines(outfile)
    outfile_bounds_list = [[start_lines[i], start_lines[i+1]] for i in range(len(start_lines)-1)]
    if slice_idx >= len(outfile_bounds_list):
        raise ValueError(f"Slice index {slice_idx} out of bounds.")
    return outfile_bounds_list[slice_idx]

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
    return get_outfile_slice_bounds(outfile)[-1][0]


def get_text_with_key_in_bounds(filepath: str | Path, key: str, start: int, end: int) -> str:
    """ Return contents of file at line with key in bounds.
    
    Return the line with the key in the file between the start and end lines.
    
    Parameters
    ----------
    filepath : str | Path
        Path to file.
    key : str
        Key to search for.
    start : int
        Start line.
    end : int
        End line.
    """

    rval = None
    texts = read_file(filepath)
    filelength = len(texts)
    if start < 0:
        start = filelength + start
    if end < 0:
        end = filelength + end
    for line, text in enumerate(texts):
        if line > start and line < end:
            if key in text:
                rval = text
                break
    if not rval is None:
        return rval
    else:
        raise ValueError(f"Could not find {key} in {filepath} within bounds {start} and {end}")


def get__from_bandfile_filepath(bandfile_filepath: Path | str, tok_idx: int) -> int:
    """ Get arbitrary integer from header of bandprojections file.
    
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
    with open(bandfile_filepath, 'r') as f:
        for iLine, line in enumerate(f):
            tokens = line.split()
            if iLine == 0:
                ret_data = int(tokens[tok_idx])
                break
    f.close()
    return ret_data