"""Module for atom and orbital indexing methods.

Methods for indexing atoms and orbitals in a system.
"""

from __future__ import annotations
from pathlib import Path
from crawfish.io.utils import format_file_path, read_file
from crawfish.core.elecdata import ElecData
from ase import Atoms


def fidcs(idcs: list[int] | int) -> list[int]:
    """Format atom indices to a list of integers.

    Format atom indices to a list of integers.

    Parameters
    ----------
    idcs : list[int] | int
        The list of indices for or index of species of interest
    """
    if type(idcs) is int:
        return [idcs]
    elif type(idcs) is list:
        return idcs
    else:
        raise ValueError("atom indices must be int or list of int")


def get_el_orb_u_dict(edata: ElecData, aidcs: list[int]) -> dict[str, dict[str, list[int]]]:
    """Return a dictionary mapping atom symbol and atomic orbital string to all relevant projection indices.

    Return a dictionary mapping atom symbol and atomic orbital string to all relevant projection indices.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest
    aidcs : list[int]
        The list of indices for atoms of interest
    """
    syms = edata.atoms.get_chemical_symbols()
    els = [syms[i] for i in aidcs]
    kmap = get_kmap_from_atoms(edata.atoms)
    labels_dict: dict[str, list[str]] = get_atom_orb_labels_dict(edata.bandfile_filepath)
    el_orbs_dict: dict[str, dict[str, list[int]]] = {}
    orbs_idx_dict = edata.orbs_idx_dict
    for i, el in enumerate(els):
        if el not in el_orbs_dict:
            el_orbs_dict[el] = {}
        for ui, u in enumerate(orbs_idx_dict[kmap[aidcs[i]]]):
            # ui is the index of the orbital in the context of the orbitals belonging to the atom
            orb = labels_dict[el][ui]
            if orb not in el_orbs_dict[el]:
                el_orbs_dict[el][orb] = []
            el_orbs_dict[el][orb].append(u)
    return el_orbs_dict


def get_atom_orb_labels_dict(bandfile_filepath: str | Path) -> dict[str, list[str]]:
    """Return a dictionary mapping each atom symbol to all atomic orbital projection string representations.

    Return a dictionary mapping each atom symbol to all atomic orbital projection string representations.
    (eg
    {
    "H": ["0s", "0px", "0py", "0pz"],
    "O": ["0s", "0px", "0py", "0pz", "0dxy", "0dxz", "0dyz", "0dx2y2", "0dz2"],
    "Pt": ["0s", "1s", "0px", "0py", "0pz", "1px", "1py", "1pz", "0dxy", "0dxz", "0dyz", "0dx2y2", "0dz2", "0fx3-3xy2", "0fyx2-yz2", "0fxz2", "0fz3", "0fyz2", "0fxyz", "0f3yx2-y3"]
    }, where the numbers are needed when using pseudopotentials with multiple valence shells of the same angular momentum
    are are NOT REPRESENTATIVE OF THE TRUE PRINCIPAL QUANTUM NUMBER.
    )

    Parameters
    ----------
    bandfile_filepath : str | Path
        The path to the bandfile
    """
    path = format_file_path(bandfile_filepath)
    bandfile = read_file(path)
    labels_dict: dict[str, list[str]] = {}

    for i, line in enumerate(bandfile):
        if i > 1:
            if "#" in line:
                break
            else:
                lsplit = line.strip().split()
                sym = lsplit[0]
                labels_dict[sym] = []
                lmax = int(lsplit[3])
                for j in range(lmax + 1):
                    refs = orb_ref_list[j]
                    nShells = int(lsplit[4 + j])
                    for k in range(nShells):
                        if nShells > 1:
                            for r in refs:
                                labels_dict[sym].append(f"{k}{r}")
                        else:
                            labels_dict[sym] += refs
    return labels_dict


def get_kmap_from_atoms(atoms: Atoms) -> list[str]:
    """Return a list of strings mapping ion index to element symbol and ion number.

    Return a list of strings mapping ion index to element symbol and ion number.
    (e.g. ["H #1", "H #2", "O #1", "O #2)

    Parameters
    ----------
    atoms : ase.Atoms
        The Atoms object of the system of interest
    """
    el_counter_dict = {}
    idx_to_key_map = []
    els = atoms.get_chemical_symbols()
    for i, el in enumerate(els):
        if el not in el_counter_dict:
            el_counter_dict[el] = 0
        el_counter_dict[el] += 1
        idx_to_key_map.append(f"{el} #{el_counter_dict[el]}")
    return idx_to_key_map


orb_ref_list = [
    ["s"],
    ["px", "py", "pz"],
    ["dxy", "dxz", "dyz", "dx2y2", "dz2"],
    ["fx3-3xy2", "fyx2-yz2", "fxz2", "fz3", "fyz2", "fxyz", "f3yx2-y3"],
]
