"""Module for atom and orbital indexing methods.

Methods for indexing atoms and orbitals in a system.
"""

from __future__ import annotations
from pathlib import Path
from crawfish.io.general import format_file_path, read_file

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from crawfish.core.elecdata import ElecData


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
    syms = edata.ion_names
    els = [syms[i] for i in aidcs]
    kmap = get_kmap_from_edata(edata)
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
    "H": ["s"],
    "O": ["s", "px", "py", "pz", "dxy", "dxz", "dyz", "dx2y2", "dz2"],
    "Pt": ["0s", "1s", "0px", "0py", "0pz", "1px", "1py", "1pz", "dxy", "dxz", "dyz", "dx2y2", "dz2", "fx3-3xy2", "fyx2-yz2", "fxz2", "fz3", "fyz2", "fxyz", "f3yx2-y3"]
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


def get_kmap_from_edata(edata: ElecData) -> list[str]:
    """Return a list of strings mapping ion index to element symbol and ion number.

    Return a list of strings mapping ion index to element symbol and ion number.
    (e.g. ["H #1", "H #2", "O #1", "O #2)

    Parameters
    ----------
    edata: ElecData
        The ElecData object of the system of interest
    """
    el_counter_dict = {}
    idx_to_key_map = []
    els = edata.ion_names
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


def get_orb_bool_func(orbs: list[str] | str | None) -> None | Callable:
    """Return a function that returns True if the orbital is in the input list.

    Return a function that returns True if the orbital is in the input list.

    Parameters
    ----------
    orbs : list[str] | str | None
        The list of orbitals of interest
    """
    orb_bool_func = None
    if orbs is not None:
        if type(orbs) is list:

            def orb_bool_func(s):
                return True in [o in s for o in orbs]
        else:

            def orb_bool_func(s):
                return orbs in s

    return orb_bool_func


def get_aidcs(edata: ElecData, idcs: list[int] | int | None, elements: list[str] | str | None) -> list[int]:
    """Return all ion indices encompassed by the input indices or elements.

    Return all ion indices encompassed by the input indices or elements.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest
    idcs : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements : list[str] | str | None
        The elements of interest. (auto-filled by idcs if None)
    """
    if elements is not None:
        if idcs is not None:
            raise ValueError("Cannot provide both idcs and elements.")
        idcs = []
        for el in elements:
            idcs += [i for i, sym in enumerate(edata.ion_names) if sym == el]
    if idcs is None:
        idcs = list(range(len(edata.ion_names)))
    if isinstance(idcs, int):
        idcs = [idcs]
    return idcs


def get_orb_idcs(
    edata: ElecData, idcs: list[int] | int | None, elements: list[str] | str | None, orbs: list[str] | str | None
) -> list[int]:
    """Return all orbital indices encompassed by the input indices, elements, or orbitals.

    Return all orbital indices encompassed by the input indices, elements, or orbitals.

    Parameters
    ----------
    edata : ElecData
        The ElecData object of the system of interest
    idcs : list[int] | int | None
        The indices of the ions of interest. (auto-filled by elements if None)
    elements : list[str] | str | None
        The elements of interest. (auto-filled by idcs if None)
    orbs : list[str] | str | None
        The orbitals of interest. (all orbitals if None)
    """
    if all(x is None for x in [idcs, elements, orbs]):
        raise ValueError("Must provide idcs, elements, or orbs.")
    idcs = get_aidcs(edata, idcs, elements)
    orb_bool_func = get_orb_bool_func(orbs)
    orb_idcs = []
    if orb_bool_func is not None:
        el_orb_u_dict = get_el_orb_u_dict(edata, idcs)
        for el in el_orb_u_dict:
            for orb in el_orb_u_dict[el]:
                if orb_bool_func(orb):
                    orb_idcs += el_orb_u_dict[el][orb]
    else:
        for idx in idcs:
            orb_idcs += edata.orbs_idx_dict[edata.kmap[idx]]
    return orb_idcs
