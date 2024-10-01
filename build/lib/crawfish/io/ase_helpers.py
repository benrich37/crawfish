from os.path import join as opj
from ase import Atoms, Atom
from ase.units import Bohr
import numpy as np
from ase.io import read, write
from crawfish.io.data_parsing import (
    get_start_lines, 
    get_input_coord_vars_from_outfile_filepath
    )
from shutil import copy as cp

def get_atoms_poscar(path):
    atoms = read(opj(path, "POSCAR"), format="vasp")
    return atoms

def get_atoms_contcar(path):
    atoms = read(opj(path, "CONTCAR"), format="vasp")
    return atoms

def get_atoms_poscar_gjf(path):
    atoms = read(opj(path, "POSCAR.gjf"), format="gaussian-in")
    return atoms

def get_atoms_contcar_gjf(path):
    atoms = read(opj(path, "CONTCAR.gjf"), format="gaussian-in")
    return atoms

def get_atoms_from_calc_dir(path):
    atoms = None
    try:
        atoms = get_atoms_from_out(opj(path, "out"))
    except:
        pass
    if atoms is None:
        try:
            cp(opj(path, "out"), opj(path, "out_read"))
            fix_out_file(opj(path, "out_read"))
            atoms = get_atoms_from_out(opj(path, "out_read"))
        except:
            pass
        if not atoms is None:
            cp(opj(path, "out_read"), opj(path, "out"))
        if atoms is None:
            for ga_func in [get_atoms_contcar_gjf, get_atoms_contcar, get_atoms_poscar_gjf, get_atoms_poscar]:
                try:
                    atoms = ga_func(path)
                except:
                    pass
                if not atoms is None:
                    break
    if atoms is None:
        raise ValueError(f"Could not retrieve atoms object from files within {path}")
    return atoms