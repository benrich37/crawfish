from os.path import join as opj
from ase.io import read
from shutil import copy as cp
import numpy as np
from crawfish.io.data_parsing import read_file, get_outfile_start_lines


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
    except ValueError as err:
        pass
    if atoms is None:
        try:
            cp(opj(path, "out"), opj(path, "out_read"))
            fix_out_file(opj(path, "out_read"))
            atoms = get_atoms_from_out(opj(path, "out_read"))
        except:
            pass
        if atoms is not None:
            cp(opj(path, "out_read"), opj(path, "out"))
        if atoms is None:
            for ga_func in [get_atoms_contcar_gjf, get_atoms_contcar, get_atoms_poscar_gjf, get_atoms_poscar]:
                try:
                    atoms = ga_func(path)
                except:
                    pass
                if atoms is not None:
                    break
    if atoms is None:
        raise ValueError(f"Could not retrieve atoms object from files within {path}")
    return atoms


def get_atoms(path):
    atoms = get_atoms_from_out(opj(path, "out"))
    return atoms


def get_atoms_from_outfile_data(names, posns, R, charges=None, E=0, momenta=None):
    atoms = Atoms()
    posns *= Bohr
    R = R.T * Bohr
    atoms.cell = R
    atoms.pbc = True
    if charges is None:
        charges = np.zeros(len(names))
    if momenta is None:
        momenta = np.zeros([len(names), 3])
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i], momentum=momenta[i]))
    atoms.E = E
    return atoms


def get_atoms_from_out(outfile):
    atoms_list = get_atoms_list_from_out(outfile)
    return atoms_list[-1]


def get_atoms_list_from_out_reset_vars(nAtoms=100, _def=100):
    R = np.zeros([3, 3])
    posns = []
    names = []
    chargeDir = {}
    active_lattice = False
    lat_row = 0
    active_posns = False
    log_vars = False
    coords = None
    new_posn = False
    active_lowdin = False
    idxMap = {}
    j = 0
    E = 0
    if nAtoms is None:
        nAtoms = _def
    charges = np.zeros(nAtoms, dtype=float)
    forces = []
    active_forces = False
    coords_forces = None
    return (
        R,
        posns,
        names,
        chargeDir,
        active_posns,
        active_lowdin,
        active_lattice,
        posns,
        coords,
        idxMap,
        j,
        lat_row,
        new_posn,
        log_vars,
        E,
        charges,
        forces,
        active_forces,
        coords_forces,
    )


def get_atoms_list_from_out_slice(outfile_filepath, i_start, i_end):
    charge_key = "oxidation-state"
    opts = []
    nAtoms = None
    (
        R,
        posns,
        names,
        chargeDir,
        active_posns,
        active_lowdin,
        active_lattice,
        posns,
        coords,
        idxMap,
        j,
        lat_row,
        new_posn,
        log_vars,
        E,
        charges,
        forces,
        active_forces,
        coords_forces,
    ) = get_atoms_list_from_out_reset_vars()
    outfile = read_file(outfile_filepath)
    for i, line in enumerate(outfile):
        if i > i_start and i < i_end:
            if new_posn:
                if "Lowdin population analysis " in line:
                    active_lowdin = True
                elif "R =" in line:
                    active_lattice = True
                elif "# Forces in" in line:
                    active_forces = True
                    coords_forces = line.split()[3]
                elif line.find("# Ionic positions in") >= 0:
                    coords = line.split()[4]
                    active_posns = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
                        lat_row += 1
                    else:
                        active_lattice = False
                        lat_row = 0
                elif active_posns:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == "ion":
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                        if tokens[1] not in idxMap:
                            idxMap[tokens[1]] = []
                        idxMap[tokens[1]].append(j)
                        j += 1
                    else:
                        posns = np.array(posns)
                        active_posns = False
                        nAtoms = len(names)
                        if len(charges) < nAtoms:
                            charges = np.zeros(nAtoms)
                ##########
                elif active_forces:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == "force":
                        forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    else:
                        forces = np.array(forces)
                        active_forces = False
                ##########
                elif "Minimize: Iter:" in line:
                    if "F: " in line:
                        E = float(line[line.index("F: ") :].split(" ")[1])
                    elif "G: " in line:
                        E = float(line[line.index("G: ") :].split(" ")[1])
                elif active_lowdin:
                    if charge_key in line:
                        look = line.rstrip("\n")[line.index(charge_key) :].split(" ")
                        symbol = str(look[1])
                        line_charges = [float(val) for val in look[2:]]
                        chargeDir[symbol] = line_charges
                        for atom in list(chargeDir.keys()):
                            for k, idx in enumerate(idxMap[atom]):
                                charges[idx] += chargeDir[atom][k]
                    elif "#" not in line:
                        active_lowdin = False
                        log_vars = True
                elif log_vars:
                    if np.sum(R) == 0.0:
                        R = get_input_coord_vars_from_outfile(outfile)[2]
                    if coords != "cartesian":
                        posns = np.dot(posns, R)
                    if len(forces) == 0:
                        forces = np.zeros([nAtoms, 3])
                    if coords_forces.lower() != "cartesian":
                        forces = np.dot(forces, R)
                    opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
                    (
                        R,
                        posns,
                        names,
                        chargeDir,
                        active_posns,
                        active_lowdin,
                        active_lattice,
                        posns,
                        coords,
                        idxMap,
                        j,
                        lat_row,
                        new_posn,
                        log_vars,
                        E,
                        charges,
                        forces,
                        active_forces,
                        coords_forces,
                    ) = get_atoms_list_from_out_reset_vars(nAtoms=nAtoms)
            elif "Computing DFT-D3 correction:" in line:
                new_posn = True
    return opts


def get_atoms_list_from_out(outfile_filepath):
    outfile = read_file(outfile_filepath)
    start_lines = get_outfile_start_lines(outfile, add_end=True)
    for i in range(len(start_lines) - 1):
        i_start = start_lines[::-1][i + 1]
        i_end = start_lines[::-1][i]
        atoms_list = get_atoms_list_from_out_slice(outfile, i_start, i_end)
        if type(atoms_list) is list:
            if len(atoms_list):
                return atoms_list
    erstr = "Failed getting atoms list from out file"
    raise ValueError(erstr)
