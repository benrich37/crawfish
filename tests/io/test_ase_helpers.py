from crawfish.utils.testing import EXAMPLE_FILES_DIR
from pathlib import Path


def test_get_atoms_list_from_out():
    from crawfish.io.ase_helpers import get_atoms_list_from_out
    from ase import Atoms

    # TODO: Replace with a longer out file
    exoutfile = Path(EXAMPLE_FILES_DIR) / "out"
    atoms_list = get_atoms_list_from_out(exoutfile)
    for atoms in atoms_list:
        assert isinstance(atoms, Atoms)
    assert len(atoms_list) == 1
    assert len(atoms_list[0]) == 2
