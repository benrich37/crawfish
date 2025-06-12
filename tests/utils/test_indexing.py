import pytest
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from crawfish.core.elecdata import ElecData
from pathlib import Path


def test_fidcs():
    from crawfish.utils.indexing import fidcs

    with pytest.raises(ValueError, match="atom indices must be int or list of int"):
        fidcs("a")
    assert fidcs(1) == [1]
    assert fidcs([1, 2]) == [1, 2]


def test_get_kmap_from_edata():
    from crawfish.utils.indexing import get_kmap_from_edata

    exdir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"
    edata = ElecData(exdir)
    kmap = get_kmap_from_edata(edata)
    assert len(kmap) == len(edata.structure.species)
    for i in range(2):
        assert kmap[i] == f"N #{i+1}"


# def test_get_kmap_from_atoms():
#     from crawfish.utils.indexing import get_kmap_from_atoms

#     exdir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"
#     edata = ElecData(exdir)
#     atoms = edata.atoms
#     kmap = get_kmap_from_atoms(atoms)
#     assert len(kmap) == len(atoms)
#     for i in range(2):
#         assert kmap[i] == f"N #{i+1}"


def test_get_atom_orb_labels_dict():
    from crawfish.utils.indexing import get_atom_orb_labels_dict

    exbandfile = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min" / "bandProjections"
    atom_orb_labels_dict = get_atom_orb_labels_dict(exbandfile)
    assert len(list(atom_orb_labels_dict.keys())) == 1
    assert "N" in atom_orb_labels_dict
    known_orb_labels = ["s", "px", "py", "pz"]
    assert len(atom_orb_labels_dict["N"]) == len(known_orb_labels)
    for orb_label in known_orb_labels:
        assert orb_label in atom_orb_labels_dict["N"]


def test_get_el_orb_u_dict():
    from crawfish.utils.indexing import get_el_orb_u_dict

    exdir = Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min"
    edata = ElecData(exdir)
    aidcs = [0, 1]
    el_orb_u_dict = get_el_orb_u_dict(edata, aidcs)
    assert len(list(el_orb_u_dict.keys())) == 1
    assert "N" in el_orb_u_dict
    known_orb_labels = ["s", "py", "pz", "px"]
    assert len(list(el_orb_u_dict["N"].keys())) == len(known_orb_labels)
    for i, orb_label in enumerate(known_orb_labels):
        assert orb_label in el_orb_u_dict["N"]
        assert len(el_orb_u_dict["N"][orb_label]) == 2  # one for each nitrogen
        assert type(el_orb_u_dict["N"][orb_label][0]) is int
        for j in range(len(edata.ion_names)):
            # orbital indices should appear in counting order
            # [atom #, subshell #, (arbitrary order within subshell checked against JDFTx source code)]
            assert j * len(known_orb_labels) + i in el_orb_u_dict["N"][orb_label]
