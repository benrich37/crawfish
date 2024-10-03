import pytest
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE


def test_get_p_uvjsabc():
    from crawfish.core.operations.matrix import get_p_uvjsabc
    import numpy as np

    nproj = 3
    nspin = 2
    nka = 4
    nkb = 3
    nkc = 2
    nbands = 3
    proj_sabcju = np.zeros([nspin, nka, nkb, nkc, nbands, nproj], dtype=COMPLEX_DTYPE)
    proj_sabcju += np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    proj_sabcju += 1j * np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    p_uvjsabc = get_p_uvjsabc(proj_sabcju)
    shape_expected = (nproj, nproj, nbands, nspin, nka, nkb, nkc)
    assert len(p_uvjsabc.shape) == len(shape_expected)
    for i in range(len(shape_expected)):
        assert p_uvjsabc.shape[i] == shape_expected[i]
    assert isinstance(p_uvjsabc[0, 0, 0, 0, 0, 0, 0], REAL_DTYPE)
    assert not isinstance(p_uvjsabc[0, 0, 0, 0, 0, 0, 0], COMPLEX_DTYPE)
    assert all(np.isclose(p_uvjsabc.flatten(), 2.0))
    p_uvjsabc = get_p_uvjsabc(proj_sabcju, orbs_u=[2])
    assert p_uvjsabc[0, 0, 0, 0, 0, 0, 0] == pytest.approx(0.0)
    proj_sabcju[0, 0, 0, 0, 0, 0] = 0.0
    p_uvjsabc = get_p_uvjsabc(proj_sabcju)
    assert p_uvjsabc[0, 0, 0, 0, 0, 0, 0] == pytest.approx(0.0)
