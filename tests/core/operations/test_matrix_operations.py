import pytest
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE
import numpy as np


def test_get_p_uvjsabc():
    from crawfish.core.operations.matrix import get_p_uvjsabc

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


def test_get_h_uvsabc():
    from crawfish.core.operations.matrix import get_h_uvsabc, get_p_uvjsabc

    nproj = 3
    nspin = 2
    nka = 1
    nkb = 1
    nkc = 1
    nbands = 2
    proj_sabcju = np.zeros([nspin, nka, nkb, nkc, nbands, nproj], dtype=COMPLEX_DTYPE)
    proj_sabcju += np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    proj_sabcju += 1j * np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    p_uvjsabc = get_p_uvjsabc(proj_sabcju)
    e_sabcj = np.ones([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE) * (-1)
    h_uvsabc = get_h_uvsabc(p_uvjsabc, e_sabcj)
    for u in range(nproj):
        for v in range(nproj):
            for s in range(nspin):
                for a in range(nka):
                    for b in range(nkb):
                        for c in range(nkc):
                            assert h_uvsabc[u, v, s, a, b, c] == pytest.approx(-2 * nbands)
    h_uvsabc = get_h_uvsabc(p_uvjsabc, e_sabcj, orbs_u=[0], orbs_v=[1])
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for u in range(nproj):
                        for v in range(nproj):
                            if u == 0 and v == 1:
                                assert h_uvsabc[u, v, s, a, b, c] == pytest.approx(-2 * nbands)
                            else:
                                assert h_uvsabc[u, v, s, a, b, c] == pytest.approx(0.0)
    p_uvjsabc[0, 2, :, :, :, :, :] *= -1
    h_uvsabc = get_h_uvsabc(p_uvjsabc, e_sabcj, orbs_u=[0], orbs_v=[1, 2])
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for u in range(nproj):
                        for v in range(nproj):
                            if u == 0 and v == 1:
                                assert h_uvsabc[u, v, s, a, b, c] == pytest.approx(-2 * nbands)
                            elif u == 0 and v == 2:
                                assert h_uvsabc[u, v, s, a, b, c] == pytest.approx(2 * nbands)
                            else:
                                assert h_uvsabc[u, v, s, a, b, c] == pytest.approx(0.0)


def test_get_pcoop_sabcj():
    from crawfish.core.operations.matrix import get_p_uvjsabc, get_pcoop_sabcj

    nproj = 3
    nspin = 2
    nka = 1
    nkb = 1
    nkc = 1
    nbands = 2
    np.ones([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE) * (-1)
    proj_sabcju = np.zeros([nspin, nka, nkb, nkc, nbands, nproj], dtype=COMPLEX_DTYPE)
    proj_sabcju += np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    proj_sabcju += 1j * np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    p_uvjsabc = get_p_uvjsabc(proj_sabcju)
    # h_uvsabc = get_h_uvsabc(p_uvjsabc, e_sabcj)
    orbs_u = [0]
    orbs_v = [1]
    pcoop_sabcj = get_pcoop_sabcj(p_uvjsabc, orbs_u, orbs_v)
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        # pcoop[s,a,b,c,j] = sum(proj_u,v)
                        # only one orbital in each set, and p_uvjsabc is 2 everywhere
                        assert pcoop_sabcj[s, a, b, c, j] == pytest.approx(2)
    # encode anti-bonding in final band by setting all P elements for that band to negative
    p_uvjsabc[:, :, -1, :, :, :, :] *= -1
    pcoop_sabcj = get_pcoop_sabcj(p_uvjsabc, orbs_u, orbs_v)
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        if j < nbands - 1:
                            # assert is expected bonding value
                            assert pcoop_sabcj[s, a, b, c, j] == pytest.approx(2)
                        else:
                            # assert is expected anti-bonding value
                            assert pcoop_sabcj[s, a, b, c, j] == pytest.approx(-2)


def test_get_pcohp_sabcj():
    from crawfish.core.operations.matrix import get_h_uvsabc, get_p_uvjsabc, get_pcohp_sabcj

    nproj = 3
    nspin = 2
    nka = 1
    nkb = 1
    nkc = 1
    nbands = 3
    e_sabcj = np.ones([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE) * (-1)
    proj_sabcju = np.zeros([nspin, nka, nkb, nkc, nbands, nproj], dtype=COMPLEX_DTYPE)
    proj_sabcju += np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    proj_sabcju += 1j * np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    p_uvjsabc = get_p_uvjsabc(proj_sabcju)
    h_uvsabc = get_h_uvsabc(p_uvjsabc, e_sabcj)
    orbs_u = [0]
    orbs_v = [1]
    pcohp_sabcj = get_pcohp_sabcj(p_uvjsabc, h_uvsabc, orbs_u, orbs_v)
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        # 2**2 comes from "2" value multiplied in twice (once from p_uvjsabc, once from h_uvsabc constructed from p_uvjsabc)
                        # nbands factor comes from summing over all bands in constructing h_uvsabc
                        # negative sign comes from the negative sign in the h_uvsabc (eigenvalues set all to -1)
                        assert pcohp_sabcj[s, a, b, c, j] == pytest.approx(-(2**2) * nbands)
    # encode anti-bonding in final band by setting all P elements for that band to negative
    p_uvjsabc[:, :, -1, :, :, :, :] *= -1
    h_uvsabc = get_h_uvsabc(p_uvjsabc, e_sabcj)
    pcohp_sabcj = get_pcohp_sabcj(p_uvjsabc, h_uvsabc, orbs_u, orbs_v)
    # first "2" for bonding elements of p_uvjsabc,
    # "-2*(nbands-1)" for part of h_uvsabc for sum over bonding bands (all but last),
    # and "-2*1" for part of h_uvsabc for sum over antibonding band
    expected_bonding = 2 * (-2 * (nbands - 1) - (-2 * 1))
    # Same as expected_bonding in magnitude as pulls from same [u,v] from h_uvsabc,
    # but negative sign from negative sign of p_uvjsabc at the antibonding band
    expected_antibonding = -2 * (-2 * (nbands - 1) - (-2 * 1))
    for s in range(nspin):
        for a in range(nka):
            for b in range(nkb):
                for c in range(nkc):
                    for j in range(nbands):
                        if j < nbands - 1:
                            assert pcohp_sabcj[s, a, b, c, j] == pytest.approx(expected_bonding)
                        else:
                            assert pcohp_sabcj[s, a, b, c, j] == pytest.approx(expected_antibonding)


def test_get_pdos_sabcj():
    nproj = 2
    nspin = 1
    nka = 1
    nkb = 1
    nkc = 1
    nbands = 3
    e_sabcj = np.ones([nspin, nka, nkb, nkc, nbands], dtype=REAL_DTYPE) * (-1)
    e_sabcj[:, :, :, :, 0] -= 1
    e_sabcj[:, :, :, :, 0] += 1
    proj_sabcju = np.zeros([nspin, nka, nkb, nkc, nbands, nproj], dtype=COMPLEX_DTYPE)
    proj_sabcju += np.ones([nspin, nka, nkb, nkc, nbands, nproj])
    proj_sabcju += 1j * np.ones([nspin, nka, nkb, nkc, nbands, nproj])
