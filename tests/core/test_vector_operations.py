import pytest
import numpy as np
from crawfish.utils.typing import REAL_DTYPE


def test_get_gauss_smear_spectrum():
    from crawfish.core.operations.vector import get_gauss_smear_spectrum

    erange = np.arange(0, 10, 0.1, dtype=REAL_DTYPE)
    e_sabcj = np.array([5], dtype=REAL_DTYPE)
    w_sabcj = np.array([1], dtype=REAL_DTYPE)
    e_sabcj.reshape([1, 1, 1, 1, 1])
    w_sabcj.reshape([1, 1, 1, 1, 1])
    cs = get_gauss_smear_spectrum(erange, e_sabcj, w_sabcj, 0.1)
    assert len(cs) == 1
    assert len(cs[0]) == len(erange)
    assert cs[0][int(len(erange) / 2)] == pytest.approx(1.0)
    assert cs[0][0] == pytest.approx(0.0)


def test_get_uneven_integrated_array():
    from crawfish.core.operations.vector import get_uneven_integrated_array

    e_sabcj = np.array([1, 2], dtype=REAL_DTYPE)
    w_sabcj = np.array([1, 1], dtype=REAL_DTYPE)
    e_sabcj = e_sabcj.reshape([2, 1, 1, 1, 1])
    w_sabcj = w_sabcj.reshape([2, 1, 1, 1, 1])
    e, integrated = get_uneven_integrated_array(e_sabcj, w_sabcj)
    assert len(e) == 2
    assert len(e) == len(integrated)
    assert integrated[-1] == pytest.approx(2)


def test_get_lti_spectrum():
    from crawfish.core.operations.vector import get_lti_spectrum

    # TODO: Test known analytical values
    e_sabcj = np.ones(9, dtype=REAL_DTYPE) * 0.5
    e_sabcj = e_sabcj.reshape([1, 3, 3, 1, 1])
    e_sabcj[:, 1:, :, :, :] += 0.2
    e_sabcj[:, :, 1:, :, :] += 0.2
    e_sabcj[:, 2, :, :, :] += 0.2
    e_sabcj[:, :, 2, :, :] += 0.2
    w_sabcj = np.ones(9, dtype=REAL_DTYPE)
    w_sabcj = w_sabcj.reshape([1, 3, 3, 1, 1])
    erange = np.arange(0, 10, 0.1, dtype=REAL_DTYPE)
    lattice = np.eye(3, dtype=REAL_DTYPE)
    cs = get_lti_spectrum(e_sabcj, erange, w_sabcj, lattice)
    assert len(cs) == 1
