import pytest


def test_gauss():
    from crawfish.core.operations.scalar import gauss

    x1 = -1
    x2 = 1
    mu = 0
    sig = 1
    assert gauss(x1, mu, sig) == pytest.approx(gauss(x2, mu, sig))
    assert gauss(x1, mu, sig) == pytest.approx(0.36787944117)
