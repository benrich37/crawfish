import pytest


def test_fidcs():
    from crawfish.utils.indexing import fidcs

    with pytest.raises(ValueError, match="atom indices must be int or list of int"):
        fidcs("a")
    assert fidcs(1) == [1]
    assert fidcs([1, 2]) == [1, 2]
