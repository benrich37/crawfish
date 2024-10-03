import pytest
import numpy as np
import re


def test_check_arr_typing_all():
    from crawfish.utils.typing import check_arr_typing_all

    with pytest.raises(ValueError, match="All arrays must be of type np.float32 or np.complex64"):
        check_arr_typing_all([np.zeros(5, dtype=np.float64), np.ones(5, dtype=np.complex64)])
    with pytest.raises(ValueError, match=re.escape("All arrays must have the same dtype (np.float32 or np.complex64)")):
        check_arr_typing_all([np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.complex64)])
    check_arr_typing_all([np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)])
    check_arr_typing_all([np.zeros(5, dtype=np.complex64), np.ones(5, dtype=np.complex64)])


def test_check_arr_typing():
    from crawfish.utils.typing import check_arr_typing

    with pytest.raises(ValueError, match="All arrays must be of type np.float32 or np.complex64"):
        check_arr_typing([np.zeros(5, dtype=np.float64), np.ones(5, dtype=np.complex64)])
    check_arr_typing([np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.complex64)])
