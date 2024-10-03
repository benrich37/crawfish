import pytest
import numpy as np
import re
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE


def test_check_arr_typing_all():
    from crawfish.utils.typing import check_arr_typing_all

    with pytest.raises(ValueError, match="All arrays must be of type REAL_DTYPE or COMPLEX_DTYPE"):
        check_arr_typing_all([np.zeros(5, dtype=np.float64), np.ones(5, dtype=COMPLEX_DTYPE)])
    with pytest.raises(
        ValueError, match=re.escape("All arrays must have the same dtype (REAL_DTYPE or COMPLEX_DTYPE)")
    ):
        check_arr_typing_all([np.zeros(5, dtype=REAL_DTYPE), np.ones(5, dtype=COMPLEX_DTYPE)])
    check_arr_typing_all([np.zeros(5, dtype=REAL_DTYPE), np.ones(5, dtype=REAL_DTYPE)])
    check_arr_typing_all([np.zeros(5, dtype=COMPLEX_DTYPE), np.ones(5, dtype=COMPLEX_DTYPE)])


def test_check_arr_typing():
    from crawfish.utils.typing import check_arr_typing

    with pytest.raises(ValueError, match="All arrays must be of type REAL_DTYPE or COMPLEX_DTYPE"):
        check_arr_typing([np.zeros(5, dtype=np.float64), np.ones(5, dtype=COMPLEX_DTYPE)])
    check_arr_typing([np.zeros(5, dtype=REAL_DTYPE), np.ones(5, dtype=COMPLEX_DTYPE)])
