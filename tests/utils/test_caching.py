import pytest
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR
from crawfish.core.elecdata import ElecData
from pathlib import Path
import numpy as np

from crawfish.utils.caching import is_same_type_robust, is_matching_value

# test1 = np.array([1,2,3])
# test2 = str(test1)
# test3 = eval(test2)

listlikes1 = [
        [0,1,2],
        np.array([0,1,2]),
        np.array([0,1,2], dtype=np.int32),
        range(3)
    ]

listlikes2 = [
        [0,1,2,3],
        np.array([0,1,2,3]),
        np.array([0,1,2,3], dtype=np.int32),
        range(4)
    ]

floatlikes1 = [
    0.1,
    np.float32(0.1),
    np.float64(0.1),
]

floatlikes2 = [
    0.2,
    np.float32(0.2),
    np.float64(0.2),
]

intlikes1 = [
    1,
    np.int32(1),
    np.int64(1),
]
intlikes2 = [
    2,
    np.int32(2),
    np.int64(2),
]

boollikes1 = [
    True,
    np.bool_(True),
]
boollikes2 = [
    False,
    np.bool_(False),
]

comparable_collections = [
    [listlikes1, listlikes2],
    [floatlikes1, floatlikes2],
    [intlikes1, intlikes2],
    [boollikes1, boollikes2],
]


def test_is_same_type_robust():
    from crawfish.utils.caching import is_same_type_robust

    for c1, c2 in comparable_collections:
        for v1 in c1:
            for v2 in c1:
                assert is_same_type_robust(v1, v2)
            for v2 in c2:
                assert is_same_type_robust(v1, v2)

def test_is_matching_value():
    from crawfish.utils.caching import is_matching_value

    for c1, c2 in comparable_collections:
        for v1 in c1:
            for v2 in c1:
                assert is_matching_value(v1, v2)
            for v2 in c2:
                assert not is_matching_value(v1, v2)
        for v1 in c2:
            for v2 in c1:
                assert not is_matching_value(v1, v2)
            for v2 in c2:
                assert is_matching_value(v1, v2)


test_metadatas = [
    {
        "a": "a",
        "b": 1,
        "c": [1, 2, 3],
        "d": np.array([1, 2, 3]),
        "e": np.float32(1.0),
        "f": np.int32(1),
        "g": np.bool_(True),
        "h": {"i": "j", "k": 2.0},
        "l": np.array([1, 2, 3], dtype=np.int32),
        "m": np.array([1, 2, 3], dtype=np.float32),
        "n": np.array([True, False, True], dtype=np.bool_),
    }
]

def test_metadata_self_consistency():
    from crawfish.utils.caching import write_generic_metadata, parse_generic_metadata, is_matching_metadata
    from os import getcwd
    cwd = Path(getcwd())
    from os import remove

    for metadata in test_metadatas:
        write_generic_metadata(cwd, metadata)
        parsed_metadata = parse_generic_metadata(cwd)
        assert is_matching_metadata(metadata, parsed_metadata)
    remove(cwd / "metadata.pickle")
