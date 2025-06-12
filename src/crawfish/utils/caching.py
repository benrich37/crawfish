from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from crawfish.utils.typing import REAL_DTYPE
from crawfish.io.general import safe_load
from typing import Any, TYPE_CHECKING
from crawfish.utils.typing import REAL_DTYPE, COMPLEX_DTYPE



class CachedFunction:
    
    def __init__(self, cache_file: str = 'cache.npz', arr_dtype: np.dtype = REAL_DTYPE, auto_save: bool = True):
        # Dictionary to store computed results
        self.cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray] = {}
        self.arr_dtype = arr_dtype
        self.auto_save = auto_save
        self.cache_file = cache_file

    def _create_key(self, oidcs1: List[int], oidcs2: List[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Create a canonical key from the input lists by sorting them.
        This ensures that different orderings of the same integers produce the same key.
        """
        return (tuple(sorted(oidcs1)), tuple(sorted(oidcs2)))

    def compute_or_retrieve(self, oidcs1: List[int], oidcs2: List[int], f) -> np.ndarray:
        """
        Compute f(oidcs1, oidcs2) if not already cached, otherwise return cached result.

        Parameters:
        oidcs1 (List[int]): First list of integers
        oidcs2 (List[int]): Second list of integers
        f: Function that takes two lists of integers and returns a numpy array

        Returns:
        np.ndarray: Result of f(oidcs1, oidcs2)
        """
        key1 = self._create_key(oidcs1, oidcs2)
        key2 = self._create_key(oidcs1, oidcs2)

        # Check if result is already cached
        for key in [key1, key2]:
            if key in self.cache:
                return self.cache[key]

        # Compute and cache result
        result = f(oidcs1, oidcs2)
        self.cache[key1] = result
        if self.auto_save:
            self.save_cache()
        return result

    def clear(self):
        """Clear the entire cache"""
        self.cache.clear()

    def cache_size(self) -> int:
        """Return the number of cached results"""
        return len(self.cache)
    
    def save_cache(self):
        """Save cache to a file using numpy's savez_compressed"""
        cache_data = {
            'keys': list(self.cache.keys()),
            'values': list(self.cache.values())
        }
        np.savez_compressed(self.cache_file, 
                          keys=np.array(cache_data['keys'], dtype=object),
                          values=np.array(cache_data['values'], dtype=object))
        
    def load_cache(self):
        """Load cache from a file"""
        loaded = safe_load(Path(self.cache_file), allow_pickle=True)
        if loaded is not None:
            keys = [tuple(tuple(k) for k in key) for key in loaded['keys']]
            values = loaded['values']
            self.cache = dict(zip(keys, values))
            for key in keys:
                self.cache[key] = np.array(self.cache[key], dtype=self.arr_dtype)


def parse_generic_metadata(cache_dir: Path) -> dict:
    """Parse the metadata from the cache directory.

    Parse the metadata from the cache directory.
    """
    metadata_file = cache_dir / "metadata.txt"
    vals = {}
    with open(metadata_file, "r") as f:
        for line in f:
            vals = eval(line.strip())
            break
    return vals

def write_generic_metadata(cache_dir: Path, metadata_dict: dict):
    """Write the metadata to the cache directory.

    Write the metadata to the cache directory.
    """
    metadata_file = cache_dir / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(str(metadata_dict))
    

def is_matching_str(v_from_file: str, v_from_args: str) -> bool:
    """Check if the string values match.

    Check if the string values match.
    """
    return v_from_file.strip() == v_from_args.strip()

_listlikes = (list, np.ndarray, range, tuple, set)
_boollikes = (bool, np.bool_)
_intlikes = (int, np.int32, np.int64)
_floatlikes = (float, np.float32, np.float64)
_realnumlikes = _floatlikes + _intlikes

def is_matching_float(v_from_file: _realnumlikes, v_from_args: _floatlikes) -> bool:
    """Check if the float values match.

    Check if the float values match.
    """
    # try:
    #     v_from_file = float(v_from_file)
    # except (ValueError, TypeError):
    #     return False
    return np.isclose(v_from_file, v_from_args)

def is_matching_int(v_from_file: _realnumlikes, v_from_args: _intlikes) -> bool:
    """Check if the int values match.

    Check if the int values match.
    """
    # try:
    #     v_from_file = int(v_from_file)
    # except (ValueError, TypeError):
    #     return False
    #return v_from_file == v_from_args
    return np.isclose(v_from_file, v_from_args)

def is_matching_bool(v_from_file: _boollikes, v_from_args: _boollikes) -> bool:
    """Check if the bool values match.

    Check if the bool values match.
    """
    return bool(v_from_file == v_from_args)


def list_contains_val(test_list: list[Any], val: Any) -> bool:
    """Check if the list contains the value.

    Check if the list contains the value.
    """
    return any([is_matching_value(v, val) for v in test_list])

def is_matching_list(v_from_file: _listlikes, v_from_args: _listlikes) -> bool:
    """Check if the list values match.

    Check if the list values match.
    """
    # Convert to lists here so they're easier to work with
    test_v_from_file = list(v_from_file)
    test_v_from_args = list(v_from_args)
    if len(test_v_from_file) != len(test_v_from_args):
        return False
    for subv in test_v_from_args:
        if not list_contains_val(test_v_from_file, subv):
            return False
    return True

def is_matching_value(
    v_from_file: Any,
    v_from_args: Any,
) -> bool:
    """Check if the values match.

    Check if the values match.
    """
    if not is_same_type_robust(v_from_file, v_from_args):
        return False
    elif v_from_args is None:
        return v_from_file is None
    elif isinstance(v_from_args, str):
        return is_matching_str(v_from_file, v_from_args)
    elif type(v_from_args) in _boollikes:
        return is_matching_bool(v_from_file, v_from_args)
    elif type(v_from_args) in _listlikes:
        return is_matching_list(v_from_file, v_from_args)
    elif isinstance(v_from_args, dict):
        return is_matching_metadata(v_from_file, v_from_args)
    # This can possibly be reduced to just comparing _realnumlikes, but keeping it explicit for now
    elif type(v_from_args) in _floatlikes:
        return is_matching_float(v_from_file, v_from_args)
    elif type(v_from_args) in _intlikes:
        return is_matching_int(v_from_file, v_from_args)
    else:
        raise ValueError(f"Unsupported type {type(v_from_args)} for value {v_from_args}.")
    


def is_same_type_robust(v1: Any, v2: Any) -> bool:
    """Check if the two values are of the same type.

    Check if the two values are of the same type.
    """
    if v1 is None:
        return v2 is None
    elif isinstance(v1, str):
        return isinstance(v2, str)
    # Bools are also recognized as real numbers, so we need to check for them first
    elif isinstance(v1, _boollikes):
        return isinstance(v2, _boollikes)
    elif isinstance(v1, _realnumlikes):
        return isinstance(v2, _realnumlikes)
    elif isinstance(v1, _listlikes):
        return isinstance(v2, _listlikes)
    elif isinstance(v1, dict):
        return isinstance(v2, dict)
    

def is_matching_metadata(
    metadata_from_file: dict[str, Any],
    metadata_from_args: dict[str, Any]
) -> bool:
    """Check if the metadata in the cache directory matches the given metadata.

    Check if the metadata in the cache directory matches the given metadata.
    """
    for key, val in metadata_from_args.items():
        if key not in metadata_from_file:
            return False
        elif not is_matching_value(metadata_from_file[key], val):
            return False
    return True

    
def is_generic_cache_dir(test_cache_dir: Path, metadata_dict: dict) -> bool:
    test_metadata_dict = parse_generic_metadata(test_cache_dir)
    return is_matching_metadata(test_metadata_dict, metadata_dict)


def get_preexisting_generic_cache_dirs(parent_dir: Path, prefix: str):
    cache_dirs = list(parent_dir.glob(f"{prefix}_*"))
    cache_dirs = [d for d in cache_dirs if d.is_dir()]
    return cache_dirs

def find_generic_cache_dir(parent_dir: Path, prefix: str, metadata_dict: dict):
    """Find the generic cache directory.
    Find the generic cache directory.
    """
    cache_dirs = get_preexisting_generic_cache_dirs(parent_dir, prefix)
    for cache_dir in cache_dirs:
        if is_generic_cache_dir(cache_dir, metadata_dict):
            return cache_dir
    return None


def write_new_generic_cache_dir(parent_dir: Path, prefix: str, metadata_dict: dict):
    """Write a new energy range cache directory.

    Write a new energy range cache directory.
    """
    cache_dirs = get_preexisting_generic_cache_dirs(parent_dir, prefix)
    cache_dir = None
    for i in range(len(cache_dirs)+1):
        cache_dir = parent_dir / f"{prefix}_{i}"
        if not cache_dir.is_dir():
            break
    if cache_dir is None:
        raise ValueError(f"Issue with get_preexisting_generic_cache_dirs not returning all valid {prefix} cache dirs in {parent_dir}.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_generic_metadata(cache_dir, metadata_dict)
    return cache_dir

    
def get_generic_cache_dir(parent_dir: Path, prefix: str, metadata_dict: dict):
    """Return the generic cache directory.

    Return the generic cache directory.
    """
    cache_dir = find_generic_cache_dir(parent_dir, prefix, metadata_dict)
    if cache_dir is not None:
        return cache_dir
    else:
        cache_dir = write_new_generic_cache_dir(parent_dir, prefix, metadata_dict)
        return cache_dir

def get_erange_cache_dir(parent_dir: Path, erange: np.ndarray):
    """Return the energy range cache directory.

    Return the energy range cache directory.
    """
    prefix = "erange"
    metadata_dict = {
        "emin": erange[0],
        "emax": erange[-1],
        "nstep": len(erange),
    }
    return get_generic_cache_dir(parent_dir, prefix, metadata_dict)

