from typing import List, Dict, Tuple
import numpy as np
from functools import lru_cache
from pathlib import Path
from crawfish.utils.typing import REAL_DTYPE
from crawfish.io.general import safe_load



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
    
    # def load_cache(self, filename: str):
    #     """Load cache from a file"""
    #     if Path(filename).exists():
    #         loaded = np.load(filename, allow_pickle=True)
    #         keys = loaded['keys']
    #         values = loaded['values']
    #         self.cache = dict(zip(keys, values))
