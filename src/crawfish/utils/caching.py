from typing import List, Dict, Tuple
import numpy as np
from functools import lru_cache

class CachedFunction:
    def __init__(self):
        # Dictionary to store computed results
        self.cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray] = {}

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
                print(f"Cache hit for key: {key}")
                return self.cache[key]

        # Compute and cache result
        print(f"Cache miss for key: {key1}")
        result = f(oidcs1, oidcs2)
        self.cache[key] = result
        return result

    def clear(self):
        """Clear the entire cache"""
        self.cache.clear()

    def cache_size(self) -> int:
        """Return the number of cached results"""
        return len(self.cache)
