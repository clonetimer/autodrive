"""Optional Gymnasium boundary, never disguises the offline protocol as Gymnasium.

The native trainer needs only shapes and contains/sample. When Gymnasium is
installed the exported classes are its real classes; otherwise this module is
an explicitly named, small offline protocol. No sys.modules monkey-patching.
"""
from __future__ import annotations
from typing import Any
import numpy as np

try:
    from gymnasium import Env
    from gymnasium.spaces import Box, Dict
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

    class Env:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None or not hasattr(self, 'np_random'):
                self.np_random = np.random.default_rng(seed)

        @property
        def unwrapped(self):
            return self

        def close(self) -> None:
            pass

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy() if shape else np.asarray(low, dtype=dtype)
            self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy() if shape else np.asarray(high, dtype=dtype)
            self.shape = self.low.shape
            self._rng = np.random.default_rng()

        def seed(self, seed=None):
            self._rng = np.random.default_rng(seed)
            return [seed]

        def sample(self):
            return self._rng.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, value) -> bool:
            arr = np.asarray(value)
            return bool(arr.shape == self.shape and arr.dtype == self.dtype and np.isfinite(arr).all()
                        and (arr >= self.low).all() and (arr <= self.high).all())

    class Dict:
        def __init__(self, spaces):
            self.spaces = spaces

        def __getitem__(self, key):
            return self.spaces[key]

        def contains(self, value) -> bool:
            return isinstance(value, dict) and set(value) == set(self.spaces) and all(
                space.contains(value[key]) for key, space in self.spaces.items())

        def sample(self):
            return {key: space.sample() for key, space in self.spaces.items()}
