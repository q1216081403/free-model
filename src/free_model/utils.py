"""Small helper utilities for the free-model project."""

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed NumPy's global RNG for reproducible runs."""
    np.random.seed(seed)


def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean / unit-variance normalization over ``axis``."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + eps)
