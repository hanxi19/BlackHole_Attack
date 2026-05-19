"""
Preprocessing modes for the black-hole attack pipeline.
Each preprocessor receives corpus vectors and returns (possibly modified) vectors.
"""

from __future__ import annotations

import numpy as np


def preprocess_default(vecs: np.ndarray) -> np.ndarray:
    """Default: no preprocessing, return vectors as-is."""
    return vecs


PREPROCESSORS = {
    "default": preprocess_default,
}


def apply_preprocess(vecs: np.ndarray, mode: str = "default") -> np.ndarray:
    if mode not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocess mode: {mode}. Available: {list(PREPROCESSORS)}")
    return PREPROCESSORS[mode](vecs)
