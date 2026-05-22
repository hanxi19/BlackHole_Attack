"""
Preprocessing modes for the black-hole attack pipeline.
Each preprocessor receives the source DataManager and returns vectors for clustering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from process_data.data_manager import DataManager


def preprocess_default(source: DataManager) -> np.ndarray:
    """Default: return corpus vectors as-is."""
    if source.corpus_vecs is None:
        raise RuntimeError("Source DataManager has no corpus vectors loaded")
    return source.corpus_vecs


def preprocess_query_trans(source: DataManager) -> np.ndarray:
    """Query-transfer: return query vectors instead of corpus vectors.

    The attack clusters and perturbs query vectors, producing adversarial
    documents that attract real user queries rather than corpus centroids.
    """
    if source.query_vecs is None:
        raise RuntimeError("Source DataManager has no query vectors loaded (query_trans mode requires queries)")
    return source.query_vecs


PREPROCESSORS = {
    "default": preprocess_default,
    "query_trans": preprocess_query_trans,
}


def apply_preprocess(source: DataManager, mode: str = "default") -> np.ndarray:
    if mode not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocess mode: {mode}. Available: {list(PREPROCESSORS)}")
    return PREPROCESSORS[mode](source)
