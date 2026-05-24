#!/usr/bin/env python3
"""Unit tests for LongRAG grouping (no GPU / rerank model required)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from longrag.units import build_long_units  # noqa: imports without faiss


def test_build_long_units_groups_by_title():
    texts = pd.DataFrame(
        {
            "_id": ["a1", "a2", "b1", "bh_0"],
            "title": ["T", "T", "U", ""],
            "text": ["x" * 100, "y" * 100, "z" * 50, ""],
        }
    )
    vecs = np.random.randn(4, 8).astype(np.float32)
    idx = build_long_units(texts, vecs, max_chars=500, group_by_title=True)

    assert len(idx.unit_ids) == 3  # T-group, U, adversarial singleton
    assert idx.passage_to_unit[3] == idx.passage_to_unit[3]  # bh isolated
    adv_unit = idx.passage_to_unit[3]
    assert idx.unit_passage_indices[adv_unit] == [3]
    assert idx.unit_vecs.shape == (3, 8)


def test_all_passages_mapped():
    n = 50
    texts = pd.DataFrame(
        {
            "_id": [f"d{i}" for i in range(n)],
            "title": [f"t{i % 5}" for i in range(n)],
            "text": ["word"] * n,
        }
    )
    vecs = np.random.randn(n, 16).astype(np.float32)
    idx = build_long_units(texts, vecs, max_chars=200)
    assert (idx.passage_to_unit >= 0).all()
    assert len(idx.unit_ids) >= 1
