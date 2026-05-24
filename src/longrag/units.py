"""
Long retrieval units for BEIR-style corpora (LongRAG-style grouping).

Groups short passages into longer units (by title and token budget), then
mean-pools member embeddings for coarse retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class LongUnitIndex:
    """Index over long retrieval units mapped back to passage row indices."""

    unit_ids: list[str]
    unit_passage_indices: list[list[int]]  # passage indices per unit
    unit_vecs: np.ndarray  # (n_units, dim) float32, L2-normalized
    passage_to_unit: np.ndarray  # (n_passages,) int unit index per passage


def _passage_char_len(row: pd.Series) -> int:
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    combined = f"{title} {text}".strip() if title else text
    return len(combined)


def build_long_units(
    corpus_texts: pd.DataFrame,
    corpus_vecs: np.ndarray,
    *,
    max_chars: int = 16_000,
    group_by_title: bool = True,
    adversarial_prefix: str = "bh_",
) -> LongUnitIndex:
    """Group passages into long units and compute unit embeddings (mean pool).

    Adversarial rows (``_id`` starting with *adversarial_prefix*) are always
    singleton units so poison vectors are not merged into benign groups.

    Args:
        corpus_texts: DataFrame with columns ``_id``, ``text``, ``title``.
        corpus_vecs: (N, D) passage embeddings.
        max_chars: Approximate max characters per long unit (~4 chars/token).
        group_by_title: Merge passages sharing the same non-empty title.
        adversarial_prefix: Prefix marking injected adversarial documents.

    Returns:
        LongUnitIndex ready for first-stage ANN search.
    """
    n = len(corpus_texts)
    if corpus_vecs.shape[0] != n:
        raise ValueError(f"corpus_vecs rows {corpus_vecs.shape[0]} != texts {n}")

    ids = corpus_texts["_id"].astype(str).tolist()
    is_adv = [doc_id.startswith(adversarial_prefix) for doc_id in ids]
    char_lens = corpus_texts.apply(_passage_char_len, axis=1).tolist()

    # Build ordered list of (passage_idx, group_key) for clustering
    order: list[int] = list(range(n))
    if group_by_title:
        titles = corpus_texts["title"].fillna("").astype(str).str.strip()

        def sort_key(i: int) -> tuple:
            if is_adv[i]:
                return (2, ids[i])  # adversarial: own bucket, stable by id
            t = titles.iloc[i]
            if t:
                return (0, t, i)
            return (1, i)

        order = sorted(order, key=sort_key)

    unit_passage_indices: list[list[int]] = []
    unit_ids: list[str] = []
    passage_to_unit = np.full(n, -1, dtype=np.int32)

    def flush_unit(members: list[int]) -> None:
        if not members:
            return
        uid = f"unit_{len(unit_ids):08d}"
        unit_ids.append(uid)
        unit_passage_indices.append(members)
        uidx = len(unit_ids) - 1
        for pi in members:
            passage_to_unit[pi] = uidx

    current: list[int] = []
    current_chars = 0
    current_key: str | None = None

    for pi in order:
        if is_adv[pi]:
            flush_unit(current)
            current, current_chars, current_key = [], 0, None
            flush_unit([pi])
            continue

        title = str(corpus_texts.iloc[pi].get("title", "") or "").strip()
        key = title if (group_by_title and title) else f"__seq_{pi}"

        if current and key != current_key:
            flush_unit(current)
            current, current_chars, current_key = [], 0, None

        plen = char_lens[pi]
        if current and current_chars + plen > max_chars:
            flush_unit(current)
            current, current_chars, current_key = [], 0, None

        if not current:
            current_key = key
        current.append(pi)
        current_chars += plen

    flush_unit(current)

    if (passage_to_unit < 0).any():
        missing = int((passage_to_unit < 0).sum())
        raise RuntimeError(f"{missing} passages were not assigned to any long unit")

    # Mean-pool + L2-normalize unit vectors
    dim = corpus_vecs.shape[1]
    unit_vecs = np.zeros((len(unit_ids), dim), dtype=np.float32)
    for uidx, members in enumerate(unit_passage_indices):
        pooled = corpus_vecs[members].mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        unit_vecs[uidx] = pooled.astype(np.float32)

    return LongUnitIndex(
        unit_ids=unit_ids,
        unit_passage_indices=unit_passage_indices,
        unit_vecs=unit_vecs,
        passage_to_unit=passage_to_unit,
    )
