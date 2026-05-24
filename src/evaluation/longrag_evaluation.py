"""
Evaluate black-hole attack under LongRAG retrieval (with / without rerank).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from process_data.data_manager import DataManager
from longrag.retriever import LongRAGRetriever  # noqa: E402
from evaluation.attack_evaluation import _compute_metrics, _count_adversarial, EvalMetrics


@dataclass
class LongRAGEvalComparison:
    """Attack metrics for dense-only vs rerank retrieval."""

    without_rerank: EvalMetrics
    with_rerank: EvalMetrics
    unit_k: int
    k: int


def evaluate_longrag_attack(
    poisoned_dm: DataManager,
    *,
    k: int = 10,
    unit_k: int = 20,
    sample: Optional[int] = None,
    max_chars: int = 16_000,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    skip_rerank: bool = False,
) -> LongRAGEvalComparison:
    """
    Run LongRAG retrieval on a poisoned index and measure attack effectiveness.

    Compares first-stage long-unit retrieval + vector rescore vs the same
    pipeline with cross-encoder reranking.
    """
    if poisoned_dm.query_vecs is None:
        raise RuntimeError("DataManager has no query vectors loaded")

    num_original = len(poisoned_dm.corpus_texts) - _count_adversarial(poisoned_dm)
    print(f"LongRAG attack evaluation")
    print(f"  original docs:     {num_original}")
    print(f"  adversarial docs:  {_count_adversarial(poisoned_dm)}")
    print(f"  unit_k={unit_k}  final_k={k}")
    print()

    retriever = LongRAGRetriever(poisoned_dm, max_chars=max_chars)
    retriever.build_units()

    print("--- LongRAG without rerank ---")
    res_dense = retriever.search(k=k, unit_k=unit_k, use_rerank=False, sample=sample)
    metrics_dense = _compute_metrics(res_dense.indices, num_original, k)
    print(f"  MO@{k}: {metrics_dense.mo_at_k:.4f}  ASR: {metrics_dense.asr:.4f}  FPR: {metrics_dense.fpr_mean:.2f}")
    print()

    if skip_rerank:
        return LongRAGEvalComparison(
            without_rerank=metrics_dense,
            with_rerank=metrics_dense,
            unit_k=unit_k,
            k=k,
        )

    from longrag.rerank import CrossEncoderReranker  # noqa: E402

    retriever.reranker = CrossEncoderReranker(model_name=rerank_model)
    print("--- LongRAG with cross-encoder rerank ---")
    res_rerank = retriever.search(k=k, unit_k=unit_k, use_rerank=True, sample=sample)
    metrics_rerank = _compute_metrics(res_rerank.indices, num_original, k)
    print(f"  MO@{k}: {metrics_rerank.mo_at_k:.4f}  ASR: {metrics_rerank.asr:.4f}  FPR: {metrics_rerank.fpr_mean:.2f}")
    print()

    return LongRAGEvalComparison(
        without_rerank=metrics_dense,
        with_rerank=metrics_rerank,
        unit_k=unit_k,
        k=k,
    )
