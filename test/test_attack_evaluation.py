#!/usr/bin/env python3
"""
Test attack evaluation: load poisoned index, run all queries, compute MO@10, ASR, FPR.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from process_data.data_manager import DataManager
from evaluation.attack_evaluation import evaluate, EvalMetrics

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
POISONED_DIR = os.path.join(ROOT, "poisoned")
DATASET_DIR = os.path.join(ROOT, "datasets")


def _load_poisoned() -> DataManager:
    dm = DataManager("contriever", "nq", vector_dir=POISONED_DIR, dataset_dir=DATASET_DIR)
    dm.load_corpus()
    dm.load_queries()
    dm.load_index()
    return dm


def test_evaluate_default():
    """MO@10, ASR, FPR on the default poisoned index."""
    print("=" * 60)
    print("TEST: evaluate default")
    dm = _load_poisoned()
    print(dm.summarize())
    print()

    metrics = evaluate(dm, k=10)

    assert metrics.k == 10
    assert metrics.num_queries == dm.query_vecs.shape[0]
    assert 0.0 <= metrics.mo_at_k <= 1.0
    assert 0.0 <= metrics.asr <= 1.0
    assert 1.0 <= metrics.fpr_mean <= 10.0
    assert len(metrics.mo_per_query) == metrics.num_queries
    assert len(metrics.fpr_per_query) == metrics.num_queries

    print()
    print("Metrics:")
    print(f"  MO@{metrics.k}:  {metrics.mo_at_k:.4f} ± {metrics.mo_at_k_std:.4f}")
    print(f"  ASR:     {metrics.asr:.4f} ({metrics.asr*100:.2f}%)")
    print(f"  FPR:     {metrics.fpr_mean:.2f} ± {metrics.fpr_std:.2f}")
    print("  PASSED\n")


def test_evaluate_different_k():
    """Verify k parameter changes the output shape and bounds."""
    print("=" * 60)
    print("TEST: evaluate k=5 and k=20")
    dm = _load_poisoned()

    for k in [5, 20]:
        metrics = evaluate(dm, k=k)
        assert metrics.k == k
        assert len(metrics.mo_per_query) == metrics.num_queries
        assert len(metrics.fpr_per_query) == metrics.num_queries
        assert 0.0 <= metrics.mo_at_k <= 1.0
        assert 1.0 <= metrics.fpr_mean <= float(k)
        print(f"  k={k}  MO@{k}={metrics.mo_at_k:.4f}  ASR={metrics.asr:.4f}  FPR={metrics.fpr_mean:.2f}")
    print("  PASSED\n")


def test_evaluate_no_index():
    """evaluate raises RuntimeError when no index is built."""
    print("=" * 60)
    print("TEST: evaluate without index")
    dm = DataManager("contriever", "hotpotqa", vector_dir=POISONED_DIR, dataset_dir=DATASET_DIR)
    dm.load_corpus()
    dm.load_queries()
    try:
        evaluate(dm, k=10)
        assert False, "should have raised RuntimeError"
    except RuntimeError as e:
        assert "no built index" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


def test_evaluate_no_queries():
    """evaluate raises RuntimeError when no queries are loaded."""
    print("=" * 60)
    print("TEST: evaluate without queries")
    dm = DataManager("contriever", "hotpotqa", vector_dir=POISONED_DIR, dataset_dir=DATASET_DIR)
    dm.load_corpus()
    try:
        evaluate(dm, k=10)
        assert False, "should have raised RuntimeError"
    except RuntimeError as e:
        assert "no query vectors" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("Attack Evaluation Test Suite\n")
    test_evaluate_default()
    # test_evaluate_different_k()
    # test_evaluate_no_index()
    # test_evaluate_no_queries()
    print("=" * 60)
    print("ALL TESTS PASSED")
