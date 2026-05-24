#!/usr/bin/env python3
"""
LongRAG + black-hole attack pipeline.

1. Run the existing black-hole poison pipeline (src/attack).
2. Evaluate attack effectiveness under LongRAG retrieval.
3. Compare vector rescore vs cross-encoder rerank.

Usage:
    python run_longrag.py
    python run_longrag.py --skip-rerank
    python run_longrag.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from process_data.data_manager import DataManager
from attack.pipeline import BlackHolePipeline
from evaluation.longrag_evaluation import evaluate_longrag_attack

# Reuse defaults from run.py
from run import (
    MODEL,
    SRC_DATASET,
    MODE,
    VICTIM_DATASET,
    PREPROCESS_MODE,
    CLUSTER_METHOD,
    N_CLUSTERS,
    BATCH_SIZE,
    NUM_COPIES,
    EPSILON,
    SEED,
    INDEX_TYPE,
    INDEX_KWARGS,
    SAMPLE_QUERIES,
    EVAL_K,
    ROOT,
    VECTOR_DIR,
    DATASET_DIR,
    RESULT_DIR,
    _make_dm,
)

# LongRAG-specific
LONG_UNIT_CHARS = 16_000
LONG_UNIT_K = 20
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RESULT_SUBDIR = "longrag"


def run_attack_pipeline() -> DataManager:
    """Reuse BlackHolePipeline from run.py."""
    print("=" * 60)
    print("  STEP 1: Load source data")
    print("=" * 60)
    source = _make_dm(SRC_DATASET)
    source.load_all()
    print(source.summarize())

    victim = None
    if MODE == "transfer":
        print()
        print("=" * 60)
        print("  STEP 1b: Load victim data (transfer mode)")
        print("=" * 60)
        victim = _make_dm(VICTIM_DATASET)
        victim.load_all()
        print(victim.summarize())

    n_clusters = N_CLUSTERS
    if n_clusters is None:
        target = victim if MODE == "transfer" and victim is not None else source
        n_clusters = max(1, len(target.corpus_texts) // 1000)
        print(f"  n_clusters auto: {n_clusters}")

    print()
    print("=" * 60)
    print("  STEP 2: Black-hole poison pipeline")
    print("=" * 60)
    pipeline = BlackHolePipeline(
        source,
        victim=victim,
        preprocess_mode=PREPROCESS_MODE,
        cluster_method=CLUSTER_METHOD,
        n_clusters=n_clusters,
        batch_size=BATCH_SIZE,
        num_copies=NUM_COPIES,
        epsilon=EPSILON,
        seed=SEED,
        index_type=INDEX_TYPE,
        **INDEX_KWARGS,
    )
    poisoned = pipeline.run()

    output_dir = ROOT / "poisoned" / f"{MODEL}_{SRC_DATASET}_longrag"
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(output_dir))
    return poisoned


def run_longrag_eval(poisoned: DataManager, *, skip_rerank: bool) -> dict:
    print()
    print("=" * 60)
    print("  STEP 3: LongRAG retrieval + attack evaluation")
    print("=" * 60)
    comparison = evaluate_longrag_attack(
        poisoned,
        k=EVAL_K,
        unit_k=LONG_UNIT_K,
        sample=SAMPLE_QUERIES,
        max_chars=LONG_UNIT_CHARS,
        rerank_model=RERANK_MODEL,
        skip_rerank=skip_rerank,
    )
    return {
        "without_rerank": comparison.without_rerank,
        "with_rerank": comparison.with_rerank,
        "unit_k": comparison.unit_k,
        "k": comparison.k,
    }


def save_longrag_results(metrics: dict) -> Path:
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
    payload = {
        "framework": "longrag",
        "config": {
            "model": MODEL,
            "mode": MODE,
            "src_dataset": SRC_DATASET,
            "victim_dataset": eval_dataset,
            "long_unit_chars": LONG_UNIT_CHARS,
            "unit_k": LONG_UNIT_K,
            "eval_k": EVAL_K,
            "rerank_model": RERANK_MODEL,
            "sample_queries": SAMPLE_QUERIES,
        },
        "attack": {
            "without_rerank": {
                f"MO@{EVAL_K}": metrics["without_rerank"].mo_at_k,
                "ASR": metrics["without_rerank"].asr,
                "FPR_mean": metrics["without_rerank"].fpr_mean,
            },
            "with_rerank": {
                f"MO@{EVAL_K}": metrics["with_rerank"].mo_at_k,
                "ASR": metrics["with_rerank"].asr,
                "FPR_mean": metrics["with_rerank"].fpr_mean,
            },
        },
    }
    result_dir = RESULT_DIR / RESULT_SUBDIR
    result_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = result_dir / f"{ts}_{MODEL}_{eval_dataset}_longrag.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {path}")
    return path


def print_summary(metrics: dict) -> None:
    print()
    print("=" * 70)
    print("  LongRAG ATTACK RESULTS")
    print("=" * 70)
    print(f"  {'Stage':<22s}  {'MO@' + str(EVAL_K):>10s}  {'ASR':>8s}  {'FPR':>8s}")
    print("  " + "-" * 52)
    for label, m in (
        ("Long unit + vector", metrics["without_rerank"]),
        ("Long unit + rerank", metrics["with_rerank"]),
    ):
        print(f"  {label:<22s}  {m.mo_at_k:>10.4f}  {m.asr:>8.4f}  {m.fpr_mean:>8.2f}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LongRAG + black-hole attack")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--src", dest="src_dataset", default=SRC_DATASET)
    p.add_argument("--mode", default=MODE, choices=["default", "transfer"])
    p.add_argument("--victim", dest="victim_dataset", default=VICTIM_DATASET)
    p.add_argument("--eval-only", action="store_true", help="Skip attack; load saved poisoned data")
    p.add_argument("--skip-rerank", action="store_true", help="Skip cross-encoder rerank stage")
    p.add_argument("--unit-k", type=int, default=LONG_UNIT_K)
    p.add_argument("--long-unit-chars", type=int, default=LONG_UNIT_CHARS)
    p.add_argument("--rerank-model", default=RERANK_MODEL)
    p.add_argument("--sample-queries", type=int, default=SAMPLE_QUERIES)
    p.add_argument("--eval-k", type=int, default=EVAL_K)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL = args.model
    SRC_DATASET = args.src_dataset
    MODE = args.mode
    VICTIM_DATASET = args.victim_dataset
    LONG_UNIT_K = args.unit_k
    LONG_UNIT_CHARS = args.long_unit_chars
    RERANK_MODEL = args.rerank_model
    SAMPLE_QUERIES = args.sample_queries
    EVAL_K = args.eval_k

    if args.eval_only:
        eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
        poison_dir = ROOT / "poisoned" / f"{MODEL}_{SRC_DATASET}_longrag"
        dm = DataManager(MODEL, eval_dataset, vector_dir=str(poison_dir), dataset_dir=str(DATASET_DIR))
        dm.load_all()
    else:
        dm = run_attack_pipeline()

    metrics = run_longrag_eval(dm, skip_rerank=args.skip_rerank)
    save_longrag_results(metrics)
    print_summary(metrics)
