#!/usr/bin/env python3
"""
Entry point: run the full black-hole attack pipeline and evaluate.

Usage:
    python run.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from process_data.data_manager import DataManager
from attack.pipeline import BlackHolePipeline
from evaluation.attack_evaluation import evaluate

# ═══════════════════════════════════════════════════════════════════════════════
#  Hyperparameters — modify here to experiment
# ═══════════════════════════════════════════════════════════════════════════════

MODEL = "contriever"
DATASET = "nq"

PREPROCESS_MODE = "default"

CLUSTER_METHOD = "minibatch_kmeans"  # "kmeans" | "minibatch_kmeans"
N_CLUSTERS = 3000
BATCH_SIZE = 4096

NUM_COPIES = 10
EPSILON = 0.001
SEED = 42

EVAL_K = 10

# ═══════════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent / "data"
VECTOR_DIR = ROOT / "vector"
DATASET_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "poisoned"
RESULT_FILE = ROOT / "results.json"


def run_pipeline() -> DataManager:
    """Load source, run attack, save poisoned data, return poisoned DataManager."""
    print("=" * 60)
    print("  STEP 1: Load source data")
    print("=" * 60)
    source = DataManager(MODEL, DATASET, vector_dir=str(VECTOR_DIR), dataset_dir=str(DATASET_DIR))
    source.load_all()
    print(source.summarize())

    print()
    print("=" * 60)
    print("  STEP 2: Run attack pipeline")
    print("=" * 60)
    pipeline = BlackHolePipeline(
        source,
        preprocess_mode=PREPROCESS_MODE,
        cluster_method=CLUSTER_METHOD,
        n_clusters=N_CLUSTERS,
        num_copies=NUM_COPIES,
        epsilon=EPSILON,
        seed=SEED,
    )
    poisoned = pipeline.run()

    print()
    print("=" * 60)
    print("  STEP 3: Save poisoned data")
    print("=" * 60)
    if OUTPUT_DIR.is_dir():
        shutil.rmtree(OUTPUT_DIR)
    pipeline.save(str(OUTPUT_DIR))

    return poisoned


def run_evaluation():
    """Load saved poisoned index and evaluate."""
    print()
    print("=" * 60)
    print("  STEP 4: Load poisoned index for evaluation")
    print("=" * 60)
    dm = DataManager(MODEL, DATASET, vector_dir=str(OUTPUT_DIR), dataset_dir=str(DATASET_DIR))
    dm.load_corpus()
    dm.load_queries()
    dm.load_index()
    print(dm.summarize())

    print()
    print("=" * 60)
    print("  STEP 5: Evaluate attack effectiveness")
    print("=" * 60)
    metrics = evaluate(dm, k=EVAL_K)

    return metrics


def save_results(metrics) -> None:
    """Persist evaluation metrics to JSON."""
    payload = {
        "config": {
            "model": MODEL,
            "dataset": DATASET,
            "preprocess_mode": PREPROCESS_MODE,
            "cluster_method": CLUSTER_METHOD,
            "n_clusters": N_CLUSTERS,
            "batch_size": BATCH_SIZE,
            "num_copies": NUM_COPIES,
            "epsilon": EPSILON,
            "seed": SEED,
        },
        "metrics": {
            f"MO@{EVAL_K}": metrics.mo_at_k,
            f"MO@{EVAL_K}_std": metrics.mo_at_k_std,
            "ASR": metrics.asr,
            "FPR_mean": metrics.fpr_mean,
            "FPR_std": metrics.fpr_std,
            "k": metrics.k,
            "num_queries": metrics.num_queries,
        },
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {RESULT_FILE}")


def print_summary(metrics) -> None:
    """Print final evaluation summary."""
    print()
    print("=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Model:      {MODEL}")
    print(f"  Dataset:    {DATASET}")
    print(f"  Clusters:   {N_CLUSTERS}")
    print(f"  Copies:     {NUM_COPIES}")
    print(f"  Epsilon:    {EPSILON}")
    print(f"  Seed:       {SEED}")
    print(f"  ———————————————————————————")
    print(f"  MO@{EVAL_K}:    {metrics.mo_at_k:.4f} ± {metrics.mo_at_k_std:.4f}")
    print(f"  ASR:       {metrics.asr:.4f} ({metrics.asr*100:.2f}%)")
    print(f"  FPR:       {metrics.fpr_mean:.2f} ± {metrics.fpr_std:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    _ = run_pipeline()
    eval_metrics = run_evaluation()
    save_results(eval_metrics)
    print_summary(eval_metrics)
