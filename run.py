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
SRC_DATASET = "nq"

# Attack mode:
#   "default"  — train and attack the same dataset (victim = src)
#   "transfer" — train on src, inject into a different dataset
MODE = "default"
VICTIM_DATASET = "hotpotqa"  # only used when MODE == "transfer"

PREPROCESS_MODE = "default"

CLUSTER_METHOD = "minibatch_kmeans"  # "kmeans" | "minibatch_kmeans"
N_CLUSTERS = 3000
BATCH_SIZE = 4096

NUM_COPIES = 10
EPSILON = 0.001
SEED = 42

INDEX_TYPE = "FlatIP"  # "FlatIP" | "IVF" | "HNSW"
INDEX_KWARGS: dict = {}  # e.g. {"nlist": 4096} for IVF, {"hnsw_M": 32} for HNSW

EVAL_K = 10

# ═══════════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent / "data"
VECTOR_DIR = ROOT / "vector"
DATASET_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "poisoned"
RESULT_FILE = ROOT / "results.json"


def _make_dm(dataset: str) -> DataManager:
    return DataManager(MODEL, dataset, vector_dir=str(VECTOR_DIR), dataset_dir=str(DATASET_DIR))


def run_pipeline() -> DataManager:
    """Load source, optionally load victim (transfer mode), run attack, save, return result."""
    # Step 1: Load source
    print("=" * 60)
    print("  STEP 1: Load source data")
    print("=" * 60)
    source = _make_dm(SRC_DATASET)
    source.load_all()
    print(source.summarize())

    # Load victim if transfer mode
    if MODE == "transfer":
        print()
        print("=" * 60)
        print("  STEP 1b: Load victim data (transfer mode)")
        print("=" * 60)
        victim = _make_dm(VICTIM_DATASET)
        victim.load_all()
        print(victim.summarize())
    else:
        victim = None

    # Step 2: Run attack pipeline
    print()
    print("=" * 60)
    print("  STEP 2: Run attack pipeline")
    print("=" * 60)
    pipeline = BlackHolePipeline(
        source,
        victim=victim,
        preprocess_mode=PREPROCESS_MODE,
        cluster_method=CLUSTER_METHOD,
        n_clusters=N_CLUSTERS,
        num_copies=NUM_COPIES,
        epsilon=EPSILON,
        seed=SEED,
        index_type=INDEX_TYPE,
        **INDEX_KWARGS,
    )
    poisoned = pipeline.run()

    # Step 3: Save
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
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET

    print()
    print("=" * 60)
    print("  STEP 4: Load poisoned index for evaluation")
    print("=" * 60)
    dm = DataManager(MODEL, eval_dataset, vector_dir=str(OUTPUT_DIR), dataset_dir=str(DATASET_DIR))
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
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
    payload = {
        "config": {
            "model": MODEL,
            "mode": MODE,
            "src_dataset": SRC_DATASET,
            "victim_dataset": eval_dataset,
            "preprocess_mode": PREPROCESS_MODE,
            "cluster_method": CLUSTER_METHOD,
            "n_clusters": N_CLUSTERS,
            "batch_size": BATCH_SIZE,
            "num_copies": NUM_COPIES,
            "epsilon": EPSILON,
            "seed": SEED,
            "index_type": INDEX_TYPE,
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
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
    print()
    print("=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Model:      {MODEL}")
    print(f"  Mode:       {MODE}")
    print(f"  Src dataset:   {SRC_DATASET}")
    print(f"  Victim dataset: {eval_dataset}")
    print(f"  Clusters:   {N_CLUSTERS}")
    print(f"  Copies:     {NUM_COPIES}")
    print(f"  Epsilon:    {EPSILON}")
    print(f"  Index:      {INDEX_TYPE}")
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
