#!/usr/bin/env python3
"""
Entry point: run the full black-hole attack pipeline and evaluate.

Usage:
    python run.py
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
from evaluation.attack_evaluation import evaluate
from evaluation.recall_evaluation import evaluate_recall

# ═══════════════════════════════════════════════════════════════════════════════
#  Hyperparameters — modify here to experiment
# ═══════════════════════════════════════════════════════════════════════════════

MODEL = "gte"
SRC_DATASET = "nq"

# Attack mode:
#   "default"  — train and attack the same dataset (victim = src)
#   "transfer" — train on src, inject into a different dataset
MODE = "default"
VICTIM_DATASET = "hotpotqa"  # only used when MODE == "transfer"

PREPROCESS_MODE = "default" # "default | query_trans"

CLUSTER_METHOD = "faiss_gpu"  # "kmeans" | "minibatch_kmeans | adaptive | faiss_gpu"
N_CLUSTERS: int | None = None  # None = auto: len(corpus)/1000 of target dataset
BATCH_SIZE = 30000

NUM_COPIES = 10
EPSILON = 0.001
SEED = 42

INDEX_TYPE = "FlatIP"  # "FlatIP" | "IVF" | "HNSW" | "IVFPQ" (used for the poisoned index saved to disk)
INDEX_KWARGS: dict = {}  # e.g. {"nlist": 4096} for IVF, {"hnsw_M": 32} for HNSW

EVAL_INDEX_TYPES = ["FlatIP", "IVF", "HNSW", "IVFPQ"]  # evaluate ALL these index types

SAMPLE_QUERIES: int | None = 3000  # None = all; set to an integer to subsample
EVAL_K = 10

# ═══════════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent / "data"
VECTOR_DIR = ROOT / "vector"
DATASET_DIR = ROOT / "datasets"
OUTPUT_DIR: Path | None = None  # set in main() based on model+dataset to avoid parallel conflicts
RESULT_DIR = ROOT / "result"

RESULT_SUBDIR = ""  # subdirectory under RESULT_DIR, e.g. "main" or "ablation"


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

    # Auto-compute n_clusters if not explicitly set
    n_clusters = N_CLUSTERS
    if n_clusters is None:
        target = victim if MODE == "transfer" and victim is not None else source
        n_clusters = max(1, len(target.corpus_texts) // 1000)
        print(f"  n_clusters auto: {n_clusters}  (corpus={len(target.corpus_texts)} // 1000)")

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
        n_clusters=n_clusters,
        batch_size=BATCH_SIZE,
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(OUTPUT_DIR))

    return poisoned


def run_evaluation():
    """Load saved poisoned data and evaluate attack effectiveness + recall across all ANN index types."""
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET

    print()
    print("=" * 60)
    print("  STEP 4: Load poisoned data for evaluation")
    print("=" * 60)
    dm = DataManager(MODEL, eval_dataset, vector_dir=str(OUTPUT_DIR), dataset_dir=str(DATASET_DIR))
    dm.load_corpus()
    dm.load_queries()
    try:
        dm.load_index()
    except FileNotFoundError:
        print("  (no saved index found; will build each type on the fly)")
    print(dm.summarize())

    print()
    print("=" * 60)
    print("  STEP 5: Attack effectiveness evaluation (all index types)")
    print("=" * 60)
    attack_metrics = evaluate(dm, k=EVAL_K, sample=SAMPLE_QUERIES, index_types=EVAL_INDEX_TYPES)

    print()
    print("=" * 60)
    print("  STEP 6: Recall evaluation (clean vs poisoned, all index types)")
    print("=" * 60)
    recall_metrics = evaluate_recall(dm, k=EVAL_K, sample=SAMPLE_QUERIES, index_types=EVAL_INDEX_TYPES)

    return attack_metrics, recall_metrics


def save_results(attack_metrics: dict, recall_metrics: dict) -> None:
    """Persist attack and recall evaluation metrics to JSON."""
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
            "eval_index_types": EVAL_INDEX_TYPES,
            "sample_queries": SAMPLE_QUERIES,
        },
        "attack": {
            idx_type: {
                f"MO@{EVAL_K}": m.mo_at_k,
                f"MO@{EVAL_K}_std": m.mo_at_k_std,
                "ASR": m.asr,
                "FPR_mean": m.fpr_mean,
                "FPR_std": m.fpr_std,
                "k": m.k,
                "num_queries": m.num_queries,
            }
            for idx_type, m in attack_metrics.items()
        },
        "recall": {
            idx_type: {
                f"Recall@{EVAL_K}_clean": r.clean.recall_at_k,
                f"Recall@{EVAL_K}_clean_std": r.clean.recall_at_k_std,
                f"Recall@{EVAL_K}_poisoned": r.poisoned.recall_at_k,
                f"Recall@{EVAL_K}_poisoned_std": r.poisoned.recall_at_k_std,
                "delta": r.delta,
                "k": r.clean.k,
                "num_queries": r.clean.num_queries,
            }
            for idx_type, r in recall_metrics.items()
        },
    }
    result_dir = RESULT_DIR / RESULT_SUBDIR if RESULT_SUBDIR else RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{MODEL}_{eval_dataset}.json"
    result_path = result_dir / filename
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {result_path}")


def print_summary(attack_metrics: dict, recall_metrics: dict) -> None:
    """Print final evaluation summary for attack and recall."""
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Model:          {MODEL}")
    print(f"  Mode:           {MODE}")
    print(f"  Src dataset:    {SRC_DATASET}")
    print(f"  Victim dataset: {eval_dataset}")
    print(f"  Clusters:       {N_CLUSTERS}")
    print(f"  Copies:         {NUM_COPIES}")
    print(f"  Epsilon:        {EPSILON}")
    print(f"  Seed:           {SEED}")

    # -- Attack metrics --
    print()
    print(f"  --- Attack Effectiveness ---")
    header = f"  {'Index':<8s}  {'MO@{EVAL_K}':>10s}  {'ASR':>8s}  {'FPR':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx_type, m in attack_metrics.items():
        print(f"  {idx_type:<8s}  {m.mo_at_k:>10.4f}  {m.asr:>8.4f}  {m.fpr_mean:>8.2f}")

    # -- Recall metrics --
    print()
    print(f"  --- Recall@{EVAL_K} (clean vs poisoned) ---")
    header = f"  {'Index':<8s}  {'Clean':>10s}  {'Poisoned':>10s}  {'Delta':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx_type, r in recall_metrics.items():
        print(f"  {idx_type:<8s}  {r.clean.recall_at_k:>10.4f}  {r.poisoned.recall_at_k:>10.4f}  {r.delta:>+8.4f}")

    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Black-hole attack pipeline")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--src", "--src-dataset", dest="src_dataset", default=SRC_DATASET)
    parser.add_argument("--mode", default=MODE, choices=["default", "transfer"])
    parser.add_argument("--victim", "--victim-dataset", dest="victim_dataset", default=VICTIM_DATASET)
    parser.add_argument("--preprocess", dest="preprocess_mode", default=PREPROCESS_MODE)
    parser.add_argument("--cluster", dest="cluster_method", default=CLUSTER_METHOD)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-copies", type=int, default=NUM_COPIES)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--index-type", default=INDEX_TYPE)
    parser.add_argument("--sample-queries", type=int, default=SAMPLE_QUERIES,
                        help="Number of queries to sample for evaluation (default: all)")
    parser.add_argument("--eval-k", type=int, default=EVAL_K)
    parser.add_argument("--eval-index-types", nargs="+", default=EVAL_INDEX_TYPES,
                        choices=["FlatIP", "IVF", "HNSW", "IVFPQ"],
                        help="ANN index types to evaluate (default: all)")
    parser.add_argument("--result-subdir", default=RESULT_SUBDIR,
                        help="Subdirectory under data/result/ for organizing experiment runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL = args.model
    SRC_DATASET = args.src_dataset
    MODE = args.mode
    VICTIM_DATASET = args.victim_dataset
    PREPROCESS_MODE = args.preprocess_mode
    CLUSTER_METHOD = args.cluster_method
    N_CLUSTERS = args.n_clusters
    BATCH_SIZE = args.batch_size
    NUM_COPIES = args.num_copies
    EPSILON = args.epsilon
    SEED = args.seed
    INDEX_TYPE = args.index_type
    EVAL_INDEX_TYPES = args.eval_index_types
    SAMPLE_QUERIES = args.sample_queries
    EVAL_K = args.eval_k
    RESULT_SUBDIR = args.result_subdir
    OUTPUT_DIR = ROOT / "poisoned" / f"{MODEL}_{SRC_DATASET}"

    _ = run_pipeline()
    attack_metrics, recall_metrics = run_evaluation()
    save_results(attack_metrics, recall_metrics)
    print_summary(attack_metrics, recall_metrics)
