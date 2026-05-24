#!/usr/bin/env bash
#
# run_multiquery_trans.sh
#
# Multi-query-transfer experiment:
#   For each victim dataset, train adversarial vectors on queries from ALL OTHER
#   datasets, then attack the victim.  This is a leave-one-dataset-out evaluation
#   of cross-dataset query-transfer attack effectiveness.
#
# Evaluates MO@10, ASR, FPR, and recall loss across all (model, dataset) combos
# and all ANN index types.
#
# Prerequisites:
#   1. Datasets downloaded  -> scripts/download_data.sh
#   2. Vectors encoded      -> scripts/encode_data.sh
#
# Results: data/result/multi_query_trans/<timestamp>_<model>_<dataset>.json
#
# Parallel mode:
#   Set NUM_GPUS to override auto-detection (default: all visible GPUs).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON="${PYTHON:-python3}"

# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment grid — modify here
# ═══════════════════════════════════════════════════════════════════════════════

MODELS=(contriever)
# DATASETS=(msmarco)
DATASETS=(msmarco nq hotpotqa trec-covid nfcorpus fiqa arguana
          touche2020 quora dbpedia scidocs fever
          climate-fever scifact)
INDEX_TYPES=(FlatIP)

# ═══════════════════════════════════════════════════════════════════════════════
#  Hyperparameters — all explicitly set, not relying on run.py defaults
# ═══════════════════════════════════════════════════════════════════════════════

MODE="transfer"                  # "transfer" — attack victim with multi-source queries
PREPROCESS_MODE="multi_query_transfer"
CLUSTER_METHOD="faiss_gpu"       # "kmeans" | "minibatch_kmeans" | "adaptive" | "faiss_gpu"
N_CLUSTERS=""                    # empty = auto (corpus//1000 of victim); set to an integer to override
BATCH_SIZE=30000
NUM_COPIES=10
EPSILON=0.001
SEED=42
INDEX_TYPE="FlatIP"              # index type saved to disk (for the poisoned index)
SAMPLE_QUERIES=3000              # number of queries to sample for evaluation (empty = all)
EVAL_K=10

# -- GPU detection -----------------------------------------------------------

if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        NUM_GPUS=1
    fi
fi

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))

echo "=============================================="
echo "  Black-Hole Attack - Multi-Query Transfer"
echo "  Models:   ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Indexes:  ${INDEX_TYPES[*]}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Total runs: ${TOTAL}"
echo ""
echo "  Mode:       ${MODE}"
echo "  Preprocess: ${PREPROCESS_MODE}"
echo "  Cluster:    ${CLUSTER_METHOD}"
echo "  n_clusters: ${N_CLUSTERS:-auto}"
echo "  Copies:     ${NUM_COPIES}"
echo "  Epsilon:    ${EPSILON}"
echo "  Seed:       ${SEED}"
echo "=============================================="
echo ""

# -- Build common args (empty N_CLUSTERS is omitted) --------------------------

COMMON_ARGS=(
    --mode "${MODE}"
    --preprocess "${PREPROCESS_MODE}"
    --cluster "${CLUSTER_METHOD}"
    --batch-size "${BATCH_SIZE}"
    --num-copies "${NUM_COPIES}"
    --epsilon "${EPSILON}"
    --seed "${SEED}"
    --index-type "${INDEX_TYPE}"
    --sample-queries "${SAMPLE_QUERIES}"
    --eval-k "${EVAL_K}"
    --eval-index-types "${INDEX_TYPES[@]}"
    --result-subdir multi_query_trans
)
if [ -n "${N_CLUSTERS}" ]; then
    COMMON_ARGS+=(--n-clusters "${N_CLUSTERS}")
fi

# -- Distribute tasks across GPUs --------------------------------------------

TASK_DIR=$(mktemp -d)
declare -a TASK_FILES
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    TASK_FILES[$gpu]="${TASK_DIR}/gpu_${gpu}.tasks"
done

task_count=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        gpu=$((task_count % NUM_GPUS))
        echo "${model} ${dataset}" >> "${TASK_FILES[$gpu]}"
        task_count=$((task_count + 1))
    done
done

echo "Task distribution:"
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    count=$(wc -l < "${TASK_FILES[$gpu]}" 2>/dev/null || echo 0)
    echo "  GPU ${gpu}: ${count} tasks"
done
echo ""

# -- Worker function ---------------------------------------------------------

run_worker() {
    local gpu_id=$1
    local task_file=$2
    local worker_failed=0

    export CUDA_VISIBLE_DEVICES=$gpu_id

    while read -r model dataset; do
        echo "============================================================"
        echo "[GPU ${gpu_id}] model=${model}  victim=${dataset}"
        echo "============================================================"
        echo ""

        if "${PYTHON}" "${PROJECT_DIR}/run.py" \
            --model "${model}" \
            --src "${dataset}" \
            --victim "${dataset}" \
            "${COMMON_ARGS[@]}"; then
            echo ""
            echo "  [GPU ${gpu_id}] OK: model=${model} victim=${dataset}"
        else
            echo ""
            echo "  [GPU ${gpu_id}] FAILED: model=${model} victim=${dataset}" >&2
            worker_failed=1
        fi
        echo ""
    done < "$task_file"

    return $worker_failed
}

# -- Launch workers ----------------------------------------------------------

declare -a PIDS
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    if [ -s "${TASK_FILES[$gpu]}" ]; then
        run_worker "$gpu" "${TASK_FILES[$gpu]}" &
        PIDS+=($!)
    fi
done

# -- Wait for all workers ----------------------------------------------------

failed=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || failed=1
done

rm -rf "$TASK_DIR"

# -- Summary ----------------------------------------------------------------

echo "============================================================"
echo "  Multi-query transfer experiment complete"
if [ "$failed" -eq 1 ]; then
    echo "  Some workers failed — check logs above for details"
    echo "  Results: ${PROJECT_DIR}/data/result/multi_query_trans/"
    exit 1
fi
echo "  All ${TOTAL} runs succeeded"
echo "  Results: ${PROJECT_DIR}/data/result/multi_query_trans/"
echo "============================================================"
