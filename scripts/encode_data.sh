#!/usr/bin/env bash
set -euo pipefail

# ── Edit these to choose models and datasets ──
MODELS=(contriever)
DATASETS=(msmarco nq hotpotqa trec-covid nfcorpus fiqa arguana
          touche2020 quora dbpedia scidocs fever
          climate-fever scifact)
# ──────────────────────────────────────────────

# Number of GPUs to use (auto-detect if not set)
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        NUM_GPUS=1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/src/process_data"
DATA_DIR="${PROJECT_DIR}/data"
DATASET_DIR="${DATA_DIR}/datasets"
VECTOR_DIR="${DATA_DIR}/vector"

PYTHON="${PYTHON:-python3}"

echo "============================================"
echo "  encode_data.sh"
echo "  models:   ${MODELS[*]}"
echo "  datasets: ${DATASETS[*]}"
echo "  gpus:     ${NUM_GPUS}"
echo "============================================"
echo ""

# ── Collect all pending tasks ──
TASK_DIR=$(mktemp -d)
declare -a TASK_FILES
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    TASK_FILES[$gpu]="${TASK_DIR}/gpu_${gpu}.tasks"
done

task_count=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ ! -f "${VECTOR_DIR}/${model}_${dataset}.npy" ]; then
            gpu=$((task_count % NUM_GPUS))
            echo "corpus ${model} ${dataset}" >> "${TASK_FILES[$gpu]}"
            task_count=$((task_count + 1))
        fi
        if [ ! -f "${VECTOR_DIR}/${model}_${dataset}_queries.npy" ]; then
            gpu=$((task_count % NUM_GPUS))
            echo "queries ${model} ${dataset}" >> "${TASK_FILES[$gpu]}"
            task_count=$((task_count + 1))
        fi
    done
done

echo "Pending tasks: ${task_count}"
echo ""

if [ "$task_count" -eq 0 ]; then
    echo "All tasks already completed, nothing to do."
    rm -rf "$TASK_DIR"
    exit 0
fi

# ── Worker: runs tasks for one GPU sequentially ──
run_worker() {
    local gpu_id=$1
    local task_file=$2

    export CUDA_VISIBLE_DEVICES=$gpu_id

    while read -r mode model dataset; do
        if [ "$mode" = "corpus" ]; then
            echo "[GPU ${gpu_id}] corpus: model=${model} dataset=${dataset}"
            "${PYTHON}" "${SRC_DIR}/encode.py" \
                --model "${model}" \
                --dataset "${dataset}" \
                --dataset-dir "${DATASET_DIR}" \
                --output-dir "${VECTOR_DIR}"
        else
            echo "[GPU ${gpu_id}] queries: model=${model} dataset=${dataset}"
            "${PYTHON}" "${SRC_DIR}/encode.py" \
                --model "${model}" \
                --dataset "${dataset}" \
                --dataset-dir "${DATASET_DIR}" \
                --output-dir "${VECTOR_DIR}" \
                --queries
        fi
    done < "$task_file"
}

# ── Launch one worker per GPU ──
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    if [ -s "${TASK_FILES[$gpu]}" ]; then
        run_worker "$gpu" "${TASK_FILES[$gpu]}" &
    fi
done

# Wait for all workers, track failures
declare -a pids
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    if [ -s "${TASK_FILES[$gpu]}" ]; then
        pids+=($!)
    fi
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done

rm -rf "$TASK_DIR"

if [ "$failed" -eq 1 ]; then
    echo "ERROR: some workers failed"
    exit 1
fi

echo ""
echo "===== Done ====="
echo "vectors: ${VECTOR_DIR}"
