#!/usr/bin/env bash
# ============================================================================
# run_transfer.sh
# Run black-hole attack across all dataset pairs (transfer + default) and
# save results under data/result/ with timestamp-based filenames.
#
# Usage:
#     bash scripts/run_transfer.sh
# ============================================================================
set -euo pipefail

# ── Config ──
DATASETS=(hotpotqa msmarco nq)
MODEL="${MODEL:-contriever}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

RUN_PY="${PROJECT_DIR}/run.py"

echo "============================================"
echo "  run_transfer.sh"
echo "  model:    ${MODEL}"
echo "  datasets: ${DATASETS[*]}"
echo "============================================"
echo ""

FAILURES=()
TOTAL=0

for src in "${DATASETS[@]}"; do
    for victim in "${DATASETS[@]}"; do
        if [ "$src" = "$victim" ]; then
            MODE="default"
            PREPROCESS="default"
        else
            MODE="transfer"
            PREPROCESS="query_trans"
        fi

        TOTAL=$((TOTAL + 1))
        echo "============================================"
        echo "  [${TOTAL}] src=${src} → victim=${victim} (mode=${MODE} preprocess=${PREPROCESS})"
        echo "============================================"

        if "${PYTHON}" "${RUN_PY}" \
            --model "${MODEL}" \
            --src "${src}" \
            --victim "${victim}" \
            --mode "${MODE}" \
            --preprocess "${PREPROCESS}"; then
            echo "  ✓ PASSED"
        else
            echo "  ✗ FAILED"
            FAILURES+=("src=${src} victim=${victim}")
        fi
        echo ""
    done
done

echo "============================================"
echo "  Done. ${TOTAL} runs total."
if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "  Failures:"
    for f in "${FAILURES[@]}"; do
        echo "    - ${f}"
    done
    exit 1
else
    echo "  All passed!"
fi
echo "============================================"
