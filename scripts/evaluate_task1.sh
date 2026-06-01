#!/usr/bin/env bash
set -euo pipefail

ABFORGE_ROOT=${ABFORGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
INPUT=${1:?Usage: scripts/evaluate_task1.sh <inference.jsonl>}

export TASK1_COUNT_FREE_EXTRA="${TASK1_COUNT_FREE_EXTRA:-1}"
export TASK1_COUNT_SOFT_RATE="${TASK1_COUNT_SOFT_RATE:-0.005}"
export TASK1_COUNT_HARD_THRESHOLD="${TASK1_COUNT_HARD_THRESHOLD:-7}"
export TASK1_COUNT_HARD_RATE="${TASK1_COUNT_HARD_RATE:-0.05}"
export TASK1_COUNT_CAP="${TASK1_COUNT_CAP:-0.5}"

python3 "$ABFORGE_ROOT/evaluation/eval_task1.py" --input "$INPUT" "${@:2}"
