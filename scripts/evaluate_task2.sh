#!/usr/bin/env bash
set -euo pipefail

ABFORGE_ROOT=${ABFORGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
INPUT=${1:?Usage: scripts/evaluate_task2.sh <inference.jsonl>}

python3 "$ABFORGE_ROOT/evaluation/eval_task2_rubric.py" --input "$INPUT" "${@:2}"
