#!/usr/bin/env bash
# Starts the unified ABForge reward server (reward/combined_reward.py).
# It exposes POST /get_reward and routes each request to the Task 1 or Task 2
# judge by the sample's `data_source` field.
set -euo pipefail

ABFORGE_ROOT=${ABFORGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ABFORGE_ROOT/reward"

: "${JUDGE_API_BASE:?Set JUDGE_API_BASE, e.g. https://api.openai.com/v1 or http://127.0.0.1:8000/v1}"
: "${JUDGE_API_KEY:?Set JUDGE_API_KEY. For local vLLM, any non-empty value is fine.}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to your OpenAI-compatible judge model}"

export COMBINED_REWARD_PORT="${COMBINED_REWARD_PORT:-6010}"

# Task 1 count-penalty knobs (see reward/task1_candidate_utils.py).
export TASK1_COUNT_FREE_EXTRA="${TASK1_COUNT_FREE_EXTRA:-1}"
export TASK1_COUNT_SOFT_RATE="${TASK1_COUNT_SOFT_RATE:-0.005}"
export TASK1_COUNT_HARD_THRESHOLD="${TASK1_COUNT_HARD_THRESHOLD:-7}"
export TASK1_COUNT_HARD_RATE="${TASK1_COUNT_HARD_RATE:-0.05}"
export TASK1_COUNT_CAP="${TASK1_COUNT_CAP:-0.5}"

echo "Unified reward server on port ${COMBINED_REWARD_PORT} (routes by data_source)"
echo "Task 1 count penalty: free_extra=${TASK1_COUNT_FREE_EXTRA}, soft=${TASK1_COUNT_SOFT_RATE}, hard_threshold=${TASK1_COUNT_HARD_THRESHOLD}, hard_rate=${TASK1_COUNT_HARD_RATE}, cap=${TASK1_COUNT_CAP}"
python3 combined_reward.py
