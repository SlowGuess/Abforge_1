#!/usr/bin/env bash
set -euo pipefail

ABFORGE_ROOT=${ABFORGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ABFORGE_ROOT/reward"

: "${JUDGE_API_BASE:?Set JUDGE_API_BASE, e.g. https://api.openai.com/v1 or http://127.0.0.1:8000/v1}"
: "${JUDGE_API_KEY:?Set JUDGE_API_KEY. For local vLLM, any non-empty value is fine.}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to your OpenAI-compatible judge model}"

export TASK2_REWARD_PORT="${TASK2_REWARD_PORT:-6011}"
python3 task2_rubric_reward.py
