# Reward Server

RL training uses a single unified reward server:

```bash
scripts/serve_reward_api.sh   # runs reward/combined_reward.py on port 6010
```

It exposes `POST /get_reward` and routes each request by the sample's
`data_source` field:

- `abforge_task1*` → Task 1 judge (`task1_reward.py`): specificity-weighted
  one-to-one objective matching with structural penalties.
- `abforge_task2*` → Task 2 rubric judge (`task2_rubric_reward.py`): weighted
  rubric score with format/length penalties.

The per-task endpoints `POST /get_reward_task1` and `POST /get_reward_task2`
are served by the same process, and `GET /health` reports the judge
configuration.

Both judges call an OpenAI-compatible judge endpoint configured through:

```bash
export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=<your-judge-model>
```

This endpoint can be a hosted API service or a local vLLM OpenAI server.
