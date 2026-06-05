# Reward Servers

The reward servers expose FastAPI endpoints used by `verl` RL training:

- Task 1: `POST /get_reward_task1`
- Task 2 rubric: `POST /get_reward_task2`

Both servers call an OpenAI-compatible judge endpoint configured through:

```bash
export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=<your-judge-model>
```

This endpoint can be a hosted API service or a local vLLM OpenAI server.
