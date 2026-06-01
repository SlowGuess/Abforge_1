# ABForge

ABForge is a post-training codebase for paper-grounded ablation design. It
builds on a lightly customized `verl` training stack and provides ABForge
data conversion, reward servers, training launchers, and evaluation scripts.

This public layout keeps ABForge-specific code outside `verl_proj/` where
possible:

- `verl_proj/`: the training framework.
- `dataprocess/`: converts ABForge JSONL data to parquet files consumed by `verl`.
- `abforge/`: external custom SFT dataset class used by the launchers.
- `reward/`: OpenAI-compatible reward servers for RL.
- `scripts/`: non-Slurm launchers for SFT, RL, reward servers, local judges, and eval.
- `evaluation/`: evaluation scripts.
- `examples/`: tiny schema examples only. Full data is available on Hugging Face.

## Data

Full training and evaluation data is available at
[`SlowGuess/abforge-data`](https://huggingface.co/datasets/SlowGuess/abforge-data).

Download the JSONL files:

```bash
huggingface-cli download SlowGuess/abforge-data \
  --repo-type dataset \
  --local-dir data
```

Then convert the training files to parquet:

```bash
python dataprocess/prepare_sft.py \
  --task 1 \
  --sft_data_path data/train/sft_task1_45961.jsonl \
  --sft_remain_path data/train/sft_raw_pool_52813.jsonl \
  --local_dir data/abforge_task1_sft

python dataprocess/prepare_sft.py \
  --task 2 \
  --sft_data_path data/train/sft_task2_37019.jsonl \
  --sft_remain_path data/train/sft_raw_pool_52813.jsonl \
  --local_dir data/abforge_task2_sft

python dataprocess/prepare_task1_rl.py \
  --input data/train/rl_task1_30000.jsonl \
  --local_dir data/abforge_task1_rl

python dataprocess/prepare_task2_rl.py \
  --input data/train/rl_task2_rubric_v2_30000.jsonl \
  --local_dir data/abforge_task2_rl
```

The held-out evaluation files are under `data/eval/`.

Task 1 RL/eval defaults use the v18.3 count penalty: one free extra bullet above GT, soft penalty 0.005 up to 7 bullets, and hard penalty 0.05 beyond 7. Override with `TASK1_COUNT_*` environment variables when needed.

## SFT

```bash
MODEL_PATH=Qwen/Qwen3-8B scripts/train_task1_sft.sh
MODEL_PATH=Qwen/Qwen3-8B scripts/train_task2_sft.sh
```

The SFT launchers use the external dataset class at
`abforge/abforge_sft_dataset.py` and pass it to `verl` via
`data.custom_cls.path`.

## Reward Servers

The reward servers use an OpenAI-compatible chat completions API. This can be
a hosted API endpoint or a local vLLM OpenAI server.

Hosted or existing endpoint:

```bash
export JUDGE_API_BASE=https://api.openai.com/v1
export JUDGE_API_KEY=...
export JUDGE_MODEL=...

scripts/serve_task1_reward_api.sh
scripts/serve_task2_reward_api.sh
```

Local vLLM judge:

```bash
JUDGE_MODEL=Qwen/Qwen3-32B TP_SIZE=2 PORT=8000 scripts/serve_local_judge_vllm.sh

export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=Qwen/Qwen3-32B
scripts/serve_task1_reward_api.sh
```

## RL

For local RL training, start the services in this order:

1. Start a judge endpoint, or point `JUDGE_API_BASE` to an existing
   OpenAI-compatible API.
2. Start the corresponding ABForge reward server.
3. Run the RL launcher with `REWARD_URL` pointing to that reward server.

Example with a local vLLM judge:

```bash
JUDGE_MODEL=Qwen/Qwen3-32B TP_SIZE=2 PORT=8000 scripts/serve_local_judge_vllm.sh

export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=Qwen/Qwen3-32B
scripts/serve_task1_reward_api.sh
scripts/serve_task2_reward_api.sh
```

Then, in another shell:

```bash
MODEL_PATH=outputs/checkpoints/task1_sft scripts/train_task1_rl.sh
MODEL_PATH=outputs/checkpoints/task2_sft scripts/train_task2_rl.sh
```

## Evaluation

The included evaluation scripts use the same OpenAI-compatible judge API
configuration as the reward servers:

```bash
export JUDGE_API_BASE=https://api.openai.com/v1
export JUDGE_API_KEY=...
export JUDGE_MODEL=...

scripts/evaluate_task1.sh outputs/task1_infer.jsonl
scripts/evaluate_task2.sh outputs/task2_infer.jsonl
```

## Notes

- No Slurm scripts are included in this public layout.
- No training data, checkpoints, logs, secrets, or machine-specific paths should
  be committed.
- Task 1 currently uses the v18.3 TM+RQ formulation: 2-6 bullets, each with a target module and a high-level research question.
