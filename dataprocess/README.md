# Data Processing

These scripts convert the released ABForge table to the parquet files consumed
by `verl`. They do not synthesize new data; prompt templates in these files
define the model-facing input format used during SFT/RL training.

`prepare_sft.py` and the two `prepare_task*_rl.py` scripts read the same table —
one row per paper, carrying both the paper context and the task supervision —
select their split by its `in_*` column, and apply their own filtering, so no
pre-filtered training file is needed. `--dataset` takes a HF dataset id, a local
directory of parquet shards, or a JSONL file with the same fields.

- `prepare_sft.py`: SFT conversion for Task 1 and Task 2 (`in_sft_task1` /
  `in_sft_task2`), applying the focus-count and length filters.
- `prepare_task1_rl.py`: RL conversion for Task 1 (`in_rl_task1`).
- `prepare_task2_rl.py`: RL conversion for Task 2 (`in_rl_task2`).
- `prepare_combined.py`: merges the per-task parquets into the unified
  (mixed 1:1) SFT/RL training sets consumed by `scripts/train_sft.sh` and
  `scripts/train_rl.sh`. Each RL row keeps its `data_source`
  (`abforge_task1` / `abforge_task2`) so the reward server can route it.

The full data lives at
`https://huggingface.co/datasets/SlowGuess/abforge-data` (the table under
`train/`, the benchmarks under `eval/`).
The files in `examples/` are tiny schema examples only.
