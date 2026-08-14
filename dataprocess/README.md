# Data Processing

These scripts convert ABForge JSONL files to parquet files consumed by `verl`.
They do not synthesize new data; prompt templates in these files define the
model-facing input format used during SFT/RL training.

- `prepare_sft.py`: SFT conversion for Task 1 and Task 2. Reads the **unified**
  table of the dataset (one row per paper, carrying both the paper context and
  the task supervision). By default (`--select run`) it takes the rows the
  released SFT run consumed, recorded in the table as `sft_run_task{1,2}`; that
  selection was made on an intermediate artifact outside the release and cannot
  be recomputed from the published text. `--select view` instead takes the whole
  `in_sft_task{1,2}` view and re-derives the focus/length filters, which yields a
  larger pool.
- `prepare_task1_rl.py`: RL conversion for Task 1.
- `prepare_task2_rl.py`: RL conversion for Task 2.
- `prepare_combined.py`: merges the per-task parquets into the unified
  (mixed 1:1) SFT/RL training sets consumed by `scripts/train_sft.sh` and
  `scripts/train_rl.sh`. Each RL row keeps its `data_source`
  (`abforge_task1` / `abforge_task2`) so the reward server can route it.

Full JSONL data should be downloaded from
`https://huggingface.co/datasets/SlowGuess/abforge-data`.
The files in `examples/` are tiny schema examples only.
