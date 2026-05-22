# Data Processing

These scripts convert ABForge JSONL files to parquet files consumed by `verl`.
They do not synthesize new data; prompt templates in these files define the
model-facing input format used during SFT/RL training.

- `prepare_sft.py`: SFT conversion for Task 1 and Task 2.
- `prepare_task1_rl.py`: RL conversion for Task 1.
- `prepare_task2_rl.py`: RL conversion for Task 2.

Full JSONL data should be downloaded from
`https://huggingface.co/datasets/SlowGuess/abforge-data`.
The files in `examples/` are tiny schema examples only.
