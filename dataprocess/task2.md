# Task 2 Defaults

- SFT train parquet: `data/abforge_task2_sft/train.parquet`
- SFT val parquet: `data/abforge_task2_sft/val.parquet`
- RL train parquet: `data/abforge_task2_rl/train.parquet`
- RL val parquet: `data/abforge_task2_rl/val.parquet`
- Training consumes the unified mixture built by `prepare_combined.py`
  (`data/abforge_combined_sft` / `data/abforge_combined_rl`).
- Reward endpoint: `http://127.0.0.1:6010/get_reward` (unified server; routed
  to the Task 2 rubric judge by `data_source`).
- Output format: `<Proposed_Plan>` with Objective, Baseline Setup, Variants,
  and Fixed Protocols & Metrics sections.
