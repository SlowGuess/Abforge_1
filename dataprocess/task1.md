# Task 1 Defaults

- SFT train parquet: `data/abforge_task1_sft/train.parquet`
- SFT val parquet: `data/abforge_task1_sft/val.parquet`
- RL train parquet: `data/abforge_task1_rl/train.parquet`
- RL val parquet: `data/abforge_task1_rl/val.parquet`
- Training consumes the unified mixture built by `prepare_combined.py`
  (`data/abforge_combined_sft` / `data/abforge_combined_rl`).
- Reward endpoint: `http://127.0.0.1:6010/get_reward` (unified server; routed
  to the Task 1 judge by `data_source`).
- Data filter: keep papers with 2-6 GT focuses by default.
- Output format: `<Result>` with 2-6 bullets. Each bullet contains
  `- Target Module: ...` and an indented `- Research Question: ...`.
- The RL reward and evaluation share an optional count-related penalty that is
  fully configurable through `TASK1_COUNT_*` environment variables (see
  `reward/task1_candidate_utils.py` for the available knobs and their defaults).
