# Task 1 Defaults

- SFT train parquet: `data/abforge_task1_sft/train.parquet`
- SFT val parquet: `data/abforge_task1_sft/val.parquet`
- RL train parquet: `data/abforge_task1_rl/train.parquet`
- RL val parquet: `data/abforge_task1_rl/val.parquet`
- Reward endpoint: `http://127.0.0.1:6013/get_reward_task1`
- Data filter: keep papers with 2-6 GT focuses by default.
- Output format: `<Result>` with 2-6 bullets. Each bullet contains
  `- Target Module: ...` and an indented `- Research Question: ...`.
- RL reward/eval count penalty default follows v18.3:
  `TASK1_COUNT_FREE_EXTRA=1`, `TASK1_COUNT_SOFT_RATE=0.005`,
  `TASK1_COUNT_HARD_THRESHOLD=7`, `TASK1_COUNT_HARD_RATE=0.05`.
