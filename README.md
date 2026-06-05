# ABForge: Post-Training for Paper-Grounded Ablation Design

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/SlowGuess/abforge-data">Hugging Face Dataset</a>&nbsp&nbsp | &nbsp&nbsp📄 <a href="https://arxiv.org/abs/XXXX.XXXXX">arXiv</a>
</p>

<!-- TODO: replace the arXiv link above once the paper is public. -->

## 📖 Abstract

<!-- TODO: replace this paragraph with the official paper abstract. -->

ABForge is a post-training codebase for **paper-grounded ablation design**. Given
a scientific paper, ABForge trains language models to (1) identify the key
ablation objectives a paper should investigate, and (2) synthesize a rigorous,
executable ablation experiment plan. The pipeline follows a standard
**SFT → GRPO** recipe on a lightly customized [`verl`](https://github.com/volcengine/verl)
training stack, using **LLM-as-judge** reward servers to score the structured
output during reinforcement learning. We release the data-conversion tools,
reward servers, training launchers, and evaluation scripts needed to reproduce
ABForge post-training.

<div align="center">
<img src="./assets/overview.png" width="80%"/>
<p><em>Overview of ABForge.</em></p>
</div>

<!-- TODO: add ./assets/overview.png. Until then this image will show as broken. -->

## 🔗 Resources

- Training & evaluation data: [`SlowGuess/abforge-data`](https://huggingface.co/datasets/SlowGuess/abforge-data)
- Model checkpoints: <!-- TODO: add HF links for released SFT / RL checkpoints, or remove this line. -->

ABForge covers two tasks:

- **Task 1 — Ablation objective identification:** identify the key ablation
  objectives / research questions a paper should investigate.
- **Task 2 — Ablation plan synthesis:** produce a rigorous ablation experiment
  plan (objective, baseline setup, variants, fixed protocols & metrics).

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/SlowGuess/Abforge_1.git
cd Abforge_1

conda create -n abforge python=3.11 -y
conda activate abforge

# Install the customized verl training stack, then ABForge dependencies.
pip install -e verl_proj
pip install -r requirements.txt
```

The repository keeps ABForge-specific code **outside** `verl_proj/` where
possible, so the framework and the project code stay decoupled.

### Data

Full training and evaluation data is available at
[`SlowGuess/abforge-data`](https://huggingface.co/datasets/SlowGuess/abforge-data).

Download the JSONL files:

```bash
huggingface-cli download SlowGuess/abforge-data \
  --repo-type dataset \
  --local-dir data
```

Then convert the training files to parquet (consumed by `verl`):

```bash
# SFT
python dataprocess/prepare_sft.py \
  --task 1 \
  --sft_data_path data/train/sft_task1_45961.jsonl \
  --sft_remain_path data/train/SFT_50K.jsonl \
  --local_dir data/abforge_task1_sft

python dataprocess/prepare_sft.py \
  --task 2 \
  --sft_data_path data/train/sft_task2_37019.jsonl \
  --sft_remain_path data/train/SFT_50K.jsonl \
  --local_dir data/abforge_task2_sft

# RL
python dataprocess/prepare_task1_rl.py \
  --input data/train/RL_task1_30K.jsonl \
  --local_dir data/abforge_task1_rl

python dataprocess/prepare_task2_rl.py \
  --input data/train/RL_task2_30K.jsonl \
  --local_dir data/abforge_task2_rl
```

The held-out evaluation files are under `data/eval/`. Use
`ablationbench_1000.jsonl` for the full benchmark and
`ablationbench_200.jsonl` for the clean 200-instance human-evaluation subset.

> **Task defaults.** Task 1 SFT/RL preprocessing keeps papers with 2–6
> ground-truth focuses by default. See
> [`configs/task1.md`](configs/task1.md) and [`configs/task2.md`](configs/task2.md)
> for the configurable defaults.

## 🛠️ Training

### SFT

```bash
MODEL_PATH=Qwen/Qwen3-8B scripts/train_task1_sft.sh
MODEL_PATH=Qwen/Qwen3-8B scripts/train_task2_sft.sh
```

The SFT launchers use the external dataset class at
`abforge/abforge_sft_dataset.py` and pass it to `verl` via
`data.custom_cls.path`.

### Reward Servers

The reward servers expose an OpenAI-compatible chat-completions judge. The judge
can be a hosted API endpoint or a local vLLM OpenAI server.

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

### RL (GRPO)

Start the services in this order:

1. Start a judge endpoint, or point `JUDGE_API_BASE` to an existing
   OpenAI-compatible API.
2. Start the corresponding ABForge reward server.
3. Run the RL launcher with `REWARD_URL` pointing to that reward server.

```bash
# 1 + 2: judge + reward servers
JUDGE_MODEL=Qwen/Qwen3-32B TP_SIZE=2 PORT=8000 scripts/serve_local_judge_vllm.sh

export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=Qwen/Qwen3-32B
scripts/serve_task1_reward_api.sh
scripts/serve_task2_reward_api.sh
```

Then, in another shell:

```bash
# 3: RL launchers (point MODEL_PATH at your SFT checkpoint)
MODEL_PATH=outputs/checkpoints/task1_sft scripts/train_task1_rl.sh
MODEL_PATH=outputs/checkpoints/task2_sft scripts/train_task2_rl.sh
```

## 📈 Evaluation

The evaluation scripts use the same OpenAI-compatible judge configuration as the
reward servers:

```bash
export JUDGE_API_BASE=https://api.openai.com/v1
export JUDGE_API_KEY=...
export JUDGE_MODEL=...

scripts/evaluate_task1.sh outputs/task1_infer.jsonl
scripts/evaluate_task2.sh outputs/task2_infer.jsonl
```

## 🗂️ Repository Layout

- `verl_proj/` — the (lightly customized) `verl` training framework.
- `dataprocess/` — converts ABForge JSONL data to parquet files consumed by `verl`.
- `abforge/` — external custom SFT dataset class used by the launchers.
- `reward/` — OpenAI-compatible reward servers for RL (Task 1 / Task 2 rubric).
- `scripts/` — launchers for SFT, RL, reward servers, local judges, and eval.
- `evaluation/` — evaluation scripts.
- `configs/` — task defaults and an environment-variable template (`env.example`).
- `examples/` — tiny schema examples only; full data lives on Hugging Face.

> **Notes.** No training data, checkpoints, logs, secrets, or machine-specific
> paths are committed. Task 1 produces 2–6 bullets, each pairing a target module
> with a high-level research question.

## 🙏 Acknowledgements

This repository builds on the excellent [verl](https://github.com/volcengine/verl)
project. ABForge code is released under the Apache-2.0 License (see
[`LICENSE`](LICENSE)).

## 📝 Citation

<!-- TODO: replace with the final paper citation once available. -->

If you find ABForge useful in your research, please cite our paper:

```bibtex
@misc{abforge2026,
      title={ABForge: Post-Training for Paper-Grounded Ablation Design},
      author={TODO},
      year={2026},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/XXXX.XXXXX},
}
```
