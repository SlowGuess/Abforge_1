# ABForge: Post-Training for Paper-Grounded Ablation Design

<p align="center">
    🤗 <a href="https://huggingface.co/SlowGuess">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📄 <a href="https://arxiv.org/abs/XXXX.XXXXX">arXiv</a>
</p>

<!-- TODO: replace the 🤗 Hugging Face and 📄 arXiv links above with the paper / HF collection URLs once public. The dataset link is in the Resources section below. -->

## 📖 Abstract

Ablation studies are central to testing scien-
tific claims, yet designing a rigorous paper-
grounded ablation remains difficult. We for-
mulate this problem as two tasks, Ablation
Objective Identification and Ablation Exper-
iment Synthesis, and introduce AblationBench,
an expert-annotated benchmark for evaluating
both. To support post-training, we construct
the ABForge post-training corpus through a
semi-automated audit-in-the-loop pipeline that
extracts ablation objectives, performs abla-
tion specification completion, synthesizes self-
reflection CoT rationales, and derives target-
specific evaluation rubrics. We then post-train
Qwen-3-8B with supervised fine-tuning fol-
lowed by rubric-based reinforcement learning.
The resulting model improves the base model
from 42.1 to 56.8 on objective identification
and from 42.0 to 62.0 on experiment synthesis
in automated evaluation, and is the strongest
open-source model in human evaluation.

<div align="center">
<img src="./assets/overview.png" width="80%"/>
<p><em>Overview of ABForge.</em></p>
</div>


## 🔗 Resources

- Training & evaluation data: [`SlowGuess/abforge-data`](https://huggingface.co/datasets/SlowGuess/abforge-data)

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

### Configuration

ABForge is driven by a few environment variables; set them in your shell or job
launcher before running the scripts below:

```bash
export ABFORGE_ROOT=/path/to/Abforge_1     # repo root
export MODEL_PATH=Qwen/Qwen3-8B            # base model to train

# OpenAI-compatible judge endpoint (hosted API or local vLLM)
export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=<your-judge-model>

# reward server ports
export TASK1_REWARD_PORT=6013
export TASK2_REWARD_PORT=6011
```

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
> [`dataprocess/task1.md`](dataprocess/task1.md) and [`dataprocess/task2.md`](dataprocess/task2.md)
> for the configurable defaults.

## 🛠️ Training

### SFT

```bash
MODEL_PATH=Qwen/Qwen3-8B scripts/train_task1_sft.sh
MODEL_PATH=Qwen/Qwen3-8B scripts/train_task2_sft.sh
```

The SFT launchers use the external dataset class at
`dataprocess/abforge_sft_dataset.py` and pass it to `verl` via
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

Local vLLM judge (serve any OpenAI-compatible model locally):

```bash
JUDGE_MODEL=<your-judge-model> TP_SIZE=2 PORT=8000 scripts/serve_local_judge_vllm.sh

export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=<your-judge-model>
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
JUDGE_MODEL=<your-judge-model> TP_SIZE=2 PORT=8000 scripts/serve_local_judge_vllm.sh

export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=<your-judge-model>
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
- `dataprocess/` — all data handling: JSONL→parquet conversion (`prepare_*.py`), the
  external SFT dataset class (`abforge_sft_dataset.py`), task defaults
  (`task1.md` / `task2.md`), and `examples/` schema samples (full data on Hugging Face).
- `reward/` — OpenAI-compatible reward servers for RL (Task 1 / Task 2 rubric).
- `scripts/` — launchers for SFT, RL, reward servers, local judges, and eval.
- `eval/` — evaluation scripts.

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
