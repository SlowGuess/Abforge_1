# ABForge: Post-Training for Paper-Grounded Ablation Design

<p align="center">
    🤗 <a href="https://huggingface.co/collections/SlowGuess/abforge-6a2ac561d0e97f11e409dd75">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📄 <a href="https://arxiv.org/abs/XXXX.XXXXX">arXiv</a>
</p>

<!-- TODO: replace the 📄 arXiv link above with the paper URL once public. -->

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
from 44.4 to 55.9 on objective identification
and from 43.4 to 62.4 on experiment synthesis
in automated evaluation, and is the strongest
open-source model in human evaluation.

<div align="center">
<img src="./assets/overview.png" width="80%"/>
<p><em>Overview of ABForge.</em></p>
</div>


## 🔗 Resources

- Models & data (Hugging Face collection): [`SlowGuess/ABForge`](https://huggingface.co/collections/SlowGuess/abforge-6a2ac561d0e97f11e409dd75)
- Released model: [`SlowGuess/ABForge-Qwen3-8B`](https://huggingface.co/SlowGuess/ABForge-Qwen3-8B) —
  one unified checkpoint for both tasks, with its
  [`-SFT`](https://huggingface.co/SlowGuess/ABForge-Qwen3-8B-SFT) and
  [`-RL`](https://huggingface.co/SlowGuess/ABForge-Qwen3-8B-RL) stage ablations
- Task-specific specialists (ablation of task sharing):
  [`-Task1`](https://huggingface.co/SlowGuess/ABForge-Qwen3-8B-Task1) /
  [`-Task2`](https://huggingface.co/SlowGuess/ABForge-Qwen3-8B-Task2), each also in `-SFT` and `-RL` variants
- Training & evaluation data: [`SlowGuess/abforge-data`](https://huggingface.co/datasets/SlowGuess/abforge-data) —
  also hosts per-paper generations and judge outputs for all 21 evaluated models under `outputs/`

ABForge covers two tasks:

- **Task 1 — Ablation objective identification:** identify the key ablation
  objectives / research questions a paper should investigate.
- **Task 2 — Ablation plan synthesis:** produce a rigorous ablation experiment
  plan (objective, baseline setup, variants, fixed protocols & metrics).

ABForge trains a **single unified model** that handles both tasks: Task 1 and
Task 2 examples are mixed at a 1:1 ratio in both the SFT and RL stages, and
during RL each rollout is routed to its task-specific reward by the sample's
`data_source` field.

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

# unified reward server port
export COMBINED_REWARD_PORT=6010
```

### Data

Full training and evaluation data is available at
[`SlowGuess/abforge-data`](https://huggingface.co/datasets/SlowGuess/abforge-data).

Download it (the canonical table plus the RL and evaluation files):

```bash
huggingface-cli download SlowGuess/abforge-data --repo-type dataset \
  --include "unified/*" "train/RL_*" "eval/*" --local-dir data
```

Convert each task's training data to parquet. The SFT scripts read the unified
table directly — pass a dataset id to stream it from the Hub, or the local
`data/unified` directory downloaded above:

```bash
# SFT
python dataprocess/prepare_sft.py --task 1 \
  --dataset data/unified \
  --local_dir data/abforge_task1_sft

python dataprocess/prepare_sft.py --task 2 \
  --dataset data/unified \
  --local_dir data/abforge_task2_sft

# RL
python dataprocess/prepare_task1_rl.py \
  --input data/train/RL_task1_30K.jsonl \
  --local_dir data/abforge_task1_rl

python dataprocess/prepare_task2_rl.py \
  --input data/train/RL_task2_30K.jsonl \
  --local_dir data/abforge_task2_rl
```

Then merge them into the unified (mixed 1:1) training sets consumed by the
training scripts:

```bash
python dataprocess/prepare_combined.py --mode sft \
  --task1_dir data/abforge_task1_sft --task2_dir data/abforge_task2_sft \
  --out_dir data/abforge_combined_sft

python dataprocess/prepare_combined.py --mode rl \
  --task1_dir data/abforge_task1_rl --task2_dir data/abforge_task2_rl \
  --out_dir data/abforge_combined_rl
```

The held-out evaluation files are under `data/eval/`. Use
`ablationbench_1000.jsonl` for the full benchmark and
`ablationbench_200.jsonl` for the clean 200-instance human-evaluation subset.

> **Task defaults.** Task 1 SFT/RL preprocessing keeps papers with 1–6
> ground-truth focuses by default, matching the released runs. The evaluation
> sets are strictly 2–6. See
> [`dataprocess/task1.md`](dataprocess/task1.md) and [`dataprocess/task2.md`](dataprocess/task2.md)
> for the configurable defaults.

## 🛠️ Training

### SFT

Fine-tune the base model on the mixed Task 1 + Task 2 SFT data:

```bash
MODEL_PATH=Qwen/Qwen3-8B scripts/train_sft.sh
```

The SFT launcher uses the external dataset class at
`dataprocess/abforge_sft_dataset.py` and passes it to `verl` via
`data.custom_cls.path`.

### Reward Server

RL uses a single unified reward server (`reward/combined_reward.py`). It
exposes `POST /get_reward` and routes each request to the Task 1 judge
(specificity-weighted objective matching + structural penalties) or the Task 2
judge (weighted rubric score + format/length penalties) based on the sample's
`data_source`. The underlying judge is an OpenAI-compatible chat-completions
endpoint — a hosted API or a local vLLM server.

Hosted or existing endpoint:

```bash
export JUDGE_API_BASE=https://api.openai.com/v1
export JUDGE_API_KEY=...
export JUDGE_MODEL=...

scripts/serve_reward_api.sh
```

Local vLLM judge (serve any OpenAI-compatible model locally):

```bash
JUDGE_MODEL=<your-judge-model> TP_SIZE=2 PORT=8000 scripts/serve_local_judge_vllm.sh

export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=<your-judge-model>
scripts/serve_reward_api.sh
```

### RL (GRPO)

Start the services in this order:

1. Start a judge endpoint, or point `JUDGE_API_BASE` to an existing
   OpenAI-compatible API.
2. Start the unified ABForge reward server.
3. Run the RL launcher with `REWARD_URL` pointing to that reward server.

```bash
# 1 + 2: judge + reward server
JUDGE_MODEL=<your-judge-model> TP_SIZE=2 PORT=8000 scripts/serve_local_judge_vllm.sh

export JUDGE_API_BASE=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MODEL=<your-judge-model>
scripts/serve_reward_api.sh
```

Then, in another shell:

```bash
# 3: RL launcher (point MODEL_PATH at your SFT checkpoint)
MODEL_PATH=outputs/checkpoints/sft scripts/train_rl.sh
```

The RL launcher runs 200 GRPO steps by default (twice a single-task schedule,
since each task is ~50% of the mixed batch) and validates/saves every 20
steps. Validation score typically peaks well before the schedule ends; select
the best-validation checkpoint rather than the last one.

## 📈 Evaluation

Evaluation is per-task and runs in two steps: generate, then judge.

**1. Generate.** `run_inference_local.py` writes the reference fields next to each
generation, so its output is already the scorer's input — no join step:

```bash
huggingface-cli download SlowGuess/abforge-data --repo-type dataset \
  --include "eval/*" --local-dir data

python run_inference_local.py --task 1 \
  --input data/eval/ablationbench_200.jsonl \
  --output outputs/task1_infer.jsonl \
  --model-path SlowGuess/ABForge-Qwen3-8B \
  --dtype bf16 --device-map auto \
  --max-new-tokens 5120 --temperature 0.0 --stop-on '</Result>'

python run_inference_local.py --task 2 \
  --input data/eval/ablationbench_200.jsonl \
  --output outputs/task2_infer.jsonl \
  --model-path SlowGuess/ABForge-Qwen3-8B \
  --dtype bf16 --device-map auto \
  --max-new-tokens 4096 --temperature 0.0 --stop-on '</Proposed_Plan>'
```

Re-running resumes where it left off; pass `--overwrite` to start the file over.

To score the **published** generations of the 21 evaluated models instead, join
them to the benchmark first — those files carry only the model output:

```bash
python eval/make_eval_input.py --task 1 \
  --generations data/outputs/task1/generations/abforge.jsonl \
  --bench data/eval/ablationbench_200.jsonl \
  --output outputs/task1_infer.jsonl
```

**2. Judge.** The evaluation scripts use the same OpenAI-compatible judge
configuration as the reward server:

```bash
export JUDGE_API_BASE=https://api.openai.com/v1
export JUDGE_API_KEY=...
export JUDGE_MODEL=...

scripts/evaluate_task1.sh outputs/task1_infer.jsonl
scripts/evaluate_task2.sh outputs/task2_infer.jsonl
```

The reported Task 1 score is `paper_score = 100 × (R + 0.5 × (P_spec − 0.70))`,
where `R` is bipartite-enforced recall over the reference focuses and `P_spec` is
the mean paper-specificity of the generated bullets. Task 2 reports `design_score`
over the fixed 10-item rubric. `adjusted_score` is a diagnostic, not the headline.

## 🗂️ Repository Layout

- `verl_proj/` — the (lightly customized) `verl` training framework.
- `dataprocess/` — all data handling: per-task JSONL→parquet conversion
  (`prepare_*.py`), the unified-mixture merge (`prepare_combined.py`), the
  external SFT dataset class (`abforge_sft_dataset.py`), task defaults
  (`task1.md` / `task2.md`), and `examples/` schema samples (full data on Hugging Face).
- `reward/` — the unified reward server for RL (`combined_reward.py`), which
  routes to the Task 1 / Task 2 rubric judges by `data_source`.
- `scripts/` — launchers for SFT, RL, the reward server, local judges, and eval.
- `eval/` — the two rubric judges (`eval_task1.py`, `eval_task2_rubric.py`) and
  `make_eval_input.py`, which joins published generations back to the benchmark.
- `run_inference_local.py` — batch generation with a local checkpoint, writing
  scorer-ready JSONL for either task.

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
