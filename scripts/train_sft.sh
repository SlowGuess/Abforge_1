#!/usr/bin/env bash
# Unified SFT: fine-tune the base model on the 1:1 Task 1 + Task 2 mixture
# (data/abforge_combined_sft, built by dataprocess/prepare_combined.py).
set -euo pipefail
set -x

ABFORGE_ROOT=${ABFORGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ABFORGE_ROOT/verl_proj"

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
TRAIN_PATH=${TRAIN_PATH:-$ABFORGE_ROOT/data/abforge_combined_sft/train.parquet}
VAL_PATH=${VAL_PATH:-$ABFORGE_ROOT/data/abforge_combined_sft/val.parquet}
SAVE_DIR=${SAVE_DIR:-$ABFORGE_ROOT/outputs/checkpoints/sft}
LOGGER=${LOGGER:-"['console']"}

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_PATH" \
  data.val_files="$VAL_PATH" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length="${MAX_LENGTH:-6144}" \
  data.truncation=right \
  data.train_batch_size="${GLOBAL_BSZ:-128}" \
  data.micro_batch_size_per_gpu="${MICRO_BSZ:-1}" \
  +data.chat_template_kwargs.enable_thinking=False \
  data.custom_cls.path="$ABFORGE_ROOT/dataprocess/abforge_sft_dataset.py" \
  data.custom_cls.name=AbforgeSFTDataset \
  optim.lr="${LR:-2e-5}" \
  optim.weight_decay="${WEIGHT_DECAY:-0.1}" \
  optim.warmup_steps_ratio="${WARMUP_RATIO:-0.03}" \
  optim.clip_grad="${CLIP_GRAD:-1.0}" \
  model.partial_pretrain="$MODEL_PATH" \
  model.trust_remote_code=True \
  model.enable_gradient_checkpointing=True \
  model.fsdp_config.cpu_offload=False \
  model.fsdp_config.offload_params=False \
  use_remove_padding=True \
  ulysses_sequence_parallel_size="${ULYSSES_SIZE:-2}" \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.project_name="${PROJECT_NAME:-abforge_sft}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_8b_sft_unified}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.save_freq="${SAVE_FREQ:--1}" \
  trainer.logger="$LOGGER" \
  "$@"
