#!/usr/bin/env bash
set -euo pipefail
set -x

ABFORGE_ROOT=${ABFORGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ABFORGE_ROOT/verl_proj"

MODEL_PATH=${MODEL_PATH:-$ABFORGE_ROOT/outputs/checkpoints/task1_sft}
TRAIN_PATH=${TRAIN_PATH:-$ABFORGE_ROOT/data/abforge_task1_rl/train.parquet}
VAL_PATH=${VAL_PATH:-$ABFORGE_ROOT/data/abforge_task1_rl/val.parquet}
REWARD_URL=${REWARD_URL:-http://127.0.0.1:6013/get_reward_task1}
LOGGER=${LOGGER:-"['console']"}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_PATH" \
  data.val_files="$VAL_PATH" \
  data.train_batch_size="${TRAIN_BSZ:-64}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-3072}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  +data.chat_template_kwargs.enable_thinking=False \
  +data.trust_remote_code=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr="${LR:-1.5e-5}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BSZ:-64}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BSZ:-2}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-16000}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BSZ:-4}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-2}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${TOP_P:-0.95}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BSZ:-4}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  reward_model.enable=False \
  reward_model.reward_api="$REWARD_URL" \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="$LOGGER" \
  trainer.project_name="${PROJECT_NAME:-abforge_task1}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_8b_grpo_task1_v18_3}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-50}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-100}" \
  trainer.default_local_dir="${SAVE_DIR:-$ABFORGE_ROOT/outputs/checkpoints/task1_rl}" \
  "$@"
