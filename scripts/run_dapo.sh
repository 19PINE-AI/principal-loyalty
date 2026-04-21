#!/usr/bin/env bash
# Launch DAPO (via verl 0.7.1) on Qwen3-8B SFT-adapted checkpoint.
#
# Prereq:
#   - runs/qwen_sft/ exists (from scripts/train_qwen_sft.py)
#   - data/verl_train.parquet + data/verl_val.parquet exist (from
#     scripts/build_verl_dataset.py)
#   - src/reward.py defines compute_score(data_source, solution_str,
#     ground_truth, extra_info) -> {"score": float, ...}
#
# DAPO specifics (vs. plain GRPO):
#   - Dynamic sampling: discard groups where all rollouts have the same
#     reward (no contrastive signal).
#   - Asymmetric clip: clip ratio high > low (allows promoting rare good
#     behaviors more aggressively).
#   - Token-level objective; no length normalization.
#
# This script is a scaffolded command — tune batch sizes / rollout-n per
# VRAM budget. With PID 4093593 holding ~45GB, we have ~52GB headroom; a
# Qwen3-8B rollout with n=4 per prompt, bs=4 should fit.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH=${PL_MODEL_PATH:-"Qwen/Qwen3-8B"}
ACTOR_INIT=${PL_ACTOR_INIT:-"runs/qwen_sft"}         # SFT adapter merged in or passed as peft_config
OUT_DIR=${PL_DAPO_DIR:-"runs/qwen_dapo"}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=false \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.response_length=384 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  data.train_files=data/verl_train.parquet \
  data.val_files=data/verl_val.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=384 \
  reward_model.reward_manager=dapo \
  reward_model.enable=False \
  custom_reward_function.path=src/reward.py \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name=principal-loyalty \
  trainer.experiment_name=qwen3-8b-dapo-v0 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=50 \
  trainer.test_freq=25 \
  trainer.total_epochs=6 \
  trainer.default_local_dir="${OUT_DIR}"
