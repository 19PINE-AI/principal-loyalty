#!/usr/bin/env bash
# DAPO leak-only ablation — validates §4.4's "cooperative branch is load-bearing" claim.
#
# Reward: -leak only. No cooperative branch, no refusal bonus.
#
# Prediction: policy collapses onto max-refusal (posture-collapse), reproducing
# the v4 regression that v4.1 DPO walked back. If sanity-cell harm and probe-item
# MI stay flat or improve under this reward, the claim is wrong. If they crater
# while adversarial leak stays near zero, the claim is confirmed.
#
# Uses the ORIGINAL (non-oversampled) v1 parquet so the delta is purely the
# reward function, not the data mix.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH=${PL_MODEL_PATH:-"runs/qwen_sft_dpo_v4_1_merged"}
OUT_DIR=${PL_DAPO_DIR:-"runs/qwen_dapo_leakonly"}

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs "${OUT_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=false \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.response_length=384 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.enforce_eager=true \
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
  custom_reward_function.name=compute_score_leakonly \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name=principal-loyalty \
  trainer.experiment_name=qwen3-8b-dapo-leakonly \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.total_epochs=5 \
  trainer.default_local_dir="${OUT_DIR}"
