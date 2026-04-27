#!/usr/bin/env bash
# DAPO-v3 on Qwen3-8B — orthogonality test from §4.4.1.
#
# §4.4.1 ablation found two reward terms tune ORTHOGONAL axes:
#   - Adv refusal bonus controls bound-leak (smaller bonus -> bound-leak ~1)
#   - Coop refusal penalty controls authoring/sanity/posture cells
#
# v3 combines:
#   - Leak-only's adv side: NO refusal bonus, just -leak
#   - v2's coop side: -1.0 penalty on cooperative refusals
#   - v2's 3x-oversampled sanity parquet (24/47 coop)
#
# Hypothesis: total harm < 25/108, plain leak < 8%, bound-leak <= 1.
# If reward terms truly orthogonal, v3 inherits each variant's strengths.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH=${PL_MODEL_PATH:-"runs/qwen_sft_dpo_v4_1_merged"}
OUT_DIR=${PL_DAPO_DIR:-"runs/qwen_dapo_v3"}

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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  data.train_files=data/verl_train_v2.parquet \
  data.val_files=data/verl_val_v2.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=384 \
  reward_model.reward_manager=dapo \
  reward_model.enable=False \
  custom_reward_function.path=src/reward.py \
  custom_reward_function.name=compute_score_v3 \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name=principal-loyalty \
  trainer.experiment_name=qwen3-8b-dapo-v3 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.total_epochs=5 \
  trainer.default_local_dir="${OUT_DIR}"
