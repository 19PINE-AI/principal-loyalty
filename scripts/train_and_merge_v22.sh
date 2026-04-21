#!/usr/bin/env bash
# Compose v2.2 DPO (v2 + reader_is_principal produce pairs), train, merge.
set -euo pipefail
cd /home/ubuntu/principal-loyalty

echo "[v22] building data/dpo_v22.jsonl"
python3 scripts/build_dpo_v22.py

echo "[v22] training DPO (SFT base -> runs/qwen_dpo_v22)"
PL_DPO_PATH=data/dpo_v22.jsonl PL_DPO_DIR=runs/qwen_dpo_v22 \
    python3 scripts/train_qwen_dpo.py

echo "[v22] merging LoRA"
python3 scripts/merge_lora.py \
    --adapter runs/qwen_dpo_v22 \
    --out runs/qwen_sft_dpo_v22_merged

echo "[v22] done — merged at runs/qwen_sft_dpo_v22_merged/"
