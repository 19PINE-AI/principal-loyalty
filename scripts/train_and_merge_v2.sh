#!/usr/bin/env bash
# Compose v2 DPO pairs, train, merge. Assumes authoring-MT v2 pair file already
# written to data/dpo_multiturn_v2_authoring.jsonl.
set -euo pipefail
cd /home/ubuntu/principal-loyalty

echo "[v2] building data/dpo_v2.jsonl"
python3 scripts/build_dpo_v2.py

echo "[v2] training DPO (SFT base -> runs/qwen_dpo_v2)"
PL_DPO_PATH=data/dpo_v2.jsonl PL_DPO_DIR=runs/qwen_dpo_v2 \
    python3 scripts/train_qwen_dpo.py

echo "[v2] merging LoRA"
python3 scripts/merge_lora.py \
    --adapter runs/qwen_dpo_v2 \
    --out runs/qwen_sft_dpo_v2_merged

echo "[v2] done — merged at runs/qwen_sft_dpo_v2_merged/"
