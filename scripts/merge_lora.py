"""Merge a LoRA adapter into the base Qwen3-8B and save as a full HF model.

Needed because vLLM is easier to drive with a full model directory than
with runtime LoRA swapping across multiple adapters — and Phase 1 needs
its subject vendor (qwen-8b-local) to point at a single served model.

Run (after SFT+DPO):
    python3 scripts/merge_lora.py \
        --adapter runs/qwen_dpo \
        --out runs/qwen_sft_dpo_merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-8B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=MODEL_ID,
                    help="Base model to apply the adapter onto. Default: %(default)s. "
                         "For DAPO adapters trained on top of the merged v4.1, pass "
                         "runs/qwen_sft_dpo_v4_1_merged.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # verl LoRA adapter dirs don't ship the tokenizer; pull it from the base.
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    peft = PeftModel.from_pretrained(base, args.adapter)
    merged = peft.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"[merge] saved -> {out}")


if __name__ == "__main__":
    main()
