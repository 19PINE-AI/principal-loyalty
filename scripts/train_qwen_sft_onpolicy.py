"""SFT training on on-policy distillation completions.

Input file format (one row per training example, jsonl):
    {
      "item_id": ...,
      "system": PLAIN_SYSTEM,
      "prompt": "<briefing + cp[0] + agent_student[1] + cp[1] + ... + cp[N-1]>",
      "completion": teacher's response at this state,
      ...
    }

This trains a QLoRA adapter on top of the input model (e.g. v4.1 endpoint),
maximizing the log-likelihood of the teacher's completion conditioned on the
student's actual prior conversation state. This is the per-turn-SFT variant
of on-policy distillation, equivalent to DeepSeek-V4's multi-turn distillation
approximation when teacher logits are not available.

Env vars (mirrors train_qwen_sft.py):
    PL_MODEL_ID       base model id or local path
    PL_SFT_PATH       on-policy SFT pairs (jsonl, with 'completion' key)
    PL_OUT_DIR        output adapter directory
    PL_SFT_EPOCHS     default 2
    PL_SFT_LR         default 5e-5
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_ID = os.environ.get("PL_MODEL_ID", "runs/qwen_sft_dpo_v4_1_merged")
ROOT = Path(__file__).resolve().parent.parent
SFT_PATH = Path(os.environ.get("PL_SFT_PATH", ROOT / "data" / "onpolicy_iter1_sft.jsonl"))
OUT_DIR = Path(os.environ.get("PL_OUT_DIR", ROOT / "runs" / "qwen_onpolicy_sft_iter1"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _merge_consecutive_roles(msgs: list[dict]) -> list[dict]:
    if not msgs:
        return msgs
    out = [dict(msgs[0])]
    for m in msgs[1:]:
        if m["role"] == out[-1]["role"]:
            out[-1] = {**out[-1], "content": (out[-1]["content"] + "\n\n" + m["content"]).strip()}
        else:
            out.append(dict(m))
    return out


def format_sft(example: dict, tokenizer) -> dict:
    msgs = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    msgs = _merge_consecutive_roles(msgs)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[sft-onpolicy] model={MODEL_ID} data={SFT_PATH} out={OUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config(),
        device_map="auto",
        dtype=torch.bfloat16,
    )

    raw = load_jsonl(SFT_PATH)
    print(f"[sft-onpolicy] {len(raw)} training points")
    ds_train = Dataset.from_list([format_sft(e, tokenizer) for e in raw])

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=int(os.environ.get("PL_SFT_EPOCHS", 2)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=float(os.environ.get("PL_SFT_LR", 5e-5)),
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_length=4096,
        gradient_checkpointing=True,
        report_to=[],
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds_train,
        peft_config=lora_config(),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"[sft-onpolicy] saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
