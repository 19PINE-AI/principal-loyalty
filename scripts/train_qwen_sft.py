"""QLoRA SFT on Qwen3-8B from principal-loyalty clean teacher traces.

Input:  data/sft_v0.jsonl (built by scripts/build_sft_dataset.py)
Output: runs/qwen_sft/ — LoRA adapter

Training objective: given PLAIN_SYSTEM + principal briefing + counterparty
turns, produce the scaffolded-teacher's assistant turns. Loss is only on
assistant tokens (SFTTrainer default with chat template masking when we
use completion-only style).

Run:
    python3 scripts/train_qwen_sft.py
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


MODEL_ID = os.environ.get("PL_MODEL_ID", "Qwen/Qwen3-8B")
ROOT = Path(__file__).resolve().parent.parent
SFT_PATH = Path(os.environ.get("PL_SFT_PATH", ROOT / "data" / "sft_v0.jsonl"))
OUT_DIR = Path(os.environ.get("PL_OUT_DIR", ROOT / "runs" / "qwen_sft"))

# Reserve a few items for eval split — training excludes these.
HELDOUT_ITEMS = {
    "pb-leak-negotbatna-01",
    "pb-capit-repq-01",
    "pb-author-testcase-01",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _merge_consecutive_roles(msgs: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages with a blank-line separator.

    Mistral's default chat template requires strict user/assistant
    alternation after an optional leading system message. Our SFT data
    has consecutive user turns (briefing + counterparty opening) which
    Qwen3's template tolerates but Mistral's doesn't. Merging is safe
    for both.
    """
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
    msgs = [{"role": "system", "content": example["system"]}]
    for m in example["messages"]:
        msgs.append({"role": m["role"], "content": m.get("content") or ""})
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

    print(f"[sft] model={MODEL_ID} data={SFT_PATH} out={OUT_DIR}")
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
    train_raw = [e for e in raw if e.get("item_id") not in HELDOUT_ITEMS]
    eval_raw = [e for e in raw if e.get("item_id") in HELDOUT_ITEMS]
    print(f"[sft] {len(raw)} total; train={len(train_raw)} eval={len(eval_raw)}")
    ds_train = Dataset.from_list([format_sft(e, tokenizer) for e in train_raw])
    ds_eval = (
        Dataset.from_list([format_sft(e, tokenizer) for e in eval_raw])
        if eval_raw else None
    )

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=int(os.environ.get("PL_SFT_EPOCHS", 3)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=float(os.environ.get("PL_SFT_LR", 2e-4)),
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if ds_eval is not None else "no",
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
        eval_dataset=ds_eval,
        peft_config=lora_config(),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"[sft] saved adapter → {OUT_DIR}")


if __name__ == "__main__":
    main()
