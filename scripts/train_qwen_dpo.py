"""DPO fine-tune on Qwen3-8B SFT-adapted checkpoint using principal-loyalty pairs.

Input:  data/dpo_v0.jsonl (preferred = scaffolded-clean; rejected = plain
        with leak/concession/harm)
Output: runs/qwen_dpo/ — LoRA adapter layered on runs/qwen_sft/

For the small-data regime (~35 pairs), TRL's DPOTrainer is the right
tool. If we scale DPO pairs substantially later, we can swap to DAPO via
verl using the same preference source (see scripts/build_verl_dataset.py
+ src/reward.py).

Run (after SFT):
    python3 scripts/train_qwen_dpo.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


MODEL_ID = os.environ.get("PL_MODEL_ID", "Qwen/Qwen3-8B")
ROOT = Path(__file__).resolve().parent.parent
DPO_PATH = Path(os.environ.get("PL_DPO_PATH", ROOT / "data" / "dpo_v0.jsonl"))
SFT_DIR = Path(os.environ.get("PL_SFT_DIR", ROOT / "runs" / "qwen_sft"))
OUT_DIR = Path(os.environ.get("PL_DPO_DIR", ROOT / "runs" / "qwen_dpo"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def format_dpo(example: dict, tokenizer) -> dict:
    prompt_msgs = [
        {"role": "system", "content": example.get("system", "You are a helpful assistant.")},
        {"role": "user", "content": example["prompt"]},
    ]
    prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[dpo] model={MODEL_ID} sft_adapter={SFT_DIR} data={DPO_PATH} out={OUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_DIR if SFT_DIR.exists() else MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config(),
        device_map="auto",
        dtype=torch.bfloat16,
    )
    peft_config = None
    if SFT_DIR.exists():
        model = PeftModel.from_pretrained(base, str(SFT_DIR), is_trainable=True)
    else:
        print(f"[dpo] no prior SFT adapter at {SFT_DIR}; attaching a fresh LoRA for DPO on the quantized base.")
        model = base
        # When the base is 4-bit quantized, DPOTrainer needs a LoRA peft_config
        # to attach trainable adapters; can't fine-tune a purely quantized model.
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    raw = load_jsonl(DPO_PATH)
    print(f"[dpo] {len(raw)} preference pairs")
    ds = Dataset.from_list([format_dpo(e, tokenizer) for e in raw])

    cfg = DPOConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=int(os.environ.get("PL_DPO_EPOCHS", 3)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=float(os.environ.get("PL_DPO_LR", 5e-6)),
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        max_length=2048,
        beta=0.1,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer_kwargs = dict(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"[dpo] saved adapter -> {OUT_DIR}")


if __name__ == "__main__":
    main()
