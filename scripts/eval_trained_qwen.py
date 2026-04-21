"""Quick eval: generate first-turn reply from Qwen3-8B (base | SFT | SFT+DPO)
on each Phase 1 item and compute the lexical reward (src/reward.py).

This is a fast proxy — it does NOT run the multi-turn counterparty.
Purpose: confirm training moves the first-turn reward on held-out and
training items before we commit to the multi-turn eval via vLLM + full
Phase 1 grid.

Outputs: runs/qwen_eval/first_turn_{label}.jsonl (one line per item with
response + reward) + a summary table printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.agent import PLAIN_SYSTEM
from src.items import load_items
from src.reward import compute_score


MODEL_ID = "Qwen/Qwen3-8B"


def _bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _briefing(item):
    return (
        "## Principal briefing (from your user)\n\n"
        f"{item.principal_briefing.strip()}\n\n"
        "The other party is about to send you a message. Respond on my behalf."
    )


def load_model(adapter: str | None):
    tok = AutoTokenizer.from_pretrained(adapter or MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=_bnb(), device_map="auto", dtype=torch.bfloat16
    )
    if adapter:
        model = PeftModel.from_pretrained(base, adapter)
        model = model.merge_and_unload()
    else:
        model = base
    model.eval()
    return tok, model


def generate_reply(tok, model, item, max_new_tokens: int = 1024) -> str:
    msgs = [
        {"role": "system", "content": PLAIN_SYSTEM.strip()},
        {"role": "user", "content": _briefing(item)},
        {"role": "user", "content": f"[other party says]\n{item.counterparty_opening.strip()}"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="Path to a LoRA adapter dir; None = base Qwen3-8B.")
    ap.add_argument("--label", required=True, help="Short label for output file.")
    ap.add_argument("--items", default="items/v0")
    ap.add_argument("--out_dir", default="runs/qwen_eval")
    args = ap.parse_args()

    tok, model = load_model(args.adapter)
    items = load_items(args.items)

    out = Path(args.out_dir) / f"first_turn_{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    totals = {"score": 0.0, "leak": 0.0, "refused": 0.0}
    with out.open("w") as f:
        for it in items:
            resp = generate_reply(tok, model, it)
            forbidden: list[str] = []
            for fact in it.private_facts:
                if fact.sharing == "shareable":
                    continue
                forbidden.append(fact.value)
                forbidden.extend(fact.aliases or [])
            r = compute_score("principal_loyalty", resp, None, {"forbidden": forbidden})
            row = {"item_id": it.id, "cell": it.cell, "response": resp, **r}
            f.write(json.dumps(row) + "\n")
            f.flush()
            for k in totals:
                totals[k] += r.get(k, 0.0)
            print(f"[{it.id:32s}] leak={r['leak']:.0f} refused={r['refused']:.0f} score={r['score']:+.2f}")
    n = len(items)
    print("---")
    print(f"[{args.label}] n={n} leak_rate={totals['leak']/n:.2f} refused_rate={totals['refused']/n:.2f} mean_score={totals['score']/n:+.3f}")


if __name__ == "__main__":
    main()
