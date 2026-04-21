"""Rescore eval_trained_qwen.py outputs with <think>...</think> stripped.

Qwen3-8B emits a <think> block (internal reasoning) before the reply.
That block is never sent to the counterparty, so it should not count as
a leak. The original eval scored the full response; here we strip the
thinking block and recompute.

Inputs: runs/qwen_eval/first_turn_{label}.jsonl
Writes: runs/qwen_eval/first_turn_{label}_stripped.jsonl (+ summary)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.items import load_items
from src.reward import compute_score


THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def strip_think(text: str) -> str:
    cleaned = THINK_RE.sub("", text).strip()
    return cleaned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--items", default="items/v0")
    ap.add_argument("--in_dir", default="runs/qwen_eval")
    args = ap.parse_args()

    items_by_id = {it.id: it for it in load_items(args.items)}
    inp = Path(args.in_dir) / f"first_turn_{args.label}.jsonl"
    outp = Path(args.in_dir) / f"first_turn_{args.label}_stripped.jsonl"

    totals = {"score": 0.0, "leak": 0.0, "refused": 0.0}
    n = 0
    with outp.open("w") as f:
        for line in inp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            it = items_by_id[row["item_id"]]
            forbidden: list[str] = []
            for fact in it.private_facts:
                if fact.sharing == "shareable":
                    continue
                forbidden.append(fact.value)
                forbidden.extend(fact.aliases or [])
            stripped = strip_think(row["response"])
            r = compute_score("principal_loyalty", stripped, None, {"forbidden": forbidden})
            out = {
                "item_id": row["item_id"],
                "cell": row["cell"],
                "response_stripped": stripped,
                **r,
            }
            f.write(json.dumps(out) + "\n")
            for k in totals:
                totals[k] += r.get(k, 0.0)
            n += 1
            print(f"[{row['item_id']:32s}] leak={r['leak']:.0f} refused={r['refused']:.0f} score={r['score']:+.2f}")
    print("---")
    print(f"[{args.label}] n={n} leak_rate={totals['leak']/n:.2f} refused_rate={totals['refused']/n:.2f} mean_score={totals['score']/n:+.3f}")


if __name__ == "__main__":
    main()
