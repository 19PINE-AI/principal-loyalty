"""Build SFT dataset from clean teacher traces.

Input:  data/teacher_traces.jsonl (from scripts/generate_teacher_traces.py)
        Each line = one clean scaffolded trajectory (zero leaks, all
        commitments held, no harm-floor fire).

Output: data/sft_v0.jsonl — one line per trajectory with:
        {
          "item_id": "...",
          "cell": "...",
          "system": <PLAIN_SYSTEM>,   # student sees the minimal prompt
          "messages": [
              {"role": "user",      "content": "## Principal briefing ..."},
              {"role": "user",      "content": "[other party says] ..."},
              {"role": "assistant", "content": "<teacher reply>"},
              ...
          ]
        }

Training intent: student learns to behave like the scaffolded teacher
(zero leaks, holds positions, signals posture) using only the PLAIN
system prompt at inference — i.e., loyalty becomes a learned disposition
rather than a runtime scaffold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import PLAIN_SYSTEM  # noqa: E402
from src.items import load_items  # noqa: E402


def _briefing_user_turn(principal_briefing: str) -> str:
    return (
        "## Principal briefing (from your user)\n\n"
        f"{principal_briefing.strip()}\n\n"
        "The other party is about to send you a message. Respond on my behalf."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/teacher_traces.jsonl")
    ap.add_argument("--items", default="items/v0")
    ap.add_argument("--out", default="data/sft_v0.jsonl")
    args = ap.parse_args()

    items = {it.id: it for it in load_items(args.items)}
    traces = [json.loads(l) for l in Path(args.traces).read_text().splitlines() if l.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("w") as f:
        for tr in traces:
            item = items.get(tr["item_id"])
            if item is None:
                continue
            messages: list[dict] = [
                {"role": "user", "content": _briefing_user_turn(item.principal_briefing)}
            ]
            for t in tr["turns"]:
                if t["speaker"] == "counterparty":
                    messages.append({"role": "user", "content": f"[other party says]\n{t['text'].strip()}"})
                elif t["speaker"] == "agent":
                    messages.append({"role": "assistant", "content": t["text"].strip()})
            # Only keep traces with at least one assistant turn.
            if not any(m["role"] == "assistant" and m["content"] for m in messages):
                continue
            example = {
                "item_id": item.id,
                "cell": item.cell,
                "subcell": item.subcell,
                "system": PLAIN_SYSTEM.strip(),
                "messages": messages,
            }
            f.write(json.dumps(example) + "\n")
            n_written += 1

    print(f"[sft] wrote {n_written} examples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
