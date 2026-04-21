"""Concatenate v1-lite DPO pairs + v2 authoring MT pairs -> data/dpo_v2.jsonl."""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parts = [
        root / "data" / "dpo_v1_lite.jsonl",
        root / "data" / "dpo_multiturn_v2_authoring.jsonl",
    ]
    out = root / "data" / "dpo_v2.jsonl"
    n = 0
    with out.open("w") as fout:
        for p in parts:
            if not p.exists():
                print(f"[build] missing {p}; skipping")
                continue
            text = p.read_text()
            fout.write(text if text.endswith("\n") else text + "\n")
            n += sum(1 for l in text.splitlines() if l.strip())
            print(f"[build] + {p.name}")
    print(f"[build] wrote {n} pairs -> {out}")


if __name__ == "__main__":
    main()
