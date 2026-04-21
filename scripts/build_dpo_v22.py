"""Concatenate v2 DPO pairs + author-to-principal produce pairs -> data/dpo_v22.jsonl.

v22 = v2 (v1-lite + probe-gated authoring MT) + reader_is_principal produce pairs.
Target: fix the authoring-to-principal over-refusal documented in the n=24 probe
without regressing the authoring-cell refuse register that v2 gets right.
"""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parts = [
        root / "data" / "dpo_v2.jsonl",
        root / "data" / "dpo_author_to_principal.jsonl",
    ]
    out = root / "data" / "dpo_v22.jsonl"
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
