"""Four-way Phase 3 comparison: v4.1 / DAPO-v1 / DAPO-v2 / DAPO-leakonly.

Answers:
  - Did v2's oversampled sanity + tuned reward close the DAPO-v1 residuals?
  - Does leak-only reward posture-collapse sanity, validating §4.4's
    "cooperative branch is load-bearing" ablation claim?
  - What's the best single checkpoint for the paper's headline?

Counterparty = claude-sonnet across all four (GPT-5 counterparty check is a
separate script, compare_phase3_counterparty_swap.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean


STEP_V2 = os.environ.get("PL_V2_STEP", "35")
STEP_LO = os.environ.get("PL_LO_STEP", "35")

STEP_V3 = os.environ.get("PL_V3_STEP", "30")

RUNS = [
    ("v4.1",       "runs/phase2_trained_v4_1/scored.jsonl"),
    ("dapo-v1",    "runs/phase3_dapo_v1_step35/scored.jsonl"),
    ("dapo-v2",    f"runs/phase3_dapo_v2_step{STEP_V2}/scored.jsonl"),
    ("leak-only",  f"runs/phase3_dapo_leakonly_step{STEP_LO}/scored.jsonl"),
    ("dapo-v3",    f"runs/phase3_dapo_v3_step{STEP_V3}/scored.jsonl"),
]
ARMS = ["plain", "prompted", "scaffolded"]


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def harm_fire(h: dict) -> bool:
    if "harm_fire" in h and isinstance(h["harm_fire"], bool):
        return h["harm_fire"]
    return any(bool(h.get(k, False)) for k in (
        "fabrication", "deception", "leaked_private_bound",
        "missed_instruction", "third_party_harm",
    ))


def summ(rows: list[dict]) -> dict:
    per: dict[str, dict] = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            per[a] = {"n": 0}
            continue
        n = len(rs)
        leaks = [r.get("leak_rate", 0.0) for r in rs]
        mi = sum(1 for r in rs if (r.get("harm") or {}).get("missed_instruction"))
        hf = sum(1 for r in rs if harm_fire(r.get("harm") or {}))
        bound = sum(1 for r in rs if (r.get("harm") or {}).get("leaked_private_bound"))
        per[a] = {
            "n": n, "leak": mean(leaks), "mi": mi,
            "mi_rate": mi/n, "harm": hf, "bound": bound,
        }
    return per


def cell_harm(rows: list[dict]) -> dict:
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("cell") or r["item_id"].split("-")[1]
        by_cell.setdefault(c, []).append(r)
    return {c: (sum(1 for r in rs if harm_fire(r.get("harm") or {})), len(rs))
            for c, rs in by_cell.items()}


def main() -> None:
    data = {}
    for label, path in RUNS:
        rows = load(path)
        if not rows:
            print(f"[skip] {label}: no rows at {path}")
            continue
        data[label] = {"all": summ(rows), "cell": cell_harm(rows)}
        print(f"{label:<12} loaded {len(rows)} rows from {path}")

    print()
    print("=" * 100)
    print("HEADLINE — total harm_fire + leak per arm")
    print("=" * 100)
    print(f"{'run':<12} | {'plain leak':<11} | {'prompted leak':<13} | {'scaff leak':<11} | {'total harm':<12} | {'bound':<6}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        total_harm = sum(s[a].get("harm", 0) for a in ARMS)
        total_bound = sum(s[a].get("bound", 0) for a in ARMS)
        total_n = sum(s[a].get("n", 0) for a in ARMS)
        print(f"{label:<12} | "
              f"{s['plain']['leak']*100:>8.1f}%   | "
              f"{s['prompted']['leak']*100:>10.1f}%   | "
              f"{s['scaffolded']['leak']*100:>8.1f}%   | "
              f"{total_harm:>3}/{total_n:<7} | "
              f"{total_bound:<6}")

    print()
    print("=" * 100)
    print("PER-CELL harm_fire")
    print("=" * 100)
    cells = sorted({c for label in data for c in data[label]["cell"]})
    header = f"{'cell':<12} | " + " | ".join(f"{l:<12}" for l, _ in RUNS if l in data)
    print(header)
    for c in cells:
        parts = []
        for label in [l for l, _ in RUNS if l in data]:
            hf, n = data[label]["cell"].get(c, (0, 0))
            parts.append(f"{hf}/{n}")
        print(f"{c:<12} | " + " | ".join(f"{p:<12}" for p in parts))

    print()
    print("=" * 100)
    print("MISSED_INSTRUCTION per arm (lower=better)")
    print("=" * 100)
    print(f"{'run':<12} | {'plain':<10} | {'prompted':<10} | {'scaffolded':<10}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        row = [f"{s[a]['mi']}/{s[a]['n']}" for a in ARMS]
        print(f"{label:<12} | {row[0]:<10} | {row[1]:<10} | {row[2]:<10}")


if __name__ == "__main__":
    main()
