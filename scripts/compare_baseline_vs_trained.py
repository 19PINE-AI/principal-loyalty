"""Phase B baseline comparison: BASE Qwen3-8B (no training) vs DAPO-v1 step_35.

Tests the §6 reviewer objection: 'what if just the scaffolded sentinel
prompt without any training does most of the work?' If baseline-scaffolded
already gets close to DAPO-v1, training contributes little. If not,
training is essential.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


RUNS = [
    ("baseline (no training)", "runs/phase3_baseline_qwen/scored.jsonl"),
    ("v4.1 (SFT+DPO)",         "runs/phase2_trained_v4_1/scored.jsonl"),
    ("dapo-v1 (best trained)", "runs/phase3_dapo_v1_step35/scored.jsonl"),
]
ARMS = ["plain", "prompted", "scaffolded"]


def load(p):
    return [json.loads(l) for l in Path(p).open()]


def harm_fire(h):
    if "harm_fire" in h and isinstance(h["harm_fire"], bool):
        return h["harm_fire"]
    return any(bool(h.get(k, False)) for k in (
        "fabrication", "deception", "leaked_private_bound",
        "missed_instruction", "third_party_harm",
    ))


def summ(rows):
    out = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        n = len(rs)
        leaks = [r.get("leak_rate", 0.0) for r in rs]
        mi = sum(1 for r in rs if (r.get("harm") or {}).get("missed_instruction"))
        hf = sum(1 for r in rs if harm_fire(r.get("harm") or {}))
        bound = sum(1 for r in rs if (r.get("harm") or {}).get("leaked_private_bound"))
        out[a] = {"n": n, "leak": mean(leaks) if leaks else 0, "mi": mi, "harm": hf, "bound": bound}
    return out


def cell(rows):
    by = {}
    for r in rows:
        c = r.get("cell") or r["item_id"].split("-")[1]
        by.setdefault(c, []).append(r)
    return {c: (sum(1 for r in rs if harm_fire(r.get("harm") or {})), len(rs))
            for c, rs in by.items()}


def main():
    data = {}
    for label, path in RUNS:
        rows = load(path)
        data[label] = {"all": summ(rows), "cell": cell(rows)}
        print(f"{label:<28}: loaded {len(rows)} rows from {path}")

    print()
    print("=" * 100)
    print("HEADLINE — leak / harm by arm (does the scaffolded sentinel suffice without training?)")
    print("=" * 100)
    for arm in ARMS:
        print(f"\n{arm}:")
        print(f"  {'run':<28} | {'leak':>7} | {'harm':>8} | {'MI':>6} | {'bound':>6}")
        for label in [l for l, _ in RUNS]:
            s = data[label]["all"][arm]
            print(f"  {label:<28} | {s['leak']*100:>6.1f}% | {s['harm']:>3}/{s['n']:<4} | {s['mi']:>3}/{s['n']:<2} | {s['bound']:>6}")

    print()
    print("=" * 100)
    print("TOTAL across 108 rollouts (baseline = none, training = some)")
    print("=" * 100)
    print(f"{'run':<28} | {'total harm':<12} | {'total MI':<10} | {'bound':<7}")
    for label in [l for l, _ in RUNS]:
        s = data[label]["all"]
        th = sum(s[a]["harm"] for a in ARMS)
        tm = sum(s[a]["mi"] for a in ARMS)
        tb = sum(s[a]["bound"] for a in ARMS)
        nt = sum(s[a]["n"] for a in ARMS)
        print(f"{label:<28} | {th}/{nt:<8} | {tm}/{nt:<6} | {tb:<7}")

    print()
    print("=" * 100)
    print("PER-CELL HARM_FIRE — does training help all cells or just specific ones?")
    print("=" * 100)
    cells = sorted({c for l in data for c in data[l]["cell"]})
    header = f"{'cell':<12} | " + " | ".join(f"{l:<28}" for l, _ in RUNS)
    print(header)
    for c in cells:
        parts = []
        for label in [l for l, _ in RUNS]:
            hf, n = data[label]["cell"].get(c, (0, 0))
            parts.append(f"{hf}/{n}")
        print(f"{c:<12} | " + " | ".join(f"{p:<28}" for p in parts))


if __name__ == "__main__":
    main()
