"""Compare step_35 harm_fire/leak across claude-sonnet vs gpt-5 counterparty.

If the leak/MI frontier is a Claude-specific dialogue artifact, swapping the
other party should flip the numbers. If it's structural, numbers should land
in the same band.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


RUNS = [
    ("cp=claude-sonnet",  "runs/phase3_dapo_v1_step35/scored.jsonl"),
    ("cp=gpt-5",          "runs/phase3_dapo_v1_step35_gpt5cp/scored.jsonl"),
    ("cp=gemini-3-flash", "runs/phase3_dapo_v1_step35_gemcp/scored.jsonl"),
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
        print(f"{label}: loaded {len(rows)} rows")

    print()
    print("=" * 96)
    print("PER-ARM HEADLINE — leak / harm_fire / missed_instruction")
    print("=" * 96)
    for arm in ARMS:
        print(f"\n{arm}:")
        print(f"  {'run':<22} | {'leak':>7} | {'harm':>8} | {'MI':>8} | {'bound':>6}")
        for label in [l for l, _ in RUNS]:
            s = data[label]["all"][arm]
            print(f"  {label:<22} | {s['leak']*100:>6.1f}% | {s['harm']:>3}/{s['n']:<4} | {s['mi']:>3}/{s['n']:<4} | {s['bound']:>6}")

    print()
    print("=" * 96)
    print("TOTAL across all 108 rollouts")
    print("=" * 96)
    print(f"{'run':<22} | {'total harm':<12} | {'total MI':<10} | {'total bound':<11}")
    for label in [l for l, _ in RUNS]:
        s = data[label]["all"]
        th = sum(s[a]["harm"] for a in ARMS)
        tm = sum(s[a]["mi"] for a in ARMS)
        tb = sum(s[a]["bound"] for a in ARMS)
        print(f"{label:<22} | {th}/108{'':<6} | {tm}/108{'':<4} | {tb:<11}")

    print()
    print("=" * 96)
    print("PER-CELL HARM — counterparty swap robustness")
    print("=" * 96)
    cells = sorted({c for l in data for c in data[l]["cell"]})
    header = f"{'cell':<12} | " + " | ".join(f"{l:<22}" for l, _ in RUNS)
    print(header)
    for c in cells:
        parts = []
        for label in [l for l, _ in RUNS]:
            hf, n = data[label]["cell"].get(c, (0, 0))
            parts.append(f"{hf}/{n}")
        print(f"{c:<12} | " + " | ".join(f"{p:<22}" for p in parts))


if __name__ == "__main__":
    main()
