"""Phase D comparison: DAPO-v1 step_35 (n=31 train) vs DAPO-v1 step_55 (n=46 train).

Tests §4.4.2's "the 25/108 ceiling is data-coverage-bounded" claim.
If n=46 training breaks below 25/108 with the same reward and same eval grid,
data expansion is the right next-step intervention.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


RUNS = [
    ("v4.1 (no DAPO)",          "runs/phase2_trained_v4_1/scored.jsonl"),
    ("DAPO-v1 step_35 (n=31)",  "runs/phase3_dapo_v1_step35/scored.jsonl"),
    ("DAPO-v1 n=51 step_55",    "runs/phase3_dapo_v1_n51_step55/scored.jsonl"),
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


def item_diff(a, b):
    """Items that improved (a fired, b clean) and regressed (a clean, b fired)."""
    def key(r):
        return (r["item_id"], r["arm"])
    map_a = {key(r): r for r in a}
    map_b = {key(r): r for r in b}
    shared = sorted(set(map_a) & set(map_b))
    improved, regressed = [], []
    for k in shared:
        ha = harm_fire(map_a[k].get("harm") or {})
        hb = harm_fire(map_b[k].get("harm") or {})
        if ha and not hb:
            improved.append(k)
        elif hb and not ha:
            regressed.append(k)
    return improved, regressed


def main():
    data = {}
    raw = {}
    for label, path in RUNS:
        rows = load(path)
        data[label] = {"all": summ(rows), "cell": cell(rows)}
        raw[label] = rows
        print(f"{label:<30}: loaded {len(rows)} rows from {path}")

    print()
    print("=" * 100)
    print("HEADLINE — does n=46 training break the 25/108 ceiling?")
    print("=" * 100)
    for arm in ARMS:
        print(f"\n{arm}:")
        print(f"  {'run':<30} | {'leak':>7} | {'harm':>9} | {'MI':>6} | {'bound':>6}")
        for label in [l for l, _ in RUNS]:
            s = data[label]["all"][arm]
            print(f"  {label:<30} | {s['leak']*100:>6.1f}% | {s['harm']:>3}/{s['n']:<5} | {s['mi']:>3}/{s['n']:<2} | {s['bound']:>6}")

    print()
    print("=" * 100)
    print("TOTAL across 108 rollouts")
    print("=" * 100)
    print(f"{'run':<30} | {'total harm':<13} | {'total MI':<10} | {'bound':<7}")
    for label in [l for l, _ in RUNS]:
        s = data[label]["all"]
        th = sum(s[a]["harm"] for a in ARMS)
        tm = sum(s[a]["mi"] for a in ARMS)
        tb = sum(s[a]["bound"] for a in ARMS)
        nt = sum(s[a]["n"] for a in ARMS)
        print(f"{label:<30} | {th}/{nt:<9} | {tm}/{nt:<7} | {tb:<7}")

    print()
    print("=" * 100)
    print("PER-CELL HARM_FIRE")
    print("=" * 100)
    cells = sorted({c for l in data for c in data[l]["cell"]})
    header = f"{'cell':<12} | " + " | ".join(f"{l:<30}" for l, _ in RUNS)
    print(header)
    for c in cells:
        parts = []
        for label in [l for l, _ in RUNS]:
            hf, n = data[label]["cell"].get(c, (0, 0))
            parts.append(f"{hf}/{n}")
        print(f"{c:<12} | " + " | ".join(f"{p:<30}" for p in parts))

    if "DAPO-v1 step_35 (n=31)" in raw and "DAPO-v1 n=51 step_55" in raw:
        improved, regressed = item_diff(raw["DAPO-v1 step_35 (n=31)"], raw["DAPO-v1 n=51 step_55"])
        print()
        print("=" * 100)
        print(f"ITEM-LEVEL DELTA n=31 → n=51:  improvements={len(improved)}  regressions={len(regressed)}")
        print("=" * 100)
        if improved:
            print("Improvements (n=31 fired -> n=51 clean):")
            for iid, arm in improved:
                print(f"  + {iid:<32} {arm}")
        if regressed:
            print("Regressions (n=31 clean -> n=51 fired):")
            for iid, arm in regressed:
                print(f"  - {iid:<32} {arm}")


if __name__ == "__main__":
    main()
