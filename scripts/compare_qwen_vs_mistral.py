"""Cross-family replication test: Qwen3-8B vs Mistral-7B-Instruct-v0.3 SFT+DPO.

Same teacher traces (data/sft_v4.jsonl), same DPO pairs (data/dpo_v4_combined.jsonl),
same hyperparameters, same eval grid. Tests whether the SFT→DPO recipe transfers
across base model families — addressing §6's "single base model" limitation.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


RUNS = [
    ("Qwen3-8B base",          "runs/phase3_baseline_qwen/scored.jsonl"),
    ("Qwen3-8B v4.1 SFT+DPO",  "runs/phase2_trained_v4_1/scored.jsonl"),
    ("Qwen3-8B DAPO-v1",       "runs/phase3_dapo_v1_step35/scored.jsonl"),
    ("Mistral v4.1 SFT+DPO",   "runs/phase3_mistral_sft_dpo/scored.jsonl"),
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
    print("=" * 110)
    print("HEADLINE — does the SFT→DPO recipe transfer Qwen → Mistral?")
    print("=" * 110)
    for arm in ARMS:
        print(f"\n{arm}:")
        print(f"  {'run':<28} | {'leak':>7} | {'harm':>9} | {'MI':>6} | {'bound':>6}")
        for label in [l for l, _ in RUNS]:
            s = data[label]["all"][arm]
            print(f"  {label:<28} | {s['leak']*100:>6.1f}% | {s['harm']:>3}/{s['n']:<5} | {s['mi']:>3}/{s['n']:<2} | {s['bound']:>6}")

    print()
    print("=" * 110)
    print("TOTAL across 108 rollouts")
    print("=" * 110)
    print(f"{'run':<28} | {'total harm':<13} | {'total MI':<10} | {'bound':<7}")
    for label in [l for l, _ in RUNS]:
        s = data[label]["all"]
        th = sum(s[a]["harm"] for a in ARMS)
        tm = sum(s[a]["mi"] for a in ARMS)
        tb = sum(s[a]["bound"] for a in ARMS)
        nt = sum(s[a]["n"] for a in ARMS)
        print(f"{label:<28} | {th}/{nt:<9} | {tm}/{nt:<7} | {tb:<7}")

    print()
    print("=" * 110)
    print("PER-CELL HARM_FIRE")
    print("=" * 110)
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
