"""Compare Phase 2 v4 vs Phase 2.1 v4.1 results.

Phase 2 v4   — SFT+DPO on original clean v4 teacher traces. Leak collapsed
              to historic low; missed_instruction regressed (posture-collapse
              against reader-is-principal probe items and cooperative sanity).
Phase 2.1    — additional DPO pass with 66 targeted missed_instruction pairs
              on top of SFT-v4. Hypothesis: walks the frontier back toward
              following principal instructions without giving up leak gains.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


RUNS = [
    ("v4",    "runs/phase2_trained_v4/scored.jsonl"),
    ("v4.1",  "runs/phase2_trained_v4_1/scored.jsonl"),
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
        holds = [r.get("commitment_hold_rate", 0.0) for r in rs]
        mi = sum(1 for r in rs if (r.get("harm") or {}).get("missed_instruction"))
        hf = sum(1 for r in rs if harm_fire(r.get("harm") or {}))
        bound = sum(1 for r in rs if (r.get("harm") or {}).get("leaked_private_bound"))
        pi = sum(1 for r in rs if (r.get("harm") or {}).get("principal_instructed_misrepresentation"))
        per[a] = {
            "n": n,
            "leak": mean(leaks),
            "hold": mean(holds),
            "mi": mi,
            "mi_rate": mi/n,
            "harm": hf,
            "harm_rate": hf/n,
            "bound": bound,
            "pi": pi,
        }
    return per


def probe_cut(rows: list[dict]) -> dict:
    probe = [r for r in rows if "to-principal" in r["item_id"]]
    nonprobe = [r for r in rows if "to-principal" not in r["item_id"]]
    return {"probe": summ(probe), "nonprobe": summ(nonprobe)}


def fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x*100:5.1f}%"


def main() -> None:
    data = {}
    for label, path in RUNS:
        rows = load(path)
        if not rows:
            print(f"[skip] {label}: no rows at {path}")
            continue
        data[label] = {"all": summ(rows), **probe_cut(rows)}
        print(f"{label}: loaded {len(rows)} rows")

    print()
    print("=" * 96)
    print("LEAK_RATE (lower=better) — mean per arm")
    print("=" * 96)
    print(f"{'version':<8} | {'plain':<10} | {'prompted':<10} | {'scaffolded':<10}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        print(f"{label:<8} | {fmt_pct(s['plain']['leak']):<10} | {fmt_pct(s['prompted']['leak']):<10} | {fmt_pct(s['scaffolded']['leak']):<10}")

    print()
    print("=" * 96)
    print("MISSED_INSTRUCTION count / n (lower=better)")
    print("=" * 96)
    print(f"{'version':<8} | {'plain':<14} | {'prompted':<14} | {'scaffolded':<14}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        row = [f"{s[a]['mi']}/{s[a]['n']} ({fmt_pct(s[a]['mi_rate']).strip()})" for a in ARMS]
        print(f"{label:<8} | {row[0]:<14} | {row[1]:<14} | {row[2]:<14}")

    print()
    print("=" * 96)
    print("HARM_FIRE count / n (lower=better)")
    print("=" * 96)
    print(f"{'version':<8} | {'plain':<14} | {'prompted':<14} | {'scaffolded':<14}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        row = [f"{s[a]['harm']}/{s[a]['n']} ({fmt_pct(s[a]['harm_rate']).strip()})" for a in ARMS]
        print(f"{label:<8} | {row[0]:<14} | {row[1]:<14} | {row[2]:<14}")

    print()
    print("=" * 96)
    print("PROBE-ITEM CUT (reader-is-principal) — MI count / n")
    print("=" * 96)
    print(f"{'version':<8} | {'probe plain':<14} | {'probe prompt':<14} | {'probe scaffo':<14} | {'nonprobe MI (all arms)':<28}")
    for label in [l for l, _ in RUNS if l in data]:
        p = data[label]["probe"]
        np_ = data[label]["nonprobe"]
        probe_row = [f"{p[a]['mi']}/{p[a]['n']}" if p[a]['n'] else "-/0" for a in ARMS]
        np_total_mi = sum(np_[a].get("mi", 0) for a in ARMS)
        np_total_n = sum(np_[a].get("n", 0) for a in ARMS)
        print(f"{label:<8} | {probe_row[0]:<14} | {probe_row[1]:<14} | {probe_row[2]:<14} | {np_total_mi}/{np_total_n}")

    print()
    print("=" * 96)
    print("LEAKED_PRIVATE_BOUND + PRINCIPAL_INSTRUCTED_MISREP (other harm flags)")
    print("=" * 96)
    print(f"{'version':<8} | {'bound (all)':<14} | {'pi_misrep (all)':<16}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        total_b = sum(s[a].get("bound", 0) for a in ARMS)
        total_pi = sum(s[a].get("pi", 0) for a in ARMS)
        print(f"{label:<8} | {total_b:<14} | {total_pi:<16}")


if __name__ == "__main__":
    main()
