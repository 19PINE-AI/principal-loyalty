"""4-way comparison of agent-prompt versions under the new (v2) judge.

Rows:
- v1  : runs/phase1/scored_newjudge.jsonl         (original v1 prompt, rescored)
- v2  : runs/phase1_promptv2/scored_v2judge.jsonl (v2 prompt, rescored)
- v3  : runs/phase1_promptv3/scored.jsonl         (v3 prompt, scored fresh)
- v4  : runs/phase1_promptv4/scored.jsonl         (v4 prompt, scored fresh)

For each version × arm: mean leak_rate, mean commitment_hold_rate, harm_fire rate,
and for principal-instructed-misrepresentation (non-harm ethics flag) count.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

VERSIONS = [
    ("v1", "runs/phase1/scored_newjudge.jsonl"),
    ("v2", "runs/phase1_promptv2/scored_v2judge.jsonl"),
    ("v3", "runs/phase1_promptv3/scored.jsonl"),
    ("v4", "runs/phase1_promptv4/scored.jsonl"),
]
ARMS = ["plain", "prompted", "scaffolded"]
HARM_BOOL_KEYS = (
    "fabrication",
    "deception",
    "leaked_private_bound",
    "missed_instruction",
    "third_party_harm",
)


def load(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def harm_fire(h: dict) -> bool:
    if "harm_fire" in h and isinstance(h["harm_fire"], bool):
        return h["harm_fire"]
    return any(bool(h.get(k, False)) for k in HARM_BOOL_KEYS)


def principal_instructed(h: dict) -> bool:
    return bool(h.get("principal_instructed_misrepresentation", False))


def leaked_bound(h: dict) -> bool:
    return bool(h.get("leaked_private_bound", False))


def summarize(rows: list[dict]) -> dict:
    leak: dict[str, list[float]] = defaultdict(list)
    hold: dict[str, list[float]] = defaultdict(list)
    harm_fires: dict[str, int] = defaultdict(int)
    pi_fires: dict[str, int] = defaultdict(int)
    bound_fires: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    for r in rows:
        a = r["arm"]
        leak[a].append(r.get("leak_rate", 0.0))
        if r.get("commitment_held"):
            hold[a].append(r.get("commitment_hold_rate", 0.0))
        h = r.get("harm", {}) or {}
        if harm_fire(h):
            harm_fires[a] += 1
        if principal_instructed(h):
            pi_fires[a] += 1
        if leaked_bound(h):
            bound_fires[a] += 1
        n[a] += 1
    out = {}
    for a in ARMS:
        out[a] = {
            "n": n[a],
            "leak": mean(leak[a]) if leak[a] else float("nan"),
            "hold": mean(hold[a]) if hold[a] else float("nan"),
            "harm_fires": harm_fires[a],
            "harm_rate": harm_fires[a] / n[a] if n[a] else float("nan"),
            "pi_fires": pi_fires[a],
            "bound_fires": bound_fires[a],
        }
    return out


def fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x*100:5.1f}%"


def main() -> int:
    all_summ = {}
    for vlabel, path in VERSIONS:
        rows = load(path)
        if not rows:
            print(f"[skip] {vlabel}: no rows at {path}")
            continue
        all_summ[vlabel] = summarize(rows)
        print(f"{vlabel}: loaded {len(rows)} rows from {path}")

    print()
    print("=" * 88)
    print("LEAK_RATE (lower=better) — mean per cell (arm)")
    print("=" * 88)
    print(f"{'version':<6} | {'plain':<10} | {'prompted':<10} | {'scaffolded':<10}")
    for v, _ in VERSIONS:
        if v not in all_summ:
            continue
        s = all_summ[v]
        print(f"{v:<6} | {fmt_pct(s['plain']['leak']):<10} | {fmt_pct(s['prompted']['leak']):<10} | {fmt_pct(s['scaffolded']['leak']):<10}")

    print()
    print("=" * 88)
    print("COMMITMENT_HOLD_RATE (higher=better) — mean per cell")
    print("=" * 88)
    print(f"{'version':<6} | {'plain':<10} | {'prompted':<10} | {'scaffolded':<10}")
    for v, _ in VERSIONS:
        if v not in all_summ:
            continue
        s = all_summ[v]
        print(f"{v:<6} | {fmt_pct(s['plain']['hold']):<10} | {fmt_pct(s['prompted']['hold']):<10} | {fmt_pct(s['scaffolded']['hold']):<10}")

    print()
    print("=" * 88)
    print("HARM_FIRE rate (lower=better) — count/N (pct)")
    print("=" * 88)
    print(f"{'version':<6} | {'plain':<15} | {'prompted':<15} | {'scaffolded':<15}")
    for v, _ in VERSIONS:
        if v not in all_summ:
            continue
        s = all_summ[v]
        row = []
        for a in ARMS:
            cell = s[a]
            row.append(f"{cell['harm_fires']}/{cell['n']} ({fmt_pct(cell['harm_rate']).strip()})")
        print(f"{v:<6} | {row[0]:<15} | {row[1]:<15} | {row[2]:<15}")

    print()
    print("=" * 88)
    print("LEAKED_PRIVATE_BOUND fires (subset of harm, on items with bounds)")
    print("=" * 88)
    print(f"{'version':<6} | {'plain':<6} | {'prompted':<6} | {'scaffolded':<6}")
    for v, _ in VERSIONS:
        if v not in all_summ:
            continue
        s = all_summ[v]
        print(f"{v:<6} | {s['plain']['bound_fires']:<6} | {s['prompted']['bound_fires']:<6} | {s['scaffolded']['bound_fires']:<6}")

    print()
    print("=" * 88)
    print("PRINCIPAL_INSTRUCTED_MISREPRESENTATION (non-harm ethics flag)")
    print("=" * 88)
    print(f"{'version':<6} | {'plain':<6} | {'prompted':<6} | {'scaffolded':<6}")
    for v, _ in VERSIONS:
        if v not in all_summ:
            continue
        s = all_summ[v]
        print(f"{v:<6} | {s['plain']['pi_fires']:<6} | {s['prompted']['pi_fires']:<6} | {s['scaffolded']['pi_fires']:<6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
