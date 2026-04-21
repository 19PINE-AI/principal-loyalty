"""Compare DPO v0 (first-turn only) vs DPO v1 (first-turn + multi-turn) vs baseline."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def bootstrap_ci(xs, n=1000, alpha=0.05):
    if not xs: return (0, 0)
    rng = random.Random(0)
    ms = []
    k = len(xs)
    for _ in range(n):
        s = [xs[rng.randrange(k)] for _ in range(k)]
        ms.append(sum(s) / k)
    ms.sort()
    return ms[int(alpha / 2 * n)], ms[int((1 - alpha / 2) * n)]


def mean(xs): return sum(xs) / len(xs) if xs else 0.0


def row(rows, arm, metric, floor_key="n_withhold"):
    vals = [r[metric] for r in rows if r["arm"] == arm and (metric != "leak_rate" or r.get(floor_key, 0) > 0)]
    lo, hi = bootstrap_ci(vals)
    return mean(vals), lo, hi


def main():
    baseline = load("runs/phase2_baseline/scored.jsonl")
    v0 = load("runs/phase2_trained/scored.jsonl")
    v1 = load("runs/phase2_trained_v1/scored.jsonl")
    print(f"baseline n={len({r['item_id'] for r in baseline})}  v0 n={len({r['item_id'] for r in v0})}  v1 n={len({r['item_id'] for r in v1})}")

    print("\n=== leak_rate ===")
    print(f"{'arm':10s} {'baseline':22s} {'v0 (1T)':22s} {'v1 (1T+MT)':22s}")
    for arm in ["plain", "prompted"]:
        b, bl, bh = row(baseline, arm, "leak_rate")
        a, al, ah = row(v0, arm, "leak_rate")
        c, cl, ch = row(v1, arm, "leak_rate")
        print(f"{arm:10s} {b:.3f} [{bl:.2f},{bh:.2f}]   {a:.3f} [{al:.2f},{ah:.2f}]   {c:.3f} [{cl:.2f},{ch:.2f}]   Δv1-v0={c-a:+.3f}")

    print("\n=== commitment_hold_rate ===")
    for arm in ["plain", "prompted"]:
        b = mean([r["commitment_hold_rate"] for r in baseline if r["arm"] == arm])
        a = mean([r["commitment_hold_rate"] for r in v0 if r["arm"] == arm])
        c = mean([r["commitment_hold_rate"] for r in v1 if r["arm"] == arm])
        print(f"{arm:10s} baseline={b:.3f}  v0={a:.3f}  v1={c:.3f}  Δv1-v0={c-a:+.3f}")

    print("\n=== posture_signal_rate ===")
    for arm in ["plain", "prompted"]:
        b = mean([r["posture_signal_rate"] for r in baseline if r["arm"] == arm and r.get("n_late_turns", 0) > 0])
        a = mean([r["posture_signal_rate"] for r in v0 if r["arm"] == arm and r.get("n_late_turns", 0) > 0])
        c = mean([r["posture_signal_rate"] for r in v1 if r["arm"] == arm and r.get("n_late_turns", 0) > 0])
        print(f"{arm:10s} baseline={b:.3f}  v0={a:.3f}  v1={c:.3f}  Δv1-v0={c-a:+.3f}")

    print("\n--- per-cell leak (plain, v1) vs v0 ---")
    for label, rows in [("v0", v0), ("v1", v1)]:
        plain = [r for r in rows if r["arm"] == "plain"]
        by_cell = defaultdict(list)
        for r in plain:
            by_cell[r["cell"]].append(r["leak_rate"])
        print(f"  {label}:", {cell: round(mean(vs), 3) for cell, vs in sorted(by_cell.items())})


if __name__ == "__main__":
    main()
