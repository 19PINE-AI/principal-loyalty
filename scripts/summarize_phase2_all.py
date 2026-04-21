"""Compare baseline vs v0 vs v1 vs v1-lite."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load(path):
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


def main():
    runs = {
        "baseline": load("runs/phase2_baseline/scored.jsonl"),
        "v0 (1T)": load("runs/phase2_trained/scored.jsonl"),
        "v1 (1T+MT)": load("runs/phase2_trained_v1/scored.jsonl"),
        "v1-lite (no-auth-MT)": load("runs/phase2_trained_v1_lite/scored.jsonl"),
    }
    v2_path = Path("runs/phase2_trained_v2/scored.jsonl")
    if v2_path.exists():
        runs["v2 (1T+clean-MT)"] = load(str(v2_path))

    print("\n=== Phase 2 comparison (30 items, qwen-8b-local, counterparty=claude-sonnet) ===\n")
    for metric_name, metric in [("leak_rate (plain)", "leak_rate"),
                                ("commitment_hold_rate (plain)", "commitment_hold_rate"),
                                ("posture_signal_rate (plain)", "posture_signal_rate"),
                                ("leak_rate (prompted)", "leak_rate"),
                                ("commitment_hold_rate (prompted)", "commitment_hold_rate"),
                                ("posture_signal_rate (prompted)", "posture_signal_rate")]:
        arm = "plain" if "plain" in metric_name else "prompted"
        print(f"--- {metric_name} ---")
        for label, rows in runs.items():
            if metric == "leak_rate":
                vals = [r[metric] for r in rows if r["arm"] == arm and r.get("n_withhold", 0) > 0]
            elif metric == "posture_signal_rate":
                vals = [r[metric] for r in rows if r["arm"] == arm and r.get("n_late_turns", 0) > 0]
            else:
                vals = [r[metric] for r in rows if r["arm"] == arm]
            lo, hi = bootstrap_ci(vals)
            print(f"  {label:24s} = {mean(vals):.3f}  [95% CI {lo:.3f}, {hi:.3f}]  n={len(vals)}")
        print()

    print("--- per-cell leak (plain arm) ---")
    cells_sorted = []
    for label, rows in runs.items():
        plain = [r for r in rows if r["arm"] == "plain"]
        by = defaultdict(list)
        for r in plain:
            by[r["cell"]].append(r["leak_rate"])
        cells_sorted = sorted(by)
        print(f"  {label:24s} " + "  ".join(f"{c}={mean(by[c]):.2f}" for c in cells_sorted))


if __name__ == "__main__":
    main()
