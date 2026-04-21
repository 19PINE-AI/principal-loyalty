"""Aggregate Phase 2 scored.jsonl files into a headline table.

Prints:
  - per-arm leak_rate / commitment_hold_rate / posture_signal_rate for trained vs baseline
  - bootstrap 95% CI over items
  - per-cell breakdown
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def bootstrap_ci(xs: list[float], n: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(0)
    means = []
    k = len(xs)
    for _ in range(n):
        sample = [xs[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return (lo, hi)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: list[dict], label: str) -> None:
    print(f"\n=== {label} (n_items={len({r['item_id'] for r in rows})}) ===")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm in sorted(by_arm):
        leaks = [r["leak_rate"] for r in by_arm[arm] if r["n_withhold"] > 0]
        holds = [r["commitment_hold_rate"] for r in by_arm[arm] if r["commitment_held"]]
        # commitment_hold_rate: fraction of positions held; use directly if positions exist
        holds_all = [r["commitment_hold_rate"] for r in by_arm[arm]]
        posture = [r["posture_signal_rate"] for r in by_arm[arm] if r.get("n_late_turns", 0) > 0]
        m_leak = mean(leaks); lo_l, hi_l = bootstrap_ci(leaks)
        m_hold = mean(holds_all); lo_h, hi_h = bootstrap_ci(holds_all)
        m_post = mean(posture) if posture else 0.0
        print(f"  arm={arm:10s}  leak={m_leak:.3f} [{lo_l:.3f},{hi_l:.3f}]  hold={m_hold:.3f} [{lo_h:.3f},{hi_h:.3f}]  posture={m_post:.3f}  n_traj={len(by_arm[arm])}")


def cell_breakdown(rows: list[dict], label: str) -> None:
    print(f"\n--- {label} per-cell leak (plain arm only) ---")
    plain = [r for r in rows if r["arm"] == "plain"]
    by_cell = defaultdict(list)
    for r in plain:
        by_cell[r["cell"]].append(r["leak_rate"])
    for cell in sorted(by_cell):
        xs = by_cell[cell]
        print(f"  {cell:12s}  leak={mean(xs):.3f}  n={len(xs)}")


def main() -> None:
    trained = load("runs/phase2_trained/scored.jsonl")
    baseline = load("runs/phase2_baseline/scored.jsonl")
    summarize(baseline, "BASELINE raw Qwen3-8B (local vLLM)")
    summarize(trained, "TRAINED Qwen3-8B SFT+DPO (local vLLM)")
    cell_breakdown(baseline, "baseline")
    cell_breakdown(trained, "trained")

    # delta table
    print("\n=== headline delta (baseline → trained) ===")
    for arm in ["plain", "prompted"]:
        b = [r["leak_rate"] for r in baseline if r["arm"] == arm and r["n_withhold"] > 0]
        t = [r["leak_rate"] for r in trained if r["arm"] == arm and r["n_withhold"] > 0]
        bh = [r["commitment_hold_rate"] for r in baseline if r["arm"] == arm]
        th = [r["commitment_hold_rate"] for r in trained if r["arm"] == arm]
        print(f"  arm={arm}: leak {mean(b):.3f} → {mean(t):.3f} (Δ={mean(t)-mean(b):+.3f})   hold {mean(bh):.3f} → {mean(th):.3f} (Δ={mean(th)-mean(bh):+.3f})")


if __name__ == "__main__":
    main()
