"""Multi-rollout eval: run the v0-grid evaluation multiple times against an
already-running vLLM server, aggregate harm/leak fires across seeds.

Each seed gets its own out_dir runs/<base>_seedN/. The harness sampling at
temperature=0.7 already provides stochasticity — we just re-run.

Usage:
    python3 scripts/multi_rollout_eval.py --base runs/phase3_dapo_v1_step35 \\
        --seeds 3 --counterparty claude-sonnet --parallel 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


def aggregate(out_dirs: list[Path]) -> dict:
    """For each (item_id, arm) cell, list harm/leak booleans across seeds."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for d in out_dirs:
        with open(d / "scored.jsonl") as f:
            for line in f:
                r = json.loads(line)
                key = (r["item_id"], r["arm"])
                cells.setdefault(key, []).append(
                    {
                        "harm": bool(r.get("harm", {}).get("harm_fire")),
                        "leak": (r.get("leak_rate") or 0) > 0,
                    }
                )
    n_seeds = len(out_dirs)
    summary = {
        "n_seeds": n_seeds,
        "n_cells": len(cells),
        "harm_per_seed": [0] * n_seeds,
        "leak_per_seed": [0] * n_seeds,
        "harm_all_seeds": 0,  # cells where ALL seeds fire
        "harm_any_seed": 0,   # cells where AT LEAST ONE seed fires
        "leak_all_seeds": 0,
        "leak_any_seed": 0,
    }
    for key, samples in cells.items():
        for i, s in enumerate(samples):
            if s["harm"]:
                summary["harm_per_seed"][i] += 1
            if s["leak"]:
                summary["leak_per_seed"][i] += 1
        if all(s["harm"] for s in samples):
            summary["harm_all_seeds"] += 1
        if any(s["harm"] for s in samples):
            summary["harm_any_seed"] += 1
        if all(s["leak"] for s in samples):
            summary["leak_all_seeds"] += 1
        if any(s["leak"] for s in samples):
            summary["leak_any_seed"] += 1
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base out dir (an existing seed-1 run)")
    p.add_argument("--extra-seeds", type=int, default=2, help="number of additional seeds to run")
    p.add_argument("--counterparty", default="claude-sonnet")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--items", default="items/v0")
    args = p.parse_args()

    base = Path(args.base)
    items = load_items(args.items)

    # Seed 1 = base dir, additional seeds = base_seedN
    seed_dirs = [base] + [Path(f"{base}_seed{i+2}") for i in range(args.extra_seeds)]

    for sd in seed_dirs[1:]:
        sd.mkdir(parents=True, exist_ok=True)
        if not (sd / "trajectories.jsonl").exists():
            print(f"[seed {sd}] running grid")
            run_grid(
                items=items,
                subjects=["qwen-8b-local"],
                arms=["plain", "prompted", "scaffolded"],
                out_path=sd / "trajectories.jsonl",
                counterparty_spec=args.counterparty,
                parallel=args.parallel,
            )
        else:
            print(f"[seed {sd}] trajectories already exist; resuming scoring")
        score_grid(
            trajectory_jsonl=sd / "trajectories.jsonl",
            items=items,
            out_path=sd / "scored.jsonl",
            resume=True,
            parallel=8,
        )

    summary = aggregate(seed_dirs)
    out_summary = base.parent / f"{base.name}_multirollout_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
