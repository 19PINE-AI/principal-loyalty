"""Score one or more trajectory files (no trajectory generation).

Useful when trajectories were generated against a vLLM server that's
since been torn down, and scoring is queued separately to manage
OpenRouter rate limits.

Usage:
    python3 scripts/score_only.py --seed-dirs runs/foo_seed4 runs/foo_seed5 --parallel 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-dirs", nargs="+", required=True)
    p.add_argument("--items", default="items/v0")
    p.add_argument("--parallel", type=int, default=8)
    args = p.parse_args()

    items = load_items(args.items)
    for sd in args.seed_dirs:
        sd_path = Path(sd)
        tj = sd_path / "trajectories.jsonl"
        sc = sd_path / "scored.jsonl"
        if not tj.exists():
            print(f"[skip] {tj} missing")
            continue
        print(f"[score] {sd}")
        score_grid(
            trajectory_jsonl=tj,
            items=items,
            out_path=sc,
            resume=True,
            parallel=args.parallel,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
