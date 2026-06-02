"""Run trajectory generation only for a list of seed dirs (no scoring).

Useful when scoring is queued separately to manage OpenRouter rate limits
while vLLM trajectory generation can run in parallel against a local server.

Usage:
    python3 scripts/run_traj_only.py --seed-dirs runs/foo_seed4 runs/foo_seed5 \
        --subject qwen-8b-local --counterparty claude-sonnet --parallel 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-dirs", nargs="+", required=True)
    p.add_argument("--subject", default="qwen-8b-local")
    p.add_argument("--counterparty", default="claude-sonnet")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--items", default="items/v0")
    args = p.parse_args()

    items = load_items(args.items)
    for sd in args.seed_dirs:
        sd_path = Path(sd)
        sd_path.mkdir(parents=True, exist_ok=True)
        tj = sd_path / "trajectories.jsonl"
        if tj.exists() and tj.stat().st_size > 0:
            print(f"[skip] {tj} exists ({tj.stat().st_size} bytes)")
            continue
        print(f"[run] {sd}")
        run_grid(
            items=items,
            subjects=[args.subject],
            arms=["plain", "prompted", "scaffolded"],
            out_path=tj,
            counterparty_spec=args.counterparty,
            parallel=args.parallel,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
