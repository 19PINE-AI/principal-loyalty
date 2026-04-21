"""Re-score the original runs/phase1/trajectories.jsonl with the new judge.
Lets us compare v1/v2/v3 prompts all under the same (new) judge.
Output: runs/phase1/scored_newjudge.jsonl
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


def main() -> int:
    items = load_items("items/v0")
    score_grid(
        trajectory_jsonl="runs/phase1/trajectories.jsonl",
        items=items,
        out_path="runs/phase1/scored_newjudge.jsonl",
        resume=True,
        parallel=8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
