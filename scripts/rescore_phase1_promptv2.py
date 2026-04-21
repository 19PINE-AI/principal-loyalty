"""Re-score existing runs/phase1_promptv2/trajectories.jsonl under the v2 judge
(src/scorer.py HARM_JUDGE_SYSTEM post-rewrite), without re-rolling out the agent.

Output: runs/phase1_promptv2/scored_v2judge.jsonl — lets us isolate how much of
the harm-regression story was judge-side vs model-side.
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
        trajectory_jsonl="runs/phase1_promptv2/trajectories.jsonl",
        items=items,
        out_path="runs/phase1_promptv2/scored_v2judge.jsonl",
        resume=True,
        parallel=8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
