"""Phase D eval: DAPO-v1 reward retrained on n=51 dataset, evaluated on v0 (n=36) grid.

Apples-to-apples comparison vs DAPO-v1 step_35 (n=31 train):
- Same v4.1 base
- Same v1 reward
- Same eval grid (v0 / 36 items × 3 arms = 108 rollouts)
- Same counterparty (claude-sonnet)
- DIFFERENT: training data expanded n=31 → n=46 (51 items minus 5 heldout)

Tests whether the 25/108 ceiling on n=31 was data-coverage-bounded.
If n=46 training breaks below 25, the §4.4.2 "data expansion is the
right next step" claim is empirically validated.

Output: runs/phase3_dapo_v1_n51_step55/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


OUT_DIR = Path("runs/phase3_dapo_v1_n51_step55")


def main() -> int:
    items = load_items("items/v0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted", "scaffolded"],
        out_path=OUT_DIR / "trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )
    score_grid(
        trajectory_jsonl=OUT_DIR / "trajectories.jsonl",
        items=items,
        out_path=OUT_DIR / "scored.jsonl",
        resume=True,
        parallel=8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
