"""Phase 3 eval: DAPO leak-only ablation checkpoint.

Reads PL_STEP env var to select global_step_N. Output:
runs/phase3_dapo_leakonly_step${PL_STEP}/.

Purpose: compare against DAPO-v1 step_35. Prediction is posture-collapse
on sanity cell, residual probe-arm MI, but leak still near zero.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


def main() -> int:
    step = os.environ.get("PL_STEP", "35")
    out_dir = Path(f"runs/phase3_dapo_leakonly_step{step}")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items("items/v0")
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted", "scaffolded"],
        out_path=out_dir / "trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )
    score_grid(
        trajectory_jsonl=out_dir / "trajectories.jsonl",
        items=items,
        out_path=out_dir / "scored.jsonl",
        resume=True,
        parallel=8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
