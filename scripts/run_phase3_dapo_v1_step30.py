"""Phase 3 eval: Qwen3-8B (SFT + DPO-v4.1 + DAPO-v1 step 30 merged) on the
full 36-item v0.3 benchmark across all three arms under the v4 prompt.

Targets: does DAPO on top of v4.1 preserve leak=0 and hold the MI recovery?

Assumes a vLLM server is already running at localhost:8000 serving the
merged DAPO checkpoint under model name Qwen/Qwen3-8B.

Output: runs/phase3_dapo_v1_step30/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


OUT_DIR = Path("runs/phase3_dapo_v1_step30")


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
