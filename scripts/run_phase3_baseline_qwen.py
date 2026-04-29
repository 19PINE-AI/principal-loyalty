"""Phase 3 baseline: BASE Qwen3-8B (no SFT/DPO/DAPO) on full 36-item × 3-arm grid.

Tests the §6 reviewer objection: "what if just the scaffolded sentinel
prompt without any training does most of the work?" Compares directly
against DAPO-v1 step_35 to isolate the contribution of training.

Output: runs/phase3_baseline_qwen/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


OUT_DIR = Path("runs/phase3_baseline_qwen")


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
