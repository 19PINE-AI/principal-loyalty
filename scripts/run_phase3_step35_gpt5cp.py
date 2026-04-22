"""Phase 3 robustness eval: step_35 DAPO checkpoint vs GPT-5 counterparty.

Addresses the §6 limitation that all prior evals use claude-sonnet as the
other party. If the leak/MI frontier is a Claude-specific dialogue artifact,
swapping the counterparty should break the Phase 3 headline. If the frontier
is structural, harm_fire/leak should land in the same range.

Output: runs/phase3_dapo_v1_step35_gpt5cp/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


OUT_DIR = Path("runs/phase3_dapo_v1_step35_gpt5cp")


def main() -> int:
    items = load_items("items/v0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted", "scaffolded"],
        out_path=OUT_DIR / "trajectories.jsonl",
        counterparty_spec="gpt-5",
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
