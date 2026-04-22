"""Phase 3 robustness eval: step_35 DAPO checkpoint vs Gemini-3-Flash counterparty.

Third counterparty vendor (Google) on top of claude-sonnet (§4.4 headline) and
gpt-5 (§counterparty-robustness). Three independent vendors exercising the
same 36-item × 3-arm grid lets us separate frontier-is-structural from
frontier-is-counterparty-specific.

Output: runs/phase3_dapo_v1_step35_gemcp/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


OUT_DIR = Path("runs/phase3_dapo_v1_step35_gemcp")


def main() -> int:
    items = load_items("items/v0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted", "scaffolded"],
        out_path=OUT_DIR / "trajectories.jsonl",
        counterparty_spec="gemini-3-flash",
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
