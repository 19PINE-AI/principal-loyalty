"""Phase 4: v4 prompted-arm baseline on a FRONTIER subject across all 36 items.

Closes the reviewer-objection gap "is your trained 8B doing more than just
applying the v4 prompt to a strong model?" by running the same v4 prompt
(agent.py PROMPTED_SYSTEM) on claude-sonnet across the full v0 grid
(36 items × 3 arms × multi-turn), with claude-sonnet as the counterparty
(matching the DAPO-v1 step_35 headline configuration).

Existing phase1_promptv4 covered claude-sonnet but only on the 16-item
Phase 1 subset; this extends to the full 36-item grid for direct
comparability with the trained-model headline numbers.

Output: runs/phase4_promptv4_frontier/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


SUBJECTS = ["claude-sonnet"]
ARMS = ["plain", "prompted", "scaffolded"]
OUT_DIR = Path("runs/phase4_promptv4_frontier")


def main() -> int:
    items = load_items("items/v0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_grid(
        items=items,
        subjects=SUBJECTS,
        arms=ARMS,
        out_path=OUT_DIR / "trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=6,
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
