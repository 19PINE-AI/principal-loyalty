"""Run Phase 1 diagnostic: 3 subjects x 3 arms x 12 items = 108 trajectories.

Subjects: claude-sonnet (4.5 in vendors.py), gpt-5-mini, gemini-3p1-flash-lite.
Counterparty simulator: claude-sonnet (fixed for variance control).
Arms: plain, prompted, scaffolded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid
from src.items import load_items


SUBJECTS = ["claude-sonnet", "gpt-5-mini", "gemini-3p1-flash-lite", "qwen-8b", "qwen-27b"]
ARMS = ["plain", "prompted", "scaffolded"]


def main() -> None:
    items = load_items("items/v0")
    print(f"loaded {len(items)} items")
    run_grid(
        items=items,
        subjects=SUBJECTS,
        arms=ARMS,
        out_path="runs/phase1/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=6,
    )


if __name__ == "__main__":
    main()
