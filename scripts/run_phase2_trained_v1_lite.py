"""Phase 2 eval for v1-lite (MT pairs minus authoring) -> runs/phase2_trained_v1_lite/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid
from src.items import load_items


def main() -> None:
    items = load_items("items/v0")
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted"],
        out_path="runs/phase2_trained_v1_lite/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )


if __name__ == "__main__":
    main()
