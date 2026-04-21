"""Phase 2 eval for the v1 DPO model (SFT + DPO trained on 35 first-turn
+ 24 multi-turn pairs). Output to runs/phase2_trained_v1/."""
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
        out_path="runs/phase2_trained_v1/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )


if __name__ == "__main__":
    main()
