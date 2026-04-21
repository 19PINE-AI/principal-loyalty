"""Score Phase 2 v2.1 trajectories."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.items import load_items
from src.scorer import score_grid


def main() -> None:
    items = load_items("items/v0")
    score_grid(
        trajectory_jsonl="runs/phase2_trained_v21/trajectories.jsonl",
        items=items,
        out_path="runs/phase2_trained_v21/scored.jsonl",
        resume=True,
        parallel=6,
    )


if __name__ == "__main__":
    main()
