"""Score the authoring-to-principal probe trajectories."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.items import load_items
from src.scorer import score_grid


def main() -> None:
    items = load_items("items/v0")
    score_grid(
        trajectory_jsonl="runs/probe_auth_to_principal/trajectories.jsonl",
        items=items,
        out_path="runs/probe_auth_to_principal/scored.jsonl",
        resume=True,
        parallel=6,
    )


if __name__ == "__main__":
    main()
