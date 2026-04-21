"""Score untrained Qwen3-8B served locally (phase2_baseline)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.items import load_items
from src.scorer import score_grid
if __name__ == "__main__":
    score_grid(
        trajectory_jsonl="runs/phase2_baseline/trajectories.jsonl",
        items=load_items("items/v0"),
        out_path="runs/phase2_baseline/scored.jsonl",
        resume=True,
        parallel=6,
    )
