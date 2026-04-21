"""Untrained Qwen3-8B served locally via vLLM — apples-to-apples control
against scripts/run_phase2_trained.py.

Same subject spec (`qwen-8b-local`) but the served model is the raw
Qwen/Qwen3-8B (not merged SFT+DPO). Same counterparty, same items, same
arms. Purpose: isolate training deltas from serving-stack deltas.
"""
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
        out_path="runs/phase2_baseline/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )


if __name__ == "__main__":
    main()
