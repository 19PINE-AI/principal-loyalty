"""Run Phase 1 multi-turn eval against the trained Qwen3-8B (SFT+DPO).

Subject: qwen-8b-local (vLLM serving runs/qwen_sft_dpo_merged on :8000).
Arms: plain (primary — "did training move plain behavior toward scaffolded?")
      + prompted (secondary — does the simple policy preamble still help on top?)
Counterparty: claude-sonnet (same as Phase 1 baseline for apples-to-apples).

Output: runs/phase2_trained/trajectories.jsonl  →  score with score_phase1.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid
from src.items import load_items


def main() -> None:
    items = load_items("items/v0")
    print(f"loaded {len(items)} items")
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted"],
        out_path="runs/phase2_trained/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )


if __name__ == "__main__":
    main()
