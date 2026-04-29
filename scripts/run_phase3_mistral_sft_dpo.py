"""Phase C eval: Mistral-7B SFT+DPO (v4.1 replication) on full 36-item × 3-arm grid.

Tests whether the SFT→DPO pipeline (built on Qwen3-8B traces) transfers
to a different base model family (Mistral). If results match Qwen3-8B
v4.1 within ±5pp on key metrics, the structural claim is supported.

vLLM serves the merged model under name 'qwen-8b-local' (the harness uses
that subject identifier for hosting-transparent evaluation).

Output: runs/phase3_mistral_sft_dpo/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


OUT_DIR = Path("runs/phase3_mistral_sft_dpo")


def main() -> int:
    items = load_items("items/v0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted", "scaffolded"],
        out_path=OUT_DIR / "trajectories.jsonl",
        counterparty_spec="claude-sonnet",
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
