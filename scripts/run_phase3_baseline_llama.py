"""Phase 3 baseline: BASE Llama-3.1-8B-Instruct (no SFT/DPO/DAPO) on 108-grid.

Third base-model arm of the cross-family replication (Qwen3-8B + Mistral-7B + Llama-3.1-8B).
Establishes the untrained Llama baseline on PrincipalBench v0 (36 items × 3 arms).

Uses NousResearch/Meta-Llama-3.1-8B-Instruct (bit-identical mirror of
meta-llama/Llama-3.1-8B-Instruct) because the meta-llama repo is gated and
the autonomous run has no HF token configured.

vLLM serves the model under name 'qwen-8b-local' (the harness uses that
subject identifier for hosting-transparent evaluation; same pattern as the
Mistral cross-family arm).

Output: runs/phase3_baseline_llama/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


OUT_DIR = Path("runs/phase3_baseline_llama")


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
