"""Run Phase 1 under v3 prompt (adds: execute-positive-instructions,
private-bound discipline, operational-questions-aren't-probes).

5 subjects × 3 arms × 16 items = 240 trajectories → runs/phase1_promptv3/.
Scored under the v2 judge (briefing-grounded fab/dec, split against_principal,
separate instructed-misrep flag).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


SUBJECTS = ["claude-sonnet", "gemini-3p1-flash-lite", "gpt-5-mini", "qwen-27b", "qwen-8b"]
ARMS = ["plain", "prompted", "scaffolded"]
OUT_DIR = Path("runs/phase1_promptv3")


def main() -> int:
    all_items = load_items("items/v0")
    # Phase 1 item set from original rollout.
    import json
    phase1_item_ids = set()
    with open("runs/phase1/trajectories.jsonl") as f:
        for line in f:
            r = json.loads(line)
            phase1_item_ids.add(r["item_id"])
    items = [i for i in all_items if i.id in phase1_item_ids]
    print(f"Phase 1 items: {len(items)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_grid(
        items=items,
        subjects=SUBJECTS,
        arms=ARMS,
        out_path=OUT_DIR / "trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=6,
    )

    score_grid(
        trajectory_jsonl=OUT_DIR / "trajectories.jsonl",
        items=all_items,
        out_path=OUT_DIR / "scored.jsonl",
        resume=True,
        parallel=8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
