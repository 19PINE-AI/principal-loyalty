"""Re-run Phase 1 prompted+scaffolded arms under the v2 prompt
(adversarial-stranger framing + decline-without-enumerating rule).

Outputs go to runs/phase1_promptv2/ so the original phase1/ artifacts stay
intact for delta analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


SUBJECTS = ["claude-sonnet", "gemini-3p1-flash-lite", "gpt-5-mini", "qwen-27b", "qwen-8b"]
ARMS = ["prompted", "scaffolded"]
OUT_DIR = Path("runs/phase1_promptv2")


def main() -> int:
    items = load_items("items/phase1") if Path("items/phase1").exists() else None
    if items is None:
        all_items = load_items("items/v0")
        # phase1 used 16 items — the first 16 non-sanity items (we load all then slice by what phase1 used)
        # The original phase1 item set is recoverable from phase1/trajectories.jsonl
        import json
        phase1_item_ids = set()
        with open("runs/phase1/trajectories.jsonl") as f:
            for line in f:
                r = json.loads(line)
                phase1_item_ids.add(r["item_id"])
        items = [i for i in all_items if i.id in phase1_item_ids]
        print(f"Loaded {len(items)} Phase 1 items (from original trajectories)")

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
        items=load_items("items/v0"),
        out_path=OUT_DIR / "scored.jsonl",
        resume=True,
        parallel=8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
