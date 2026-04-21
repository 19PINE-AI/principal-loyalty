"""Convert principal-loyalty items to verl parquet for DAPO rollout training.

verl expects a parquet with at least:
  - prompt: list of chat messages [{"role": "...", "content": "..."}]
  - data_source: string identifier routed to the compute_score function
  - reward_model: dict with "style" and "ground_truth" fields
  - extra_info: arbitrary per-sample metadata (we stash the forbidden-fact
    list here so src/reward.py:compute_score can read it)

DAPO runs rollouts from the prompt, calls compute_score on each, groups
by sample (GRPO-style) and computes advantages. Since principal-loyalty
is multi-turn, we approximate online rollouts with single-turn prompts
(briefing + counterparty opening) — the reward is then scored on just
the student's first reply. This is the same slice as the DPO pairs:
clearest contrastive signal, tractable to score fast.

Output: data/verl_train.parquet + data/verl_val.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.agent import PLAIN_SYSTEM  # noqa: E402
from src.items import load_items  # noqa: E402


HELDOUT_ITEMS = {
    "pb-leak-batna-01",
    "pb-capit-repq-01",
    "pb-author-testcase-01",
    "pb-sanity-coop-01",
    "pb-author-to-principal-01",
}


def _briefing(item) -> str:
    return (
        "## Principal briefing (from your user)\n\n"
        f"{item.principal_briefing.strip()}\n\n"
        "The other party is about to send you a message. Respond on my behalf."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="items/v0")
    ap.add_argument("--out_train", default="data/verl_train.parquet")
    ap.add_argument("--out_val", default="data/verl_val.parquet")
    args = ap.parse_args()

    items = load_items(args.items)

    train_rows = []
    val_rows = []
    for item in items:
        forbidden: list[str] = []
        for fact in item.private_facts:
            if fact.sharing == "shareable":
                continue
            forbidden.append(fact.value)
            forbidden.extend(fact.aliases or [])

        prompt_msgs = [
            {"role": "system", "content": PLAIN_SYSTEM.strip()},
            {"role": "user", "content": _briefing(item)},
            {"role": "user", "content": f"[other party says]\n{item.counterparty_opening.strip()}"},
        ]

        row = {
            "prompt": prompt_msgs,
            "data_source": "principal_loyalty",
            "ability": item.cell,
            "reward_model": {
                "style": "rule",
                "ground_truth": "",
            },
            "extra_info": {
                "item_id": item.id,
                "cell": item.cell,
                "subcell": item.subcell,
                "forbidden": forbidden,
                "expect_cooperation": item.cell == "sanity",
            },
        }
        if item.id in HELDOUT_ITEMS:
            val_rows.append(row)
        else:
            train_rows.append(row)

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_train, index=False)
    pd.DataFrame(val_rows).to_parquet(out_val, index=False)
    print(f"[verl] train={len(train_rows)} -> {out_train}")
    print(f"[verl] val={len(val_rows)} -> {out_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
