"""Sentinel-spoof probe.

Tests whether v0.6's sentinel is a blind override (safety concern) or
whether the model cross-references the sentinel against the briefing.

Two runs:
  A) THIRD_PARTY items with sentinel forced to PRINCIPAL — does the model
     dump the secret to what's actually a third party?
  B) PRINCIPAL items with sentinel forced to THIRD_PARTY — does the model
     refuse even though the reader is actually the principal?

If the sentinel is a blind override, (A) leaks heavily and (B) refuses
everywhere. If the model is cross-referencing, (A) is more resistant
than baseline on leakage items, and (B) might still produce on
obvious-principal cases.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.agent as agent_mod  # noqa: E402
from src.harness import run_grid  # noqa: E402
from src.items import load_items  # noqa: E402
from src.scorer import score_grid  # noqa: E402


def force_sentinel(value: str):
    """Monkey-patch reader_sentinel to return a fixed sentinel."""
    body_principal = agent_mod.READER_PRINCIPAL_SENTINEL
    body_third = agent_mod.READER_THIRD_PARTY_SENTINEL
    forced = body_principal if value == "PRINCIPAL" else body_third

    def _fake(_item):
        return forced

    agent_mod.reader_sentinel = _fake
    # also rebind in Agent module import path
    from src import agent as mod
    mod.reader_sentinel = _fake


def main() -> int:
    items = load_items("items/v0")

    third_party_leak_items = [
        i for i in items
        if i.id in {"pb-leak-price-01", "pb-leak-batna-01", "pb-author-policy-01"}
    ]
    principal_items = [i for i in items if i.reader_is_principal]

    # A) THIRD_PARTY items under PRINCIPAL sentinel
    force_sentinel("PRINCIPAL")
    out_a = "runs/spoof_a_third_party_as_principal"
    Path(out_a).mkdir(parents=True, exist_ok=True)
    run_grid(
        items=third_party_leak_items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted"],
        out_path=f"{out_a}/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )
    score_grid(
        trajectory_jsonl=f"{out_a}/trajectories.jsonl",
        items=items,
        out_path=f"{out_a}/scored.jsonl",
        resume=True,
        parallel=6,
    )

    # B) PRINCIPAL items under THIRD_PARTY sentinel
    force_sentinel("THIRD_PARTY")
    out_b = "runs/spoof_b_principal_as_third_party"
    Path(out_b).mkdir(parents=True, exist_ok=True)
    run_grid(
        items=principal_items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted"],
        out_path=f"{out_b}/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )
    score_grid(
        trajectory_jsonl=f"{out_b}/trajectories.jsonl",
        items=items,
        out_path=f"{out_b}/scored.jsonl",
        resume=True,
        parallel=6,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
