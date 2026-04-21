"""Focused probe: 2 authoring-to-principal sanity items across baseline+v2.

Output lives in runs/probe_auth_to_principal/ to keep the n=30 headline set clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_grid
from src.items import load_items


PROBE_IDS = {"pb-author-to-principal-01", "pb-author-to-principal-02"}


def main() -> None:
    all_items = load_items("items/v0")
    items = [i for i in all_items if i.id in PROBE_IDS]
    assert len(items) == 2, f"expected 2 items, got {len(items)}: {[i.id for i in items]}"
    run_grid(
        items=items,
        subjects=["qwen-8b-local"],
        arms=["plain", "prompted"],
        out_path="runs/probe_auth_to_principal/trajectories.jsonl",
        counterparty_spec="claude-sonnet",
        parallel=4,
    )


if __name__ == "__main__":
    main()
