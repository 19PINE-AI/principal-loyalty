"""Held-out vs trained-on item split: cell-level harm/leak breakdown.

Reads the verl train/val parquets to get the (n=31 train, n=5 val) item split,
then re-tabulates harm and leak from the headline scored.jsonl files split by
which side each item was on.

Defuses "is this just memorization?" — if the leak/harm reduction generalizes
to held-out items, the recipe is doing real work, not just memorizing.

Run after re-running broken evals so phase2_trained_v4_1 and phase3_dapo_v1_step35
contain real multi-turn rollouts.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def load_split(train_pq: str, val_pq: str) -> tuple[set, set]:
    train_items = set()
    val_items = set()
    for r in pq.read_table(train_pq).to_pylist():
        train_items.add(r["extra_info"]["item_id"])
    for r in pq.read_table(val_pq).to_pylist():
        val_items.add(r["extra_info"]["item_id"])
    return train_items, val_items


def tabulate(scored_path: str, train_items: set, val_items: set) -> dict:
    rows = []
    with open(scored_path) as f:
        for line in f:
            rows.append(json.loads(line))

    # cell × split → (harm_fires, leak_fires, n)
    cell_split = defaultdict(lambda: {"harm": 0, "leak": 0, "n": 0})
    arm_split = defaultdict(lambda: {"harm": 0, "leak": 0, "n": 0})
    overall = defaultdict(lambda: {"harm": 0, "leak": 0, "n": 0})

    for r in rows:
        item = r["item_id"]
        if item in train_items:
            split = "train"
        elif item in val_items:
            split = "val"
        else:
            split = "unknown"

        cell = r.get("cell", "?")
        arm = r.get("arm", "?")
        harm = bool(r.get("harm", {}).get("harm_fire"))
        leak = (r.get("leak_rate") or 0) > 0

        cell_split[(cell, split)]["harm"] += int(harm)
        cell_split[(cell, split)]["leak"] += int(leak)
        cell_split[(cell, split)]["n"] += 1

        arm_split[(arm, split)]["harm"] += int(harm)
        arm_split[(arm, split)]["leak"] += int(leak)
        arm_split[(arm, split)]["n"] += 1

        overall[split]["harm"] += int(harm)
        overall[split]["leak"] += int(leak)
        overall[split]["n"] += 1

    return {"cell": dict(cell_split), "arm": dict(arm_split), "overall": dict(overall)}


def fmt(n_fire: int, n_total: int) -> str:
    if n_total == 0:
        return "—"
    return f"{n_fire}/{n_total} ({100*n_fire/n_total:.0f}%)"


def report(label: str, scored_path: str, train_items: set, val_items: set) -> None:
    if not Path(scored_path).exists():
        print(f"\n## {label}\n  (missing: {scored_path})")
        return

    t = tabulate(scored_path, train_items, val_items)

    print(f"\n## {label}")
    print(f"   {scored_path}")

    print("\n   Overall:")
    for split in ("train", "val", "unknown"):
        d = t["overall"].get(split, {"harm": 0, "leak": 0, "n": 0})
        if d["n"] > 0:
            print(
                f"     {split:8s}  harm: {fmt(d['harm'], d['n'])}    "
                f"leak: {fmt(d['leak'], d['n'])}"
            )

    print("\n   By cell × split (harm | leak):")
    cells = sorted({c for (c, _) in t["cell"]})
    for cell in cells:
        tr = t["cell"].get((cell, "train"), {"harm": 0, "leak": 0, "n": 0})
        va = t["cell"].get((cell, "val"), {"harm": 0, "leak": 0, "n": 0})
        print(
            f"     {cell:12s}  train {fmt(tr['harm'], tr['n'])} | {fmt(tr['leak'], tr['n'])}    "
            f"  val {fmt(va['harm'], va['n'])} | {fmt(va['leak'], va['n'])}"
        )

    print("\n   By arm × split (harm | leak):")
    for arm in ("plain", "prompted", "scaffolded"):
        tr = t["arm"].get((arm, "train"), {"harm": 0, "leak": 0, "n": 0})
        va = t["arm"].get((arm, "val"), {"harm": 0, "leak": 0, "n": 0})
        print(
            f"     {arm:12s}  train {fmt(tr['harm'], tr['n'])} | {fmt(tr['leak'], tr['n'])}    "
            f"  val {fmt(va['harm'], va['n'])} | {fmt(va['leak'], va['n'])}"
        )


def main() -> int:
    train_items, val_items = load_split("data/verl_train.parquet", "data/verl_val.parquet")
    print(f"Train items (n={len(train_items)}):", sorted(train_items))
    print(f"Val items   (n={len(val_items)}):", sorted(val_items))

    headlines = [
        ("Untrained baseline", "runs/phase3_baseline_qwen/scored.jsonl"),
        ("v4.1 SFT+DPO endpoint", "runs/phase2_trained_v4_1/scored.jsonl"),
        ("DAPO-v1 step_35 (HEADLINE)", "runs/phase3_dapo_v1_step35/scored.jsonl"),
        ("DAPO-v1 step_30", "runs/phase3_dapo_v1_step30/scored.jsonl"),
        ("DAPO-v2 step_55", "runs/phase3_dapo_v2_step55/scored.jsonl"),
        ("DAPO-leak-only step_35", "runs/phase3_dapo_leakonly_step35/scored.jsonl"),
        ("DAPO-v3 step_30", "runs/phase3_dapo_v3_step30/scored.jsonl"),
        ("DAPO-v3 step_55", "runs/phase3_dapo_v3_step55/scored.jsonl"),
        ("DAPO-v1 n=51 step_55", "runs/phase3_dapo_v1_n51_step55/scored.jsonl"),
        ("Mistral SFT+DPO", "runs/phase3_mistral_sft_dpo/scored.jsonl"),
    ]
    for label, path in headlines:
        report(label, path, train_items, val_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
