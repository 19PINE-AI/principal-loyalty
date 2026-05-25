#!/usr/bin/env python3
"""Held-out per-arm Wilcoxon: does the bimodal split hold per-arm on held-out
items? Same H2 design as scripts/per_arm_xvendor_wilcoxon.py but using
runs/phase4_promptv4_*_heldout/scored.jsonl on items/v0_75.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from scipy.stats import wilcoxon

ROOT = Path("runs")

HELDOUT_VENDORS = {
    "calibrated": [
        ("deepseek_heldout", "DeepSeek"),
        ("gemini3p1_lite_heldout", "Gemini-3p1-flash-lite"),
        ("claude_opus_heldout", "Claude-Opus"),
        ("claude_heldout", "Claude-Sonnet"),
        ("qwen32b_openrouter_heldout", "Qwen3-32B"),
        ("llama70b_heldout", "Llama-3.1-70B"),
        ("gemini3flash_heldout", "Gemini-3-flash"),
    ],
    "intermediate": [("glm46_heldout", "GLM-4.6")],
    "over_refuse": [
        ("gpt5mini_heldout", "GPT-5-mini"),
        ("qwen27b_heldout", "Qwen3.5-27B"),
        ("gpt5_heldout", "GPT-5"),
    ],
}


def load_scored(label):
    p = ROOT / f"phase4_promptv4_{label}" / "scored.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def harm_of(r):
    return int(bool((r.get("harm") or {}).get("harm_fire")))


def item_id(r):
    return r.get("item_id") or r.get("item", {}).get("id")


def main():
    print("=== Held-out per-arm Wilcoxon (items/v0_75, 24 items) ===\n")

    all_rows = {}
    for cluster, labels in HELDOUT_VENDORS.items():
        for label, name in labels:
            rows = load_scored(label)
            if not rows:
                print(f"[skip] no scored for {label}")
                continue
            all_rows[label] = rows

    print("\n## H1' (over_refuse held-out): v4 (prompted) vs plain")
    diffs = []
    for label, name in HELDOUT_VENDORS["over_refuse"]:
        if label not in all_rows:
            continue
        plain = {item_id(r): harm_of(r) for r in all_rows[label] if r.get("arm") == "plain"}
        prompted = {item_id(r): harm_of(r) for r in all_rows[label] if r.get("arm") == "prompted"}
        common = sorted(set(plain) & set(prompted))
        for iid in common:
            diffs.append(prompted[iid] - plain[iid])
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_zero = sum(1 for d in diffs if d == 0)
    print(f"    n_pairs={len(diffs)} pos={n_pos} neg={n_neg} zero={n_zero}")
    if diffs and (n_pos + n_neg) > 0:
        res = wilcoxon(diffs, zero_method="wilcox")
        print(f"    Wilcoxon: stat={res.statistic:.2f}  p={res.pvalue:.4f}")

    print("\n## H2 (held-out): calibrated cluster vs over-refuse cluster, per arm")
    print("    paired by item; rate = mean across cluster vendors on that item")
    for arm in ("plain", "prompted", "scaffolded"):
        per_item = defaultdict(lambda: defaultdict(list))
        for cluster in ("calibrated", "over_refuse"):
            for label, _ in HELDOUT_VENDORS[cluster]:
                if label not in all_rows:
                    continue
                for r in all_rows[label]:
                    if r.get("arm") != arm:
                        continue
                    per_item[item_id(r)][cluster].append(harm_of(r))
        cal_rates, over_rates = [], []
        for iid in sorted(per_item):
            cal = per_item[iid].get("calibrated") or []
            over = per_item[iid].get("over_refuse") or []
            if not cal or not over:
                continue
            cal_rates.append(mean(cal))
            over_rates.append(mean(over))
        if not cal_rates:
            continue
        diffs = [o - c for o, c in zip(over_rates, cal_rates)]
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_zero = sum(1 for d in diffs if d == 0)
        res = wilcoxon(diffs, zero_method="wilcox")
        p_str = f"p={res.pvalue:.2e}" if res.pvalue < 0.001 else f"p={res.pvalue:.4f}"
        print(
            f"  arm={arm:12s}  n_items={len(cal_rates):3d}  "
            f"cal={mean(cal_rates):.2f}  over={mean(over_rates):.2f}  "
            f"pos={n_pos} neg={n_neg} zero={n_zero}  {p_str}"
        )


if __name__ == "__main__":
    main()
