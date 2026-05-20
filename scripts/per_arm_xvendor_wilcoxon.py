#!/usr/bin/env python3
"""Tier B.5: per-arm cross-vendor Wilcoxon.

Two hypotheses we test:

  H1 (v4 effect on calibrated): Across the calibrated cluster, paired by item,
      does the v4 'prompted' arm differ in harm rate from the 'plain' arm?

  H2 (calibrated vs over-refuse split): Per arm (plain, prompted, scaffolded),
      paired by item, do calibrated and over-refuse vendor mean harm rates
      differ?  Pair = mean across vendors in each cluster, on the same item.

Both via paired Wilcoxon signed-rank with continuity correction.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy.stats import wilcoxon

ROOT = Path("runs")

VENDORS = {
    "calibrated": [
        ("deepseek", "deepseek"),
        ("gemini3p1_lite", "Gemini-3p1-flash-lite"),
        ("claude_opus", "Claude-Opus"),
        ("gemini25flash", "Gemini-2.5-flash"),
        ("gemini3flash", "Gemini-3-flash"),
        ("qwen32b_openrouter", "Qwen3-32B"),
        ("frontier", "Claude-Sonnet"),
        ("llama70b", "Llama-3.1-70B"),
        ("mistral_large", "Mistral-Large"),
    ],
    "intermediate": [("glm46", "GLM-4.6")],
    "over_refuse": [
        ("gpt5_nano", "GPT-5-nano (OR)"),
        ("gpt5mini", "GPT-5-mini"),
        ("gpt5", "GPT-5"),
        ("qwen27b", "Qwen3.5-27B"),
    ],
}


def load_scored(label: str) -> list[dict]:
    p = ROOT / f"phase4_promptv4_{label}" / "scored.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def harm_of(r):
    return int(bool((r.get("harm") or {}).get("harm_fire")))


def item_id(r):
    return r.get("item_id") or r.get("item", {}).get("id")


def per_item_harm(rows, arm):
    """Return {item_id: harm 0/1} for rows in the given arm."""
    return {item_id(r): harm_of(r) for r in rows if r.get("arm") == arm}


def main():
    print("=== Tier B.5: per-arm cross-vendor Wilcoxon ===\n")

    all_rows: dict[str, list[dict]] = {}
    for cluster, labels in VENDORS.items():
        for label, name in labels:
            rows = load_scored(label)
            if not rows:
                print(f"[skip] no scored for {label}")
                continue
            all_rows[label] = rows

    # ---- H1 / H1': v4 effect within each cluster, paired by (vendor, item) ----
    for cluster_name in ("calibrated", "over_refuse"):
        print(f"\n## H1 ({cluster_name}): v4 (prompted) vs plain")
        print("    paired by (vendor, item), Wilcoxon signed-rank")
        diffs = []
        for label, name in VENDORS[cluster_name]:
            if label not in all_rows:
                continue
            plain = per_item_harm(all_rows[label], "plain")
            prompted = per_item_harm(all_rows[label], "prompted")
            common = sorted(set(plain) & set(prompted))
            for iid in common:
                diffs.append(prompted[iid] - plain[iid])
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_zero = sum(1 for d in diffs if d == 0)
        print(f"    n_pairs={len(diffs)} pos={n_pos} neg={n_neg} zero={n_zero}")
        if diffs and (n_pos + n_neg) > 0:
            try:
                res = wilcoxon(diffs, zero_method="wilcox")
                print(f"    Wilcoxon: stat={res.statistic:.2f}  p={res.pvalue:.4f}")
            except Exception as e:
                print(f"    Wilcoxon error: {e}")

    # ---- H2: calibrated vs over_refuse, per arm, paired by item ----
    print("\n## H2: calibrated cluster vs over-refuse cluster, per arm")
    print("    paired by item; rate = mean across cluster vendors on that item")
    for arm in ("plain", "prompted", "scaffolded"):
        # per-item, per-cluster rate
        per_item = defaultdict(lambda: defaultdict(list))  # iid -> cluster -> [0/1, ...]
        for cluster, labels in [("calibrated", VENDORS["calibrated"]),
                                 ("over_refuse", VENDORS["over_refuse"])]:
            for label, _ in labels:
                if label not in all_rows:
                    continue
                for r in all_rows[label]:
                    if r.get("arm") != arm:
                        continue
                    iid = item_id(r)
                    per_item[iid][cluster].append(harm_of(r))
        cal_rates, over_rates = [], []
        for iid in sorted(per_item):
            cal = per_item[iid].get("calibrated") or []
            over = per_item[iid].get("over_refuse") or []
            if not cal or not over:
                continue
            cal_rates.append(mean(cal))
            over_rates.append(mean(over))
        if not cal_rates:
            print(f"  arm={arm}: no paired items")
            continue
        diffs = [o - c for o, c in zip(over_rates, cal_rates)]
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_zero = sum(1 for d in diffs if d == 0)
        try:
            res = wilcoxon(diffs, zero_method="wilcox")
            p = f"p={res.pvalue:.2e}" if res.pvalue < 0.001 else f"p={res.pvalue:.4f}"
            stat = f"stat={res.statistic:.1f}"
        except Exception as e:
            p = f"err={e}"
            stat = ""
        print(
            f"  arm={arm:12s}  n_items={len(cal_rates):3d}  "
            f"cal={mean(cal_rates):.2f}  over={mean(over_rates):.2f}  "
            f"pos={n_pos} neg={n_neg} zero={n_zero}  {stat}  {p}"
        )


if __name__ == "__main__":
    main()
