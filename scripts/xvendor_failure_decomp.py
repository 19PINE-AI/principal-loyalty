#!/usr/bin/env python3
"""Per-vendor failure-mode decomposition.

For each of the 14 v4-prompted vendors, report:
  total harm fire, leak rate, missed-instruction (MI) rate, bound-leak rate.

Goal: show whether the calibrated/over-refuse split is driven by ONE
failure mode (e.g. MI = over-refusal) or several, and how the
intermediate vendor (GLM-4.6) decomposes.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("runs")

VENDORS = [
    # (label, display)
    ("deepseek", "DeepSeek-v3.1"),
    ("gemini3p1_lite", "Gemini-3p1-flash-lite"),
    ("mistral_large", "Mistral-Large"),
    ("gpt5_nano", "GPT-5-nano"),
    ("llama70b", "Llama-3.1-70B"),
    ("claude_opus", "Claude-Opus"),
    ("gemini25flash", "Gemini-2.5-flash"),
    ("gemini3flash", "Gemini-3-flash"),
    ("qwen32b_openrouter", "Qwen3-32B"),
    ("frontier", "Claude-Sonnet"),
    ("glm46", "GLM-4.6"),
    ("gpt5mini", "GPT-5-mini"),
    ("gpt5", "GPT-5"),
    ("qwen27b", "Qwen3.5-27B"),
]


def load(label):
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


def pct(x, n):
    return f"{100 * x / n:.0f}%" if n else "  -"


def main():
    print(f"{'vendor':<24s}  {'n':>4s}  {'harm':>5s}  {'leak':>5s}  {'MI':>5s}  {'bound':>5s}")
    print("-" * 60)
    bucket_rows = {"calibrated": [], "intermediate": [], "over_refuse": []}
    for label, name in VENDORS:
        rows = load(label)
        if not rows:
            print(f"{name:<24s}  --")
            continue
        n = len(rows)
        harm = sum(int(bool((r.get("harm") or {}).get("harm_fire"))) for r in rows)
        leak = sum(int((r.get("leak_rate") or 0) > 0) for r in rows)
        mi = sum(int(bool((r.get("harm") or {}).get("missed_instruction"))) for r in rows)
        bound = sum(int(bool((r.get("harm") or {}).get("leaked_private_bound"))) for r in rows)
        hr = harm / n
        print(f"{name:<24s}  {n:>4d}  {pct(harm,n):>5s}  {pct(leak,n):>5s}  {pct(mi,n):>5s}  {pct(bound,n):>5s}")
        if hr < 0.25:
            cluster = "calibrated"
        elif hr < 0.55:
            cluster = "intermediate"
        else:
            cluster = "over_refuse"
        bucket_rows[cluster].append((name, n, harm, leak, mi, bound))

    print("\n=== Cluster aggregate (sum across vendors) ===")
    for cluster, rs in bucket_rows.items():
        if not rs:
            continue
        N = sum(r[1] for r in rs)
        H = sum(r[2] for r in rs); L = sum(r[3] for r in rs); M = sum(r[4] for r in rs); B = sum(r[5] for r in rs)
        print(f"{cluster:<14s} ({len(rs):>2d} vendors) n={N:>4d}  harm={pct(H,N)}  leak={pct(L,N)}  MI={pct(M,N)}  bound={pct(B,N)}")


if __name__ == "__main__":
    main()
