"""Print the analysis tables for the top-tier-conference push:
 - Per-arm and per-cell breakdown for each of: untrained-qwen, qwen-v4.1,
   qwen-dapo-v1-step35, mistral-sft-dpo, mistral-dapo-v1, llama-untrained,
   phase4-frontier (claude-sonnet+v4-prompt).
 - Multi-seed paired Wilcoxon at whatever n is currently on disk.

Output: prints a markdown-formatted summary suitable for paper appendix.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = [
    ("untrained-qwen3-8b", "runs/phase3_baseline_qwen/scored.jsonl"),
    ("untrained-llama-3.1-8b", "runs/phase3_baseline_llama/scored.jsonl"),
    ("qwen v4.1 SFT+DPO (off-policy)", "runs/phase2_trained_v4_1/scored.jsonl"),
    ("qwen DAPO-v1 step35 (single seed)", "runs/phase3_dapo_v1_step35/scored.jsonl"),
    ("qwen DAPO-v2 step55", "runs/phase3_dapo_v2_step55/scored.jsonl"),
    ("qwen DAPO-leakonly step35", "runs/phase3_dapo_leakonly_step35/scored.jsonl"),
    ("qwen DAPO-v3 step55", "runs/phase3_dapo_v3_step55/scored.jsonl"),
    ("qwen DAPO-v1 n=51 step55", "runs/phase3_dapo_v1_n51_step55/scored.jsonl"),
    ("qwen on-policy SFT iter1 (Variant B)", "runs/phase5_onpolicy_sft_iter1/scored.jsonl"),
    ("qwen on-policy DPO iter1 (Variant C)", "runs/phase5_onpolicy_dpo_iter1/scored.jsonl"),
    ("qwen on-policy SFT iter2 (best)", "runs/phase5_onpolicy_sft_iter2/scored.jsonl"),
    ("qwen on-policy SFT iter3 (plateau)", "runs/phase5_onpolicy_sft_iter3/scored.jsonl"),
    ("qwen on-policy SFT leak-axis only", "runs/phase5_onpolicy_sft_leak_iter1/scored.jsonl"),
    ("qwen on-policy SFT MI-axis only", "runs/phase5_onpolicy_sft_mi_iter1/scored.jsonl"),
    ("mistral SFT+DPO", "runs/phase3_mistral_sft_dpo/scored.jsonl"),
    ("mistral DAPO-v1 step35", "runs/phase3_mistral_dapo_v1_step35/scored.jsonl"),
    ("llama SFT+DPO", "runs/phase3_llama_sft_dpo/scored.jsonl"),
    ("phase 4: claude-sonnet + v4 prompt", "runs/phase4_promptv4_frontier/scored.jsonl"),
]


def summarize(p: str) -> dict | None:
    if not p or not Path(p).exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    n = len(rows)
    out = {"n": n, "harm_total": 0, "leak_total": 0, "bound_total": 0, "mi_total": 0,
           "arms": {}}
    for r in rows:
        h = r.get("harm") or {}
        out["harm_total"] += int(bool(h.get("harm_fire")))
        out["leak_total"] += int((r.get("leak_rate") or 0) > 0)
        out["bound_total"] += int(bool(h.get("leaked_private_bound")))
        out["mi_total"] += int(bool(h.get("missed_instruction")))
    for arm, arows in by_arm.items():
        arm_n = len(arows)
        harm = sum(int(bool((r.get("harm") or {}).get("harm_fire"))) for r in arows)
        leak = sum(int((r.get("leak_rate") or 0) > 0) for r in arows)
        bound = sum(int(bool((r.get("harm") or {}).get("leaked_private_bound"))) for r in arows)
        mi = sum(int(bool((r.get("harm") or {}).get("missed_instruction"))) for r in arows)
        leak_rate = sum((r.get("leak_rate") or 0) for r in arows) / max(1, arm_n)
        out["arms"][arm] = {"n": arm_n, "harm": harm, "leak": leak, "leak_rate": leak_rate,
                            "bound": bound, "mi": mi}
    return out


def main() -> int:
    print("# Variant comparison (current state of runs/)")
    print()
    print("| variant | n | harm | leak | bound | MI | plain h/l | prompted h/l | scaffolded h/l |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, p in VARIANTS:
        s = summarize(p) if p else None
        if not s:
            print(f"| {name} | --- | --- | --- | --- | --- | --- | --- | --- |")
            continue
        a = s["arms"]
        def cell(arm):
            x = a.get(arm)
            if not x: return "---"
            return f"{x['harm']}/{x['leak']}"
        print(f"| {name} | {s['n']} | {s['harm_total']} | {s['leak_total']} | {s['bound_total']} | {s['mi_total']} | "
              f"{cell('plain')} | {cell('prompted')} | {cell('scaffolded')} |")
    print()
    print("Notes:")
    print("- harm/leak in arm columns are fire counts (out of 36 each).")
    print("- bound = leaked_private_bound subflag count.")
    print("- MI = missed_instruction subflag count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
