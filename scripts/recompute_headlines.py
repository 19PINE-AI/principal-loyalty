"""Recompute every paper headline number from current scored.jsonl files,
plus an audit column flagging contaminated evals (single-turn / auth-error).

Output: stdout table + writes runs/HEADLINES.md for paste-into-paper.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: str) -> tuple[list, dict]:
    if not Path(path).exists():
        return [], {"missing": True}
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    errs = sum(
        "error" in (r.get("early_end_reason") or "").lower() for r in rows
    )
    turns = Counter(r.get("turns", 0) for r in rows)
    return rows, {
        "n": len(rows),
        "errs": errs,
        "turns": dict(turns),
        "is_truncated": (
            errs == len(rows) > 0
            or (len(turns) == 1 and 1 in turns)
        ),
    }


def aggregate(rows: list) -> dict:
    harm = sum(bool(r.get("harm", {}).get("harm_fire")) for r in rows)
    leak = sum((r.get("leak_rate") or 0) > 0 for r in rows)
    bound = sum(bool(r.get("harm", {}).get("leaked_private_bound")) for r in rows)
    mi = sum(bool(r.get("harm", {}).get("missed_instruction")) for r in rows)
    by_arm = defaultdict(lambda: {"n": 0, "leak": 0, "harm": 0})
    by_cell = defaultdict(lambda: {"n": 0, "leak": 0, "harm": 0})
    for r in rows:
        a = r.get("arm", "?")
        c = r.get("cell", "?")
        by_arm[a]["n"] += 1
        by_arm[a]["leak"] += int((r.get("leak_rate") or 0) > 0)
        by_arm[a]["harm"] += int(bool(r.get("harm", {}).get("harm_fire")))
        by_cell[c]["n"] += 1
        by_cell[c]["leak"] += int((r.get("leak_rate") or 0) > 0)
        by_cell[c]["harm"] += int(bool(r.get("harm", {}).get("harm_fire")))
    return {
        "harm": harm,
        "leak": leak,
        "bound_leak": bound,
        "missed_instruction": mi,
        "n": len(rows),
        "by_arm": dict(by_arm),
        "by_cell": dict(by_cell),
    }


HEADLINES = [
    ("untrained baseline (Qwen3-8B)", "runs/phase3_baseline_qwen/scored.jsonl"),
    ("DPO endpoint v4 (post-SFT+DPO v4)", "runs/phase2_trained_v4/scored.jsonl"),
    ("DPO endpoint v4.1 (HEADLINE BASELINE)", "runs/phase2_trained_v4_1/scored.jsonl"),
    ("DAPO-v1 step_30", "runs/phase3_dapo_v1_step30/scored.jsonl"),
    ("DAPO-v1 step_35 (HEADLINE)", "runs/phase3_dapo_v1_step35/scored.jsonl"),
    ("DAPO-v2 step_55", "runs/phase3_dapo_v2_step55/scored.jsonl"),
    ("DAPO-leak-only step_35", "runs/phase3_dapo_leakonly_step35/scored.jsonl"),
    ("DAPO-v3 step_30", "runs/phase3_dapo_v3_step30/scored.jsonl"),
    ("DAPO-v3 step_55", "runs/phase3_dapo_v3_step55/scored.jsonl"),
    ("DAPO-v1 n=51 step_55", "runs/phase3_dapo_v1_n51_step55/scored.jsonl"),
    ("counterparty: Claude (DAPO-v1 step_35)", "runs/phase3_dapo_v1_step35/scored.jsonl"),
    ("counterparty: GPT-5", "runs/phase3_dapo_v1_step35_gpt5cp/scored.jsonl"),
    ("counterparty: Gemini-3-flash", "runs/phase3_dapo_v1_step35_gemcp/scored.jsonl"),
    ("Mistral SFT+DPO v4.1", "runs/phase3_mistral_sft_dpo/scored.jsonl"),
]


def main() -> int:
    lines = ["# Headline numbers (recomputed)\n"]
    print(f"{'label':50s}  {'audit':12s}  {'harm':>10s}  {'leak':>10s}  arm-leak (plain/prompted/scaffolded)")
    for label, path in HEADLINES:
        rows, audit = load(path)
        if audit.get("missing"):
            print(f"{label:50s}  MISSING")
            lines.append(f"- **{label}** — *missing*")
            continue
        if audit.get("is_truncated"):
            audit_tag = f"BROKEN_AUTH ({audit['errs']}/{audit['n']})"
        else:
            audit_tag = "OK"
        agg = aggregate(rows)
        n = agg["n"]
        plain = agg["by_arm"].get("plain", {"leak": 0, "n": 0})
        prom = agg["by_arm"].get("prompted", {"leak": 0, "n": 0})
        scaf = agg["by_arm"].get("scaffolded", {"leak": 0, "n": 0})
        plain_pct = 100 * plain["leak"] / plain["n"] if plain["n"] else 0
        prom_pct = 100 * prom["leak"] / prom["n"] if prom["n"] else 0
        scaf_pct = 100 * scaf["leak"] / scaf["n"] if scaf["n"] else 0
        print(
            f"{label:50s}  {audit_tag:12s}  {agg['harm']:>4d}/{n:<4d}  "
            f"{agg['leak']:>4d}/{n:<4d}  "
            f"{plain_pct:5.1f}% / {prom_pct:5.1f}% / {scaf_pct:5.1f}%  "
            f"bound={agg['bound_leak']} MI={agg['missed_instruction']}"
        )
        lines.append(
            f"- **{label}** — harm {agg['harm']}/{n}, leak {agg['leak']}/{n} "
            f"(plain {plain_pct:.1f}%, prompted {prom_pct:.1f}%, scaffolded {scaf_pct:.1f}%) "
            f"bound={agg['bound_leak']} MI={agg['missed_instruction']} "
            f"[{audit_tag}]"
        )

    Path("runs/HEADLINES.md").write_text("\n".join(lines))
    print(f"\nwrote runs/HEADLINES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
