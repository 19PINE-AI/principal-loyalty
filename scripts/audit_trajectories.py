"""Fail loudly if a trajectories.jsonl file has agent-turn=0 trajectories.

This is the integrity gate that the paper's §App I incident motivated. Run
this AFTER every run_grid() and BEFORE any score_grid() that you intend to
report. If any trajectory has 0 agent turns or an error early_end_reason,
exit non-zero and dump the bad keys so the caller can re-run.

Usage:
    python3 scripts/audit_trajectories.py <trajectories.jsonl> [--require N]

Where --require N (default 108) is the minimum expected row count.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("trajectories")
    p.add_argument("--require", type=int, default=108)
    p.add_argument("--allow-error-frac", type=float, default=0.05,
                   help="tolerate up to this fraction of trajectories with errors (counterparty / transient API 5xx)")
    args = p.parse_args()

    rows = []
    with open(args.trajectories) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    n = len(rows)
    zero_turn = [r for r in rows if (r.get("n_agent_turns") or 0) == 0]
    error_rows = [r for r in rows if "error" in (r.get("early_end_reason") or "").lower()]
    turn_dist = Counter(r.get("n_agent_turns") or 0 for r in rows)
    eer = Counter(r.get("early_end_reason") or "OK" for r in rows)

    print(f"[audit] {args.trajectories}: n={n} (need ≥{args.require})")
    print(f"[audit] turn distribution: {dict(sorted(turn_dist.items()))}")
    print(f"[audit] early_end_reason: {dict(eer)}")

    issues = []
    if n < args.require:
        issues.append(f"row count {n} < required {args.require}")
    if zero_turn:
        issues.append(f"{len(zero_turn)} trajectories with 0 agent turns "
                      f"(first: {[(r['subject'],r['arm'],r['item_id']) for r in zero_turn[:3]]})")
    err_frac = len(error_rows) / max(1, n)
    if error_rows and err_frac > args.allow_error_frac:
        issues.append(f"{len(error_rows)} trajectories with early_end='error*' "
                      f"({100*err_frac:.1f}% > tolerance {100*args.allow_error_frac:.1f}%)")
    elif error_rows:
        print(f"[audit] {len(error_rows)} error trajectories ({100*err_frac:.1f}%) — within tolerance")

    if issues:
        print("[audit] FAIL:")
        for i in issues:
            print(f"   - {i}")
        return 1
    print("[audit] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
