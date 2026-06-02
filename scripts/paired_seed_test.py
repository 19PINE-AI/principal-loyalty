"""Paired Wilcoxon signed-rank test on cell-level harm/leak fire counts
across two checkpoints, at matched seed budget.

For each (item_id, arm) cell we count fires-out-of-N-seeds for each
checkpoint. Wilcoxon signed-rank on the paired differences is robust to
the non-normal distribution of small fire-count integers and respects
per-cell pairing.

Usage:
    python3 scripts/paired_seed_test.py \\
        --a runs/phase2_trained_v4_1 --a-seeds 5 \\
        --b runs/phase3_dapo_v1_step35 --b-seeds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def seed_dirs(base: Path, n_seeds: int) -> list[Path]:
    out = [base]
    for i in range(2, n_seeds + 1):
        out.append(Path(f"{base}_seed{i}"))
    return out


def cell_fires(dirs: list[Path], key: str) -> dict[tuple[str, str], int]:
    """Count fires (0..n_seeds) for each (item, arm) cell on a given key."""
    out: dict[tuple[str, str], int] = {}
    for d in dirs:
        p = d / "scored.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                k = (r["item_id"], r["arm"])
                if key == "harm":
                    v = bool(r.get("harm", {}).get("harm_fire"))
                elif key == "leak":
                    v = (r.get("leak_rate") or 0) > 0
                elif key == "bound":
                    v = bool(r.get("harm", {}).get("leaked_private_bound"))
                elif key == "mi":
                    v = bool(r.get("harm", {}).get("missed_instruction"))
                else:
                    raise ValueError(f"unknown key {key}")
                out[k] = out.get(k, 0) + int(v)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="checkpoint A base dir")
    p.add_argument("--b", required=True, help="checkpoint B base dir")
    p.add_argument("--a-seeds", type=int, default=5)
    p.add_argument("--b-seeds", type=int, default=5)
    args = p.parse_args()

    a_dirs = seed_dirs(Path(args.a), args.a_seeds)
    b_dirs = seed_dirs(Path(args.b), args.b_seeds)

    for key in ["harm", "leak", "bound", "mi"]:
        a = cell_fires(a_dirs, key)
        b = cell_fires(b_dirs, key)
        cells = sorted(set(a) & set(b))
        a_vec = [a[c] for c in cells]
        b_vec = [b[c] for c in cells]
        a_total = sum(a_vec)
        b_total = sum(b_vec)
        a_per_seed = a_total / args.a_seeds
        b_per_seed = b_total / args.b_seeds
        # Robust: count cells fire-under-all and fire-under-any
        a_all = sum(1 for c in cells if a[c] == args.a_seeds)
        b_all = sum(1 for c in cells if b[c] == args.b_seeds)
        a_any = sum(1 for c in cells if a[c] > 0)
        b_any = sum(1 for c in cells if b[c] > 0)
        diffs = [bi - ai for ai, bi in zip(a_vec, b_vec)]
        nonzero = [d for d in diffs if d != 0]
        result = {
            "key": key,
            "n_cells": len(cells),
            "a_per_seed_mean": round(a_per_seed, 1),
            "b_per_seed_mean": round(b_per_seed, 1),
            "a_robust(all_seeds_fire)": a_all,
            "b_robust(all_seeds_fire)": b_all,
            "a_any(>=1_seed_fires)": a_any,
            "b_any(>=1_seed_fires)": b_any,
            "n_nonzero_pairs": len(nonzero),
        }
        if HAVE_SCIPY and len(nonzero) >= 1:
            w = wilcoxon(a_vec, b_vec, zero_method="zsplit", alternative="two-sided")
            result["wilcoxon_W"] = round(float(w.statistic), 2)
            result["wilcoxon_p"] = round(float(w.pvalue), 4)
        else:
            result["wilcoxon"] = "scipy not available or no nonzero pairs"
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
