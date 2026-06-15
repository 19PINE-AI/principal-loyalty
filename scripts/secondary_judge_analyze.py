"""Paired Wilcoxon under the SECONDARY judge (claude-haiku) vs the primary
(gpt-5-mini), for the Mechanism-2 headline (base vs per-token-KL iter1),
n=5 seeds, 36-item core. Confirms the harm-gain direction is judge-robust.

Reads runs/secondary_judge_rejudge.jsonl (from secondary_judge_rejudge.py)
and the primary scored.jsonl files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import recompute_all as ra  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ra.CORE = ra.core_ids()
BASE = ["phase2_trained_v4_1"] + [f"phase2_trained_v4_1_seed{i}" for i in range(2, 6)]
KL = ["phase5_pertoken_kl_iter1"] + [f"phase5_pertoken_kl_iter1_seed{i}" for i in range(2, 6)]


def secondary_records():
    recs = [json.loads(l) for l in open(ROOT / "runs" / "secondary_judge_rejudge.jsonl") if l.strip()]
    return [r for r in recs if "error" not in r]


def cell_fires_secondary(dirs, key="harm_fire"):
    recs = secondary_records()
    by_dir = {}
    for r in recs:
        by_dir.setdefault(r["run_dir"], {})[(r["item_id"], r["arm"])] = r
    agg = {}
    for d in dirs:
        rows = by_dir.get(d, {})
        for (it, arm), r in rows.items():
            if it not in ra.CORE:
                continue
            agg[(it, arm)] = agg.get((it, arm), 0) + int(bool(r.get(key)))
    return agg


def cell_fires_primary(dirs, key="harm"):
    mf = ra.multiseed_cell_fires_dirs(dirs, key) if hasattr(ra, "multiseed_cell_fires_dirs") else None
    if mf is not None:
        return mf
    agg = {}
    for d in dirs:
        rows = ra.scope(ra.load(ROOT / "runs" / d / "scored.jsonl"))
        for r in rows:
            k = (r["item_id"], r["arm"])
            agg[k] = agg.get(k, 0) + int(ra.fire(r, key))
    return agg


def paired(a, b):
    cells = sorted(set(a) & set(b))
    av = [a[c] for c in cells]
    bv = [b[c] for c in cells]
    if not any(x != y for x, y in zip(av, bv)):
        return None, sum(av), sum(bv), len(cells)
    w = wilcoxon(av, bv, zero_method="zsplit", alternative="two-sided")
    return float(w.pvalue), sum(av), sum(bv), len(cells)


def main():
    n_sec = len(secondary_records())
    print(f"secondary records: {n_sec} (expect ~1080)\n")

    print("=== HARM (judge-determined harm_fire), n=5 seeds, 36-core ===")
    for judge, fa, fb in [
        ("PRIMARY  (gpt-5-mini)", cell_fires_primary(BASE, "harm"), cell_fires_primary(KL, "harm")),
        ("SECONDARY (claude-haiku)", cell_fires_secondary(BASE), cell_fires_secondary(KL)),
    ]:
        p, base_tot, kl_tot, nc = paired(fa, fb)
        bps, kps = base_tot / 5, kl_tot / 5
        sig = "  *SIGNIFICANT" if (p is not None and p < 0.05) else ""
        print(f"  {judge:26s} base {bps:5.1f}/seed -> KL {kps:5.1f}/seed   "
              f"p={p:.4f}{sig}  (n_cells={nc})" if p is not None
              else f"  {judge:26s} base {bps:.1f} -> KL {kps:.1f}  (no nonzero diffs)")

    print("\n=== sub-flag: missed-instruction, secondary judge ===")
    fa = cell_fires_secondary(BASE, "missed_instruction")
    fb = cell_fires_secondary(KL, "missed_instruction")
    p, bt, kt, nc = paired(fa, fb)
    print(f"  SECONDARY MI: base {bt/5:.1f}/seed -> KL {kt/5:.1f}/seed   p={p}")


if __name__ == "__main__":
    main()
