"""Three-judge inter-rater agreement (pairwise Cohen's κ + Fleiss κ).

Re-judges the n=60 trajectory subset that was already scored by gpt-5-mini
(primary) and claude-haiku (secondary). Adds claude-sonnet (tertiary) and
reports:
  - pairwise raw agreement and Cohen's κ for all 3 pairs
  - Fleiss κ across all 3 raters
per sub-flag and any-fire.

Reuses scripts/dual_judge_kappa.py:judge_harm to keep the prompt + parsing
identical.

Usage:
  python3 scripts/three_judge_kappa.py \\
    --scored runs/phase2_baseline/scored.jsonl \\
    --trajectories runs/phase2_baseline/trajectories.jsonl \\
    --items items/v0 \\
    --secondary-kappa runs/phase2_baseline/harm_kappa.json \\
    --tertiary claude-sonnet \\
    --out runs/phase2_baseline/three_judge_kappa.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.items import load_items  # noqa: E402
from src.vendors import get_vendor  # noqa: E402

from scripts.dual_judge_kappa import judge_harm  # noqa: E402


FLAGS = ["fabrication", "deception", "against_principal", "any_fire"]


def cohen_kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1:
        return agree, 1.0
    kappa = (agree - pe) / (1 - pe)
    return agree, kappa


def fleiss_kappa(ratings: list[list[int]]) -> float:
    """Fleiss κ on N items × K raters with binary categories (0/1)."""
    N = len(ratings)
    if N == 0:
        return 0.0
    K = len(ratings[0])
    # n_ij = number of raters who assigned category j to item i
    pj = [0.0, 0.0]
    pj_count = [0, 0]
    Pi = []
    for row in ratings:
        n0 = sum(1 for r in row if r == 0)
        n1 = K - n0
        pj_count[0] += n0
        pj_count[1] += n1
        Pi.append((n0 * (n0 - 1) + n1 * (n1 - 1)) / (K * (K - 1)) if K > 1 else 0)
    total = K * N
    pj[0] = pj_count[0] / total
    pj[1] = pj_count[1] / total
    P_bar = sum(Pi) / N
    P_e_bar = pj[0] ** 2 + pj[1] ** 2
    if P_e_bar >= 1:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--items", default="items/v0")
    p.add_argument("--secondary-kappa", required=True,
                   help="path to existing dual_judge_kappa.py output (has secondary results)")
    p.add_argument("--tertiary", default="claude-sonnet")
    p.add_argument("--out", required=True)
    p.add_argument("--max-n", type=int, default=60)
    p.add_argument("--parallel", type=int, default=4)
    args = p.parse_args()

    items = {it.id: it for it in load_items(args.items)}
    scored = [json.loads(l) for l in Path(args.scored).read_text().splitlines() if l.strip()]
    trajs = [json.loads(l) for l in Path(args.trajectories).read_text().splitlines() if l.strip()]
    traj_by_key = {(t["subject"], t["arm"], t["item_id"]): t for t in trajs}
    sec = json.loads(Path(args.secondary_kappa).read_text())
    # The secondary kappa file has per-evaluation calls cached implicitly; we need
    # to re-derive its labels from a re-run if not stored. Simpler: just re-run BOTH
    # secondary (claude-haiku) AND tertiary (claude-sonnet) for parity.

    # Take the first max_n scored rows as our subset
    subset = scored[:args.max_n]
    print(f"[3-judge] subset n={len(subset)}; secondary={sec.get('secondary_judge')}; tertiary={args.tertiary}")

    primary_judge = get_vendor("gpt-5-mini")
    secondary_judge = get_vendor(sec.get("secondary_judge", "claude-haiku"))
    tertiary_judge = get_vendor(args.tertiary)

    def _judge_row(row, judge):
        key = (row["subject"], row["arm"], row["item_id"])
        traj = traj_by_key.get(key)
        if not traj:
            return None
        it = items.get(row["item_id"])
        if not it:
            return None
        try:
            return judge_harm(traj, it, judge)
        except Exception as e:
            print(f"  FAIL {key}: {type(e).__name__}: {e}")
            return None

    # Run all 3 judges in parallel
    primary_labels: dict[int, dict] = {}
    secondary_labels: dict[int, dict] = {}
    tertiary_labels: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=args.parallel * 3) as ex:
        futs = {}
        for i, row in enumerate(subset):
            futs[ex.submit(_judge_row, row, primary_judge)] = (i, "primary")
            futs[ex.submit(_judge_row, row, secondary_judge)] = (i, "secondary")
            futs[ex.submit(_judge_row, row, tertiary_judge)] = (i, "tertiary")
        for fut in as_completed(futs):
            i, which = futs[fut]
            r = fut.result()
            if r is None:
                continue
            if which == "primary":
                primary_labels[i] = r
            elif which == "secondary":
                secondary_labels[i] = r
            else:
                tertiary_labels[i] = r

    indices = sorted(set(primary_labels) & set(secondary_labels) & set(tertiary_labels))
    print(f"[3-judge] all-3 labels for {len(indices)} of {len(subset)}")

    out = {
        "primary_judge": "gpt-5-mini",
        "secondary_judge": sec.get("secondary_judge"),
        "tertiary_judge": args.tertiary,
        "scored_path": args.scored,
        "n_evaluations": len(indices),
        "pairs": {},
        "fleiss": {},
        "per_judge_positive_rate": {},
    }

    for flag in FLAGS:
        a = [int(bool(primary_labels[i].get(flag))) for i in indices]
        b = [int(bool(secondary_labels[i].get(flag))) for i in indices]
        c = [int(bool(tertiary_labels[i].get(flag))) for i in indices]
        out["per_judge_positive_rate"][flag] = {
            "primary": sum(a) / max(1, len(a)),
            "secondary": sum(b) / max(1, len(b)),
            "tertiary": sum(c) / max(1, len(c)),
        }
        ag_pri_sec, k_pri_sec = cohen_kappa(a, b)
        ag_pri_ter, k_pri_ter = cohen_kappa(a, c)
        ag_sec_ter, k_sec_ter = cohen_kappa(b, c)
        out["pairs"][flag] = {
            "primary_vs_secondary": {"agreement": ag_pri_sec, "kappa": k_pri_sec},
            "primary_vs_tertiary":  {"agreement": ag_pri_ter, "kappa": k_pri_ter},
            "secondary_vs_tertiary":{"agreement": ag_sec_ter, "kappa": k_sec_ter},
        }
        ratings = list(zip(a, b, c))
        out["fleiss"][flag] = fleiss_kappa([list(r) for r in ratings])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[3-judge] wrote {args.out}")
    print(json.dumps(out["pairs"], indent=2))
    print("Fleiss κ:", out["fleiss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
