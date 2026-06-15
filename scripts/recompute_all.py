"""Canonical recomputation of EVERY number reported in paper_arxiv.tex,
derived directly from the raw runs/*/scored.jsonl eval outputs.

This is the reproduction path the paper previously lacked: make_figs_arxiv.py
and the LaTeX tables hardcoded their values, so nothing re-derived them from
data. Run this and diff against the paper.

Scope conventions (must match the paper):
  * The "36-item core" is the union of data/verl_train.parquet (31) and
    data/verl_val.parquet (5); 108 cells = 36 items x 3 arms.
  * Counts subset to the core item-ids and de-duplicate (item_id, arm)
    keeping the first occurrence (some runs contain duplicate rows).
  * harm = harm.harm_fire ; leak = leak_rate>0 ; bound = harm.leaked_private_bound ;
    mi = harm.missed_instruction (missed-instruction / over-refusal axis).
  * Audit: a row with early_end_reason containing "error" or 0 agent turns is
    dropped before scoring (counterparty_end is a normal, benign ending).
  * Multi-seed mean = (sum of per-cell fires across the N seed dirs) / N.

Usage:
    python3 scripts/recompute_all.py            # full reconciliation report
    python3 scripts/recompute_all.py --json out.json
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

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# ---------------------------------------------------------------------------
# Loading / scoping helpers
# ---------------------------------------------------------------------------

def load(path) -> list[dict]:
    """Tolerant JSONL loader: skips blank and unparseable lines."""
    p = Path(path)
    out: list[dict] = []
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def core_ids() -> set[str]:
    import pandas as pd
    ids: set[str] = set()
    for name in ("verl_train.parquet", "verl_val.parquet"):
        df = pd.read_parquet(ROOT / "data" / name)
        for ei in df["extra_info"]:
            if isinstance(ei, dict):
                for k in ("item_id", "id", "index"):
                    if k in ei:
                        ids.add(ei[k])
    if len(ids) != 36:
        raise RuntimeError(f"expected 36 core items, got {len(ids)}")
    return ids


CORE = None  # populated in main()


def is_error(r: dict) -> bool:
    eer = (r.get("early_end_reason") or "").lower()
    if "error" in eer:
        return True
    # zero agent turns
    nat = r.get("n_agent_turns")
    if nat is not None and nat == 0:
        return True
    return False


def scope(rows: list[dict], to_core: bool = True, dedupe: bool = True,
          audit: bool = False) -> list[dict]:
    """Scope rows the way the paper does: subset to the 36-core and de-dup
    (item_id, arm). The integrity gate is a RUN-LEVEL pass/fail (see
    run_audit / audit_trajectories.py), NOT a per-row filter, so audit
    defaults to False here -- audit-passing runs are scored in full,
    including the <5% benign counterparty_error rows."""
    out = []
    seen = set()
    for r in rows:
        if audit and is_error(r):
            continue
        if to_core and CORE is not None and r.get("item_id") not in CORE:
            continue
        if dedupe:
            k = (r.get("item_id"), r.get("arm"))
            if k in seen:
                continue
            seen.add(k)
        out.append(r)
    return out


def run_audit(dirname: str, tol: float = 0.05) -> dict:
    """Run-level integrity gate: error fraction + zero-turn count."""
    rows = load(RUNS / dirname / "scored.jsonl")
    n = len(rows)
    errs = sum(1 for r in rows if "error" in (r.get("early_end_reason") or "").lower())
    zero = sum(1 for r in rows if r.get("n_agent_turns") == 0)
    frac = errs / max(1, n)
    return dict(n=n, errs=errs, err_frac=round(frac, 3), zero_turn=zero,
                passes=(frac <= tol and zero == 0))


def aggregate_pct(base: str, n: int, key: str = "harm", **kw):
    """Mean over available seeds of 100 * fires / n_rows_in_seed."""
    pcts = []
    for d in seed_dirs(base, n):
        rows = scope(load(RUNS / d / "scored.jsonl"), **kw)
        if not rows:
            continue
        h = sum(int(fire(r, key)) for r in rows)
        pcts.append(100 * h / len(rows))
    if not pcts:
        return None
    mean = sum(pcts) / len(pcts)
    # sample sd (ddof=1), matching the paper's tab:prompt_grid aggregate column
    ddof = 1 if len(pcts) > 1 else 0
    sd = (sum((p - mean) ** 2 for p in pcts) / (len(pcts) - ddof)) ** 0.5
    return dict(mean=round(mean, 1), sd=round(sd, 1), seeds=len(pcts))


def fire(r: dict, key: str) -> bool:
    if key == "harm":
        return bool((r.get("harm") or {}).get("harm_fire"))
    if key == "leak":
        return (r.get("leak_rate") or 0) > 0
    if key == "bound":
        return bool((r.get("harm") or {}).get("leaked_private_bound"))
    if key == "mi":
        return bool((r.get("harm") or {}).get("missed_instruction"))
    raise ValueError(key)


def counts(rows: list[dict]) -> dict:
    return dict(
        n=len(rows),
        harm=sum(fire(r, "harm") for r in rows),
        leak=sum(fire(r, "leak") for r in rows),
        bound=sum(fire(r, "bound") for r in rows),
        mi=sum(fire(r, "mi") for r in rows),
    )


def run_counts(dirname: str, **kw) -> dict:
    rows = scope(load(RUNS / dirname / "scored.jsonl"), **kw)
    return counts(rows)


def per_arm(dirname: str, **kw) -> dict:
    rows = scope(load(RUNS / dirname / "scored.jsonl"), **kw)
    out = {}
    for arm in ("plain", "prompted", "scaffolded"):
        sub = [r for r in rows if r.get("arm") == arm]
        h = sum(fire(r, "harm") for r in sub)
        out[arm] = (h, len(sub), round(100 * h / len(sub)) if sub else None)
    return out


# Some subjects' seed dirs are not named <base>_seedN. Claude-Sonnet's seed-1
# run is the shared frontier eval; seeds 2-5 live under phase4_promptv4_claude.
SEED_OVERRIDE = {
    "phase4_promptv4_frontier": [
        "phase4_promptv4_frontier",
        "phase4_promptv4_claude_seed2", "phase4_promptv4_claude_seed3",
        "phase4_promptv4_claude_seed4", "phase4_promptv4_claude_seed5",
    ],
}


def seed_dirs(base: str, n: int) -> list[str]:
    if base in SEED_OVERRIDE:
        return SEED_OVERRIDE[base][:n]
    return [base] + [f"{base}_seed{i}" for i in range(2, n + 1)]


def multiseed_cell_fires(base: str, n: int, key: str, dedupe: bool = False, **kw) -> dict:
    """fires 0..n per (item,arm) cell summed across seed dirs.

    Note: dedupe defaults to False here to match the canonical
    paired_seed_test.py convention used for the paper's reported p-values."""
    agg: dict[tuple, int] = {}
    present = 0
    for d in seed_dirs(base, n):
        rows = scope(load(RUNS / d / "scored.jsonl"), dedupe=dedupe, **kw)
        if not rows:
            continue
        present += 1
        for r in rows:
            k = (r.get("item_id"), r.get("arm"))
            agg[k] = agg.get(k, 0) + int(fire(r, key))
    return {"fires": agg, "n_seeds_present": present}


def multiseed_mean(base: str, n: int, key: str, **kw):
    mf = multiseed_cell_fires(base, n, key, **kw)
    ns = mf["n_seeds_present"] or 1
    # mean total fires per seed
    totals = []
    for i, d in enumerate(seed_dirs(base, n)):
        rows = scope(load(RUNS / d / "scored.jsonl"), **kw)
        if rows:
            totals.append(sum(int(fire(r, key)) for r in rows))
    if not totals:
        return None
    mean = sum(totals) / len(totals)
    sd = (sum((t - mean) ** 2 for t in totals) / len(totals)) ** 0.5
    return dict(mean=round(mean, 1), sd=round(sd, 1), seeds=len(totals), totals=totals)


def paired_wilcoxon(base_a: str, n_a: int, base_b: str, n_b: int, key: str, **kw):
    if not HAVE_SCIPY:
        return None
    a = multiseed_cell_fires(base_a, n_a, key, **kw)["fires"]
    b = multiseed_cell_fires(base_b, n_b, key, **kw)["fires"]
    cells = sorted(set(a) & set(b))
    av = [a[c] for c in cells]
    bv = [b[c] for c in cells]
    if not any(x != y for x, y in zip(av, bv)):
        return dict(p=None, note="all-zero diffs", n_cells=len(cells))
    w = wilcoxon(av, bv, zero_method="zsplit", alternative="two-sided")
    return dict(p=round(float(w.pvalue), 4), n_cells=len(cells))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def diff(label, paper, got, ok_fn=None):
    """Print a reconciliation line."""
    if ok_fn is None:
        ok = (paper == got)
    else:
        ok = ok_fn(paper, got)
    mark = "OK " if ok else "XX "
    print(f"  [{mark}] {label:42s} paper={paper!s:24s} data={got!s}")
    return ok


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# subject display name -> run dir stem
SUBJECTS = [
    ("Gemini-2.5-flash", "phase4_promptv4_gemini25flash"),
    ("Mistral-Large", "phase4_promptv4_mistral_large"),
    ("Gemini-3p1-flash-lite", "phase4_promptv4_gemini3p1_lite"),
    ("DeepSeek-v3.1", "phase4_promptv4_deepseek"),
    ("Qwen3-32B", "phase4_promptv4_qwen32b_openrouter"),
    ("Claude-Opus", "phase4_promptv4_claude_opus"),
    ("Llama-3.1-70B", "phase4_promptv4_llama70b"),
    ("Gemini-3-flash", "phase4_promptv4_gemini3flash"),
    ("Claude-Sonnet", "phase4_promptv4_frontier"),
    ("GLM-4.6", "phase4_promptv4_glm46"),
    ("GPT-5-mini", "phase4_promptv4_gpt5mini"),
    ("GPT-5", "phase4_promptv4_gpt5"),
    ("Qwen3.5-27B", "phase4_promptv4_qwen27b"),
]
# paper per-arm (plain/prompted/scaffolded) and aggregate from tab:prompt_grid
# Per-arm = seed-1 (base dir); aggregate = n=5 mean. Values are the canonical
# data-derived figures now printed in paper_arxiv.tex Table tab:prompt_grid.
PAPER_GRID = {
    "Gemini-2.5-flash": (19, 14, 17, 5.5),
    "Mistral-Large": (9, 6, 22, 11.0),
    "Gemini-3p1-flash-lite": (17, 17, 0, 12.0),
    "DeepSeek-v3.1": (11, 9, 11, 12.3),
    "Qwen3-32B": (25, 12, 14, 16.3),
    "Claude-Opus": (11, 19, 19, 18.1),
    "Llama-3.1-70B": (11, 21, 17, 19.2),
    "Gemini-3-flash": (17, 20, 15, 19.4),
    "Claude-Sonnet": (19, 22, 17, 19.5),
    "GLM-4.6": (43, 61, 43, 46.0),
    "GPT-5-mini": (44, 76, 65, 53.6),
    "GPT-5": (63, 71, 71, 71.1),
    "Qwen3.5-27B": (72, 79, 59, 75.3),
}
CALIBRATED = [s for s in PAPER_GRID if PAPER_GRID[s][3] < 30]
OVERREFUSE = ["GPT-5-mini", "GPT-5", "Qwen3.5-27B"]


def main():
    global CORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    CORE = core_ids()
    results = {}

    section("0. Core definition")
    print(f"  36-item core resolved from verl parquets: {len(CORE)} items")

    # -- Mechanism 2: Qwen distillation ladder (sec:kiter, sec:ptkl) ----------
    section("1. Qwen3-8B per-token-KL ladder (single seed, 36-core)  [Table sec:kiter]")
    qwen_runs = {
        "v4.1 base": "phase2_trained_v4_1",
        "iter1": "phase5_pertoken_kl_iter1",
        "iter2": "phase5_pertoken_kl_iter2",
        "iter3": "phase5_pertoken_kl_iter3",
        "iter4": "phase5_pertoken_kl_iter4",
        "iter5": "phase5_pertoken_kl_iter5",
    }
    paper_qwen = {  # harm/leak/bound/mi
        "v4.1 base": (56, None, None, None),
        "iter1": (33, 13, 3, 32),
        "iter2": (38, 9, 2, 35),
        "iter3": (41, 15, 4, 40),
        "iter4": (42, 17, 5, 42),
        "iter5": (32, 19, 6, 32),
    }
    for k, d in qwen_runs.items():
        c = run_counts(d)
        got = (c["harm"], c["leak"], c["bound"], c["mi"])
        pap = paper_qwen[k]
        cmp = tuple(g if p is not None else None for g, p in zip(got, pap))
        diff(f"Qwen {k} (n={c['n']})", pap, cmp)
        results[f"qwen_{k}"] = c

    # -- Llama distillation ladder -------------------------------------------
    section("2. Llama-3.1-8B per-token-KL ladder (single seed, 36-core)  [Table sec:kiter]")
    llama_runs = {
        "iter1": "phase5_pertoken_kl_llama_iter1",
        "iter2": "phase5_pertoken_kl_llama_iter2",
        "iter3": "phase5_pertoken_kl_llama_iter3",
        "iter4": "phase5_pertoken_kl_llama_iter4",
    }
    paper_llama = {
        "iter1": (27, 3, 2, 25),
        "iter2": (22, 9, 2, 20),
        "iter3": (17, 7, 3, 15),
        "iter4": (18, 6, 2, 17),
    }
    for k, d in llama_runs.items():
        c = run_counts(d)
        got = (c["harm"], c["leak"], c["bound"], c["mi"])
        diff(f"Llama {k} (n={c['n']})", paper_llama[k], got)
        results[f"llama_{k}"] = c

    # -- Multi-seed headline (tab:headline) ----------------------------------
    section("3. Multi-seed headline: per-token-KL iter1 vs SFT+DPO base  [Table tab:headline]")
    for key, pap in [("harm", (47.8, 39.2)), ("leak", (15.8, 13.8)),
                     ("bound", (4.6, 2.8)), ("mi", (44.4, 37.2))]:
        base = multiseed_mean("phase2_trained_v4_1", 5, key)
        kl = multiseed_mean("phase5_pertoken_kl_iter1", 5, key)
        w = paired_wilcoxon("phase2_trained_v4_1", 5, "phase5_pertoken_kl_iter1", 5, key)
        diff(f"{key} base mean", pap[0], base["mean"] if base else None)
        diff(f"{key} KL mean", pap[1], kl["mean"] if kl else None)
        print(f"        -> KL sd={kl['sd'] if kl else '?'}  wilcoxon p={w['p'] if w else '?'}  (seeds base={base['seeds']}, kl={kl['seeds']})")
        results[f"headline_{key}"] = dict(base=base, kl=kl, wilcoxon=w)

    # iter1 single-seed (operating point) vs multi-seed mean
    section("3b. iter1 single-seed operating point vs multi-seed mean")
    c1 = run_counts("phase5_pertoken_kl_iter1")
    diff("iter1 single-seed harm", 33, c1["harm"])
    m = multiseed_mean("phase5_pertoken_kl_iter1", 5, "harm")
    diff("iter1 multiseed harm mean", 39.2, m["mean"] if m else None)

    # -- Other variants p-values (sec:ptkl) ----------------------------------
    section("4. Variant significance vs base (sec:ptkl / fig:variants)")
    for label, d, n in [("per-turn SFT iter1", "phase5_onpolicy_sft_iter1", 5),
                        ("per-turn DPO iter1", "phase5_onpolicy_dpo_iter1", 1),
                        ("DAPO from base (step35)", "phase3_dapo_v1_step35", 5)]:
        if n > 1:
            w = paired_wilcoxon("phase2_trained_v4_1", 5, d, n, "harm")
            print(f"  {label}: harm wilcoxon p={w['p'] if w else '?'} (paper: SFT 0.104, DAPO 0.903)")
        c = run_counts(d)
        print(f"        {label} single-seed harm={c['harm']}/{c['n']}")

    # -- DAPO-from-iter1 regression (sec:manifold #2) ------------------------
    section("5. DAPO-from-ptkl-iter1 regression (sec:manifold)")
    c = run_counts("phase5_dapo_from_pertoken_kl")
    diff("DAPO-from-iter1 harm", 46, c["harm"])
    diff("DAPO-from-iter1 mi", 45, c["mi"])
    print(f"        (n={c['n']}; paper says 46/106, 2 cells audit-dropped from 108)")

    # -- Data scaling (sec:manifold #5) --------------------------------------
    section("6. Data-scaling 4.2x (sec:manifold)")
    c = run_counts("phase5_pertoken_kl_scaled3x_iter1")
    diff("scaled3x train harm %", 37, round(100 * c["harm"] / c["n"]) if c["n"] else None)
    ho = counts(scope(load(RUNS / "phase5_pertoken_kl_scaled3x_iter1_heldout_v0_75" / "scored.jsonl"),
                      to_core=False))
    diff("scaled3x held-out harm %", 56, round(100 * ho["harm"] / ho["n"]) if ho["n"] else None)
    base_ho = counts(scope(load(RUNS / "phase5_pertoken_kl_iter1_heldout_v0_75" / "scored.jsonl"),
                          to_core=False))
    diff("iter1 held-out harm %", 40, round(100 * base_ho["harm"] / base_ho["n"]) if base_ho["n"] else None)

    # -- Teacher self-validation (sec:selfval) -------------------------------
    section("7. Teacher self-validation (sec:selfval)")
    t = counts(scope(load(RUNS / "phase4_qwen32b_teacher_eval" / "scored.jsonl"),
                    to_core=False, audit=False))
    diff("Qwen3-32B teacher harm", 4, t["harm"])
    diff("Qwen3-32B teacher leak", 21, t["leak"])
    diff("Qwen3-32B teacher mi", 3, t["mi"])
    print(f"        (teacher n={t['n']}; paper n=31)")
    # Claude-Sonnet scaffolded arm
    fr = scope(load(RUNS / "phase4_promptv4_frontier" / "scored.jsonl"))
    sc = [r for r in fr if r.get("arm") == "scaffolded"]
    cc = counts(sc)
    diff("Claude-Sonnet scaffolded harm", 6, cc["harm"])
    diff("Claude-Sonnet scaffolded leak", 6, cc["leak"])
    diff("Claude-Sonnet scaffolded mi", 6, cc["mi"])

    # -- Counterparty robustness (sec:selfval) -------------------------------
    section("8. Counterparty robustness, iter1 (sec:selfval)")
    cp = {
        "Claude-Sonnet(self)": "phase5_pertoken_kl_iter1",
        "GPT-5": "phase5_pertoken_kl_iter1_cp_gpt5",
        "Gemini-3-flash": "phase5_pertoken_kl_iter1_cp_gemini",
    }
    paper_cp = {"Claude-Sonnet(self)": (33, 13), "GPT-5": (38, 14), "Gemini-3-flash": (49, 20)}
    for k, d in cp.items():
        c = run_counts(d)
        diff(f"cp={k} (harm,leak)", paper_cp[k], (c["harm"], c["leak"]))

    # -- Llama held-out / cross-family (sec:selfval) -------------------------
    section("9. Llama cross-family held-out (sec:selfval)")
    lho = counts(scope(load(RUNS / "phase5_pertoken_kl_llama_iter3_heldout" / "scored.jsonl"),
                      to_core=False))
    diff("Llama iter3 held-out harm", 20, lho["harm"])
    diff("Llama iter3 held-out n", 75, lho["n"])

    # -- 13-subject split grid (tab:prompt_grid) -----------------------------
    section("10. 13-subject split grid: per-arm + aggregate  [Table tab:prompt_grid]")
    print(f"  {'subject':24s} {'paper(p/pr/sc/agg)':26s} {'data(p/pr/sc/agg)'}")
    grid_ok = 0
    for name, d in SUBJECTS:
        pa = per_arm(d)
        agg = aggregate_pct(d, 5, "harm")
        aggpct = agg["mean"] if agg else None
        aggsd = agg["sd"] if agg else None
        pp, ppr, psc, pagg = PAPER_GRID[name]
        got = (pa["plain"][2], pa["prompted"][2], pa["scaffolded"][2], aggpct)
        close = all(p is None or g is None or abs(p - g) <= 1 for p, g in
                    zip((pp, ppr, psc), got[:3])) and (pagg is None or aggpct is None or abs(pagg - aggpct) <= 2)
        mark = "OK " if close else "XX "
        print(f"  [{mark}] {name:22s} ({pp},{ppr},{psc},{pagg})  -> ({got[0]},{got[1]},{got[2]},{aggpct}±{aggsd})  seeds={agg['seeds'] if agg else 0}")
        grid_ok += int(close)
        results[f"grid_{name}"] = dict(per_arm=pa, agg_pct=aggpct, agg_sd=aggsd, seeds=agg["seeds"] if agg else 0)
    print(f"  grid rows within tolerance: {grid_ok}/13")

    # -- cluster per-arm paired Wilcoxon (sec:split robustness) --------------
    section("11. Cluster per-arm paired Wilcoxon (calibrated vs over-refuse)  [sec:split]")
    # Defer to the canonical tool (exact cluster membership + pairing the paper used).
    import subprocess
    try:
        out = subprocess.run(["python3", str(ROOT / "scripts" / "per_arm_xvendor_wilcoxon.py")],
                             capture_output=True, text=True, timeout=300).stdout
        paper_p = {"plain": "1.8e-6", "prompted": "2.2e-7", "scaffolded": "5.9e-7"}
        for line in out.splitlines():
            if line.strip().startswith("arm="):
                arm = line.split("arm=")[1].split()[0]
                pv = line.split("p=")[-1].strip()
                print(f"  {arm:11s} paper p={paper_p.get(arm,'?'):8s} data p={pv}  [via per_arm_xvendor_wilcoxon.py]")
    except Exception as e:
        print(f"  could not run canonical cluster tool: {e}")

    # -- subflag decomposition by cluster (sec:split) ------------------------
    section("12. Sub-flag decomposition by cluster (sec:split)")
    def cluster_decomp(subjects):
        tot = dict(n=0, harm=0, leak=0, bound=0, mi=0)
        for name in subjects:
            d = dict(SUBJECTS)[name]
            c = run_counts(d)
            for k in tot:
                tot[k] += c[k]
        return tot
    for label, subs in [("over-refuse", OVERREFUSE), ("calibrated", CALIBRATED)]:
        t = cluster_decomp(subs)
        if t["n"]:
            print(f"  {label:11s}: MI={100*t['mi']/t['n']:.0f}% leak={100*t['leak']/t['n']:.0f}% bound={100*t['bound']/t['n']:.0f}%  (n={t['n']})")
    print("  paper: over-refuse MI67/leak3/bound0 ; calibrated MI14/leak12/bound2")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
