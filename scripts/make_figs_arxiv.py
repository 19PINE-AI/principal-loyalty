"""Generate publication-quality figures for the arXiv draft.

Style: clean, two-column compatible, ColorBrewer palette.
Output: figures/arxiv_fig{1..6}.pdf
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path("/home/ubuntu/principal-loyalty")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load(p):
    if not Path(p).exists(): return []
    return [json.loads(l) for l in open(p)]


def counts(rows):
    n = len(rows)
    harm = sum(int(bool((r.get("harm") or {}).get("harm_fire"))) for r in rows)
    leak = sum(int((r.get("leak_rate") or 0) > 0) for r in rows)
    bound = sum(int(bool((r.get("harm") or {}).get("leaked_private_bound"))) for r in rows)
    mi = sum(int(bool((r.get("harm") or {}).get("missed_instruction"))) for r in rows)
    return dict(n=n, harm=harm, leak=leak, bound=bound, mi=mi)


# ============================================================
# Figure 1: Manifold scatter — variants on (leak %, MI %) plane
# ============================================================
def fig1_manifold():
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    variants = [
        ("Qwen-8B untrained",   "runs/phase3_baseline_qwen/scored.jsonl",        "tab:gray",    "x"),
        ("Qwen v4.1 SFT+DPO",   "runs/phase2_trained_v4_1/scored.jsonl",         "tab:red",     "o"),
        ("Qwen DAPO-v1 step35", "runs/phase3_dapo_v1_step35/scored.jsonl",       "tab:orange",  "s"),
        ("Per-turn SFT iter1",  "runs/phase5_onpolicy_sft_iter1/scored.jsonl",   "tab:purple",  "D"),
        ("Per-turn SFT iter2",  "runs/phase5_onpolicy_sft_iter2/scored.jsonl",   "tab:purple",  "v"),
        ("Per-token KL iter1",  "runs/phase5_pertoken_kl_iter1/scored.jsonl",    "tab:blue",    "*"),
        ("Per-token KL iter2",  "runs/phase5_pertoken_kl_iter2/scored.jsonl",    "tab:blue",    "P"),
        ("Per-token KL iter3",  "runs/phase5_pertoken_kl_iter3/scored.jsonl",    "tab:blue",    "X"),
        ("Claude + v4 (gold)",  "runs/phase4_promptv4_frontier/scored.jsonl",    "tab:green",   "^"),
    ]
    for name, path, color, marker in variants:
        rows = load(path)
        if not rows: continue
        c = counts(rows)
        leak_pct = 100 * c["leak"] / c["n"]
        mi_pct = 100 * c["mi"] / c["n"]
        ax.scatter(leak_pct, mi_pct, s=160 if marker == "*" else 80,
                   c=color, marker=marker, edgecolors="black", linewidths=0.5,
                   label=f"{name} ({c['harm']}/{c['n']})", zorder=3)
    # Draw approximate manifold (low-leak/low-MI corner)
    ax.axhline(20, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvline(20, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.text(2, 6, "Pareto-better\nregion", fontsize=8, color="gray", alpha=0.6, style="italic")
    ax.set_xlabel("Leak rate (%)")
    ax.set_ylabel("Missed-instruction rate (%)")
    ax.set_title("Leak/MI manifold across PrincipalBench variants (label: harm/n)")
    ax.legend(loc="upper right", framealpha=0.95, ncol=1, fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig1_manifold.pdf", bbox_inches="tight")
    plt.close()
    print("[fig1] saved")


# ============================================================
# Figure 2: K=3 trajectory (per-token KL) — two optima
# ============================================================
def fig2_kiter():
    iters = ["v4.1\nbase", "iter1", "iter2", "iter3"]
    harm = [56, 33, 38, 41]
    leak = [18, 13,  9, 15]
    bound = [4,  3,  2,  4]
    mi    = [51, 32, 35, 40]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(iters))
    ax.plot(x, harm, marker="o", linewidth=2, label="harm", color="tab:red")
    ax.plot(x, mi,   marker="s", linewidth=2, label="MI",   color="tab:orange")
    ax.plot(x, leak, marker="^", linewidth=2, label="leak", color="tab:blue")
    ax.plot(x, bound, marker="D", linewidth=2, label="bound", color="tab:purple")

    # Annotate optima
    ax.annotate("harm-min", xy=(1, 33), xytext=(0.3, 26),
                arrowprops=dict(arrowstyle="->", color="tab:red", alpha=0.6),
                fontsize=8, color="tab:red")
    ax.annotate("leak/bound-min", xy=(2, 9), xytext=(2.5, 18),
                arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=0.6),
                fontsize=8, color="tab:blue")

    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("Failures per 108 trajectories")
    ax.set_title("K-iteration trajectory of per-token KL distillation (Variant 3)")
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig2_kiter.pdf", bbox_inches="tight")
    plt.close()
    print("[fig2] saved")


# ============================================================
# Figure 3: Multi-seed paired Wilcoxon — V3 iter1 + iter2 vs v4.1
# ============================================================
def fig3_wilcoxon():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    metrics = ["harm", "leak", "bound", "MI"]
    v41_mean = [47.8, 15.8, 4.6, 44.4]
    v3i1_mean = [39.2, 13.8, 2.8, 37.2]
    v3i1_err = [4.0, 1.8, 1.5, 3.6]
    v3i2_mean = [41.3, 11.7, 2.7, 40.0]
    v3i2_err = [3.5, 3.8, 0.6, 5.0]
    p_iter1 = [0.0114, 0.534, 0.385, 0.055]
    p_iter2 = [0.0121, 0.214, 0.717, 0.114]

    x = np.arange(len(metrics))
    w = 0.27
    ax.bar(x - w, v41_mean, w, color="tab:red", alpha=0.7, label="v4.1 base (n=5)")
    bars1 = ax.bar(x, v3i1_mean, w, yerr=v3i1_err, color="tab:blue", alpha=0.85,
                   capsize=3, label="V3 iter1 (n=5)")
    bars2 = ax.bar(x + w, v3i2_mean, w, yerr=v3i2_err, color="tab:cyan", alpha=0.85,
                   capsize=3, label="V3 iter2 (n=3)")

    # Annotate p-values above bars
    for i, (p1, p2) in enumerate(zip(p_iter1, p_iter2)):
        tag1 = f"p={p1:.3f}" + ("*" if p1 < 0.05 else "")
        tag2 = f"p={p2:.3f}" + ("*" if p2 < 0.05 else "")
        ax.text(i, max(v3i1_mean[i] + v3i1_err[i], v3i2_mean[i] + v3i2_err[i]) + 2.5,
                f"{tag1}\n{tag2}",
                ha="center", fontsize=7, color="black",
                fontweight="bold" if p1 < 0.05 or p2 < 0.05 else "normal")

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Mean fires per 108 (across eval seeds)")
    ax.set_title("Multi-seed paired Wilcoxon vs v4.1 baseline\n(both per-token KL stopping points hit p < 0.05 on harm)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, max(v41_mean) * 1.45)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig3_wilcoxon.pdf", bbox_inches="tight")
    plt.close()
    print("[fig3] saved")


# ============================================================
# Figure 4: Teacher comparison — Qwen3-32B+v4 vs Claude+v4 (scaffolded only)
# ============================================================
def fig4_teacher():
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    metrics = ["harm", "leak", "bound", "MI"]
    # Qwen3-32B teacher (scaffolded only, n=31): 4, 21, 0, 3
    qwen_pct = [100*4/31, 100*21/31, 100*0/31, 100*3/31]
    # Claude+v4 (scaffolded only, n=36): 6, 6, 1, 6
    claude_pct = [100*6/36, 100*6/36, 100*1/36, 100*6/36]
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, claude_pct, w, color="tab:green", alpha=0.85, label="Claude-Sonnet + v4 (n=36)")
    ax.bar(x + w/2, qwen_pct, w, color="tab:blue", alpha=0.85, label="Qwen3-32B-AWQ + v4 (n=31)")
    for i, (c, q) in enumerate(zip(claude_pct, qwen_pct)):
        ax.text(i - w/2, c + 2, f"{c:.0f}%", ha="center", fontsize=8)
        ax.text(i + w/2, q + 2, f"{q:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Fire rate (%) on scaffolded arm")
    ax.set_title("Teacher self-validation (scaffolded only)\nopen-weight teacher trades leak for harm/MI")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig4_teacher.pdf", bbox_inches="tight")
    plt.close()
    print("[fig4] saved")


# ============================================================
# Figure 5: Counterparty + held-out robustness
# ============================================================
def fig5_robustness():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.3))

    # Left: counterparty
    cps = ["Claude\n(default)", "GPT-5", "Gemini-3\n-flash"]
    pt_kl_harm = [33, 38, 49]
    pt_kl_leak = [13, 14, 20]
    sft_harm   = [36, 34, 41]
    sft_leak   = [24, 17, 30]
    x = np.arange(len(cps))
    w = 0.36
    ax1.bar(x - w/2, pt_kl_harm, w, color="tab:blue", alpha=0.85, label="Per-token KL iter1")
    ax1.bar(x + w/2, sft_harm, w, color="tab:purple", alpha=0.85, label="Per-turn SFT iter2")
    ax1.set_xticks(x); ax1.set_xticklabels(cps)
    ax1.set_ylabel("Harm fires (out of 108)")
    ax1.set_title("Counterparty robustness")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: held-out gap
    sets = ["Training\n(v0, n=108)", "Held-out\n(v0_75, n=72)"]
    pt_kl_iter1 = [33/108*100, 29/72*100]
    sft_iter2   = [36/108*100, 25/72*100]
    scaled3x    = [50/104*100, 40/72*100]
    x2 = np.arange(len(sets))
    w2 = 0.27
    ax2.bar(x2 - w2, pt_kl_iter1, w2, color="tab:blue", alpha=0.85, label="Per-token KL iter1 (113 pts)")
    ax2.bar(x2,      sft_iter2,   w2, color="tab:purple", alpha=0.85, label="Per-turn SFT iter2 (113 pts)")
    ax2.bar(x2 + w2, scaled3x,    w2, color="tab:cyan",   alpha=0.85, label="Per-token KL scaled3x (372 pts)")
    for i in range(2):
        ax2.text(i - w2, pt_kl_iter1[i] + 1, f"{pt_kl_iter1[i]:.0f}%", ha="center", fontsize=7)
        ax2.text(i,      sft_iter2[i] + 1,   f"{sft_iter2[i]:.0f}%",   ha="center", fontsize=7)
        ax2.text(i + w2, scaled3x[i] + 1,    f"{scaled3x[i]:.0f}%",    ha="center", fontsize=7)
    ax2.set_xticks(x2); ax2.set_xticklabels(sets)
    ax2.set_ylabel("Harm rate (%)")
    ax2.set_title("Held-out generalization + data scaling")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig5_robustness.pdf", bbox_inches="tight")
    plt.close()
    print("[fig5] saved")


# ============================================================
# Figure 6: Sample-efficiency / variant comparison
# ============================================================
def fig6_variants():
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    # Distillation variant comparison
    labels = ["v4.1\nbase",
              "Per-turn DPO\n(V2)",
              "Per-turn SFT\n(V1) iter1",
              "Per-turn SFT\n(V1) iter2",
              "Per-token KL\n(V3) iter1",
              "Per-token KL\n(V3) iter2",
              "Claude + v4\n(gold teacher)"]
    harm = [56, 54, 44, 36, 33, 38, 21]
    colors = ["tab:red", "tab:orange", "tab:purple", "mediumpurple",
              "tab:blue", "tab:cyan", "tab:green"]
    sig = ["", "", "p=.10", "p=.10", "p=.011*", "p=.012*", ""]

    x = np.arange(len(labels))
    bars = ax.bar(x, harm, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, (h, s) in enumerate(zip(harm, sig)):
        ax.text(i, h + 1.5, f"{h}", ha="center", fontsize=9, fontweight="bold")
        if s:
            ax.text(i, -3, s, ha="center", fontsize=7, color="black",
                    style="italic" if "*" not in s else "normal",
                    fontweight="bold" if "*" in s else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Harm fires (out of 108)")
    ax.set_title("Distillation variant ladder on Qwen3-8B (* = p<0.05 paired Wilcoxon vs v4.1)")
    ax.set_ylim(-5, 65)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig6_variants.pdf", bbox_inches="tight")
    plt.close()
    print("[fig6] saved")


if __name__ == "__main__":
    fig1_manifold()
    fig2_kiter()
    fig3_wilcoxon()
    fig4_teacher()
    fig5_robustness()
    fig6_variants()
    print("DONE — figures in", FIG_DIR)
