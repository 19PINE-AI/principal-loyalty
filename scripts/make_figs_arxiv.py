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
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2,
    "patch.linewidth": 0.8,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.92,
})

# Visual palette — three semantic colors used across figures
C_BASE     = "#C0504D"  # warm red — baselines / problem
C_MECH1    = "#4F81BD"  # blue — Mechanism 1 / scaffold
C_MECH2    = "#9BBB59"  # green — Mechanism 2 / per-token KL teacher
C_TRAINED  = "#8064A2"  # purple — trained students
C_GOLD     = "#E8A33D"  # gold — gold-teacher / Claude
C_NEUTRAL  = "#7F7F7F"  # gray — neutral / ancillary

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
# Figure 0: Teaser — 3-role architecture of multi-party loyalty
# ============================================================
def fig0_problem():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.set_xlim(0, 21); ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, label, sub, color, edge):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
            linewidth=1.5, edgecolor=edge, facecolor=color, alpha=0.92))
        ax.text(x + w/2, y + h - 0.45, label, ha="center", va="top",
                fontsize=11, fontweight="bold")
        ax.text(x + w/2, y + 0.45, sub, ha="center", va="bottom",
                fontsize=8.5, style="italic", color="#444")

    # Three role boxes: narrower + larger gaps so arrow labels fit between
    # without bleeding into box edges.
    box(0.3, 3.4, 3.4, 2.4, "PRINCIPAL",   "user / company\nyou represent",    "#FBEEEA", "#C0504D")
    box(8.8, 3.4, 3.4, 2.4, "AGENT",       "LLM acting\non P's behalf",        "#EAF1F8", "#4F81BD")
    box(17.3, 3.4, 3.4, 2.4, "COUNTERPARTY","other person /\norg (may conflict)","#EFE6F3", "#8064A2")

    # P <-> A — back-and-forth: instructions one way, results the other
    a1a = FancyArrowPatch((3.7, 4.85), (8.8, 4.85),
                          arrowstyle="-|>", mutation_scale=13,
                          color="#C0504D", linewidth=1.5)
    a1b = FancyArrowPatch((8.8, 4.15), (3.7, 4.15),
                          arrowstyle="-|>", mutation_scale=13,
                          color="#C0504D", linewidth=1.5)
    ax.add_patch(a1a); ax.add_patch(a1b)
    ax.text(6.25, 5.20, "briefing, requests", ha="center", fontsize=8, color="#C0504D")
    ax.text(6.25, 3.55, "results, clarifications", ha="center", fontsize=8, color="#C0504D")

    # A <-> C — represents one way, probes/pressure the other
    a2 = FancyArrowPatch((12.2, 4.85), (17.3, 4.85),
                         arrowstyle="-|>", mutation_scale=13,
                         color="#666", linewidth=1.5)
    a3 = FancyArrowPatch((17.3, 4.15), (12.2, 4.15),
                         arrowstyle="-|>", mutation_scale=13,
                         color="#8064A2", linewidth=1.5)
    ax.add_patch(a2); ax.add_patch(a3)
    ax.text(14.75, 5.20, "represents", ha="center", fontsize=8, color="#666")
    ax.text(14.75, 3.55, "probes, pressure", ha="center", fontsize=8, color="#8064A2")

    # Loyalty axes panel (bottom)
    ax.add_patch(FancyBboxPatch(
        (0.6, 0.2), 19.8, 2.6, boxstyle="round,pad=0.10,rounding_size=0.20",
        linewidth=1.0, edgecolor="#888", facecolor="#FAFAFA"))
    ax.text(10.5, 2.35, "loyalty failure axes  —  the agent fails by...",
            ha="center", fontsize=10, fontweight="bold", color="#222")
    axes_list = [
        ("× leak",        "revealing\nwithheld facts"),
        ("× capitulate",  "conceding\nunder pressure"),
        ("× posture",     "signaling\nmotivation"),
        ("× over-refuse", "declining\nP-authorized asks"),
    ]
    xs = [3.0, 8.0, 13.0, 18.0]
    for x, (k, v) in zip(xs, axes_list):
        ax.text(x, 1.55, k, ha="center", fontsize=10, fontweight="bold", color="#C0504D")
        ax.text(x, 0.75, v, ha="center", fontsize=8.5, color="#444", style="italic")

    plt.savefig(FIG_DIR / "arxiv_fig0_problem.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close()
    print("[fig0] saved")


# ============================================================
# Figure 0b: Six failure modes — annotated grid
# ============================================================
def fig0b_cells():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 18); ax.set_ylim(0, 7.4)
    ax.set_aspect("equal")
    ax.axis("off")

    cells = [
        # (col, row, name, axis_tag, example_lines, face, edge)
        (0, 1, "LEAKAGE",     "× leak",
         ['C: "what\'s the minimum?"', 'A: "$12,000."'],
         "#FBEEEA", "#C0504D"),
        (1, 1, "CAPITULATION","× capitulate / bound",
         ['C: "take $11,500 or I walk"', 'A: "$11,500 works."'],
         "#FBEEEA", "#C0504D"),
        (2, 1, "POSTURE",     "× posture",
         ['C: "are you flexible?"', 'A: "she needs to sell."'],
         "#FBEEEA", "#C0504D"),
        (0, 0, "AUTHORING",   "× leak (via artifact)",
         ['P: "draft my listing."', 'A: "...willing to take $12k."'],
         "#FBEEEA", "#C0504D"),
        (1, 0, "MODERATION",  "× third-party leak",
         ['C: "who else saw it?"', 'A: names other witnesses.'],
         "#FBEEEA", "#C0504D"),
        (2, 0, "SANITY",      "× over-refuse",
         ['P: "draft my self-review."', 'A: "can\'t share private info."'],
         "#EAF1F8", "#4F81BD"),
    ]

    cw, ch = 6.0, 3.10
    pad = 0.10
    gap_y = 0.20
    for (col, row, name, axis, lines, face, edge) in cells:
        x = 0.0 + col * cw
        y = 0.2 + row * (ch + gap_y)
        ax.add_patch(FancyBboxPatch(
            (x + pad, y + pad), cw - 2*pad, ch - 2*pad,
            boxstyle="round,pad=0.08,rounding_size=0.16",
            linewidth=1.3, edgecolor=edge, facecolor=face, alpha=0.92))
        ax.text(x + cw/2, y + ch - 0.45, name,
                ha="center", va="top", fontsize=11,
                fontweight="bold", color=edge)
        ax.text(x + cw/2, y + ch - 1.00, axis,
                ha="center", va="top", fontsize=9,
                style="italic", color=edge)
        for li, line in enumerate(lines):
            ax.text(x + cw/2, y + 1.10 - li * 0.45, line,
                    ha="center", va="center", fontsize=8.5, color="#222",
                    family="monospace")

    ax.text(9.0, 7.10, "Six failure cells  (one benchmark cell each)",
            ha="center", fontsize=12, fontweight="bold")

    plt.savefig(FIG_DIR / "arxiv_fig0b_cells.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close()
    print("[fig0b] saved")


# ============================================================
# Figure 1: Manifold scatter — variants on (leak %, MI %) plane
# ============================================================
def fig1_manifold():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    variants = [
        ("Qwen-8B untrained",   "runs/phase3_baseline_qwen/scored.jsonl",        C_NEUTRAL,  "x"),
        ("Qwen v4.1 SFT+DPO",   "runs/phase2_trained_v4_1/scored.jsonl",         C_BASE,     "o"),
        ("Qwen DAPO-v1",        "runs/phase3_dapo_v1_step35/scored.jsonl",       C_GOLD,     "s"),
        ("Per-turn SFT i1",     "runs/phase5_onpolicy_sft_iter1/scored.jsonl",   C_TRAINED,  "D"),
        ("Per-turn SFT i2",     "runs/phase5_onpolicy_sft_iter2/scored.jsonl",   C_TRAINED,  "v"),
        ("Per-token KL i1",     "runs/phase5_pertoken_kl_iter1/scored.jsonl",    C_MECH1,    "*"),
        ("Per-token KL i2",     "runs/phase5_pertoken_kl_iter2/scored.jsonl",    C_MECH1,    "P"),
        ("Per-token KL i3",     "runs/phase5_pertoken_kl_iter3/scored.jsonl",    C_MECH1,    "X"),
        ("Claude + scaffold",   "runs/phase4_promptv4_frontier/scored.jsonl",    C_MECH2,    "^"),
    ]
    handles = []
    for name, path, color, marker in variants:
        rows = load(path)
        if not rows: continue
        c = counts(rows)
        leak_pct = 100 * c["leak"] / c["n"]
        mi_pct = 100 * c["mi"] / c["n"]
        size = 220 if marker == "*" else 90
        edge = "black" if marker not in ("x",) else color
        sc = ax.scatter(leak_pct, mi_pct, s=size,
                        c=color, marker=marker, edgecolors=edge, linewidths=0.5,
                        label=f"{name} ({c['harm']}/{c['n']})", zorder=3)
        handles.append(sc)
    # Shaded "Pareto-better" corner
    ax.axhline(20, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axvline(20, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhspan(0, 20, xmin=0, xmax=20/45, alpha=0.05, color="#3a8c3a")
    # No inline label — the shaded corner is self-evident; explaining it
    # in the caption avoids overlap with axis ticks / data markers.
    ax.set_xlabel("Leak rate (%)")
    ax.set_ylabel("Missed-instruction rate (%)")
    ax.set_title("Leak / MI floor across variants  (label: harm/n)")
    # Legend in a column outside the data area to avoid overlap.
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95, handletextpad=0.5,
              borderpad=0.4, labelspacing=0.4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig1_manifold.pdf", bbox_inches="tight")
    plt.close()
    print("[fig1] saved")


# ============================================================
# Figure 2: K=3 trajectory (per-token KL) — two optima
# ============================================================
def fig2_kiter():
    iters = ["v4.1\nbase", "iter1", "iter2", "iter3", "iter4", "iter5"]
    harm  = [56, 33, 38, 41, 42, 32]
    leak  = [18, 13,  9, 15, 17, 19]
    bound = [ 4,  3,  2,  4,  5,  6]
    mi    = [51, 32, 35, 40, 42, 32]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(iters))
    ax.plot(x, harm,  marker="o", linewidth=2.2, label="harm",  color=C_BASE)
    ax.plot(x, mi,    marker="s", linewidth=2.2, label="MI",    color=C_GOLD)
    ax.plot(x, leak,  marker="^", linewidth=2.2, label="leak",  color=C_MECH1)
    ax.plot(x, bound, marker="D", linewidth=2.2, label="bound", color=C_TRAINED)

    # Annotate optima — placed clear of data lines
    ax.annotate("harm-min", xy=(1, 33), xytext=(0.05, 22),
                arrowprops=dict(arrowstyle="->", color=C_BASE, alpha=0.6),
                fontsize=9, color=C_BASE)
    ax.annotate("leak/bound-min", xy=(2, 9), xytext=(2.6, 1.5),
                arrowprops=dict(arrowstyle="->", color=C_MECH1, alpha=0.6),
                fontsize=9, color=C_MECH1)

    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("Failures per 108 trajectories")
    ax.set_title("Per-token KL on Qwen3-8B across iterations")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 62)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig2_kiter.pdf", bbox_inches="tight")
    plt.close()
    print("[fig2] saved")


# ============================================================
# Figure 3: Multi-seed paired Wilcoxon — V3 iter1 + iter2 vs v4.1
# ============================================================
def fig3_wilcoxon():
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    metrics = ["harm", "leak", "bound", "MI"]
    v41_mean  = [47.8, 15.8, 4.6, 44.4]
    v3i1_mean = [39.2, 13.8, 2.8, 37.2]
    v3i1_err  = [4.0, 1.8, 1.5, 3.6]
    v3i2_mean = [41.5, 11.2, 2.5, 40.5]
    v3i2_err  = [3.0, 3.5, 0.5, 4.5]
    p_iter1   = [0.0114, 0.534, 0.385, 0.055]
    p_iter2   = [0.0436, 0.177, 0.592, 0.214]

    x = np.arange(len(metrics))
    w = 0.27
    ax.bar(x - w, v41_mean,  w, color=C_BASE,    alpha=0.85, label="v4.1 base (n=5)")
    ax.bar(x,     v3i1_mean, w, yerr=v3i1_err, color=C_MECH1, alpha=0.85,
           capsize=3, label="KL iter1 (n=5)")
    ax.bar(x + w, v3i2_mean, w, yerr=v3i2_err, color=C_MECH2, alpha=0.85,
           capsize=3, label="KL iter2 (n=3)")

    # Compact p-value annotations above each pair
    for i, (p1, p2) in enumerate(zip(p_iter1, p_iter2)):
        y_top = max(v3i1_mean[i] + v3i1_err[i], v3i2_mean[i] + v3i2_err[i])
        tag1 = f"{p1:.3f}" + ("*" if p1 < 0.05 else "")
        tag2 = f"{p2:.3f}" + ("*" if p2 < 0.05 else "")
        ax.text(i, y_top + 1.5, f"i1: {tag1}\ni2: {tag2}",
                ha="center", fontsize=8,
                fontweight="bold" if (p1 < 0.05 or p2 < 0.05) else "normal")

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Mean fires per 108 (n=5 seeds)")
    ax.set_title("Multi-seed paired Wilcoxon vs v4.1 base")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(v41_mean) * 1.50)
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
    qwen_pct   = [100*4/31, 100*21/31, 100*0/31, 100*3/31]
    claude_pct = [100*6/36, 100*6/36,  100*1/36, 100*6/36]
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, claude_pct, w, color=C_MECH2, alpha=0.88, label="Claude-Sonnet (n=36)")
    ax.bar(x + w/2, qwen_pct,   w, color=C_MECH1, alpha=0.88, label="Qwen3-32B (n=31)")
    for i, (c, q) in enumerate(zip(claude_pct, qwen_pct)):
        ax.text(i - w/2, c + 2.5, f"{c:.0f}%", ha="center", fontsize=9)
        ax.text(i + w/2, q + 2.5, f"{q:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Fire rate (%) on scaffolded arm")
    ax.set_title("Teacher self-validation: open-weight trades leak for harm")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig4_teacher.pdf", bbox_inches="tight")
    plt.close()
    print("[fig4] saved")


# ============================================================
# Figure 5: Counterparty + held-out robustness
# ============================================================
def fig5_robustness():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    # Left: counterparty robustness
    cps = ["Claude\n(default)", "GPT-5", "Gemini-3\nflash"]
    pt_kl_harm = [33, 38, 49]
    sft_harm   = [36, 34, 41]
    x = np.arange(len(cps))
    w = 0.36
    ax1.bar(x - w/2, pt_kl_harm, w, color=C_MECH1,   alpha=0.88, label="Per-token KL i1")
    ax1.bar(x + w/2, sft_harm,   w, color=C_TRAINED, alpha=0.88, label="Per-turn SFT i2")
    for i in range(len(cps)):
        ax1.text(i - w/2, pt_kl_harm[i] + 1.0, f"{pt_kl_harm[i]}", ha="center", fontsize=9)
        ax1.text(i + w/2, sft_harm[i] + 1.0,   f"{sft_harm[i]}",   ha="center", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(cps)
    ax1.set_ylabel("Harm fires / 108")
    ax1.set_title("Counterparty robustness")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylim(0, 60)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: held-out gap + data scaling
    sets = ["Training\n(36 items)", "Held-out\n(24 items)"]
    pt_kl_iter1 = [33/108*100, 29/72*100]
    sft_iter2   = [36/108*100, 25/72*100]
    scaled3x    = [50/104*100, 40/72*100]
    x2 = np.arange(len(sets))
    w2 = 0.27
    ax2.bar(x2 - w2, pt_kl_iter1, w2, color=C_MECH1,   alpha=0.88, label="KL i1 (113 pts)")
    ax2.bar(x2,      sft_iter2,   w2, color=C_TRAINED, alpha=0.88, label="SFT i2 (113 pts)")
    ax2.bar(x2 + w2, scaled3x,    w2, color=C_GOLD,    alpha=0.88, label="KL scaled3x (372 pts)")
    for i in range(2):
        ax2.text(i - w2, pt_kl_iter1[i] + 1.2, f"{pt_kl_iter1[i]:.0f}%", ha="center", fontsize=9)
        ax2.text(i,      sft_iter2[i]   + 1.2, f"{sft_iter2[i]:.0f}%",   ha="center", fontsize=9)
        ax2.text(i + w2, scaled3x[i]    + 1.2, f"{scaled3x[i]:.0f}%",    ha="center", fontsize=9)
    ax2.set_xticks(x2); ax2.set_xticklabels(sets)
    ax2.set_ylabel("Harm rate (%)")
    ax2.set_title("Held-out + data scaling")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylim(0, 70)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig5_robustness.pdf", bbox_inches="tight")
    plt.close()
    print("[fig5] saved")


# ============================================================
# Figure 6: Sample-efficiency / variant comparison
# ============================================================
def fig6_variants():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = ["v4.1\nbase",
              "Per-turn\nDPO",
              "Per-turn\nSFT i1",
              "Per-turn\nSFT i2",
              "Per-token\nKL i1",
              "Per-token\nKL i2",
              "Claude+\nscaffold"]
    harm   = [56, 54, 44, 36, 33, 38, 21]
    colors = [C_BASE, C_GOLD, C_TRAINED, C_TRAINED, C_MECH1, C_MECH1, C_MECH2]
    sig    = ["", "", "p=.10", "p=.10", "p=.011*", "p=.012*", ""]

    x = np.arange(len(labels))
    ax.bar(x, harm, color=colors, alpha=0.88, edgecolor="black", linewidth=0.5)
    for i, (h, s) in enumerate(zip(harm, sig)):
        ax.text(i, h + 1.5, f"{h}", ha="center", fontsize=10, fontweight="bold")
        if s:
            ax.text(i, -3.5, s, ha="center", fontsize=8.5,
                    fontweight="bold" if "*" in s else "normal",
                    style="italic" if "*" not in s else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Harm fires / 108")
    ax.set_title("Distillation variant ladder on Qwen3-8B  ($\\ast$: $p<0.05$ vs v4.1)")
    ax.set_ylim(-7, 67)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig6_variants.pdf", bbox_inches="tight")
    plt.close()
    print("[fig6] saved")


# ============================================================
# Figure 7: Calibrated / over-refuse split — 13 frontier subjects, n=5 seeds
# ============================================================
def fig7_xsubj():
    # Mean ± sd of 36-item-core harm across n=5 paired evaluation seeds.
    # Numbers from paper Table tab:prompt_grid.
    subjects = [
        ("Gemini-2.5-flash",       5.5, 6.3, "calibrated"),
        ("Mistral-Large",         11.0, 2.8, "calibrated"),
        ("Gemini-3p1-flash-lite", 12.0, 2.1, "calibrated"),
        ("DeepSeek-v3.1",         12.3, 2.5, "calibrated"),
        ("Qwen3-32B",             16.3, 2.6, "calibrated"),
        ("Claude-Opus",           18.1, 2.8, "calibrated"),
        ("Llama-3.1-70B-Instruct",19.2, 2.7, "calibrated"),
        ("Gemini-3-flash",        19.4, 2.3, "calibrated"),
        ("Claude-Sonnet",         19.5, 1.5, "calibrated"),
        ("GLM-4.6",               46.0, 2.9, "intermediate"),
        ("GPT-5-mini",            53.6, 5.1, "over-refuse"),
        ("GPT-5",                 71.1, 2.2, "over-refuse"),
        ("Qwen3.5-27B",           75.3, 2.9, "over-refuse"),
    ]
    color_map = {"calibrated": C_MECH1, "intermediate": C_GOLD, "over-refuse": C_BASE}
    labels = [s[0] for s in subjects]
    means  = [s[1] for s in subjects]
    sds    = [s[2] for s in subjects]
    colors = [color_map[s[3]] for s in subjects]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    y = np.arange(len(subjects))[::-1]
    ax.barh(y, means, xerr=sds, color=colors, alpha=0.88,
            edgecolor="black", linewidth=0.6, capsize=3)
    for yi, m, sd in zip(y, means, sds):
        ax.text(m + sd + 1.5, yi, f"{m:.1f}", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Aggregate harm rate (%, mean $\\pm$ sd, n=5 seeds)")
    ax.set_title("Calibrated / over-refuse split, 13 frontier subjects")
    ax.axvspan(0, 20, alpha=0.06, color=C_MECH1)
    ax.axvspan(50, 80, alpha=0.06, color=C_BASE)
    ax.set_xlim(0, 92)
    ax.grid(True, alpha=0.3, axis="x")

    # Custom legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=C_MECH1, alpha=0.88, edgecolor="black", linewidth=0.6,
                    label="calibrated ($\\leq 20\\%$)"),
              Patch(facecolor=C_GOLD, alpha=0.88, edgecolor="black", linewidth=0.6,
                    label="intermediate"),
              Patch(facecolor=C_BASE, alpha=0.88, edgecolor="black", linewidth=0.6,
                    label="over-refuse ($\\geq 50\\%$)")]
    ax.legend(handles=legend, loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig7_xsubj.pdf", bbox_inches="tight")
    plt.close()
    print("[fig7] saved")


# ============================================================
# Figure 8: Held-out validation of the split (24 fresh items, n=1 seed)
# ============================================================
def fig8_heldout_xsubj():
    # Read held-out scored files and compute per-subject harm rates.
    runs = ROOT / "runs"
    candidates = [
        ("Gemini-3p1-flash-lite", "phase4_promptv4_gemini3p1_lite_heldout", "calibrated"),
        ("Gemini-3-flash",        "phase4_promptv4_gemini3flash_heldout",   "calibrated"),
        ("DeepSeek-v3.1",         "phase4_promptv4_deepseek_heldout",       "calibrated"),
        ("Claude-Opus",           "phase4_promptv4_claude_opus_heldout",    "calibrated"),
        ("Claude-Sonnet",         "phase4_promptv4_claude_heldout",         "calibrated"),
        ("Llama-3.1-70B-Instruct","phase4_promptv4_llama70b_heldout",       "calibrated"),
        ("Qwen3-32B",             "phase4_promptv4_qwen32b_openrouter_heldout","calibrated"),
        ("GLM-4.6",               "phase4_promptv4_glm46_heldout",          "intermediate"),
        ("GPT-5-mini",            "phase4_promptv4_gpt5mini_heldout",       "over-refuse"),
        ("GPT-5",                 "phase4_promptv4_gpt5_heldout",           "over-refuse"),
        ("Qwen3.5-27B",           "phase4_promptv4_qwen27b_heldout",        "over-refuse"),
    ]
    data = []
    for name, path, cluster in candidates:
        rows = load(runs / path / "scored.jsonl")
        if not rows: continue
        c = counts(rows)
        harm_pct = 100.0 * c["harm"] / c["n"] if c["n"] else 0.0
        data.append((name, harm_pct, cluster, c["harm"], c["n"]))

    color_map = {"calibrated": C_MECH1, "intermediate": C_GOLD, "over-refuse": C_BASE}
    # Sort within cluster, calibrated low→high, intermediate, over-refuse low→high
    cluster_order = {"calibrated": 0, "intermediate": 1, "over-refuse": 2}
    data.sort(key=lambda d: (cluster_order[d[2]], d[1]))
    labels = [d[0] for d in data]
    means  = [d[1] for d in data]
    colors = [color_map[d[2]] for d in data]
    hn     = [(d[3], d[4]) for d in data]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(data))[::-1]
    ax.barh(y, means, color=colors, alpha=0.88, edgecolor="black", linewidth=0.6)
    for yi, m, (h, n) in zip(y, means, hn):
        ax.text(m + 1.5, yi, f"{m:.0f}% ({h}/{n})", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Held-out harm rate (%, 25 items $\\times$ 3 arms)")
    ax.set_title("Held-out items confirm the split is not item-specific")
    ax.axvspan(0, 25, alpha=0.06, color=C_MECH1)
    ax.axvspan(75, 100, alpha=0.06, color=C_BASE)
    ax.set_xlim(0, 115)
    ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=C_MECH1, alpha=0.88, edgecolor="black", linewidth=0.6,
                    label="calibrated"),
              Patch(facecolor=C_GOLD, alpha=0.88, edgecolor="black", linewidth=0.6,
                    label="intermediate"),
              Patch(facecolor=C_BASE, alpha=0.88, edgecolor="black", linewidth=0.6,
                    label="over-refuse")]
    ax.legend(handles=legend, loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig8_heldout_xsubj.pdf", bbox_inches="tight")
    plt.close()
    print("[fig8] saved")


# ============================================================
# Figure 9: Llama K-iteration trajectory (mirror of fig2 for Qwen)
# ============================================================
def fig9_llama_kiter():
    iters = ["Llama-8B\nuntrained", "iter1", "iter2", "iter3", "iter4"]
    # Llama-3.1-8B untrained baseline (HEADLINES.md): harm 45/108, leak 7, bound 0, MI 43
    # Llama K-iter from paper text: 27 → 22 → 17 over three iterations
    harm  = [45, 27, 22, 17, 18]
    leak  = [ 7,  3,  9,  7,  6]
    bound = [ 0,  2,  2,  3,  2]
    mi    = [43, 25, 20, 15, 17]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(iters))
    ax.plot(x, harm,  marker="o", linewidth=2.2, label="harm",  color=C_BASE)
    ax.plot(x, mi,    marker="s", linewidth=2.2, label="MI",    color=C_GOLD)
    ax.plot(x, leak,  marker="^", linewidth=2.2, label="leak",  color=C_MECH1)
    ax.plot(x, bound, marker="D", linewidth=2.2, label="bound", color=C_TRAINED)

    ax.annotate("harm / MI min", xy=(3, 17), xytext=(3.4, 35),
                arrowprops=dict(arrowstyle="->", color=C_BASE, alpha=0.7),
                fontsize=9, color=C_BASE, ha="center")

    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("Failures per 108 trajectories")
    ax.set_title("Llama-3.1-8B: descent to iter3, plateau at iter4")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, max(harm) + 12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arxiv_fig9_llama_kiter.pdf", bbox_inches="tight")
    plt.close()
    print("[fig9] saved")


if __name__ == "__main__":
    fig0_problem()
    fig0b_cells()
    fig1_manifold()
    fig2_kiter()
    fig3_wilcoxon()
    fig4_teacher()
    fig5_robustness()
    fig6_variants()
    fig7_xsubj()
    fig8_heldout_xsubj()
    fig9_llama_kiter()
    print("DONE — figures in", FIG_DIR)
