"""Regenerate paper figures at their AAAI display sizes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUT = Path(__file__).resolve().parent / "figures"
BLUE = "#4F81BD"
GOLD = "#E8A33D"
RED = "#C0504D"
COLORS = {"calibrated": BLUE, "intermediate": GOLD, "over-refuse": RED}

# Multi-seed results used by the multi-axis comparison plot.
MULTISEED_METRICS = ["harm", "leak", "bound", "MI"]
MULTISEED_BASE = [47.8, 15.8, 4.6, 44.4]
MULTISEED_KL_ITER1 = [39.2, 13.8, 2.8, 37.2]
MULTISEED_KL_ITER1_ERR = [4.0, 1.8, 1.5, 3.6]
MULTISEED_KL_ITER2 = [41.5, 11.2, 2.5, 40.5]
MULTISEED_KL_ITER2_ERR = [3.0, 3.5, 0.5, 4.5]
MULTISEED_BASE_N = 5
MULTISEED_KL_ITER1_N = 5
MULTISEED_KL_ITER2_N = 4

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "savefig.dpi": 300,
        # AAAI forbids Type 3 fonts, including fonts embedded in figures.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def core_split() -> None:
    rows = [
        ("Gemini-2.5-flash", 5.5, 6.3, "calibrated"),
        ("Mistral-Large", 11.0, 2.8, "calibrated"),
        ("Gemini-3p1-flash-lite", 12.0, 2.1, "calibrated"),
        ("DeepSeek-v3.1", 12.3, 2.5, "calibrated"),
        ("Qwen3-32B", 16.5, 2.6, "calibrated"),
        ("Claude-Opus", 18.1, 2.8, "calibrated"),
        ("Llama-3.1-70B-Instruct", 19.2, 2.7, "calibrated"),
        ("Gemini-3-flash", 19.4, 2.3, "calibrated"),
        ("Claude-Sonnet", 19.5, 1.5, "calibrated"),
        ("GLM-4.6", 46.0, 2.9, "intermediate"),
        ("GPT-5-mini", 53.6, 5.1, "over-refuse"),
        ("GPT-5", 71.1, 2.2, "over-refuse"),
        ("Qwen3.5-27B", 75.3, 2.9, "over-refuse"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    y = np.arange(len(rows))[::-1]
    means = [r[1] for r in rows]
    sds = [r[2] for r in rows]
    ax.barh(
        y,
        means,
        xerr=sds,
        color=[COLORS[r[3]] for r in rows],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
    )
    for yi, mean, sd in zip(y, means, sds):
        ax.text(mean + sd + 1.2, yi, f"{mean:.1f}", va="center", fontsize=9.2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Aggregate harm rate (%)\nmean $\\pm$ sd, n=5 seeds", fontsize=9.2)
    ax.tick_params(axis="both", labelsize=9.2)
    ax.axvspan(0, 20, alpha=0.06, color=BLUE)
    ax.axvspan(50, 80, alpha=0.06, color=RED)
    ax.set_xlim(0, 110)
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.88, edgecolor="black", linewidth=0.5, label="selective ($\\leq 20\\%$)"),
            Patch(facecolor=GOLD, alpha=0.88, edgecolor="black", linewidth=0.5, label="intermediate"),
            Patch(facecolor=RED, alpha=0.88, edgecolor="black", linewidth=0.5, label="over-refusing ($\\geq 50\\%$)"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        borderpad=0.35,
        handlelength=0.9,
        columnspacing=0.7,
        fontsize=9.2,
    )
    fig.subplots_adjust(left=0.48, right=0.98, top=0.67, bottom=0.16)
    fig.savefig(OUT / "arxiv_fig7_xsubj.pdf")
    plt.close(fig)


def heldout_split() -> None:
    # Preserve all labels from the original figure exactly.
    rows = [
        ("DeepSeek-v3.1", 9, 6, 68, "calibrated"),
        ("Gemini-3p1-flash-lite", 12, 8, 69, "calibrated"),
        ("Claude-Opus", 15, 11, 71, "calibrated"),
        ("Claude-Sonnet", 16, 11, 70, "calibrated"),
        ("Qwen3-32B", 18, 12, 68, "calibrated"),
        ("Llama-3.1-70B-Instruct", 20, 13, 66, "calibrated"),
        ("Gemini-3-flash", 24, 16, 67, "calibrated"),
        ("GLM-4.6", 49, 34, 69, "intermediate"),
        ("GPT-5-mini", 76, 53, 70, "over-refuse"),
        ("Qwen3.5-27B", 78, 53, 68, "over-refuse"),
        ("GPT-5", 93, 67, 72, "over-refuse"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    y = np.arange(len(rows))[::-1]
    means = [r[1] for r in rows]
    ax.barh(
        y,
        means,
        color=[COLORS[r[4]] for r in rows],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.5,
    )
    for yi, (_, mean, harm, n, _) in zip(y, rows):
        ax.text(mean + 1.3, yi, f"{mean:.0f}% ({harm}/{n})", va="center", fontsize=9.2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Held-out harm rate (%)\n25 items $\\times$ 3 arms", fontsize=9.2)
    ax.tick_params(axis="both", labelsize=9.2)
    ax.axvspan(0, 25, alpha=0.06, color=BLUE)
    ax.axvspan(75, 100, alpha=0.06, color=RED)
    ax.set_xlim(0, 145)
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.88, edgecolor="black", linewidth=0.5, label="calibrated"),
            Patch(facecolor=GOLD, alpha=0.88, edgecolor="black", linewidth=0.5, label="intermediate"),
            Patch(facecolor=RED, alpha=0.88, edgecolor="black", linewidth=0.5, label="over-refuse"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        borderpad=0.35,
        handlelength=0.9,
        columnspacing=0.7,
        fontsize=9.2,
    )
    fig.subplots_adjust(left=0.48, right=0.98, top=0.68, bottom=0.17)
    fig.savefig(OUT / "arxiv_fig8_heldout_xsubj.pdf")
    plt.close(fig)


def combined_subject_split() -> None:
    """Align core and held-out results in one full-width two-panel figure."""
    rows = [
        ("Gemini-2.5-flash", 5.5, 6.3, None, None, None, "calibrated"),
        ("Mistral-Large", 11.0, 2.8, None, None, None, "calibrated"),
        ("Gemini-3p1-flash-lite", 12.0, 2.1, 12, 8, 69, "calibrated"),
        ("DeepSeek-v3.1", 12.3, 2.5, 9, 6, 68, "calibrated"),
        ("Qwen3-32B", 16.5, 2.6, 18, 12, 68, "calibrated"),
        ("Claude-Opus", 18.1, 2.8, 15, 11, 71, "calibrated"),
        ("Llama-3.1-70B-Instruct", 19.2, 2.7, 20, 13, 66, "calibrated"),
        ("Gemini-3-flash", 19.4, 2.3, 24, 16, 67, "calibrated"),
        ("Claude-Sonnet", 19.5, 1.5, 16, 11, 70, "calibrated"),
        ("GLM-4.6", 46.0, 2.9, 49, 34, 69, "intermediate"),
        ("GPT-5-mini", 53.6, 5.1, 76, 53, 70, "over-refuse"),
        ("GPT-5", 71.1, 2.2, 93, 67, 72, "over-refuse"),
        ("Qwen3.5-27B", 75.3, 2.9, 78, 53, 68, "over-refuse"),
    ]
    fig, (ax_core, ax_held) = plt.subplots(
        1, 2, figsize=(7.15, 3.35), sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.13},
    )
    y = np.arange(len(rows))[::-1]
    colors = [COLORS[r[6]] for r in rows]

    core_means = [r[1] for r in rows]
    core_sds = [r[2] for r in rows]
    ax_core.barh(y, core_means, xerr=core_sds, color=colors, alpha=0.88,
                 edgecolor="black", linewidth=0.5, capsize=2)
    for yi, mean, sd in zip(y, core_means, core_sds):
        ax_core.text(mean + sd + 1.1, yi, f"{mean:.1f}", va="center", fontsize=9.4)
    ax_core.set_yticks(y, [r[0] for r in rows])
    ax_core.tick_params(axis="y", labelsize=9.4, length=0, pad=3)
    ax_core.set_title("(a) 36-item core", fontsize=9.4, fontweight="bold", pad=4)
    ax_core.set_xlabel(r"Five-seed mean harm rate (%)  $\pm 1\sigma$", fontsize=9.4)
    ax_core.set_xlim(0, 92)

    held_rows = [(yi, r) for yi, r in zip(y, rows) if r[3] is not None]
    ax_held.barh(
        [yi for yi, _ in held_rows], [r[3] for _, r in held_rows],
        color=[COLORS[r[6]] for _, r in held_rows], alpha=0.88,
        edgecolor="black", linewidth=0.5,
    )
    for yi, r in held_rows:
        if r[3] >= 65:
            ax_held.text(r[3] - 1.5, yi, f"{r[3]:.0f}%  ({r[4]}/{r[5]})",
                         ha="right", va="center", fontsize=9.4, color="white")
        else:
            ax_held.text(r[3] + 1.2, yi, f"{r[3]:.0f}%  ({r[4]}/{r[5]})",
                         va="center", fontsize=9.4)
    for yi, r in zip(y, rows):
        if r[3] is None:
            ax_held.text(2, yi, "not measured", va="center", fontsize=9.4,
                         color="#666666", style="italic")
    ax_held.tick_params(axis="y", left=False, labelleft=False)
    ax_held.set_title("(b) 24-item held-out grid", fontsize=9.4,
                      fontweight="bold", pad=4)
    ax_held.set_xlabel("Harm rate among valid cells (%)", fontsize=9.4)
    ax_held.set_xlim(0, 105)
    ax_held.set_xticks([0, 25, 50, 75, 100])

    for ax in (ax_core, ax_held):
        ax.tick_params(axis="x", labelsize=9.4)
        ax.grid(True, alpha=0.25, axis="x", linewidth=0.5)
        ax.set_axisbelow(True)

    fig.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.88, edgecolor="black", linewidth=0.5,
                  label="selective"),
            Patch(facecolor=GOLD, alpha=0.88, edgecolor="black", linewidth=0.5,
                  label="intermediate"),
            Patch(facecolor=RED, alpha=0.88, edgecolor="black", linewidth=0.5,
                  label="over-refusing"),
        ],
        loc="upper center", bbox_to_anchor=(0.58, 0.995), ncol=3,
        frameon=False, handlelength=1.0, columnspacing=1.4, fontsize=9.4,
    )
    fig.subplots_adjust(left=0.265, right=0.995, top=0.82, bottom=0.16)
    fig.savefig(OUT / "arxiv_fig_subject_split.pdf")
    plt.close(fig)


def pareto_frontier() -> None:
    rows = [
        ("Qwen-8B untrained", 76.0, 22.4, 28, 107, "#7F7F7F", "x"),
        ("Qwen SFT+DPO", 100 * 18 / 108, 100 * 51 / 108, 56, 108, RED, "o"),
        ("Qwen DAPO", 100 * 19 / 108, 100 * 34 / 108, 37, 108, GOLD, "s"),
        ("Per-turn SFT iter1", 100 * 17 / 107, 100 * 43 / 107, 44, 107, "#8064A2", "D"),
        ("Per-turn SFT iter2", 100 * 24 / 105, 100 * 32 / 105, 36, 105, "#8064A2", "v"),
        ("Per-token KL iter1", 100 * 13 / 108, 100 * 32 / 108, 33, 108, BLUE, "*"),
        ("Per-token KL iter2", 100 * 9 / 106, 100 * 35 / 106, 38, 106, BLUE, "P"),
        ("Per-token KL iter3", 100 * 15 / 108, 100 * 40 / 108, 41, 108, BLUE, "X"),
        ("Claude + scaffold", 100 * 17 / 108, 100 * 21 / 108, 21, 108, "#9BBB59", "^"),
    ]
    fig = plt.figure(figsize=(3.35, 3.50))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.8, 1.70], hspace=0.30)
    ax = fig.add_subplot(grid[0])
    key_ax = fig.add_subplot(grid[1])
    xmax = 28.0
    handles = []
    points = []
    plotted = {}
    for name, leak, mi, harm, n, color, marker in rows:
        x = xmax - 1.7 if leak > xmax else leak
        size = 95 if marker == "*" else 40
        edge = color if marker == "x" else "black"
        handle = ax.scatter(
            x,
            mi,
            s=size,
            c=color,
            marker=marker,
            edgecolors=edge,
            linewidths=0.5,
            label=name,
            zorder=4,
            clip_on=False,
        )
        handles.append(handle)
        plotted[name] = (x, mi)
        if leak <= xmax:
            points.append((leak, mi))

    front = sorted(
        p
        for p in points
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points)
    )
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax.plot(fx, fy, color="#3a8c3a", linestyle="--", linewidth=1.3, zorder=2)
    ax.fill_between([0.0] + fx + [xmax], 0, [fy[0]] + fy + [fy[-1]], color="#3a8c3a", alpha=0.07)
    ax.axhline(20, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvline(20, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.text(6.0, 6.4, "jointly favorable\ncorner is empty", ha="center", fontsize=9.0, style="italic", color="#2f6f2f")

    ax.set_xlabel("Leak rate (%)")
    ax.set_ylabel("Missed-instruction rate (%)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 53)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    # Integrated marker/variant/harm key beneath the plot. This replaces the
    # legend so exact harm fractions do not crowd either the data or labels.
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.axis("off")
    key_ax.text(0.07, 0.97, "marker", fontsize=9.0, fontweight="bold", ha="center", va="top")
    key_ax.text(0.22, 0.97, "variant", fontsize=9.0, fontweight="bold", va="top")
    key_ax.text(0.97, 0.97, "harm/valid", fontsize=9.0, fontweight="bold", ha="right", va="top")
    key_ax.plot([0.0, 1.0], [0.88, 0.88], color="black", lw=0.5, clip_on=False)
    for row_i, (name, _, _, harm, n, color, marker) in enumerate(rows):
        y = 0.81 - row_i * 0.095
        size = 52 if marker == "*" else 22
        key_ax.scatter(
            0.07,
            y,
            s=size,
            c=color,
            marker=marker,
            edgecolors=color if marker == "x" else "black",
            linewidths=0.5,
            clip_on=False,
        )
        key_ax.text(0.22, y, name, fontsize=9.0, va="center")
        key_ax.text(0.97, y, f"{harm}/{n}", fontsize=9.0, ha="right", va="center")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.92, bottom=0.02)
    fig.savefig(OUT / "arxiv_fig1_manifold.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def qwen_iteration_trajectory() -> None:
    iterations = ["SFT+DPO\nbase", "iter1", "iter2", "iter3", "iter4", "iter5"]
    harm = [56, 33, 38, 41, 42, 32]
    leak = [18, 13, 9, 15, 17, 19]
    bound = [4, 3, 2, 4, 5, 6]
    missed_instruction = [51, 32, 35, 40, 42, 32]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(iterations))
    ax.plot(x, harm, marker="o", linewidth=2.2, label="harm", color=RED)
    ax.plot(x, missed_instruction, marker="s", linewidth=2.2, label="MI", color=GOLD)
    ax.plot(x, leak, marker="^", linewidth=2.2, label="leak", color=BLUE)
    ax.plot(x, bound, marker="D", linewidth=2.2, label="bound", color="#8064A2")
    ax.annotate(
        "harm-min",
        xy=(1, 33),
        xytext=(0.04, 21),
        arrowprops=dict(arrowstyle="->", color=RED, alpha=0.6),
        fontsize=9,
        color=RED,
    )
    ax.annotate(
        "leak/bound-min",
        xy=(2, 9),
        xytext=(2.55, 16),
        arrowprops=dict(arrowstyle="->", color=BLUE, alpha=0.6),
        fontsize=9,
        color=BLUE,
        ha="center",
    )
    ax.set_xticks(x, iterations)
    ax.set_ylabel("Failures per 108 trajectories")
    ax.set_title("Per-token KL on Qwen3-8B across iterations")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 62)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "arxiv_fig2_kiter.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def multi_seed_comparison() -> None:
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    x = np.arange(len(MULTISEED_METRICS))
    width = 0.27
    ax.bar(x - width, MULTISEED_BASE, width, color=RED, alpha=0.85,
           label=f"SFT+DPO base ($n={MULTISEED_BASE_N}$)")
    ax.bar(x, MULTISEED_KL_ITER1, width, yerr=MULTISEED_KL_ITER1_ERR,
           color=BLUE, alpha=0.85, capsize=3,
           label=f"KL iter1 ($n={MULTISEED_KL_ITER1_N}$)")
    ax.bar(x + width, MULTISEED_KL_ITER2, width, yerr=MULTISEED_KL_ITER2_ERR,
           color="#9BBB59", alpha=0.85, capsize=3,
           label=f"KL iter2 ($n={MULTISEED_KL_ITER2_N}$)")
    ax.set_xticks(x, MULTISEED_METRICS)
    ax.tick_params(axis="both", labelsize=9.2)
    ax.set_ylabel("Mean fires per 108", fontsize=9.2)
    ax.legend(loc="upper center", ncol=1, fontsize=9.2, borderaxespad=0.35,
              columnspacing=0.8, handletextpad=0.4)
    ax.set_ylim(0, max(MULTISEED_BASE) * 1.50)
    ax.grid(True, alpha=0.3, axis="y")
    fig.text(0.58, 0.91, "Multi-seed comparison with SFT+DPO base",
             ha="center", va="top", fontsize=9.2)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.14)
    fig.savefig(OUT / "arxiv_fig3_seed_comparison.pdf")
    plt.close(fig)


def teacher_robustness() -> None:
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(13.0, 4.2),
        gridspec_kw=dict(wspace=0.32, left=0.075, right=0.995, bottom=0.20, top=0.70),
    )
    metrics = ["harm", "leak", "bound", "MI"]
    qwen = [100 * 4 / 31, 100 * 21 / 31, 0, 100 * 3 / 31]
    claude = [100 * 6 / 36, 100 * 6 / 36, 100 * 1 / 36, 100 * 6 / 36]
    x = np.arange(4)
    width = 0.36
    axa.bar(x - width / 2, claude, width, color="#9BBB59", alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Claude")
    axa.bar(x + width / 2, qwen, width, color=BLUE, alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Qwen teacher")
    for i, (c, q) in enumerate(zip(claude, qwen)):
        axa.text(i - width / 2, c + 2, f"{c:.0f}", ha="center", fontsize=16.75)
        axa.text(i + width / 2, q + 2, f"{q:.0f}", ha="center", fontsize=16.75)
    axa.set_xticks(x, metrics)
    axa.set_ylabel("Failure rate (%)", fontsize=16.75)
    axa.legend(loc="lower left", bbox_to_anchor=(0, 1.075), ncol=2,
               frameon=False, fontsize=16.75, handlelength=1.2,
               columnspacing=0.8, handletextpad=0.4)
    axa.set_ylim(0, 100)

    counterparties = ["Claude\n(default)", "GPT-5", "Gemini-3\nflash"]
    kl_harm, sft_harm = [33, 38, 49], [36, 34, 41]
    x = np.arange(3)
    axb.bar(x - width / 2, kl_harm, width, color=BLUE, alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Per-token KL iter1")
    axb.bar(x + width / 2, sft_harm, width, color="#8064A2", alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Per-turn SFT iter2")
    for i in range(3):
        axb.text(i - width / 2, kl_harm[i] + 1, str(kl_harm[i]), ha="center", fontsize=16.75)
        axb.text(i + width / 2, sft_harm[i] + 1, str(sft_harm[i]), ha="center", fontsize=16.75)
    axb.set_xticks(x, counterparties)
    axb.set_ylabel("Harm failures / 108", fontsize=16.75)
    axb.set_ylim(0, 60)

    sets = ["Training\n(36 items)", "Held-out\n(24 items)"]
    kl, sft, scaled = [31, 40], [34, 35], [37, 56]
    x = np.arange(2)
    width2 = 0.27
    axc.bar(x - width2, kl, width2, color=BLUE, alpha=0.88, edgecolor="black",
            linewidth=0.5, label="KL iter1")
    axc.bar(x, sft, width2, color="#8064A2", alpha=0.88, edgecolor="black",
            linewidth=0.5, label="SFT iter2")
    axc.bar(x + width2, scaled, width2, color=GOLD, alpha=0.88, edgecolor="black",
            linewidth=0.5, label="Scaled KL")
    for i in range(2):
        axc.text(i - width2, kl[i] + 1.2, str(kl[i]), ha="center", fontsize=16.75)
        axc.text(i, sft[i] + 1.2, str(sft[i]), ha="center", fontsize=16.75)
        axc.text(i + width2, scaled[i] + 1.2, str(scaled[i]), ha="center", fontsize=16.75)
    axc.set_xticks(x, sets)
    axc.set_ylabel("Harm rate (%)", fontsize=16.75)
    axc.set_ylim(0, 70)

    for ax in (axa, axb, axc):
        ax.tick_params(axis="both", labelsize=16.75)
        ax.grid(True, alpha=0.3, axis="y")
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.88, edgecolor="black", linewidth=0.5,
                  label="KL iter1"),
            Patch(facecolor="#8064A2", alpha=0.88, edgecolor="black", linewidth=0.5,
                  label="SFT iter2"),
            Patch(facecolor=GOLD, alpha=0.88, edgecolor="black", linewidth=0.5,
                  label="Scaled KL"),
        ],
        loc="upper center", bbox_to_anchor=(0.68, 0.88), ncol=3,
        frameon=False, fontsize=16.75, handlelength=1.2,
        columnspacing=1.0, handletextpad=0.4,
    )
    for x, y, title in zip(
        (0.20, 0.525, 0.845),
        (0.96, 0.96, 0.96),
        ("(a) Teacher comparison", "(b) Counterparty robustness", "(c) Held-out scaling"),
    ):
        fig.text(x, y, title, ha="center", va="top", fontsize=16.75)
    fig.savefig(OUT / "arxiv_fig_teacher_robust.pdf")
    plt.close(fig)


if __name__ == "__main__":
    pareto_frontier()
    qwen_iteration_trajectory()
    multi_seed_comparison()
    teacher_robustness()
    combined_subject_split()
