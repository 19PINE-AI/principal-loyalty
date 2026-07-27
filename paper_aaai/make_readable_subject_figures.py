"""Regenerate the two subject-split plots at AAAI single-column size."""

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
    fig, ax = plt.subplots(figsize=(3.35, 2.8))
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
        ax.text(mean + sd + 1.2, yi, f"{mean:.1f}", va="center", fontsize=9.0)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Aggregate harm rate (%, mean $\\pm$ sd, n=5 seeds)")
    ax.axvspan(0, 20, alpha=0.06, color=BLUE)
    ax.axvspan(50, 80, alpha=0.06, color=RED)
    ax.set_xlim(0, 92)
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.88, edgecolor="black", linewidth=0.5, label="selective ($\\leq 20\\%$)"),
            Patch(facecolor=GOLD, alpha=0.88, edgecolor="black", linewidth=0.5, label="intermediate"),
            Patch(facecolor=RED, alpha=0.88, edgecolor="black", linewidth=0.5, label="over-refusing ($\\geq 50\\%$)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        borderpad=0.35,
        handlelength=0.9,
        columnspacing=0.7,
    )
    fig.subplots_adjust(left=0.39, right=0.98, top=0.86, bottom=0.18)
    fig.savefig(OUT / "arxiv_fig7_xsubj.pdf", bbox_inches="tight", pad_inches=0.02)
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
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
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
        ax.text(mean + 1.3, yi, f"{mean:.0f}% ({harm}/{n})", va="center", fontsize=9.0)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Held-out harm rate (%, 25 items $\\times$ 3 arms)")
    ax.axvspan(0, 25, alpha=0.06, color=BLUE)
    ax.axvspan(75, 100, alpha=0.06, color=RED)
    ax.set_xlim(0, 115)
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.88, edgecolor="black", linewidth=0.5, label="calibrated"),
            Patch(facecolor=GOLD, alpha=0.88, edgecolor="black", linewidth=0.5, label="intermediate"),
            Patch(facecolor=RED, alpha=0.88, edgecolor="black", linewidth=0.5, label="over-refuse"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        borderpad=0.35,
        handlelength=0.9,
        columnspacing=0.7,
    )
    fig.subplots_adjust(left=0.39, right=0.98, top=0.85, bottom=0.19)
    fig.savefig(OUT / "arxiv_fig8_heldout_xsubj.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def pareto_frontier() -> None:
    rows = [
        ("Qwen-8B untrained", 76.0, 22.4, 28, 107, "#7F7F7F", "x"),
        ("Qwen v4.1 SFT+DPO", 100 * 18 / 108, 100 * 51 / 108, 56, 108, RED, "o"),
        ("Qwen DAPO-v1", 100 * 19 / 108, 100 * 34 / 108, 37, 108, GOLD, "s"),
        ("Per-turn SFT i1", 100 * 17 / 107, 100 * 43 / 107, 44, 107, "#8064A2", "D"),
        ("Per-turn SFT i2", 100 * 24 / 105, 100 * 32 / 105, 36, 105, "#8064A2", "v"),
        ("Per-token KL i1", 100 * 13 / 108, 100 * 32 / 108, 33, 108, BLUE, "*"),
        ("Per-token KL i2", 100 * 9 / 106, 100 * 35 / 106, 38, 106, BLUE, "P"),
        ("Per-token KL i3", 100 * 15 / 108, 100 * 40 / 108, 41, 108, BLUE, "X"),
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


def distillation_variants() -> None:
    labels = [
        "v4.1\nbase",
        "\nPer-turn\nDPO",
        "Per-turn\nSFT i1",
        "\nPer-turn\nSFT i2",
        "Per-token\nKL i1",
        "\nPer-token\nKL i2",
        "Claude+\nscaffold",
    ]
    harm = [56, 54, 44, 36, 33, 38, 21]
    significance = ["", "", "$p=.10$", "$p=.10$", "$p=.011^*$", "$p=.012^*$", ""]
    colors = [RED, GOLD, "#8064A2", "#8064A2", BLUE, BLUE, "#9BBB59"]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(labels))
    ax.bar(x, harm, color=colors, alpha=0.88, edgecolor="black", linewidth=0.5)
    for i, (value, p_value) in enumerate(zip(harm, significance)):
        ax.text(i, value + 1.5, str(value), ha="center", va="bottom", fontsize=13, fontweight="bold")
        if p_value:
            ax.text(i, value + 5.5, p_value, ha="center", va="bottom", fontsize=11.5)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=13, pad=3)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Harm fires / 108", fontsize=14)
    ax.set_title("Distillation variant ladder on Qwen3-8B ($^*$: $p<0.05$ vs. v4.1)", fontsize=14)
    ax.set_ylim(0, 67)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(OUT / "arxiv_fig6_variants.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def paired_wilcoxon() -> None:
    metrics = ["harm", "leak", "bound", "MI"]
    base = [47.8, 15.8, 4.6, 44.4]
    iter1 = [39.2, 13.8, 2.8, 37.2]
    iter1_err = [4.0, 1.8, 1.5, 3.6]
    iter2 = [41.5, 11.2, 2.5, 40.5]
    iter2_err = [3.0, 3.5, 0.5, 4.5]
    p_iter1 = [0.0114, 0.534, 0.385, 0.055]
    p_iter2 = [0.0436, 0.177, 0.592, 0.214]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(metrics))
    width = 0.27
    ax.bar(x - width, base, width, color=RED, alpha=0.85, label="v4.1 base ($n{=}5$)")
    ax.bar(x, iter1, width, yerr=iter1_err, color=BLUE, alpha=0.85,
           capsize=3, label="KL iter1 ($n{=}5$)")
    ax.bar(x + width, iter2, width, yerr=iter2_err, color="#9BBB59", alpha=0.85,
           capsize=3, label="KL iter2 ($n{=}4$)")
    annotation_y = [52.0, None, None, 49.0]
    for i, (p1, p2) in enumerate(zip(p_iter1, p_iter2)):
        y_top = max(iter1[i] + iter1_err[i], iter2[i] + iter2_err[i])
        tag1 = f"{p1:.3f}" + ("*" if p1 < 0.05 else "")
        tag2 = f"{p2:.3f}" + ("*" if p2 < 0.05 else "")
        label_y = annotation_y[i] if annotation_y[i] is not None else y_top + 1.5
        ax.text(i, label_y, f"i1: {tag1}\ni2: {tag2}", ha="center",
                fontsize=11, fontweight="bold" if (p1 < 0.05 or p2 < 0.05) else "normal")
    ax.set_xticks(x, metrics)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylabel("Mean fires per 108 ($n{=}5$ seeds)", fontsize=12)
    ax.set_title("Multi-seed paired Wilcoxon vs v4.1 base", fontsize=12)
    ax.legend(loc="upper center", ncol=3, fontsize=9.5, borderaxespad=0.35,
              columnspacing=0.8, handletextpad=0.4)
    ax.set_ylim(0, max(base) * 1.50)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUT / "arxiv_fig3_wilcoxon.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def teacher_robustness() -> None:
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(13.0, 3.5),
        gridspec_kw=dict(wspace=0.18, left=0.045, right=0.995, bottom=0.19, top=0.84),
    )
    metrics = ["harm", "leak", "bound", "MI"]
    qwen = [100 * 4 / 31, 100 * 21 / 31, 0, 100 * 3 / 31]
    claude = [100 * 6 / 36, 100 * 6 / 36, 100 * 1 / 36, 100 * 6 / 36]
    x = np.arange(4)
    width = 0.36
    axa.bar(x - width / 2, claude, width, color="#9BBB59", alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Claude-Sonnet ($n{=}36$)")
    axa.bar(x + width / 2, qwen, width, color=BLUE, alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Qwen3-32B teacher ($n{=}31$)")
    for i, (c, q) in enumerate(zip(claude, qwen)):
        axa.text(i - width / 2, c + 2, f"{c:.0f}", ha="center", fontsize=11)
        axa.text(i + width / 2, q + 2, f"{q:.0f}", ha="center", fontsize=11)
    axa.set_xticks(x, metrics)
    axa.set_ylabel("Fire rate (%), scaffolded arm", fontsize=11.5)
    axa.set_title("(a) Teacher self-validation", fontsize=12.5)
    axa.legend(loc="upper left", fontsize=10.5)
    axa.set_ylim(0, 100)

    counterparties = ["Claude\n(default)", "GPT-5", "Gemini-3\nflash"]
    kl_harm, sft_harm = [33, 38, 49], [36, 34, 41]
    x = np.arange(3)
    axb.bar(x - width / 2, kl_harm, width, color=BLUE, alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Per-token KL i1")
    axb.bar(x + width / 2, sft_harm, width, color="#8064A2", alpha=0.88,
            edgecolor="black", linewidth=0.5, label="Per-turn SFT i2")
    for i in range(3):
        axb.text(i - width / 2, kl_harm[i] + 1, str(kl_harm[i]), ha="center", fontsize=11)
        axb.text(i + width / 2, sft_harm[i] + 1, str(sft_harm[i]), ha="center", fontsize=11)
    axb.set_xticks(x, counterparties)
    axb.set_ylabel("Harm fires / 108", fontsize=11.5)
    axb.set_title("(b) Counterparty robustness", fontsize=12.5)
    axb.legend(loc="upper left", fontsize=10.5)
    axb.set_ylim(0, 60)

    sets = ["Training\n(36 items)", "Held-out\n(24 items)"]
    kl, sft, scaled = [31, 40], [34, 35], [37, 56]
    x = np.arange(2)
    width2 = 0.27
    axc.bar(x - width2, kl, width2, color=BLUE, alpha=0.88, edgecolor="black",
            linewidth=0.5, label="KL i1 (113 pts)")
    axc.bar(x, sft, width2, color="#8064A2", alpha=0.88, edgecolor="black",
            linewidth=0.5, label="SFT i2 (113 pts)")
    axc.bar(x + width2, scaled, width2, color=GOLD, alpha=0.88, edgecolor="black",
            linewidth=0.5, label="KL scaled3$\\times$ (480 pts)")
    for i in range(2):
        axc.text(i - width2, kl[i] + 1.2, str(kl[i]), ha="center", fontsize=11)
        axc.text(i, sft[i] + 1.2, str(sft[i]), ha="center", fontsize=11)
        axc.text(i + width2, scaled[i] + 1.2, str(scaled[i]), ha="center", fontsize=11)
    axc.set_xticks(x, sets)
    axc.set_ylabel("Harm rate (%)", fontsize=11.5)
    axc.set_title("(c) Held-out + data scaling", fontsize=12.5)
    axc.legend(loc="upper left", fontsize=10.5)
    axc.set_ylim(0, 70)

    for ax in (axa, axb, axc):
        ax.tick_params(axis="both", labelsize=10.5)
        ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(OUT / "arxiv_fig_teacher_robust.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    pareto_frontier()
    distillation_variants()
    paired_wilcoxon()
    teacher_robustness()
    core_split()
    heldout_split()
