"""Regenerate the role diagram at its AAAI display size."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "arxiv_fig0_problem.pdf"
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(figsize=(9.0, 3.8))
ax.set_xlim(0, 21)
ax.set_ylim(0, 7.5)
ax.set_aspect("equal")
ax.axis("off")


def box(x, label, sub, face, edge, w=4.6):
    y, h = 4.8, 2.5
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.92,
    ))
    ax.text(x + w / 2, 6.95, label, ha="center", va="top",
            fontsize=11.5, fontweight="bold")
    ax.text(x + w / 2, 5.15, sub, ha="center", va="bottom",
            fontsize=10, style="italic", color="#444", linespacing=1.05)


box(0.2, "PRINCIPAL", "user / company\nyou represent", "#FBEEEA", "#C0504D", w=4.2)
box(8.5, "AGENT", "LLM acting\non P's behalf", "#EAF1F8", "#4F81BD", w=4.2)
box(16.0, "COUNTERPARTY", "other party\n(may conflict)", "#EFE6F3", "#8064A2", w=4.8)

for start, end, y, color in [
    ((4.4, 6.45), (8.5, 6.45), 6.45, "#C0504D"),
    ((8.5, 5.65), (4.4, 5.65), 5.65, "#C0504D"),
    ((12.7, 6.45), (16.0, 6.45), 6.45, "#666"),
    ((16.0, 5.65), (12.7, 5.65), 5.65, "#8064A2"),
]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 color=color, linewidth=1.5))

ax.text(6.45, 6.90, "briefing,\nrequests", ha="center", va="center",
        fontsize=9.5, color="#C0504D", linespacing=0.9)
ax.text(6.45, 5.05, "results,\nclarifications", ha="center", va="center",
        fontsize=9.5, color="#C0504D", linespacing=0.9)
ax.text(14.35, 6.90, "represents", ha="center", fontsize=9.5, color="#666")
ax.text(14.35, 5.05, "probes,\npressure", ha="center", va="center",
        fontsize=9.5, color="#8064A2", linespacing=0.9)

ax.add_patch(FancyBboxPatch(
    (0.6, 1.1), 19.8, 2.7, boxstyle="round,pad=0.10,rounding_size=0.20",
    linewidth=1.0, edgecolor="#888", facecolor="#FAFAFA",
))
ax.text(10.5, 3.35, "loyalty failure axes  —  the agent fails by...",
        ha="center", fontsize=11, fontweight="bold", color="#222")

labels = [
    ("× leak", "revealing\nwithheld facts"),
    ("× capitulate", "conceding\nunder pressure"),
    ("× posture", "signaling\nmotivation"),
    ("× over-refuse", "declining\nP-authorized asks"),
]
for x, (head, sub) in zip([3.0, 8.0, 13.0, 18.0], labels):
    ax.text(x, 2.65, head, ha="center", fontsize=10.5, fontweight="bold", color="#C0504D")
    ax.text(x, 1.55, sub, ha="center", va="bottom", fontsize=9.5,
            color="#444", style="italic", linespacing=1.05)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
