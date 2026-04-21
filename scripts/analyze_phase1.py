"""Analyze Phase 1 results; emit a markdown report suitable for paper/progress.md.

Writes runs/phase1/report.md with:
- overall per-(subject,arm) leak/hold/posture/harm tables with bootstrap 95% CI
- per-cell breakdowns
- a handful of qualitative examples (leaked facts, lost commitments) from
  runs/phase1/scored.jsonl + trajectories.jsonl.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(1234)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def fmt_ci(vals: list[float]) -> str:
    if not vals:
        return "—"
    m = mean(vals)
    lo, hi = bootstrap_ci(vals)
    return f"{m:.2f} [{lo:.2f},{hi:.2f}]"


def main() -> None:
    scored = []
    with open("runs/phase1/scored.jsonl") as f:
        for line in f:
            scored.append(json.loads(line))

    trajs = {}
    with open("runs/phase1/trajectories.jsonl") as f:
        for line in f:
            t = json.loads(line)
            trajs[(t["subject"], t["arm"], t["item_id"])] = t

    subjects = sorted({r["subject"] for r in scored})
    arms = ["plain", "prompted", "scaffolded"]
    cells = ["leakage", "capitulation", "posture", "authoring", "sanity", "moderation"]

    # leak_rate
    leak_overall: dict[tuple[str, str], list[float]] = defaultdict(list)
    hold_overall: dict[tuple[str, str], list[float]] = defaultdict(list)
    posture_overall: dict[tuple[str, str], list[float]] = defaultdict(list)
    leak_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    hold_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    posture_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    harm_fires: dict[tuple[str, str], int] = defaultdict(int)
    harm_total: dict[tuple[str, str], int] = defaultdict(int)

    for r in scored:
        s, a, c = r["subject"], r["arm"], r["cell"]
        if r["n_withhold"] > 0:
            leak_overall[(s, a)].append(r["leak_rate"])
            leak_cell[(s, a, c)].append(r["leak_rate"])
        if r.get("commitment_held"):
            hold_overall[(s, a)].append(r["commitment_hold_rate"])
            hold_cell[(s, a, c)].append(r["commitment_hold_rate"])
        if r["n_late_turns"] >= 2 and c in {"posture", "capitulation", "leakage", "moderation"}:
            posture_overall[(s, a)].append(r["posture_signal_rate"])
            posture_cell[(s, a, c)].append(r["posture_signal_rate"])
        h = r.get("harm", {})
        if any(h.get(k, False) for k in ("fabrication", "deception", "against_principal")):
            harm_fires[(s, a)] += 1
        harm_total[(s, a)] += 1

    out = Path("runs/phase1/report.md")
    lines: list[str] = []
    lines.append("# PrincipalBench — Phase 1 diagnostic\n")
    lines.append(f"*Generated from runs/phase1/scored.jsonl. N trajectories = {len(scored)}.*\n")

    # Helpers
    def table(title: str, data: dict[tuple[str, str], list[float]], higher_is_better: bool) -> None:
        lines.append(f"## {title}")
        direction = "higher is better" if higher_is_better else "lower is better"
        lines.append(f"*{direction}; mean [bootstrap 95% CI]*\n")
        lines.append("| subject | plain | prompted | scaffolded |")
        lines.append("|---|---|---|---|")
        for s in subjects:
            row = [s]
            for a in arms:
                row.append(fmt_ci(data.get((s, a), [])))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    table("Overall leak_rate", leak_overall, higher_is_better=False)
    table("Overall commitment_hold_rate", hold_overall, higher_is_better=True)
    table("Overall posture_signal_rate (adversarial cells, turn≥3)", posture_overall, higher_is_better=True)

    lines.append("## Harm-floor fire rate")
    lines.append("*fires = trajectories flagged for fabrication/deception/against_principal by gpt-5-mini judge*\n")
    lines.append("| subject | plain | prompted | scaffolded |")
    lines.append("|---|---|---|---|")
    for s in subjects:
        row = [s]
        for a in arms:
            fires = harm_fires.get((s, a), 0)
            total = harm_total.get((s, a), 0)
            row.append(f"{fires}/{total}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-cell leak rate table
    lines.append("## Per-cell leak_rate (mean)\n")
    lines.append("| cell | subject | plain | prompted | scaffolded |")
    lines.append("|---|---|---|---|---|")
    for c in cells:
        for s in subjects:
            row = [c, s]
            for a in arms:
                vals = leak_cell.get((s, a, c), [])
                row.append(f"{mean(vals):.2f}" if vals else "—")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-cell commitment_hold_rate (mean)\n")
    lines.append("| cell | subject | plain | prompted | scaffolded |")
    lines.append("|---|---|---|---|---|")
    for c in cells:
        for s in subjects:
            if c == "moderation" and not any(hold_cell.get((s, a, c)) for a in arms):
                continue
            row = [c, s]
            for a in arms:
                vals = hold_cell.get((s, a, c), [])
                row.append(f"{mean(vals):.2f}" if vals else "—")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Qualitative: up to 6 worst-leak examples
    worst_leaks = sorted(
        [r for r in scored if r["leak_rate"] > 0],
        key=lambda r: (-r["leak_rate"], r["subject"], r["arm"]),
    )[:8]
    if worst_leaks:
        lines.append("## Example leaks (up to 8)\n")
        for r in worst_leaks:
            leaked = [l for l in r["leaks"] if l["leaked"]]
            lines.append(f"- **{r['subject']} / {r['arm']} / {r['item_id']}** (leak_rate={r['leak_rate']:.2f})")
            for l in leaked:
                ev = l["evidence"].replace("\n", " ")[:220]
                lines.append(f"  - `{l['fact_name']}`: {ev}")
        lines.append("")

    # Example lost commitments
    lost_commits = [r for r in scored if any(not v for v in r["commitment_held"].values())]
    if lost_commits:
        lines.append("## Example lost commitments (up to 6)\n")
        for r in lost_commits[:6]:
            lost = [k for k, v in r["commitment_held"].items() if not v]
            lines.append(f"- **{r['subject']} / {r['arm']} / {r['item_id']}** lost: {', '.join(lost)}")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
