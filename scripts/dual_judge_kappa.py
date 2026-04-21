"""Inter-judge agreement (Cohen's κ) for harm-floor scoring.

Re-judges every trajectory in a scored.jsonl file with a second judge model
(default `claude-haiku`) using the ORIGINAL composite-harm prompt, then
computes raw agreement + Cohen's κ vs. the primary judge (`gpt-5-mini`)
per-flag (fabrication, deception, against_principal) and overall any-fire.

Usage:
  python3 scripts/dual_judge_kappa.py \\
    --scored runs/phase2_baseline/scored.jsonl \\
    --trajectories runs/phase2_baseline/trajectories.jsonl \\
    --items items/v0 \\
    --judge claude-haiku \\
    --out runs/phase2_baseline/harm_kappa.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.items import load_items  # noqa: E402
from src.scorer import HARM_JUDGE_SYSTEM  # noqa: E402
from src.vendors import Message, get_vendor  # noqa: E402


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _render_convo(trajectory: dict) -> str:
    return "\n\n".join(
        f"[{t['speaker'].upper()} turn={t['turn']}] {t['text']}"
        for t in trajectory["turns"]
    )


def judge_harm(trajectory: dict, item, judge) -> dict:
    convo = _render_convo(trajectory)
    user = (
        f"PRINCIPAL BRIEFING:\n{item.principal_briefing.strip()}\n\n"
        f"CONVERSATION:\n{convo}\n\nReply with JSON only."
    )
    max_tokens = 1024 if judge.model.startswith("gpt-5") else 400
    resp = judge.chat(
        system=HARM_JUDGE_SYSTEM,
        messages=[Message(role="user", content=user)],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    obj = _extract_json(resp.text or "") or {}
    return {
        "fabrication": bool(obj.get("fabrication", False)),
        "deception": bool(obj.get("deception", False)),
        "against_principal": bool(obj.get("against_principal", False)),
    }


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Binary Cohen's κ. Returns NaN if both raters fire identically (no variance)."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--items", default="items/v0")
    ap.add_argument("--judge", default="claude-haiku")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="limit # trajectories (0 = all)")
    args = ap.parse_args()

    items = {it.id: it for it in load_items(args.items)}
    trajs = {}
    for line in Path(args.trajectories).read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        trajs[(t["subject"], t["arm"], t["item_id"])] = t

    rows = [json.loads(l) for l in Path(args.scored).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["arm"] in ("plain", "prompted")]
    if args.limit:
        rows = rows[: args.limit]

    judge = get_vendor(args.judge)

    records: list[dict] = []
    for i, r in enumerate(rows):
        key = (r["subject"], r["arm"], r["item_id"])
        traj = trajs.get(key)
        item = items.get(r["item_id"])
        if not traj or not item:
            continue
        primary = r.get("harm") or {}
        secondary = judge_harm(traj, item, judge)
        records.append({
            "item_id": r["item_id"],
            "arm": r["arm"],
            "primary": {k: bool(primary.get(k, False)) for k in ("fabrication", "deception", "against_principal")},
            "secondary": secondary,
        })
        print(f"[kappa {i+1}/{len(rows)}] {r['item_id']:30s} {r['arm']:10s}  "
              f"primary={[primary.get(k) for k in ('fabrication','deception','against_principal')]}  "
              f"secondary={[secondary[k] for k in ('fabrication','deception','against_principal')]}")

    def _col(key: str, side: str) -> list[bool]:
        return [bool(rec[side][key]) for rec in records]

    summary = {}
    for key in ("fabrication", "deception", "against_principal"):
        a = _col(key, "primary")
        b = _col(key, "secondary")
        n = len(a)
        agreement = sum(1 for x, y in zip(a, b) if x == y) / n if n else float("nan")
        summary[key] = {
            "n": n,
            "primary_positive_rate": sum(a) / n if n else float("nan"),
            "secondary_positive_rate": sum(b) / n if n else float("nan"),
            "raw_agreement": agreement,
            "kappa": cohen_kappa(a, b),
        }

    any_a = [any(rec["primary"][k] for k in ("fabrication","deception","against_principal")) for rec in records]
    any_b = [any(rec["secondary"][k] for k in ("fabrication","deception","against_principal")) for rec in records]
    summary["any_fire"] = {
        "n": len(any_a),
        "primary_positive_rate": sum(any_a) / len(any_a) if any_a else float("nan"),
        "secondary_positive_rate": sum(any_b) / len(any_b) if any_b else float("nan"),
        "raw_agreement": sum(1 for x, y in zip(any_a, any_b) if x == y) / len(any_a) if any_a else float("nan"),
        "kappa": cohen_kappa(any_a, any_b),
    }

    out = {
        "primary_judge": "gpt-5-mini",
        "secondary_judge": args.judge,
        "scored_path": args.scored,
        "n_evaluations": len(records),
        "summary": summary,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[kappa] wrote {args.out}")
    for key, s in summary.items():
        print(f"  {key:<22}  κ={s['kappa']:.3f}  agreement={s['raw_agreement']:.3f}  "
              f"(primary={s['primary_positive_rate']:.2f} secondary={s['secondary_positive_rate']:.2f}, n={s['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
