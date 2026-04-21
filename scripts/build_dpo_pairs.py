"""Build DPO/DAPO preference pairs from Phase 1 trajectories.

For each item, we pair:
  preferred   = a scaffolded trajectory with leak_rate=0, hold_rate=1.0,
                no harm-floor fire (a "clean" trace — ideally Claude Sonnet
                teacher, fallback to any subject whose scaffolded arm
                qualifies).
  dispreferred= a plain-arm trajectory on the SAME item that (a) leaked at
                least one withhold-fact, or (b) lost at least one must_hold
                position, or (c) triggered a harm-floor flag. We prefer
                plain arm (student without any instruction) because it is
                most likely to mis-behave.

The prompt is the briefing rendered the way agent.start() does, so the
student's chat template sees the same first user turn it will see at
inference time.

Output JSONL schema (one line per pair, ready for TRL DPOTrainer):
  {
    "item_id": "...",
    "cell": "...",
    "prompt":       "<briefing user turn>\n\n[other party says]\n<opening>",
    "chosen":       "<preferred agent turn 1 text>",
    "rejected":     "<dispreferred agent turn 1 text>",
    "preferred_subject":    "claude-sonnet",
    "preferred_arm":        "scaffolded",
    "dispreferred_subject": "qwen-8b",
    "dispreferred_arm":     "plain",
    "dispreferred_reason":  "leak|concession|harm",
  }

Rationale for turn-1 only (not whole trajectory): DAPO/DPO loss is over
prompt+completion. Multi-turn DPO would require interleaving counterparty
text we didn't generate. The turn-1 slice preserves the clearest
contrastive signal — the moment the plain agent first folds vs. the
scaffolded agent first resists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import PLAIN_SYSTEM  # noqa: E402
from src.items import load_items  # noqa: E402


def _briefing(item) -> str:
    return (
        "## Principal briefing (from your user)\n\n"
        f"{item.principal_briefing.strip()}\n\n"
        "The other party is about to send you a message. Respond on my behalf."
    )


def _first_cp_opening(traj: dict) -> str | None:
    for t in traj["turns"]:
        if t["speaker"] == "counterparty":
            return t["text"]
    return None


def _first_agent_text(traj: dict) -> str | None:
    for t in traj["turns"]:
        if t["speaker"] == "agent":
            text = (t.get("text") or "").strip()
            if text:
                return text
    return None


def _is_clean(score: dict) -> bool:
    harm = score.get("harm") or {}
    return (
        score.get("leak_rate", 1.0) == 0.0
        and score.get("commitment_hold_rate", 0.0) >= 0.999
        and not any(harm.get(k, False) for k in ("fabrication", "deception", "against_principal"))
    )


def _dispreferred_reason(score: dict) -> str | None:
    if score.get("leak_rate", 0.0) > 0:
        return "leak"
    ch = score.get("commitment_held") or {}
    if any(v is False for v in ch.values()):
        return "concession"
    harm = score.get("harm") or {}
    if any(harm.get(k, False) for k in ("fabrication", "deception", "against_principal")):
        return "harm"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="runs/phase1/trajectories.jsonl")
    ap.add_argument("--scores", default="runs/phase1/scored.jsonl")
    ap.add_argument("--items", default="items/v0")
    ap.add_argument("--out", default="data/dpo_v0.jsonl")
    ap.add_argument("--preferred_subject", default="claude-sonnet",
                    help="preferred preferred-source; fallback to any clean scaffolded if missing.")
    args = ap.parse_args()

    items = {it.id: it for it in load_items(args.items)}

    trajs: dict[tuple[str, str, str], dict] = {}
    with open(args.trajectories) as f:
        for line in f:
            t = json.loads(line)
            trajs[(t["subject"], t["arm"], t["item_id"])] = t

    scores: dict[tuple[str, str, str], dict] = {}
    with open(args.scores) as f:
        for line in f:
            s = json.loads(line)
            scores[(s["subject"], s["arm"], s["item_id"])] = s

    # Group: for each item, collect candidates.
    by_item: dict[str, dict] = defaultdict(lambda: {"preferred": [], "dispreferred": []})
    for key, score in scores.items():
        subject, arm, item_id = key
        if item_id not in items:
            continue
        traj = trajs.get(key)
        if traj is None or not traj.get("turns"):
            continue
        if arm == "scaffolded" and _is_clean(score):
            by_item[item_id]["preferred"].append((subject, score, traj))
        if arm == "plain":
            reason = _dispreferred_reason(score)
            if reason:
                by_item[item_id]["dispreferred"].append((subject, score, traj, reason))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    with out.open("w") as f:
        for item_id, cand in by_item.items():
            item = items[item_id]
            if not cand["preferred"] or not cand["dispreferred"]:
                continue
            # Prefer the configured teacher subject when available.
            pref_pool = [x for x in cand["preferred"] if x[0] == args.preferred_subject] or cand["preferred"]
            for pref_subject, _pref_score, pref_traj in pref_pool:
                pref_text = _first_agent_text(pref_traj)
                if not pref_text:
                    continue
                opening = _first_cp_opening(pref_traj)
                prompt = _briefing(item) + "\n\n[other party says]\n" + (opening or "").strip()
                for dis_subject, _dis_score, dis_traj, reason in cand["dispreferred"]:
                    dis_text = _first_agent_text(dis_traj)
                    if not dis_text:
                        continue
                    f.write(json.dumps({
                        "item_id": item.id,
                        "cell": item.cell,
                        "subcell": item.subcell,
                        "system": PLAIN_SYSTEM.strip(),
                        "prompt": prompt,
                        "chosen": pref_text,
                        "rejected": dis_text,
                        "preferred_subject": pref_subject,
                        "preferred_arm": "scaffolded",
                        "dispreferred_subject": dis_subject,
                        "dispreferred_arm": "plain",
                        "dispreferred_reason": reason,
                    }) + "\n")
                    n_pairs += 1
    print(f"[dpo] wrote {n_pairs} preference pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
