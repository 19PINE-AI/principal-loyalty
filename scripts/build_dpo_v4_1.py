"""Phase 2.1: build DPO pairs targeting the missed_instruction regression.

For each (item, arm) row in runs/phase2_trained_v4/scored.jsonl where the v4-
trained student triggered `missed_instruction=true`, we produce a DPO pair:

  prompt   = briefing user-turn + "[other party says]\n<counterparty opening>"
  chosen   = teacher's first agent turn on the same item (from
             data/teacher_traces_v4.jsonl — we trust the teacher scaffolded
             rollout that passed the clean filter)
  rejected = v4-trained student's first agent turn (the refusal)

Items with no teacher trace are skipped — they can be regenerated separately
if needed. This targets the specific failure mode: posture-collapse refusals
against legitimate principal requests (reader_is_principal probe items and
cooperative sanity items).

Output: data/dpo_v4_1.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import PLAIN_SYSTEM  # noqa: E402
from src.items import load_items  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / "runs/phase2_trained_v4/scored.jsonl"
TRAJS = ROOT / "runs/phase2_trained_v4/trajectories.jsonl"
TEACHER = ROOT / "data/teacher_traces_v4.jsonl"
OUT = ROOT / "data/dpo_v4_1.jsonl"


def briefing(item) -> str:
    return (
        "## Principal briefing (from your user)\n\n"
        f"{item.principal_briefing.strip()}\n\n"
        "The other party is about to send you a message. Respond on my behalf."
    )


def first_agent_text(turns: list[dict]) -> str | None:
    for t in turns:
        if t.get("speaker") == "agent":
            text = (t.get("text") or "").strip()
            if text:
                return text
    return None


def first_cp_text(turns: list[dict]) -> str | None:
    for t in turns:
        if t.get("speaker") == "counterparty":
            text = (t.get("text") or "").strip()
            if text:
                return text
    return None


def main() -> int:
    items = {it.id: it for it in load_items(ROOT / "items/v0")}

    # Student's failing trajectories keyed by (item_id, arm)
    student_trajs = {}
    with TRAJS.open() as f:
        for line in f:
            t = json.loads(line)
            student_trajs[(t["item_id"], t["arm"])] = t

    # Teacher clean rollouts grouped by item_id
    teacher_by_item: dict[str, list[dict]] = defaultdict(list)
    with TEACHER.open() as f:
        for line in f:
            tr = json.loads(line)
            teacher_by_item[tr["item_id"]].append(tr)

    # Collect missed_instruction regression rows
    pairs = 0
    skipped_no_teacher = 0
    skipped_no_turns = 0
    seen_prompts: set[str] = set()  # dedupe identical (item, chosen prefix) pairs
    with OUT.open("w") as fout, SCORED.open() as fin:
        for line in fin:
            r = json.loads(line)
            harm = r.get("harm") or {}
            if not harm.get("missed_instruction"):
                continue
            item = items.get(r["item_id"])
            if item is None:
                continue
            student = student_trajs.get((r["item_id"], r["arm"]))
            if student is None:
                skipped_no_turns += 1
                continue
            rejected = first_agent_text(student.get("turns") or [])
            if not rejected:
                skipped_no_turns += 1
                continue
            teacher_rollouts = teacher_by_item.get(r["item_id"], [])
            if not teacher_rollouts:
                skipped_no_teacher += 1
                continue

            cp_opening = first_cp_text(student.get("turns") or [])
            if not cp_opening:
                cp_opening = (item.counterparty_opening or "").strip()
            prompt = briefing(item) + "\n\n[other party says]\n" + cp_opening.strip()

            # Pair against every teacher rollout for this item — light augmentation.
            for tr in teacher_rollouts:
                chosen = first_agent_text(tr.get("turns") or [])
                if not chosen:
                    continue
                key = (r["item_id"], r["arm"], chosen[:200], rejected[:200])
                if key in seen_prompts:
                    continue
                seen_prompts.add(key)
                fout.write(json.dumps({
                    "item_id": item.id,
                    "cell": item.cell,
                    "subcell": item.subcell,
                    "system": PLAIN_SYSTEM.strip(),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "preferred_subject": "claude-sonnet",
                    "preferred_arm": "scaffolded",
                    "dispreferred_subject": "qwen-8b-local",
                    "dispreferred_arm": r["arm"],
                    "dispreferred_reason": "missed_instruction",
                }) + "\n")
                pairs += 1

    print(f"[v4.1] wrote {pairs} pairs -> {OUT}")
    print(f"  skipped (no teacher trace): {skipped_no_teacher}")
    print(f"  skipped (no student turns): {skipped_no_turns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
