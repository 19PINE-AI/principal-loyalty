"""One iteration of on-policy distillation.

This script implements the multi-turn approximation of the Thinking Machines /
DeepSeek-V4 on-policy distillation algorithm, adapted for an API-only teacher
(claude-sonnet) where exact per-token KL is not available.

ALGORITHM (per iteration k):
  1. Student (current adapter, served by vLLM under name 'qwen-8b-local' on
     port 8000) samples its own trajectory T_i on each training item i, under
     the PLAIN system prompt.
  2. For each agent turn N in T_i:
       - Build conversation history h_{1:N-1} = briefing + cp[0] + agent[1] + ... + cp[N-1]
       - Ask TEACHER (claude-sonnet, v4 scaffolded arm) for its response a_T at h_{1:N-1}
       - Record SFT training point: (PLAIN_SYSTEM, h_{1:N-1}, a_T)
       - Optionally: record DPO pair (chosen=a_T, rejected=student_a[N])
  3. Write out:
       - data/onpolicy_sft_iterK.jsonl   (Variant 1: per-turn SFT, dense supervision)
       - data/onpolicy_dpo_iterK.jsonl   (Variant 2: per-turn DPO, comparison)

WHY THIS IS ON-POLICY:
  Each training point's CONTEXT is drawn from the student's actual sampled
  trajectory. The teacher provides corrections AT THE STUDENT'S STATE, not at
  hypothetical states the teacher would generate from scratch. As the student
  improves, the failure-state distribution shifts and the training points
  follow.

WHY MULTI-TURN:
  First-turn-only pairs (the v4.1 walkback approach) only correct opening
  behavior. Multi-turn provides K-3 training points per item per iteration,
  including failures that emerge mid-conversation under sustained pressure.
  Equivalent to per-token in spirit; per-turn in resolution.

WHY AN API TEACHER IS OK:
  We don't need teacher logits. The teacher's response IS the training target.
  Per-turn SFT loss = -log pi_S(a_T | h_{1:N-1}) is well-defined without
  teacher logits.

WHY PLAIN SYSTEM PROMPT:
  The goal of distillation is to internalize the v4 rules into the WEIGHTS.
  Training under PLAIN means the student learns to behave as if v4 were
  applied, even when the v4 rules are not in the prompt at inference time.

Args:
  --out-prefix STR        prefix for output files (e.g., 'data/onpolicy_iter1')
  --items DIR             items dir (default items/v0)
  --n-samples N           samples per item (default 1)
  --temperature T         student sample temperature (default 1.0)
  --parallel P            parallel rollouts (default 4)
  --heldout I1 I2 ...     item IDs to exclude
  --max-turns-per-item N  cap per-trajectory training points (default 6)
  --student-arm ARM       arm under which student samples (default plain)
  --teacher-arm ARM       arm under which teacher generates (default scaffolded;
                            this is the v4-prompted teacher)

Output:
  ${OUT_PREFIX}_sft.jsonl  per-turn SFT training points (Variant 1)
  ${OUT_PREFIX}_dpo.jsonl  per-turn DPO pairs (Variant 2)
  ${OUT_PREFIX}_stats.json summary stats

Side effect: assumes student vLLM is already serving on localhost:8000 as
'qwen-8b-local'.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Agent, PLAIN_SYSTEM, reader_sentinel  # noqa: E402
from src.harness import run_trajectory  # noqa: E402
from src.items import load_items, Item  # noqa: E402
from src.scorer import score_trajectory  # noqa: E402
from src.vendors import get_vendor  # noqa: E402


DEFAULT_HELDOUT = [
    "pb-author-testcase-01",
    "pb-author-to-principal-01",
    "pb-capit-repq-01",
    "pb-leak-batna-01",
    "pb-sanity-coop-01",
]


def briefing_for(item: Item) -> str:
    return (
        "## Principal briefing (from your user)\n\n"
        f"{item.principal_briefing.strip()}\n\n"
        "The other party is about to send you a message. Respond on my behalf."
    )


def teacher_response_at_state(item: Item, history: list[dict],
                                teacher_spec: str = "claude-sonnet",
                                arm: str = "scaffolded") -> str:
    """Get teacher's response given a conversation history.

    history is a list of {speaker, text} dicts from the student's trajectory
    (cp[0], agent[1], cp[1], agent[2], ..., cp[N-1]).

    Returns the teacher's next agent message (turn N).
    """
    teacher_vendor = get_vendor(teacher_spec)
    teacher_agent = Agent(vendor=teacher_vendor, item=item, arm=arm)
    teacher_agent.start()
    # Replay history: feed in counterparty messages; for each cp message,
    # if we're not at the last one, also "replay" the student's agent reply
    # so teacher's internal state reflects the conversation.
    cp_messages = [t for t in history if t.get("speaker") == "counterparty"]
    student_agent_messages = [t for t in history if t.get("speaker") == "agent"]
    if not cp_messages:
        return ""
    # cp[0]: open the conversation; teacher generates agent[1]
    # but we want teacher's agent[N], not agent[1]. So we feed cp[0]..cp[N-1]
    # in sequence, accepting teacher's response after the LAST cp; but
    # interleaving requires we discard teacher's intermediate replies and
    # use the STUDENT's intermediate replies as if the teacher said them.
    #
    # Simplest implementation: build the teacher Agent's _messages list
    # directly to mirror the student's trajectory, then ask for one new turn.
    #
    # We use the Agent class to handle the chat-template details.
    from src.vendors import Message
    # Already has briefing in _messages[0]. Now alternate cp/agent_student
    # up to cp[N-1].
    for i in range(len(cp_messages) - 1):
        cp_msg = cp_messages[i]
        teacher_agent._messages.append(Message(
            role="user", content=f"[other party says]\n{cp_msg['text'].strip()}"
        ))
        if i < len(student_agent_messages):
            teacher_agent._messages.append(Message(
                role="assistant", content=student_agent_messages[i]["text"]
            ))
    # Final cp[N-1] turn: ask teacher to respond
    final_cp = cp_messages[-1]
    reply = teacher_agent.reply(final_cp["text"])
    return reply.text


def extract_turn_pairs(item: Item, traj: dict, max_turns: int = 6):
    """Yield (turn_idx, history_up_to_cp, student_agent_response) tuples
    for each agent turn in the trajectory (up to max_turns).
    """
    turns = traj.get("turns") or []
    # Iterate agent turns
    n_emitted = 0
    cp_so_far = []
    agent_so_far = []
    for t in turns:
        if t["speaker"] == "counterparty":
            cp_so_far.append(t)
        elif t["speaker"] == "agent":
            agent_so_far.append(t)
            # history at this point = briefing + cp[0..len(cp_so_far)-1] + agent[1..len(agent_so_far)-1]
            # (i.e., all cp messages up to and including the one before this agent turn)
            history = list(cp_so_far)
            # interleave with student's prior agent messages
            full_history = []
            for i, c in enumerate(cp_so_far):
                full_history.append({"speaker": "counterparty", "text": c["text"]})
                if i < len(agent_so_far) - 1:
                    full_history.append({"speaker": "agent", "text": agent_so_far[i]["text"]})
            yield (t["turn"], full_history, t["text"])
            n_emitted += 1
            if n_emitted >= max_turns:
                break


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--items", default="items/v0")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--heldout", nargs="*", default=DEFAULT_HELDOUT)
    p.add_argument("--max-turns-per-item", type=int, default=6)
    p.add_argument("--student-arm", default="plain")
    p.add_argument("--teacher-arm", default="scaffolded")
    p.add_argument("--counterparty", default="claude-sonnet")
    p.add_argument("--teacher-spec", default="claude-sonnet")
    p.add_argument("--require-failure", action="store_true",
                   help="only emit training points from trajectories that fired leak or MI")
    args = p.parse_args()

    items = load_items(args.items)
    heldout = set(args.heldout)
    train_items = [it for it in items if it.id not in heldout]
    print(f"[on-policy] {len(train_items)} train items × {args.n_samples} samples; "
          f"student arm={args.student_arm} teacher arm={args.teacher_arm}")

    # Step 1: Student samples trajectories
    tasks = [(it, s) for it in train_items for s in range(args.n_samples)]
    trajs: list[tuple[Item, int, dict]] = []
    lock = Lock()
    t_start = time.time()

    def _run_traj(item, sample_idx):
        return (item, sample_idx, run_trajectory(
            subject_spec="qwen-8b-local",
            arm=args.student_arm,
            item=item,
            counterparty_spec=args.counterparty,
        ))

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(_run_traj, item, s) for (item, s) in tasks]
        for fut in as_completed(futs):
            try:
                tup = fut.result()
                with lock:
                    trajs.append(tup)
                    if len(trajs) % 10 == 0:
                        print(f"  [{len(trajs)}/{len(tasks)}] student-traj done")
            except Exception as e:
                print(f"  FAIL student-traj: {type(e).__name__}: {e}")
    print(f"[on-policy] generated {len(trajs)} trajectories in {time.time()-t_start:.0f}s")

    # AUDIT: print turn distribution and early_end_reason; abort if degenerate.
    # trajs is a list of (item, sample_idx, trajectory_dict) — 3-tuples.
    from collections import Counter as _C
    turn_dist = _C((t.get("n_agent_turns") or 0) for (_, _, t) in trajs)
    eer_dist = _C((t.get("early_end_reason") or "OK") for (_, _, t) in trajs)
    print(f"[on-policy] AUDIT turn_dist={dict(sorted(turn_dist.items()))}")
    print(f"[on-policy] AUDIT early_end={dict(eer_dist)}")
    n_with_agent = sum(1 for (_, _, t) in trajs if (t.get("n_agent_turns") or 0) > 0)
    if n_with_agent == 0:
        print("[on-policy] FATAL: all trajectories have 0 agent turns. Refusing to generate distillation pairs from empty data.")
        return 2

    # Persist trajectories.
    traj_out = Path(args.out_prefix + "_trajectories.jsonl")
    traj_out.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_out, "w") as f:
        for (_, _, t) in trajs:
            f.write(json.dumps(t) + "\n")
    print(f"[on-policy] persisted trajectories → {traj_out}")

    # Step 2: Score (to identify failures if --require-failure)
    probe = get_vendor("gpt-5-mini")
    judge = get_vendor("gpt-5-mini")
    scored: list[tuple[Item, int, dict, object]] = []

    def _score(traj_tup):
        item, s, t = traj_tup
        sc = score_trajectory(t, item, probe, judge)
        return (item, s, t, sc)

    if args.require_failure:
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel * 2) as ex:
            futs = [ex.submit(_score, tt) for tt in trajs]
            for fut in as_completed(futs):
                try:
                    scored.append(fut.result())
                except Exception as e:
                    print(f"  FAIL score: {type(e).__name__}: {e}")
        print(f"[on-policy] scored {len(scored)} in {time.time()-t_start:.0f}s")
        # filter to failures
        scored = [tup for tup in scored
                  if (tup[3].leak_rate or 0) > 0 or tup[3].harm.get("missed_instruction")]
        print(f"[on-policy] {len(scored)} failing trajectories (using only these)")
    else:
        scored = [(item, s, t, None) for (item, s, t) in trajs]

    # Step 3: For each turn in each (failing) trajectory, query teacher
    sft_points = []   # variant 1: per-turn SFT (PLAIN_SYSTEM, history, teacher response)
    dpo_pairs = []    # variant 2: per-turn DPO (chosen=teacher, rejected=student)
    teacher_calls = 0
    t_start = time.time()

    def _process(item, sample_idx, traj, _sc):
        nonlocal teacher_calls
        local_sft, local_dpo = [], []
        for turn_idx, history, student_resp in extract_turn_pairs(item, traj, args.max_turns_per_item):
            if not student_resp:
                continue
            try:
                teacher_resp = teacher_response_at_state(
                    item, history,
                    teacher_spec=args.teacher_spec, arm=args.teacher_arm,
                )
                with lock:
                    teacher_calls += 1
            except Exception as e:
                print(f"  FAIL teacher {item.id} turn={turn_idx}: {type(e).__name__}: {e}")
                continue
            if not teacher_resp.strip():
                continue
            # Build a prompt string for SFT/DPO storage
            cp_msgs = [h for h in history if h["speaker"] == "counterparty"]
            agent_msgs = [h for h in history if h["speaker"] == "agent"]
            prompt_parts = [briefing_for(item)]
            for i, cp in enumerate(cp_msgs):
                prompt_parts.append(f"[other party says]\n{cp['text'].strip()}")
                if i < len(agent_msgs):
                    prompt_parts.append(f"[you replied]\n{agent_msgs[i]['text'].strip()}")
            prompt = "\n\n".join(prompt_parts)
            common = {
                "item_id": item.id, "cell": item.cell, "subcell": item.subcell,
                "sample_idx": sample_idx, "turn_idx": turn_idx,
                "system": PLAIN_SYSTEM.strip(),
                "prompt": prompt,
            }
            local_sft.append({**common, "completion": teacher_resp})
            local_dpo.append({**common, "chosen": teacher_resp, "rejected": student_resp,
                              "preferred_subject": args.teacher_spec,
                              "dispreferred_subject": "qwen-8b-local"})
        return local_sft, local_dpo

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(_process, item, s, t, sc) for (item, s, t, sc) in scored]
        for fut in as_completed(futs):
            try:
                ls, ld = fut.result()
                with lock:
                    sft_points.extend(ls)
                    dpo_pairs.extend(ld)
            except Exception as e:
                print(f"  FAIL process: {type(e).__name__}: {e}")

    print(f"[on-policy] {teacher_calls} teacher calls ({time.time()-t_start:.0f}s); "
          f"{len(sft_points)} SFT points, {len(dpo_pairs)} DPO pairs")

    # Step 4: Write outputs
    out_prefix = args.out_prefix
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_prefix}_sft.jsonl", "w") as fout:
        for sp in sft_points:
            fout.write(json.dumps(sp) + "\n")
    with open(f"{out_prefix}_dpo.jsonl", "w") as fout:
        for dp in dpo_pairs:
            fout.write(json.dumps(dp) + "\n")
    with open(f"{out_prefix}_stats.json", "w") as fout:
        json.dump({
            "n_trajs": len(trajs),
            "n_failing_trajs": len(scored) if args.require_failure else None,
            "n_sft_points": len(sft_points),
            "n_dpo_pairs": len(dpo_pairs),
            "teacher_calls": teacher_calls,
            "student_arm": args.student_arm,
            "teacher_arm": args.teacher_arm,
        }, fout, indent=2)
    print(f"[on-policy] wrote → {out_prefix}_sft.jsonl, {out_prefix}_dpo.jsonl, {out_prefix}_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
