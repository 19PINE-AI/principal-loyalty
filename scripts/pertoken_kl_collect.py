"""Collect per-token teacher top-K logprob distributions for the canonical
Thinking Machines / DeepSeek-V4 on-policy distillation algorithm.

Algorithm:
  For each agent turn N in each student trajectory T_i:
    1. Build chat-formatted (history_up_to_cp[N-1]) under PLAIN system prompt.
       This is the STUDENT's input context. Apply teacher's chat template with
       enable_thinking=False to avoid spurious think-tag distributions.
    2. Append the student's actual sampled response a_S[N] (no think tags) and
       chat-format with add_generation_prompt=False (which closes with im_end).
    3. Send full token sequence to teacher (Qwen3-32B-AWQ in scaffolded/v4 arm
       via system prompt) requesting prompt_logprobs=K. Teacher returns the
       per-position top-K distribution over its vocab for every input token.
    4. Slice out the response token positions; save (input_ids, response_start,
       response_ids, teacher_topk[K positions]).

The teacher arm: we want teacher logprobs UNDER THE V4 SCAFFOLDED PROMPT. The
v4 prompt is in the system slot. Because the student is trained under PLAIN
system, we substitute system in the teacher request to v4 SCAFFOLDED -- this
is the on-policy-distillation-of-prompted-teacher trick. The student then
learns to behave as if v4 were applied, even with plain prompt at inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from transformers import AutoTokenizer

from src.agent import PLAIN_SYSTEM, PROMPTED_SYSTEM, SCAFFOLDED_SYSTEM_HEADER  # noqa: E402
from src.items import load_items, Item  # noqa: E402


TEACHER_URL = os.environ.get("TEACHER_URL", "http://localhost:8001/v1/completions")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL_NAME", "Qwen/Qwen3-32B")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "Qwen/Qwen3-32B")

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


def build_messages(item: Item, history: list[dict], student_response: str,
                   teacher_system: str) -> list[dict]:
    """Build chat messages for teacher: system=teacher_system,
    user=briefing, then alternating cp/agent ending with student's response
    as the final assistant turn.

    history is the conversation BEFORE the student's response of interest,
    as a list of {speaker, text} dicts.
    """
    msgs = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": briefing_for(item)},
    ]
    # Walk history; merge consecutive same-role into one bubble (chat templates
    # tolerate it for Qwen).
    for h in history:
        role = "user" if h["speaker"] == "counterparty" else "assistant"
        text = h["text"]
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] = (msgs[-1]["content"] + "\n\n" + text).strip()
        else:
            msgs.append({"role": role, "content": text})
    # Append student's response as final assistant turn
    if msgs and msgs[-1]["role"] == "assistant":
        msgs[-1]["content"] = (msgs[-1]["content"] + "\n\n" + student_response).strip()
    else:
        msgs.append({"role": "assistant", "content": student_response})
    return msgs


def extract_response_token_range(tokenizer, messages: list[dict]) -> tuple[list[int], int]:
    """Return (full_token_ids, response_start_idx).

    response_start_idx is the index where the LAST assistant turn's content
    begins.
    """
    # Tokenize everything up to but excluding the last assistant turn, with
    # add_generation_prompt=True (so we get the "<|im_start|>assistant\n"
    # header included but not yet content).
    msgs_before = messages[:-1]
    prefix_text = tokenizer.apply_chat_template(
        msgs_before,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    response_start = len(prefix_ids)
    return full_ids, response_start


def query_teacher_logprobs(token_ids: list[int], top_k: int = 20,
                             max_retries: int = 3) -> dict:
    """POST to vLLM /v1/completions with prompt_token_ids + prompt_logprobs."""
    payload = {
        "model": TEACHER_MODEL,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": top_k,
        "logprobs": 0,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(TEACHER_URL, json=payload, timeout=300)
            if r.status_code == 200:
                return r.json()
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"  [retry {attempt+1}] teacher request failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"teacher request failed after {max_retries} retries: {r.text[:200]}")


def collect_for_item(item: Item, traj: dict, tokenizer, top_k: int,
                       teacher_system: str, max_turns: int):
    """Process one trajectory, yield per-turn KL data."""
    turns = traj.get("turns") or []
    cp_so_far: list[dict] = []
    agent_so_far: list[dict] = []

    for t in turns:
        if t["speaker"] == "counterparty":
            cp_so_far.append(t)
        elif t["speaker"] == "agent":
            agent_so_far.append(t)
            if len(agent_so_far) > max_turns:
                break
            # history = all cp + all prior agent turns
            history: list[dict] = []
            for i, c in enumerate(cp_so_far):
                history.append({"speaker": "counterparty", "text": c["text"]})
                if i < len(agent_so_far) - 1:
                    history.append({"speaker": "agent", "text": agent_so_far[i]["text"]})
            student_response = t["text"]
            if not student_response or not student_response.strip():
                continue

            messages = build_messages(item, history, student_response, teacher_system)
            try:
                full_ids, response_start = extract_response_token_range(tokenizer, messages)
            except Exception as e:
                print(f"  [skip] tokenization failed for {item.id} turn {t['turn']}: {e}")
                continue

            if len(full_ids) > 8000:
                print(f"  [skip] too long for {item.id} turn {t['turn']}: {len(full_ids)} tokens")
                continue
            response_len = len(full_ids) - response_start
            if response_len < 2:
                continue

            try:
                resp = query_teacher_logprobs(full_ids, top_k=top_k)
            except Exception as e:
                print(f"  [skip] teacher query failed for {item.id} turn {t['turn']}: {e}")
                continue

            prompt_logprobs = resp["choices"][0].get("prompt_logprobs") or []
            # prompt_logprobs is a list of dicts (one per input token, except
            # the very first which is None). Each dict maps str(token_id) -> {
            #     "logprob": float, "rank": int, "decoded_token": str }.
            # Wait, actually vLLM returns it as {tok_id_str: {"logprob": ..., "decoded_token": ..., "rank": ...}}
            # Slice to response positions
            if len(prompt_logprobs) != len(full_ids):
                # Some servers (e.g. AWQ Mixtral with different tokenizer settings)
                # return a different length. Skip rather than abort the whole batch.
                print(f"  [skip] prompt_logprobs length {len(prompt_logprobs)} != tokens {len(full_ids)} for {item.id} turn {t['turn']}")
                continue

            response_topk = []
            for pos in range(response_start, len(full_ids)):
                slot = prompt_logprobs[pos]
                if not slot:
                    response_topk.append({})
                    continue
                # slot is {token_id_str: {logprob, rank, decoded_token}}
                # Compress to {int(tok_id): float(logprob)} top-K
                compact = {}
                for tok_id_str, info in slot.items():
                    if isinstance(info, dict):
                        lp = info.get("logprob")
                    else:
                        lp = float(info)
                    try:
                        compact[int(tok_id_str)] = float(lp)
                    except (ValueError, TypeError):
                        continue
                response_topk.append(compact)

            yield {
                "item_id": item.id,
                "cell": getattr(item, "cell", None),
                "subcell": getattr(item, "subcell", None),
                "turn_idx": t["turn"],
                "input_ids": full_ids,
                "response_start": response_start,
                "response_len": response_len,
                "teacher_topk_logprobs": response_topk,
            }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True,
                   help="path to student trajectories.jsonl from a prior on-policy run")
    p.add_argument("--items-dir", default="items/v0")
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-turns-per-item", type=int, default=4)
    p.add_argument("--teacher-arm", default="scaffolded",
                   choices=["plain", "prompted", "scaffolded"])
    p.add_argument("--heldout", nargs="*", default=DEFAULT_HELDOUT)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of trajectories processed (for sanity runs)")
    args = p.parse_args()

    if args.teacher_arm == "scaffolded":
        teacher_system = SCAFFOLDED_SYSTEM_HEADER
    elif args.teacher_arm == "prompted":
        teacher_system = PROMPTED_SYSTEM
    else:
        teacher_system = PLAIN_SYSTEM

    items = {it.id: it for it in load_items(args.items_dir)}
    trajs = []
    with open(args.trajectories) as f:
        for line in f:
            traj = json.loads(line)
            if traj["item_id"] in args.heldout:
                continue
            if traj["item_id"] not in items:
                continue
            # Only use plain-arm student trajectories — those are the "raw" student
            if traj.get("arm") != "plain":
                continue
            trajs.append(traj)
    if args.limit:
        trajs = trajs[:args.limit]
    print(f"[collect] {len(trajs)} student trajectories to process")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume support: skip (item_id, turn_idx) already in out
    done: set = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["item_id"], d["turn_idx"]))
                except Exception:
                    pass
    print(f"[collect] resume: {len(done)} already done")

    lock = Lock()
    written = 0
    t0 = time.time()

    def process_one(traj):
        nonlocal written
        item = items[traj["item_id"]]
        local_records = []
        for rec in collect_for_item(item, traj, tokenizer, args.top_k,
                                       teacher_system, args.max_turns_per_item):
            if (rec["item_id"], rec["turn_idx"]) in done:
                continue
            local_records.append(rec)
        if not local_records:
            return 0
        with lock:
            with open(out_path, "a") as f:
                for rec in local_records:
                    f.write(json.dumps(rec) + "\n")
            written += len(local_records)
            elapsed = time.time() - t0
            print(f"  [{written}] {item.id}: +{len(local_records)} turns (elapsed {elapsed:.0f}s)")
        return len(local_records)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(process_one, t) for t in trajs]
        for _ in as_completed(futs):
            pass

    print(f"[collect] DONE. {written} new turn-records written. Total {written + len(done)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
