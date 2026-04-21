"""Trajectory runner.

Runs one (subject, arm, item) cell by alternating counterparty and agent turns
for up to item.max_turns. Writes one JSONL line per trajectory to the runs
directory.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Iterable

from .agent import Agent
from .counterparty import Counterparty
from .items import Item
from .vendors import Vendor, get_vendor


def run_trajectory(
    subject_spec: str,
    arm: str,
    item: Item,
    counterparty_spec: str = "claude-sonnet",
    counterparty_vendor: Vendor | None = None,
) -> dict:
    """Run one trajectory. Returns a dict ready to be written as a JSONL line."""
    subject = get_vendor(subject_spec)
    cp_vendor = counterparty_vendor or get_vendor(counterparty_spec)

    agent = Agent(vendor=subject, item=item, arm=arm)
    counterparty = Counterparty(vendor=cp_vendor, item=item)
    agent.start()

    turns: list[dict] = []
    current_counterparty = counterparty.opening()
    turns.append({"speaker": "counterparty", "turn": 0, "text": current_counterparty})

    t0 = time.time()
    early_end_reason = ""
    for turn_idx in range(1, item.max_turns + 1):
        try:
            agent_reply = agent.reply(current_counterparty)
        except Exception as e:  # noqa: BLE001
            early_end_reason = f"agent_error:{type(e).__name__}:{str(e)[:200]}"
            break
        turns.append({
            "speaker": "agent",
            "turn": turn_idx,
            "text": agent_reply.text,
            "loyalty_state": dataclasses.asdict(agent.state) if arm == "scaffolded" else None,
        })
        agent.update_state(current_counterparty, agent_reply.text)

        try:
            cp_reply = counterparty.reply(agent_reply.text)
        except Exception as e:  # noqa: BLE001
            early_end_reason = f"counterparty_error:{type(e).__name__}:{str(e)[:200]}"
            break
        if cp_reply.end:
            early_end_reason = "counterparty_end"
            break
        current_counterparty = cp_reply.text
        turns.append({"speaker": "counterparty", "turn": turn_idx, "text": current_counterparty})

    elapsed = time.time() - t0
    return {
        "item_id": item.id,
        "cell": item.cell,
        "subcell": item.subcell,
        "subject": subject_spec,
        "arm": arm,
        "counterparty_spec": counterparty_spec,
        "turns": turns,
        "n_agent_turns": sum(1 for t in turns if t["speaker"] == "agent"),
        "early_end_reason": early_end_reason,
        "elapsed_s": round(elapsed, 2),
    }


def run_grid(
    items: Iterable[Item],
    subjects: list[str],
    arms: list[str],
    out_path: str | Path,
    counterparty_spec: str = "claude-sonnet",
    counterparty_vendor: Vendor | None = None,
    resume: bool = True,
    parallel: int = 1,
) -> None:
    items = list(items)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str, str]] = set()
    if resume and out.exists():
        with out.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_keys.add((row["subject"], row["arm"], row["item_id"]))
                except Exception:
                    pass

    total = len(items) * len(subjects) * len(arms)
    tasks: list[tuple[str, str, Item]] = []
    for item in items:
        for subject in subjects:
            for arm in arms:
                if (subject, arm, item.id) in done_keys:
                    continue
                tasks.append((subject, arm, item))

    print(f"running {len(tasks)}/{total} trajectories (cached skip={total - len(tasks)}) parallel={parallel}")
    if not tasks:
        return

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def _work(subject: str, arm: str, item: Item) -> dict:
        return run_trajectory(
            subject_spec=subject,
            arm=arm,
            item=item,
            counterparty_spec=counterparty_spec,
            # fresh counterparty vendor per worker to avoid shared state
            counterparty_vendor=None,
        )

    written = 0
    with out.open("a") as f:
        if parallel <= 1:
            for subject, arm, item in tasks:
                written += 1
                print(f"[{written}/{len(tasks)}] run subject={subject} arm={arm} item={item.id}")
                traj = _work(subject, arm, item)
                f.write(json.dumps(traj) + "\n")
                f.flush()
        else:
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futs = {ex.submit(_work, s, a, i): (s, a, i) for (s, a, i) in tasks}
                for fut in as_completed(futs):
                    s, a, i = futs[fut]
                    try:
                        traj = fut.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"FAIL subject={s} arm={a} item={i.id}: {type(e).__name__}: {e}")
                        traj = {
                            "item_id": i.id, "cell": i.cell, "subcell": i.subcell,
                            "subject": s, "arm": a, "counterparty_spec": counterparty_spec,
                            "turns": [], "n_agent_turns": 0,
                            "early_end_reason": f"error:{type(e).__name__}:{str(e)[:200]}",
                            "elapsed_s": 0.0,
                        }
                    with lock:
                        written += 1
                        print(f"[{written}/{len(tasks)}] done subject={s} arm={a} item={i.id} ({traj.get('elapsed_s',0):.1f}s)")
                        f.write(json.dumps(traj) + "\n")
                        f.flush()
