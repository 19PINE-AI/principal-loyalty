"""Item-level harm breakdown for DAPO-v1 step_35.

Answers:
  1. Which items × arms dominate the 25/108 residual harm?
  2. Which items are clean across all 3 arms (3-arm zero) — these are the
     ones the model has truly mastered.
  3. Which items are dirty across multiple arms — potential targets for
     data expansion or v3-style targeted reward.
  4. What are the harm-firing signatures? (fabrication / deception /
     bound-leak / missed-instruction / third-party / pi-misrep)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


SCORED = "runs/phase3_dapo_v1_step35/scored.jsonl"
ARMS = ["plain", "prompted", "scaffolded"]
SUB_FLAGS = [
    "fabrication", "deception", "leaked_private_bound",
    "missed_instruction", "third_party_harm",
    "principal_instructed_misrepresentation",
]


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).open()]


def harm_fire(h: dict) -> bool:
    if "harm_fire" in h and isinstance(h["harm_fire"], bool):
        return h["harm_fire"]
    return any(bool(h.get(k, False)) for k in (
        "fabrication", "deception", "leaked_private_bound",
        "missed_instruction", "third_party_harm",
    ))


def main() -> None:
    rows = load(SCORED)
    print(f"loaded {len(rows)} rows from {SCORED}\n")

    # Build (item_id, arm) -> harm dict
    by_item_arm: dict[tuple[str, str], dict] = {}
    for r in rows:
        by_item_arm[(r["item_id"], r["arm"])] = r

    items = sorted({r["item_id"] for r in rows})

    # Per-item: how many arms fire harm
    item_fire_count: dict[str, int] = defaultdict(int)
    item_arm_flags: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for item in items:
        for arm in ARMS:
            r = by_item_arm.get((item, arm))
            if not r:
                continue
            h = r.get("harm") or {}
            if harm_fire(h):
                item_fire_count[item] += 1
                flags = [f for f in SUB_FLAGS if h.get(f)]
                item_arm_flags[item].append((arm, flags))

    # Bucket items by arm-fire count
    by_count: dict[int, list[str]] = defaultdict(list)
    for item in items:
        by_count[item_fire_count[item]].append(item)

    print("=" * 90)
    print("ITEM-LEVEL HARM CONCENTRATION on dapo-v1 step_35")
    print("=" * 90)
    print(f"\nItems with 3-arm clean (all arms harm-free): {len(by_count[0])} / {len(items)}")
    if by_count[0]:
        print("  " + ", ".join(by_count[0]))

    for n in [3, 2, 1]:
        print(f"\nItems firing on {n} of 3 arms ({len(by_count[n])} items):")
        for item in by_count[n]:
            cell = next(r for r in rows if r["item_id"] == item).get("cell", "?")
            print(f"  [{cell:<13}] {item}")
            for arm, flags in item_arm_flags[item]:
                print(f"      {arm:<10} {','.join(flags)}")

    print("\n" + "=" * 90)
    print("PER-CELL CONCENTRATION — harm fires per cell, normalized by n_rollouts")
    print("=" * 90)
    cell_harm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cell_harm[r["cell"]].append(r)
    for cell, rs in sorted(cell_harm.items()):
        n = len(rs)
        fire = sum(1 for r in rs if harm_fire(r.get("harm") or {}))
        # Most common flag in this cell
        flag_counts: dict[str, int] = defaultdict(int)
        for r in rs:
            for f in SUB_FLAGS:
                if (r.get("harm") or {}).get(f):
                    flag_counts[f] += 1
        top_flags = sorted(flag_counts.items(), key=lambda x: -x[1])[:3]
        flag_str = ", ".join(f"{f}={c}" for f, c in top_flags) or "—"
        print(f"  {cell:<13} {fire:>2}/{n:<3}  ({fire/n*100:>5.1f}%)   top: {flag_str}")

    print("\n" + "=" * 90)
    print("FLAG MIX across full benchmark — what KIND of harm dominates?")
    print("=" * 90)
    total_flag: dict[str, int] = defaultdict(int)
    for r in rows:
        h = r.get("harm") or {}
        for f in SUB_FLAGS:
            if h.get(f):
                total_flag[f] += 1
    fired = sum(1 for r in rows if harm_fire(r.get("harm") or {}))
    print(f"\nharm_fire total: {fired}/{len(rows)}")
    for f, c in sorted(total_flag.items(), key=lambda x: -x[1]):
        print(f"  {f:<40} {c:>3}")

    # Where does data expansion help?
    print("\n" + "=" * 90)
    print("DATA-EXPANSION TARGETS — items firing on >=2 arms")
    print("=" * 90)
    high_value = [it for it, n in item_fire_count.items() if n >= 2]
    print(f"\n{len(high_value)} items account for {sum(item_fire_count[it] for it in high_value)} "
          f"of {fired} total fires ({sum(item_fire_count[it] for it in high_value)/fired*100:.0f}%).")
    print("These items, expanded with paired teacher-good / student-bad examples, would")
    print("hit the highest leverage. Sorted by cell + arms-fired:")
    for it in sorted(high_value, key=lambda x: (-item_fire_count[x], x)):
        cell = next(r for r in rows if r["item_id"] == it).get("cell", "?")
        n = item_fire_count[it]
        flags_per_arm = "; ".join(f"{a}: {','.join(fs) or 'clean'}"
                                  for a, fs in item_arm_flags[it])
        print(f"  [{cell:<13}] {it:<32} fires={n}/3   {flags_per_arm}")


if __name__ == "__main__":
    main()
