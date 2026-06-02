"""Score the on-policy generation trajectories and route SFT points into
per-axis pools (leak / MI / bound / mixed).

For each (item_id) in data/onpolicy_iter1_trajectories.jsonl, run the same
probe + judge pipeline used by the headline scorer. Then for each SFT point
in data/onpolicy_iter1_sft.jsonl, attribute it to the failure axes of its
parent trajectory.

Output files (in data/):
  onpolicy_iter1_sft_leak.jsonl    — only points from leak-failing trajectories
  onpolicy_iter1_sft_mi.jsonl      — only points from MI-failing trajectories
  onpolicy_iter1_sft_bound.jsonl   — only points from bound-failing trajectories
  onpolicy_iter1_sft_mixed.jsonl   — all points (= existing onpolicy_iter1_sft.jsonl, copy)

Side effect: writes data/onpolicy_iter1_datagen_scored.jsonl with the per-trajectory
scoring of the data-generation trajectories.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.items import load_items, Item  # noqa: E402
from src.scorer import score_trajectory  # noqa: E402
from src.vendors import get_vendor  # noqa: E402


TRAJS = Path("data/onpolicy_iter1_trajectories.jsonl")
SFT = Path("data/onpolicy_iter1_sft.jsonl")
SCORED_OUT = Path("data/onpolicy_iter1_datagen_scored.jsonl")


def main() -> int:
    if not TRAJS.exists():
        print(f"[route] missing {TRAJS}"); return 1
    if not SFT.exists():
        print(f"[route] missing {SFT}"); return 1

    items = {it.id: it for it in load_items("items/v0")}
    trajectories = [json.loads(l) for l in TRAJS.read_text().splitlines() if l.strip()]
    print(f"[route] {len(trajectories)} trajectories, {len(items)} items")

    # Skip already-scored trajectories if SCORED_OUT exists
    already = {}
    if SCORED_OUT.exists():
        for l in SCORED_OUT.read_text().splitlines():
            r = json.loads(l)
            already[(r["item_id"], r.get("subject", "qwen-8b-local"), r.get("arm", "plain"))] = r

    probe = get_vendor("gpt-5-mini")
    judge = get_vendor("gpt-5-mini")

    lock = Lock()
    scored: list[dict] = []
    t_start = time.time()

    def _score_one(t):
        item = items.get(t["item_id"])
        if not item:
            return None
        key = (t["item_id"], t.get("subject", "qwen-8b-local"), t.get("arm", "plain"))
        if key in already:
            return already[key]
        sc = score_trajectory(t, item, probe, judge)
        # convert dataclass -> dict for JSON
        from dataclasses import asdict
        return asdict(sc)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_score_one, t) for t in trajectories]
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is None:
                    continue
                with lock:
                    scored.append(r)
                    if len(scored) % 10 == 0:
                        print(f"  [{len(scored)}/{len(trajectories)}]")
            except Exception as e:
                print(f"  FAIL: {type(e).__name__}: {e}")

    SCORED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORED_OUT, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    print(f"[route] scored {len(scored)} in {time.time()-t_start:.0f}s → {SCORED_OUT}")

    # Build axis-failure index per item_id
    by_item: dict[str, dict] = {}
    for r in scored:
        by_item[r["item_id"]] = {
            "leak": (r.get("leak_rate") or 0) > 0,
            "mi": bool((r.get("harm") or {}).get("missed_instruction")),
            "bound": bool((r.get("harm") or {}).get("leaked_private_bound")),
            "any_harm": bool((r.get("harm") or {}).get("harm_fire")),
        }

    # Stats
    n_leak = sum(1 for v in by_item.values() if v["leak"])
    n_mi = sum(1 for v in by_item.values() if v["mi"])
    n_bound = sum(1 for v in by_item.values() if v["bound"])
    n_any = sum(1 for v in by_item.values() if v["any_harm"])
    print(f"[route] failure modes per trajectory (n={len(by_item)}):")
    print(f"  leak={n_leak}  MI={n_mi}  bound={n_bound}  any_harm={n_any}")

    # Route SFT points
    sft_points = [json.loads(l) for l in SFT.read_text().splitlines() if l.strip()]
    print(f"[route] {len(sft_points)} SFT points to route")

    pools = {"leak": [], "mi": [], "bound": [], "mixed": []}
    for p in sft_points:
        iid = p["item_id"]
        v = by_item.get(iid)
        pools["mixed"].append(p)
        if v is None:
            continue
        if v["leak"]:
            pools["leak"].append(p)
        if v["mi"]:
            pools["mi"].append(p)
        if v["bound"]:
            pools["bound"].append(p)

    for name, pts in pools.items():
        out = Path(f"data/onpolicy_iter1_sft_{name}.jsonl")
        with open(out, "w") as f:
            for p in pts:
                f.write(json.dumps(p) + "\n")
        print(f"  pool '{name}': {len(pts)} points → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
