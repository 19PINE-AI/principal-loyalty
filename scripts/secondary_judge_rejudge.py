"""Full secondary-judge (claude-haiku) re-judge of the Mechanism-2 headline.

Re-scores every trajectory of the SFT+DPO base and the per-token-KL iter1
student, across all 5 evaluation seeds, with the SECONDARY judge, using the
exact same score_harm() composition as the primary pipeline
(harm_fire = any of fabrication/deception/leaked_private_bound/
missed_instruction/third_party_harm). This lets us re-run the paired
Wilcoxon under a different judge and check the gain direction is stable.

Resumable: appends one JSON record per (run_dir, item_id, arm) to the output
and skips already-done keys on restart. Concurrent via a thread pool.

Usage:
    python3 scripts/secondary_judge_rejudge.py            # full run
    python3 scripts/secondary_judge_rejudge.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.items import load_items          # noqa: E402
from src.scorer import score_harm          # noqa: E402
from src.vendors import get_vendor          # noqa: E402

BASE_DIRS = ["phase2_trained_v4_1"] + [f"phase2_trained_v4_1_seed{i}" for i in range(2, 6)]
KL_DIRS = ["phase5_pertoken_kl_iter1"] + [f"phase5_pertoken_kl_iter1_seed{i}" for i in range(2, 6)]
ALL_DIRS = BASE_DIRS + KL_DIRS

OUT = ROOT / "runs" / "secondary_judge_rejudge.jsonl"


def load_jsonl(p):
    out = []
    if Path(p).exists():
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge", default="claude-haiku")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_path = Path(args.out)

    items = {getattr(i, "id"): i for i in load_items("items/v0")}
    judge = get_vendor(args.judge)

    done = {(r["run_dir"], r["item_id"], r["arm"]) for r in load_jsonl(out_path)}
    print(f"[rejudge] {len(done)} cells already done; judge={judge.model}")

    work = []
    for d in ALL_DIRS:
        tp = ROOT / "runs" / d / "trajectories.jsonl"
        for tr in load_jsonl(tp):
            key = (d, tr["item_id"], tr["arm"])
            if key in done:
                continue
            if tr["item_id"] not in items:
                continue  # not in the released item set
            work.append((d, tr))
    print(f"[rejudge] {len(work)} cells to judge across {len(ALL_DIRS)} dirs")

    lock = threading.Lock()
    fout = open(out_path, "a")
    n_done = [0]
    n_err = [0]

    def task(d, tr):
        item = items[tr["item_id"]]
        try:
            flags = score_harm(tr, item, judge)
            rec = {
                "run_dir": d, "item_id": tr["item_id"], "arm": tr["arm"],
                "harm_fire": bool(flags["harm_fire"]),
                "fabrication": bool(flags["fabrication"]),
                "deception": bool(flags["deception"]),
                "leaked_private_bound": bool(flags["leaked_private_bound"]),
                "missed_instruction": bool(flags["missed_instruction"]),
                "third_party_harm": bool(flags["third_party_harm"]),
            }
        except Exception as e:  # transient API failure -> record, retry next run
            rec = {"run_dir": d, "item_id": tr["item_id"], "arm": tr["arm"],
                   "error": str(e)[:160]}
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(task, d, tr) for d, tr in work]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                if "error" in rec:
                    n_err[0] += 1
                else:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    n_done[0] += 1
                    if n_done[0] % 50 == 0:
                        print(f"[rejudge] {n_done[0]} done, {n_err[0]} errors")
    fout.close()
    print(f"[rejudge] FINISHED: {n_done[0]} new cells written, {n_err[0]} errors -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
