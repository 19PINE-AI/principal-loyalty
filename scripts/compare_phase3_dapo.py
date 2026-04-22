"""Compare Phase 2.1 (SFT+DPO-v4.1) vs Phase 3 (DAPO-v1 step 35 on top of v4.1).

Does GRPO/DAPO with MI-aware reward ratchet policy beyond what DPO achieved,
while preserving leak=0? Or does it regress one side of the frontier?
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


RUNS = [
    ("v4.1",      "runs/phase2_trained_v4_1/scored.jsonl"),
    ("dapo-s30",  "runs/phase3_dapo_v1_step30/scored.jsonl"),
    ("dapo-s35",  "runs/phase3_dapo_v1_step35/scored.jsonl"),
]
ARMS = ["plain", "prompted", "scaffolded"]


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def harm_fire(h: dict) -> bool:
    if "harm_fire" in h and isinstance(h["harm_fire"], bool):
        return h["harm_fire"]
    return any(bool(h.get(k, False)) for k in (
        "fabrication", "deception", "leaked_private_bound",
        "missed_instruction", "third_party_harm",
    ))


def summ(rows: list[dict]) -> dict:
    per: dict[str, dict] = {}
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            per[a] = {"n": 0}
            continue
        n = len(rs)
        leaks = [r.get("leak_rate", 0.0) for r in rs]
        holds = [r.get("commitment_hold_rate", 0.0) for r in rs]
        mi = sum(1 for r in rs if (r.get("harm") or {}).get("missed_instruction"))
        hf = sum(1 for r in rs if harm_fire(r.get("harm") or {}))
        bound = sum(1 for r in rs if (r.get("harm") or {}).get("leaked_private_bound"))
        pi = sum(1 for r in rs if (r.get("harm") or {}).get("principal_instructed_misrepresentation"))
        per[a] = {
            "n": n,
            "leak": mean(leaks),
            "hold": mean(holds),
            "mi": mi,
            "mi_rate": mi/n,
            "harm": hf,
            "harm_rate": hf/n,
            "bound": bound,
            "pi": pi,
        }
    return per


def probe_cut(rows: list[dict]) -> dict:
    probe = [r for r in rows if "to-principal" in r["item_id"]]
    nonprobe = [r for r in rows if "to-principal" not in r["item_id"]]
    return {"probe": summ(probe), "nonprobe": summ(nonprobe)}


def cell_cut(rows: list[dict]) -> dict:
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("cell") or r["item_id"].split("-")[1]
        by_cell.setdefault(c, []).append(r)
    return {c: summ(rs) for c, rs in by_cell.items()}


def fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x*100:5.1f}%"


def item_diff(a: list[dict], b: list[dict]) -> None:
    """Print items that changed harm status between a -> b."""
    def key(r):
        return (r["item_id"], r["arm"])
    map_a = {key(r): r for r in a}
    map_b = {key(r): r for r in b}
    shared = sorted(set(map_a) & set(map_b))

    regressions = []
    improvements = []
    for k in shared:
        ra, rb = map_a[k], map_b[k]
        ha = harm_fire(ra.get("harm") or {})
        hb = harm_fire(rb.get("harm") or {})
        if ha and not hb:
            improvements.append((k, ra.get("harm") or {}, rb.get("harm") or {}))
        elif hb and not ha:
            regressions.append((k, ra.get("harm") or {}, rb.get("harm") or {}))

    def fmt_flags(h: dict) -> str:
        flags = [k for k in ("fabrication","deception","leaked_private_bound","missed_instruction","third_party_harm") if h.get(k)]
        return ",".join(flags) or "-"

    print(f"\nITEM-LEVEL DELTA (v4.1 -> dapo-v1)  improvements={len(improvements)}  regressions={len(regressions)}")
    if improvements:
        print("  improvements (v4.1 fired -> dapo-v1 clean):")
        for (iid, arm), ha, hb in improvements:
            print(f"    + {iid:<32} {arm:<10}  was=[{fmt_flags(ha)}]")
    if regressions:
        print("  regressions (v4.1 clean -> dapo-v1 fired):")
        for (iid, arm), ha, hb in regressions:
            print(f"    - {iid:<32} {arm:<10}  now=[{fmt_flags(hb)}]")


def main() -> None:
    data = {}
    raw = {}
    for label, path in RUNS:
        rows = load(path)
        if not rows:
            print(f"[skip] {label}: no rows at {path}")
            continue
        data[label] = {"all": summ(rows), **probe_cut(rows), "cell": cell_cut(rows)}
        raw[label] = rows
        print(f"{label}: loaded {len(rows)} rows")

    print()
    print("=" * 96)
    print("LEAK_RATE (lower=better) — mean per arm")
    print("=" * 96)
    print(f"{'version':<10} | {'plain':<10} | {'prompted':<10} | {'scaffolded':<10}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        print(f"{label:<10} | {fmt_pct(s['plain']['leak']):<10} | {fmt_pct(s['prompted']['leak']):<10} | {fmt_pct(s['scaffolded']['leak']):<10}")

    print()
    print("=" * 96)
    print("MISSED_INSTRUCTION count / n (lower=better)")
    print("=" * 96)
    print(f"{'version':<10} | {'plain':<14} | {'prompted':<14} | {'scaffolded':<14}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        row = [f"{s[a]['mi']}/{s[a]['n']} ({fmt_pct(s[a]['mi_rate']).strip()})" for a in ARMS]
        print(f"{label:<10} | {row[0]:<14} | {row[1]:<14} | {row[2]:<14}")

    print()
    print("=" * 96)
    print("HARM_FIRE count / n (lower=better)")
    print("=" * 96)
    print(f"{'version':<10} | {'plain':<14} | {'prompted':<14} | {'scaffolded':<14}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        row = [f"{s[a]['harm']}/{s[a]['n']} ({fmt_pct(s[a]['harm_rate']).strip()})" for a in ARMS]
        print(f"{label:<10} | {row[0]:<14} | {row[1]:<14} | {row[2]:<14}")

    print()
    print("=" * 96)
    print("PROBE-ITEM CUT (reader-is-principal) — MI count / n")
    print("=" * 96)
    print(f"{'version':<10} | {'probe plain':<14} | {'probe prompt':<14} | {'probe scaffo':<14} | {'nonprobe MI (all arms)':<28}")
    for label in [l for l, _ in RUNS if l in data]:
        p = data[label]["probe"]
        np_ = data[label]["nonprobe"]
        probe_row = [f"{p[a]['mi']}/{p[a]['n']}" if p[a]['n'] else "-/0" for a in ARMS]
        np_total_mi = sum(np_[a].get("mi", 0) for a in ARMS)
        np_total_n = sum(np_[a].get("n", 0) for a in ARMS)
        print(f"{label:<10} | {probe_row[0]:<14} | {probe_row[1]:<14} | {probe_row[2]:<14} | {np_total_mi}/{np_total_n}")

    print()
    print("=" * 96)
    print("LEAKED_PRIVATE_BOUND + PRINCIPAL_INSTRUCTED_MISREP (other harm flags)")
    print("=" * 96)
    print(f"{'version':<10} | {'bound (all)':<14} | {'pi_misrep (all)':<16}")
    for label in [l for l, _ in RUNS if l in data]:
        s = data[label]["all"]
        total_b = sum(s[a].get("bound", 0) for a in ARMS)
        total_pi = sum(s[a].get("pi", 0) for a in ARMS)
        print(f"{label:<10} | {total_b:<14} | {total_pi:<16}")

    print()
    print("=" * 96)
    print("PER-CELL HARM_FIRE count / n")
    print("=" * 96)
    cells = sorted({c for label in data for c in data[label]["cell"]})
    header = f"{'cell':<12} | " + " | ".join(f"{l:<20}" for l, _ in RUNS if l in data)
    print(header)
    for c in cells:
        parts = []
        for label in [l for l, _ in RUNS if l in data]:
            s = data[label]["cell"].get(c, {})
            hf = sum(s.get(a, {}).get("harm", 0) for a in ARMS)
            n = sum(s.get(a, {}).get("n", 0) for a in ARMS)
            parts.append(f"{hf}/{n}")
        print(f"{c:<12} | " + " | ".join(f"{p:<20}" for p in parts))

    if "v4.1" in raw and "dapo-s35" in raw:
        print("\n### v4.1 -> dapo-s35")
        item_diff(raw["v4.1"], raw["dapo-s35"])
    if "v4.1" in raw and "dapo-s30" in raw:
        print("\n### v4.1 -> dapo-s30")
        item_diff(raw["v4.1"], raw["dapo-s30"])
    if "dapo-s30" in raw and "dapo-s35" in raw:
        print("\n### dapo-s30 -> dapo-s35")
        item_diff(raw["dapo-s30"], raw["dapo-s35"])


if __name__ == "__main__":
    main()
