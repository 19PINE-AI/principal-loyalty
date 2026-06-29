#!/usr/bin/env python3
"""
Extract paper data into compact JSON files for the static React site.

Emits files into website/public/data/:
  - items.json                 — all 75+ test items, lightly normalized
  - items_held.json            — held-out subset (25 items)
  - subjects.json              — 13 frontier subjects with cluster + harm
  - subject_arms.json          — per-subject per-arm aggregates
  - subject_held.json          — held-out cross-subject grid
  - variants.json              — distillation variant ladder
  - kiter.json                 — K-iteration (Qwen + Llama)
  - cells.json                 — six failure cell definitions
  - manifold.json              — leak vs MI scatter points
  - trajectories/<file>.json   — curated sample conversations (one per cell)
"""

from __future__ import annotations
import json, os, re, sys
from pathlib import Path

ROOT = Path("/home/ubuntu/principal-loyalty")
OUT  = ROOT / "website" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "trajectories").mkdir(exist_ok=True)


# ---------- helpers ----------
def jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for l in f:
            if not l.strip():
                continue
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                continue  # skip truncated/partial lines
    return out


def load_items(dirname: str):
    p = ROOT / "items" / dirname
    if not p.exists():
        return []
    out = []
    for f in sorted(p.glob("*.json")):
        d = json.loads(f.read_text())
        out.append({
            "id": d["id"],
            "cell": d.get("cell", "unknown"),
            "subcell": d.get("subcell", ""),
            "title": d.get("title", d["id"]),
            "principal_briefing": d.get("principal_briefing", ""),
            "private_facts": [
                {"name": pf.get("name", ""), "value": pf.get("value", ""),
                 "sharing": pf.get("sharing", ""), "category": pf.get("category", ""),
                 "aliases": pf.get("aliases", [])[:5]}
                for pf in d.get("private_facts", [])
            ],
            "stated_positions": d.get("stated_positions", []),
            "private_bounds": d.get("private_bounds", []),
            "counterparty_persona": d.get("counterparty_persona", ""),
            "counterparty_strategy": d.get("counterparty_strategy", ""),
            "counterparty_opening": d.get("counterparty_opening", ""),
            "max_turns": d.get("max_turns", 8),
            "posture_pressure": d.get("posture_pressure", False),
        })
    return out


def aggregate(rows):
    """Aggregate scored rows into {n, harm, leak, bound, MI} fires."""
    n = len(rows)
    harm = leak = bound = mi = 0
    for r in rows:
        if bool((r.get("harm") or {}).get("harm_fire")):
            harm += 1
        if (r.get("leak_rate") or 0) > 0:
            leak += 1
        if bool((r.get("harm") or {}).get("leaked_private_bound")):
            bound += 1
        if bool((r.get("harm") or {}).get("missed_instruction")):
            mi += 1
    return {"n": n, "harm": harm, "leak": leak, "bound": bound, "mi": mi}


def per_arm(rows):
    arms = {}
    for r in rows:
        a = r.get("arm", "unknown")
        arms.setdefault(a, []).append(r)
    return {a: aggregate(rs) for a, rs in arms.items()}


def per_cell(rows):
    cells = {}
    for r in rows:
        c = r.get("cell", "unknown")
        cells.setdefault(c, []).append(r)
    return {c: aggregate(rs) for c, rs in cells.items()}


# ---------- items ----------
items_v0   = load_items("v0")     # 50 training items
items_held = load_items("v0_75")  # 25 held-out items

# Merge & tag for the items browser
for it in items_v0:   it["split"] = "training"
for it in items_held: it["split"] = "held-out"
all_items = items_v0 + items_held
(OUT / "items.json").write_text(json.dumps(all_items, indent=1))
(OUT / "items_held.json").write_text(json.dumps(items_held, indent=1))
print(f"items: {len(items_v0)} training + {len(items_held)} held-out = {len(all_items)} total")


# ---------- cells (taxonomy) ----------
cells = [
    {"id": "leakage",     "color": "leak",   "label": "Leakage",     "desc": "Direct revelation of a withheld fact in a chat turn."},
    {"id": "capitulation","color": "capit",  "label": "Capitulation","desc": "Conceding a stated position under social/temporal pressure."},
    {"id": "posture",     "color": "post",   "label": "Posture",     "desc": "Telegraphing motivation, urgency, or flexibility without naming a fact."},
    {"id": "authoring",   "color": "author", "label": "Authoring",   "desc": "Drafting an artifact containing the leak/concession."},
    {"id": "moderation",  "color": "moder",  "label": "Moderation",  "desc": "Mishandling third-party confidentiality (witness, patient, mediator)."},
    {"id": "sanity",      "color": "sanity", "label": "Sanity",      "desc": "Cooperative items where over-refusal is the only failure."},
]
(OUT / "cells.json").write_text(json.dumps(cells, indent=1))


# ---------- subjects (Table 1 / 2) ----------
# Frozen from the multi-seed n=5 aggregate (paper Table 2). Single-seed plain/
# prompted/scaffolded come from runs/phase4_promptv4_<subject>/scored.jsonl.
subjects_meta = [
    # (subject_key, display, cluster, run_dir, run_dir_heldout, plain, prompted, scaffolded, agg_mean, agg_sd)
    ("gemini25flash",       "Gemini-2.5-flash",         "calibrated",  "phase4_promptv4_gemini25flash",     None,                                       19, 14, 17,  5.5, 6.3),
    ("mistral_large",       "Mistral-Large",            "calibrated",  "phase4_promptv4_mistral_large",     None,                                        9,  6, 22, 11.0, 2.8),
    ("gemini3p1_lite",      "Gemini-3p1-flash-lite",    "calibrated",  "phase4_promptv4_gemini3p1_lite",    "phase4_promptv4_gemini3p1_lite_heldout",   17, 17,  0, 12.0, 2.1),
    ("deepseek",            "DeepSeek-v3.1",            "calibrated",  "phase4_promptv4_deepseek",          "phase4_promptv4_deepseek_heldout",         11,  9, 11, 12.3, 2.5),
    ("qwen32b",             "Qwen3-32B",                "calibrated",  "phase4_promptv4_qwen32b_openrouter","phase4_promptv4_qwen32b_openrouter_heldout",25, 12, 14, 16.5, 2.6),
    ("claude_opus",         "Claude-Opus",              "calibrated",  "phase4_promptv4_claude_opus",       "phase4_promptv4_claude_opus_heldout",      11, 19, 19, 18.1, 2.8),
    ("llama70b",            "Llama-3.1-70B",            "calibrated",  "phase4_promptv4_llama70b",          "phase4_promptv4_llama70b_heldout",         11, 21, 17, 19.2, 2.7),
    ("gemini3flash",        "Gemini-3-flash",           "calibrated",  "phase4_promptv4_gemini3flash",      "phase4_promptv4_gemini3flash_heldout",     17, 20, 15, 19.4, 2.3),
    ("claude_sonnet",       "Claude-Sonnet",            "calibrated",  "phase4_promptv4_claude_seed2",      "phase4_promptv4_claude_heldout",           19, 22, 17, 19.5, 1.5),
    ("glm46",               "GLM-4.6",                  "intermediate","phase4_promptv4_glm46",             "phase4_promptv4_glm46_heldout",            43, 61, 43, 46.0, 2.9),
    ("gpt5mini",            "GPT-5-mini",               "over-refuse", "phase4_promptv4_gpt5mini",          "phase4_promptv4_gpt5mini_heldout",         44, 76, 65, 53.6, 5.1),
    ("gpt5",                "GPT-5",                    "over-refuse", "phase4_promptv4_gpt5",              "phase4_promptv4_gpt5_heldout",             63, 71, 71, 71.1, 2.2),
    ("qwen35_27b",          "Qwen3.5-27B",              "over-refuse", "phase4_promptv4_qwen27b",           "phase4_promptv4_qwen27b_heldout",          72, 79, 59, 75.3, 2.9),
]

subjects = []
subject_arms = {}
subject_held = []

for key, disp, cluster, run, run_h, plain, prom, scaf, mean, sd in subjects_meta:
    sub = {
        "key": key, "display": disp, "cluster": cluster,
        "plain": plain, "prompted": prom, "scaffolded": scaf,
        "mean": mean, "sd": sd,
    }
    subjects.append(sub)

    if run:
        rows = jsonl(ROOT / "runs" / run / "scored.jsonl")
        if rows:
            subject_arms[key] = {
                "display": disp, "cluster": cluster,
                "agg": aggregate(rows),
                "per_arm": per_arm(rows),
                "per_cell": per_cell(rows),
            }

    if run_h:
        rows = jsonl(ROOT / "runs" / run_h / "scored.jsonl")
        if rows:
            agg = aggregate(rows)
            subject_held.append({
                "key": key, "display": disp, "cluster": cluster,
                "n": agg["n"], "harm": agg["harm"], "leak": agg["leak"],
                "bound": agg["bound"], "mi": agg["mi"],
                "harm_pct": round(100.0 * agg["harm"] / max(agg["n"], 1), 1),
            })

(OUT / "subjects.json").write_text(json.dumps(subjects, indent=1))
(OUT / "subject_arms.json").write_text(json.dumps(subject_arms, indent=1))
(OUT / "subject_held.json").write_text(json.dumps(subject_held, indent=1))
print(f"subjects: {len(subjects)} | per-arm runs: {len(subject_arms)} | held-out runs: {len(subject_held)}")


# ---------- variants ladder (paper Fig 4) ----------
variants = [
    {"name": "v4.1 base",      "harm": 56, "color": "#94a3b8", "sig": "",        "kind": "base"},
    {"name": "Per-turn DPO",   "harm": 54, "color": "#fbbf24", "sig": "n.s.",    "kind": "variant"},
    {"name": "Per-turn SFT i1","harm": 44, "color": "#84cc16", "sig": "p=.10",   "kind": "variant"},
    {"name": "Per-turn SFT i2","harm": 36, "color": "#84cc16", "sig": "p=.10",   "kind": "variant"},
    {"name": "Per-token KL i1","harm": 33, "color": "#7c3aed", "sig": "p=.011*", "kind": "mechanism"},
    {"name": "Per-token KL i2","harm": 38, "color": "#7c3aed", "sig": "p=.044*", "kind": "mechanism"},
    {"name": "Claude + scaffold","harm":21,"color": "#0891b2", "sig": "",        "kind": "mechanism"},
]
(OUT / "variants.json").write_text(json.dumps(variants, indent=1))


# ---------- K-iteration (paper Table on §5.3) ----------
kiter = {
    "qwen": [
        {"iter": 1, "harm": 33, "leak": 13, "bound": 3, "mi": 32},
        {"iter": 2, "harm": 38, "leak":  9, "bound": 2, "mi": 35},
        {"iter": 3, "harm": 41, "leak": 15, "bound": 4, "mi": 40},
        {"iter": 4, "harm": 42, "leak": 17, "bound": 5, "mi": 42},
        {"iter": 5, "harm": 32, "leak": 19, "bound": 6, "mi": 32},
    ],
    "llama": [
        {"iter": 1, "harm": 27, "leak": 3, "bound": 2, "mi": 25},
        {"iter": 2, "harm": 22, "leak": 9, "bound": 2, "mi": 20},
        {"iter": 3, "harm": 17, "leak": 7, "bound": 3, "mi": 15},
        {"iter": 4, "harm": 18, "leak": 6, "bound": 2, "mi": 17},
    ],
}
(OUT / "kiter.json").write_text(json.dumps(kiter, indent=1))


# ---------- manifold scatter (leak vs MI on 108-cell aggregate) ----------
# Curated set of operating points across mechanisms and variants.
manifold = [
    {"label": "SFT+DPO base",        "leak": 16, "mi": 44, "harm": 56, "kind": "base"},
    {"label": "Per-turn SFT i2",     "leak": 13, "mi": 30, "harm": 36, "kind": "variant"},
    {"label": "Per-turn DPO",        "leak": 14, "mi": 42, "harm": 54, "kind": "variant"},
    {"label": "Per-token KL i1",     "leak": 13, "mi": 32, "harm": 33, "kind": "mechanism"},
    {"label": "Per-token KL i2",     "leak":  9, "mi": 35, "harm": 38, "kind": "mechanism"},
    {"label": "Per-token KL i3",     "leak": 15, "mi": 40, "harm": 41, "kind": "mechanism"},
    {"label": "Per-token KL i4",     "leak": 17, "mi": 42, "harm": 42, "kind": "mechanism"},
    {"label": "Per-token KL i5",     "leak": 19, "mi": 32, "harm": 32, "kind": "mechanism"},
    {"label": "DAPO from KL i1",     "leak": 18, "mi": 45, "harm": 46, "kind": "rl"},
    {"label": "Claude + scaffold",   "leak": 17, "mi": 21, "harm": 21, "kind": "scaffold"},
    {"label": "Llama KL i3",         "leak":  7, "mi": 15, "harm": 17, "kind": "mechanism"},
]
(OUT / "manifold.json").write_text(json.dumps(manifold, indent=1))


# ---------- curated sample trajectories ----------
# One representative trajectory per cell, from a calibrated subject under the
# prompted arm. We pick the first item alphabetically in each cell.
def load_trajectory(run_dir: str, item_id: str, arm: str = "prompted"):
    p = ROOT / "runs" / run_dir / "trajectories.jsonl"
    if not p.exists():
        return None
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("item_id") == item_id and d.get("arm") == arm:
                return d
    return None


sample_runs = [
    ("phase4_promptv4_claude_opus", "claude-opus"),
    ("phase4_promptv4_deepseek",    "deepseek-v3.1"),
    ("phase4_promptv4_gpt5",        "gpt-5"),
]

traj_index = []
items_by_cell = {}
for it in items_v0:
    items_by_cell.setdefault(it["cell"], []).append(it)

for cell, its in items_by_cell.items():
    its.sort(key=lambda x: x["id"])
    item = its[0]
    for run_dir, sub_disp in sample_runs:
        for arm in ("plain", "prompted", "scaffolded"):
            t = load_trajectory(run_dir, item["id"], arm)
            if t:
                # also fetch the score
                score = None
                for r in jsonl(ROOT / "runs" / run_dir / "scored.jsonl"):
                    if r.get("item_id") == item["id"] and r.get("arm") == arm:
                        score = r
                        break
                fname = f"{cell}_{sub_disp}_{arm}.json"
                out_doc = {
                    "item_id": item["id"],
                    "item_title": item["title"],
                    "cell": cell,
                    "subject": sub_disp,
                    "arm": arm,
                    "turns": t.get("turns", []),
                    "score": {
                        "harm_fire": bool((score or {}).get("harm", {}).get("harm_fire", False)),
                        "leak_rate": (score or {}).get("leak_rate", 0),
                        "leaked_private_bound": bool((score or {}).get("harm", {}).get("leaked_private_bound", False)),
                        "missed_instruction": bool((score or {}).get("harm", {}).get("missed_instruction", False)),
                        "notes": (score or {}).get("harm", {}).get("notes", ""),
                    } if score else None,
                }
                (OUT / "trajectories" / fname).write_text(json.dumps(out_doc, indent=1))
                traj_index.append({
                    "file": fname, "cell": cell, "subject": sub_disp,
                    "arm": arm, "item_id": item["id"], "item_title": item["title"],
                    "harm_fire": out_doc["score"]["harm_fire"] if out_doc["score"] else None,
                })

(OUT / "trajectories_index.json").write_text(json.dumps(traj_index, indent=1))
print(f"trajectories: {len(traj_index)} sample conversations")


# ---------- Wilcoxon (paper Figure 3) ----------
# Multi-seed paired Wilcoxon vs SFT+DPO base.
# Iter-1 from logs/pertoken_kl_paired_seed_test.log (n=5).
# Iter-2 from paper §5.3 (n=4 matched seeds; one seed dropped under GPU contention).
wilcoxon = {
    "iter1": {
        "n_seeds": 5,
        "metrics": [
            {"key": "harm",  "label": "Harm",  "base": 47.8, "kl": 39.2, "kl_sd": 4.0, "p": 0.0114, "robust_base": 15, "robust_kl": 6},
            {"key": "leak",  "label": "Leak",  "base": 15.8, "kl": 13.8, "kl_sd": 1.8, "p": 0.534,  "robust_base":  0, "robust_kl": 0},
            {"key": "bound", "label": "Bound", "base":  4.6, "kl":  2.8, "kl_sd": 1.5, "p": 0.385,  "robust_base":  0, "robust_kl": 0},
            {"key": "mi",    "label": "MI",    "base": 44.4, "kl": 37.2, "kl_sd": 3.6, "p": 0.055,  "robust_base": 15, "robust_kl": 5},
        ],
    },
    "iter2": {
        "n_seeds": 4,
        "metrics": [
            {"key": "harm",  "label": "Harm",  "base": 48.5, "kl": 41.5, "kl_sd": 3.6, "p": 0.0436},
        ],
    },
}
(OUT / "wilcoxon.json").write_text(json.dumps(wilcoxon, indent=1))


# ---------- Teacher self-validation (paper Figure 4) ----------
# Scaffolded arm, audit-gated.
teacher = {
    "metrics": [
        {"key": "harm", "label": "Harm",            "claude": 6,  "claude_n": 36, "qwen": 4,  "qwen_n": 31},
        {"key": "leak", "label": "Leak",            "claude": 6,  "claude_n": 36, "qwen": 21, "qwen_n": 31},
        {"key": "mi",   "label": "Missed-instruct", "claude": 6,  "claude_n": 36, "qwen": 3,  "qwen_n": 31},
    ],
    "subjects": {
        "claude":  {"display": "Claude-Sonnet + scaffold",     "color": "#0891b2"},
        "qwen":    {"display": "Qwen3-32B-AWQ + scaffold (open teacher)", "color": "#7c3aed"},
    },
}
(OUT / "teacher.json").write_text(json.dumps(teacher, indent=1))


# ---------- Counterparty robustness & held-out generalization (paper Figure 5) ----------
robustness = {
    "counterparty": [
        # PerTokenKL iter1 swept over three counterparty models
        {"counterparty": "Claude-Sonnet", "color": "#0891b2", "harm": 33, "leak": 13},
        {"counterparty": "GPT-5",         "color": "#10b981", "harm": 38, "leak": 14},
        {"counterparty": "Gemini-3-flash","color": "#f59e0b", "harm": 49, "leak": 20},
    ],
    "heldout": [
        # Training-set vs held-out harm for each recipe (% on the 36/25 item sets)
        {"recipe": "Per-token KL i1",  "color": "#7c3aed", "training": 30.6, "heldout": 40.3},
        {"recipe": "Per-turn SFT i2",  "color": "#84cc16", "training": 33.3, "heldout": 36.0},
        {"recipe": "Llama KL i3",      "color": "#0891b2", "training": 15.7, "heldout": 26.7},
    ],
}
(OUT / "robustness.json").write_text(json.dumps(robustness, indent=1))


# ---------- headline numbers (used in Overview hero) ----------
headline = {
    "claude_sonnet_scaffolded_harm": 21,
    "claude_sonnet_scaffolded_harm_pct": 19.4,
    "qwen8b_pertoken_kl_iter1_harm": 33,
    "qwen8b_pertoken_kl_iter1_leak": 13,
    "qwen8b_pertoken_kl_iter1_mi":   32,
    "llama8b_pertoken_kl_iter3_harm": 17,
    "frontier_subjects": 13,
    "items_total": 75,
    "items_training": 50,
    "items_heldout": 25,
    "calibrated_n": 9,
    "calibrated_range": "5.5–19.5%",
    "overrefuse_n": 3,
    "overrefuse_range": "53.6–75.3%",
    "wilcoxon_plain_p": "1.8e-6",
    "wilcoxon_prompted_p": "2.2e-7",
    "wilcoxon_scaffolded_p": "5.9e-7",
    "pertoken_kl_p": 0.0114,
}
(OUT / "headline.json").write_text(json.dumps(headline, indent=1))


# ---------- comprehensive per-item explorer (all subjects × arms) ----------
# For every benchmark item, gather each evaluated subject's full transcript and
# the judge's verdict under each prompt arm, so readers can inspect the raw
# agent responses and the evaluation behind every cell of the result matrix.
EXPLORER_SUBJECTS = [
    ("claude_sonnet",  "Claude-Sonnet",         "calibrated",   ["phase4_promptv4_claude_seed2",        "phase4_promptv4_claude_heldout"]),
    ("claude_opus",    "Claude-Opus",           "calibrated",   ["phase4_promptv4_claude_opus",         "phase4_promptv4_claude_opus_heldout"]),
    ("deepseek",       "DeepSeek-v3.1",         "calibrated",   ["phase4_promptv4_deepseek",            "phase4_promptv4_deepseek_heldout"]),
    ("gemini25flash",  "Gemini-2.5-flash",      "calibrated",   ["phase4_promptv4_gemini25flash"]),
    ("gemini3flash",   "Gemini-3-flash",        "calibrated",   ["phase4_promptv4_gemini3flash",        "phase4_promptv4_gemini3flash_heldout"]),
    ("gemini3p1_lite", "Gemini-3p1-flash-lite", "calibrated",   ["phase4_promptv4_gemini3p1_lite",      "phase4_promptv4_gemini3p1_lite_heldout"]),
    ("llama70b",       "Llama-3.1-70B",         "calibrated",   ["phase4_promptv4_llama70b",            "phase4_promptv4_llama70b_heldout"]),
    ("mistral_large",  "Mistral-Large",         "calibrated",   ["phase4_promptv4_mistral_large",       "phase4_promptv4_mistral_large_heldout"]),
    ("qwen32b",        "Qwen3-32B",             "calibrated",   ["phase4_promptv4_qwen32b_openrouter",  "phase4_promptv4_qwen32b_openrouter_heldout"]),
    ("glm46",          "GLM-4.6",               "intermediate", ["phase4_promptv4_glm46",               "phase4_promptv4_glm46_heldout"]),
    ("gpt5mini",       "GPT-5-mini",            "over-refuse",  ["phase4_promptv4_gpt5mini",            "phase4_promptv4_gpt5mini_heldout"]),
    ("gpt5",           "GPT-5",                 "over-refuse",  ["phase4_promptv4_gpt5",                "phase4_promptv4_gpt5_heldout"]),
    ("qwen35_27b",     "Qwen3.5-27B",           "over-refuse",  ["phase4_promptv4_qwen27b",             "phase4_promptv4_qwen27b_heldout"]),
]
ARMS = ["plain", "prompted", "scaffolded"]
ARM_ORDER = {a: i for i, a in enumerate(ARMS)}
CLUSTER_ORDER = {"calibrated": 0, "intermediate": 1, "over-refuse": 2}


def clean_turns(turns):
    out = []
    for t in turns:
        sp = t.get("speaker", "")
        if sp not in ("agent", "counterparty", "principal"):
            continue
        out.append({"speaker": sp, "text": (t.get("text") or "").strip()})
    return out


explorer = {}
for it in all_items:
    explorer[it["id"]] = {
        "id": it["id"], "cell": it["cell"], "subcell": it["subcell"],
        "title": it["title"], "split": it["split"],
        "principal_briefing": it["principal_briefing"],
        "private_facts": it["private_facts"],
        "private_bounds": it["private_bounds"],
        "stated_positions": it["stated_positions"],
        "counterparty_persona": it["counterparty_persona"],
        "counterparty_strategy": it["counterparty_strategy"],
        "counterparty_opening": it["counterparty_opening"],
        "runs": [],
    }

for key, disp, cluster, run_dirs in EXPLORER_SUBJECTS:
    for rd in run_dirs:
        scored = {}
        for r in jsonl(ROOT / "runs" / rd / "scored.jsonl"):
            scored[(r.get("item_id"), r.get("arm"))] = r
        for t in jsonl(ROOT / "runs" / rd / "trajectories.jsonl"):
            iid, arm = t.get("item_id"), t.get("arm")
            if iid not in explorer:
                continue
            s = scored.get((iid, arm)) or {}
            h = s.get("harm") or {}
            explorer[iid]["runs"].append({
                "subject": key, "display": disp, "cluster": cluster, "arm": arm,
                "scored": bool(s),
                "harm_fire": bool(h.get("harm_fire")),
                "leak_rate": s.get("leak_rate", 0) or 0,
                "n_leaked": s.get("n_leaked", 0) or 0,
                "n_withhold": s.get("n_withhold", 0) or 0,
                "leaked_private_bound": bool(h.get("leaked_private_bound")),
                "missed_instruction": bool(h.get("missed_instruction")),
                "fabrication": bool(h.get("fabrication")),
                "deception": bool(h.get("deception")),
                "against_principal": bool(h.get("against_principal")),
                "notes": h.get("notes", ""),
                "leaks": [
                    {"fact": lk.get("fact_name", ""), "leaked": bool(lk.get("leaked")),
                     "evidence": (lk.get("evidence") or "")[:400]}
                    for lk in (s.get("leaks") or [])
                ],
                "n_agent_turns": t.get("n_agent_turns", 0),
                "early_end_reason": t.get("early_end_reason", ""),
                "turns": clean_turns(t.get("turns", [])),
            })

(OUT / "explorer").mkdir(exist_ok=True)
explorer_index = []
for iid, doc in explorer.items():
    doc["runs"].sort(key=lambda r: (CLUSTER_ORDER.get(r["cluster"], 9), r["display"], ARM_ORDER.get(r["arm"], 9)))
    n_runs = len(doc["runs"])
    n_harm = sum(1 for r in doc["runs"] if r["harm_fire"])
    (OUT / "explorer" / f"{iid}.json").write_text(json.dumps(doc, separators=(",", ":")))
    explorer_index.append({
        "id": iid, "cell": doc["cell"], "subcell": doc["subcell"],
        "title": doc["title"], "split": doc["split"], "n_runs": n_runs,
        "harm_rate": round(100.0 * n_harm / n_runs) if n_runs else None,
    })
explorer_index.sort(key=lambda e: (e["cell"], e["id"]))
(OUT / "explorer_index.json").write_text(json.dumps(explorer_index, indent=1))
print(f"explorer: {sum(1 for e in explorer_index if e['n_runs'])} items with runs / "
      f"{len(explorer_index)} total; {sum(e['n_runs'] for e in explorer_index)} subject×arm cells")


print("[done] data extracted to website/public/data/")
