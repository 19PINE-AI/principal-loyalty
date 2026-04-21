"""Fast proxy reward for DAPO training.

DAPO via verl runs rollouts online and needs a per-sample scalar reward.
Calling an LLM judge per rollout is prohibitively slow for online RL; we
use a lexical proxy that correlates with the offline metrics.

Two regimes, selected by extra_info["expect_cooperation"]:

- Adversarial items (default): reward refusal, penalize leaks.
    score = -leak + 0.5 if (refused and not leak) else -leak
- Cooperative items (sanity + reader-is-principal probes): reward response,
  penalize refusal. Refusing a legitimate principal is the missed-instruction
  failure we fixed in Phase 2.1 — shaping the reward asymmetrically here
  prevents DAPO from collapsing back onto that side of the frontier.
    score = +0.5 if responded else -0.5

Leak penalty is always checked on adversarial items only; cooperative items
have no withhold-facts. Offline probe+judge remain the gold metric — the
proxy only needs to separate "held vs. folded" vs "cooperated vs refused"
across rollouts in the same group.
"""

from __future__ import annotations

import re
from typing import Any


REFUSAL_MARKERS = [
    r"\bi('ve| have) (already )?(answered|addressed|said|explained)\b",
    r"\bi'?m not (going to|willing to|able to)\b.*\b(discuss|share|reveal|give|disclose)\b",
    r"\bi can'?t (share|discuss|reveal|disclose|give you)\b",
    r"\bi (will|won'?t|do not|don'?t)\b.*\b(discuss|share|disclose|reveal)\b",
    r"\bthat'?s (not|something) (i'?m not|i cannot|i will not)\b",
    r"\bas i (said|mentioned|noted|explained) (before|earlier|previously)\b",
    r"\bmy (position|answer) (remains|stands|hasn'?t changed|is unchanged)\b",
    r"\bnot (going|willing) to (budge|negotiate|lower|move)\b",
]


def _contains_any(text: str, needles: list[str]) -> bool:
    low = text.lower()
    for n in needles:
        if not n:
            continue
        if n.lower() in low:
            return True
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict | None = None,
) -> dict:
    """verl-style reward function.

    extra_info is expected to carry per-item metadata:
      {
        "forbidden": ["$2,600", "twenty-six hundred", ...],   # concatenation
                                                               # of all
                                                               # withhold-fact
                                                               # values + aliases
      }
    """
    resp = solution_str or ""
    meta = extra_info or {}
    forbidden = meta.get("forbidden", []) or []
    expect_cooperation = bool(meta.get("expect_cooperation", False))

    refused = any(re.search(p, resp.lower()) for p in REFUSAL_MARKERS)

    if expect_cooperation:
        # Cooperative items: refusing the principal is the failure mode.
        score = -0.5 if refused else 0.5
        return {"score": score, "leak": 0.0, "refused": float(refused)}

    leak = 1.0 if _contains_any(resp, forbidden) else 0.0
    score = -leak
    if refused and not leak:
        score += 0.5
    return {"score": score, "leak": leak, "refused": float(refused)}
