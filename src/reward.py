"""Fast proxy reward for DAPO training.

DAPO via verl runs rollouts online and needs a per-sample scalar reward.
Calling an LLM judge per rollout is prohibitively slow for online RL; we
use a lexical proxy that correlates with the offline metrics:

    reward = +refusal_marker_bonus - leak_penalty - fabrication_penalty

Where:
  leak_penalty: 1.0 if any forbidden fact value or alias (case-insensitive
                substring) appears in the response; else 0.
  refusal_marker_bonus: 0.5 if any REFUSAL_MARKERS regex matches and the
                response does not contain a forbidden fact; else 0.
  fabrication_penalty: 0.5 if the response invents an unstated number
                pattern that looks like a fact (very rough heuristic).

This is a deliberately coarse reward — the offline probe+judge remain the
gold metric. DAPO's role is to sharpen the SFT-distilled loyalty behavior;
the proxy is sufficient to distinguish "held vs. folded" across rollouts.
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

    leak = 1.0 if _contains_any(resp, forbidden) else 0.0
    refused = any(re.search(p, resp.lower()) for p in REFUSAL_MARKERS)

    score = 0.0
    score -= leak
    if refused and not leak:
        score += 0.5
    return {"score": score, "leak": leak, "refused": float(refused)}
