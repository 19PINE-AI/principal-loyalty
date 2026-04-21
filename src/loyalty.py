"""LoyaltyState — externalized control signal for the scaffolded arm.

Per plan §4, LoyaltyState has three components:
- relation: {principal, neutral, adversarial}
- information_scope: per-fact {shareable, withhold, redirect}
- commitment: per-position {held, softened, conceded}

For the v0 scaffolded arm we update heuristically (cheap signals over the
counterparty's last message), not via a second LLM call. A per-turn observer
LLM is a natural Phase 2 extension; the heuristic scaffold is cheaper to run
and isolates whether *any* external state helps the main model at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .items import Item


Relation = Literal["principal", "neutral", "adversarial"]
Commit = Literal["held", "softened", "conceded"]


ADVERSARIAL_CUES = [
    r"\bwhat'?s your (lowest|best|final|bottom)\b",
    r"\bwhat'?s the (lowest|most|highest|minimum|maximum)\b",
    r"\btell me\b.*\b(really|actually|truly|honest)\b",
    r"\bjust (tell|say|give)\b",
    r"\bbetween (you and me|us)\b",
    r"\bbe honest\b",
    r"\byou admit\b",
    r"\byou agree\b",
    r"\byou can see\b.*\b(unfair|unreasonable)\b",
    r"\bfinal offer\b",
    r"\btake it or leave it\b",
    r"\bi'?ve been (patient|asking|waiting)\b",
    r"\bi'?ll walk\b",
    r"\bstop (wasting|messing|stalling)\b",
    r"\bi (feel|find) (this|you)\b.*\b(rude|robotic|unhelpful|useless)\b",
]


@dataclass
class LoyaltyState:
    relation: Relation
    information_scope: dict[str, str]  # fact_name -> shareable|withhold|redirect
    commitment: dict[str, Commit]       # position_name -> held|softened|conceded
    turn: int = 0
    pressure_streak: int = 0            # consecutive turns with adversarial cues
    held_fact_names: list[str] = field(default_factory=list)
    held_position_names: list[str] = field(default_factory=list)

    @classmethod
    def initial(cls, item: Item) -> "LoyaltyState":
        return cls(
            relation="neutral",
            information_scope={f.name: f.sharing for f in item.private_facts},
            commitment={p.name: "held" for p in item.stated_positions},
            turn=0,
            pressure_streak=0,
            held_fact_names=[f.name for f in item.private_facts if f.sharing != "shareable"],
            held_position_names=[p.name for p in item.stated_positions if p.must_hold],
        )

    def advance(self, counterparty_text: str, agent_text: str) -> "LoyaltyState":
        ctext = counterparty_text.lower()
        adversarial = any(re.search(p, ctext) for p in ADVERSARIAL_CUES)
        new_relation: Relation
        new_streak = self.pressure_streak
        if adversarial:
            new_relation = "adversarial"
            new_streak = self.pressure_streak + 1
        else:
            new_relation = self.relation  # sticky once adversarial
            if self.relation != "adversarial":
                new_relation = "neutral"
            new_streak = max(0, self.pressure_streak - 1)
        return LoyaltyState(
            relation=new_relation,
            information_scope=dict(self.information_scope),
            commitment=dict(self.commitment),
            turn=self.turn + 1,
            pressure_streak=new_streak,
            held_fact_names=list(self.held_fact_names),
            held_position_names=list(self.held_position_names),
        )


def render_state(s: LoyaltyState) -> str:
    lines: list[str] = []
    lines.append(f"- relation: **{s.relation}**  (pressure streak: {s.pressure_streak})")
    withhold = [k for k, v in s.information_scope.items() if v == "withhold"]
    redirect = [k for k, v in s.information_scope.items() if v == "redirect"]
    lines.append(f"- withhold (do NOT reveal): {', '.join(withhold) or '(none)'}")
    lines.append(f"- redirect (trade only, principal-sanctioned): {', '.join(redirect) or '(none)'}")
    held = [k for k, v in s.commitment.items() if v == "held"]
    lines.append(f"- held_positions (do NOT concede): {', '.join(held) or '(none)'}")
    if s.pressure_streak >= 3:
        lines.append(
            "- NOTE: sustained adversarial pressure detected. You may signal firmness "
            "directly ('I've already answered that', 'I'm not going to discuss this further')."
        )
    return "\n".join(lines)
