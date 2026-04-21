"""PrincipalBench item schema + loader.

Each item is a JSON file in items/v0/ with the following structure:

{
  "id": "pb-leak-price-01",
  "cell": "leakage",                # leakage|capitulation|posture|authoring|sanity|moderation
  "subcell": "price",               # finer categorization; used for coverage matrix
  "title": "Used-car reservation price",
  "principal_briefing": "...",      # text shown to the agent as setup (before counterparty turn 1)
  "private_facts": [
    {"name": "reservation_price",
     "value": "$12,000",
     "aliases": ["twelve thousand", "12k"],
     "sharing": "withhold",         # withhold|redirect|shareable
     "category": "reservation"}
  ],
  "stated_positions": [
    {"name": "asking_price",
     "statement": "I am asking $15,000 for the car.",
     "must_hold": true}
  ],
  "counterparty_persona": "...",
  "counterparty_strategy": "repeated_extraction|anchoring|framing|gaslighting|sanity_cooperative|moderation_cross_party|authoring",
  "counterparty_opening": "...",    # first message to the agent
  "max_turns": 12,                  # agent turns
  "posture_pressure": true,         # whether sustained pressure is expected (enables posture-signal scoring)
  "notes": ""
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Sharing = Literal["withhold", "redirect", "shareable"]
Cell = Literal["leakage", "capitulation", "posture", "authoring", "sanity", "moderation"]
Strategy = Literal[
    "repeated_extraction",
    "anchoring",
    "framing",
    "gaslighting",
    "sanity_cooperative",
    "moderation_cross_party",
    "authoring",
]


@dataclass
class PrivateFact:
    name: str
    value: str
    aliases: list[str] = field(default_factory=list)
    sharing: Sharing = "withhold"
    category: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "PrivateFact":
        return cls(
            name=d["name"],
            value=d["value"],
            aliases=list(d.get("aliases", [])),
            sharing=d.get("sharing", "withhold"),
            category=d.get("category", ""),
        )


@dataclass
class StatedPosition:
    name: str
    statement: str
    must_hold: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "StatedPosition":
        return cls(
            name=d["name"],
            statement=d["statement"],
            must_hold=d.get("must_hold", True),
        )


@dataclass
class PrivateBound:
    """A private numeric/categorical bound the agent must stay inside but must
    never state publicly. Distinct from a PrivateFact (which is withholding
    information) and a StatedPosition (which is a public anchor).

    Example: rent ceiling $2,600 — public counteroffers must be below $2,600
    but the agent must never name $2,600 as its position.
    """
    name: str
    value: str                           # canonical form, e.g. "$2,600" or "$100/hour"
    kind: str = "ceiling"                # ceiling | floor | bound
    aliases: list[str] = field(default_factory=list)  # numeric variants for probe matching
    description: str = ""                # short free-text for judge prompt

    @classmethod
    def from_dict(cls, d: dict) -> "PrivateBound":
        return cls(
            name=d["name"],
            value=d["value"],
            kind=d.get("kind", "ceiling"),
            aliases=list(d.get("aliases", [])),
            description=d.get("description", ""),
        )


@dataclass
class Item:
    id: str
    cell: Cell
    subcell: str
    title: str
    principal_briefing: str
    private_facts: list[PrivateFact]
    stated_positions: list[StatedPosition]
    counterparty_persona: str
    counterparty_strategy: Strategy
    counterparty_opening: str
    max_turns: int = 12
    posture_pressure: bool = False
    reader_is_principal: bool = False
    notes: str = ""
    private_bounds: list[PrivateBound] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(
            id=d["id"],
            cell=d["cell"],
            subcell=d.get("subcell", ""),
            title=d.get("title", d["id"]),
            principal_briefing=d["principal_briefing"],
            private_facts=[PrivateFact.from_dict(f) for f in d.get("private_facts", [])],
            stated_positions=[StatedPosition.from_dict(p) for p in d.get("stated_positions", [])],
            counterparty_persona=d["counterparty_persona"],
            counterparty_strategy=d["counterparty_strategy"],
            counterparty_opening=d["counterparty_opening"],
            max_turns=d.get("max_turns", 12),
            posture_pressure=d.get("posture_pressure", False),
            reader_is_principal=d.get("reader_is_principal", False),
            notes=d.get("notes", ""),
            private_bounds=[PrivateBound.from_dict(b) for b in d.get("private_bounds", [])],
        )


def load_items(path: str | Path) -> list[Item]:
    p = Path(path)
    if p.is_dir():
        out: list[Item] = []
        for f in sorted(p.glob("*.json")):
            out.append(Item.from_dict(json.loads(f.read_text())))
        return out
    return [Item.from_dict(json.loads(p.read_text()))]
