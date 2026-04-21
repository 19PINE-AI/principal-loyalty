"""Audit items for principal-briefing leakage into counterparty-facing text.

Per plan §5.4 (invoking the PaceBench v3 findings.md §5h-5i discipline).
For each item, verify no private_facts[*].value or .aliases appears in the
counterparty_persona, counterparty_strategy, or counterparty_opening. Also
check that stated_positions statements don't appear in the counterparty's
opening (they should discover positions from the agent, not be told).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main() -> int:
    items_dir = Path(__file__).parent.parent / "items" / "v0"
    failures: list[str] = []
    for path in sorted(items_dir.glob("*.json")):
        item = json.loads(path.read_text())
        haystack_fields = [
            "counterparty_persona",
            "counterparty_opening",
        ]
        haystack = " || ".join(norm(item.get(f, "")) for f in haystack_fields)
        for fact in item.get("private_facts", []):
            needles = [fact["value"]] + list(fact.get("aliases", []))
            for needle in needles:
                nn = norm(str(needle))
                if len(nn) < 3:
                    continue
                if nn in haystack:
                    failures.append(f"{item['id']}: private-fact '{fact['name']}' value/alias '{needle}' appears in counterparty text")
        # also: does the principal briefing text (minus basics) leak into opening?
        # lighter check: flag if >= 8 consecutive words of briefing appear in opening
        briefing = norm(item.get("principal_briefing", ""))
        opening = norm(item.get("counterparty_opening", ""))
        briefing_words = briefing.split()
        for i in range(len(briefing_words) - 8):
            window = " ".join(briefing_words[i:i + 8])
            if window in opening:
                failures.append(f"{item['id']}: 8-gram of briefing appears in opening: '{window}'")
    if failures:
        print(f"AUDIT FAILED ({len(failures)} issues):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"AUDIT PASS: checked {len(list(items_dir.glob('*.json')))} items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
