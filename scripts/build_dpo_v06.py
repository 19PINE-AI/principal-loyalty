"""Build dpo_v06.jsonl — reader-identity sentinel architectural fix.

Takes existing pair files and rewrites each pair's `system` field to include
a sentinel line derived from the item's `reader_is_principal` flag:

- reader_is_principal=True  -> [READER: PRINCIPAL]   + PLAIN or PROMPTED body
- reader_is_principal=False -> [READER: THIRD_PARTY] + PLAIN or PROMPTED body

This matches what Agent._system() will emit at rollout time, so DPO sees the
exact runtime prompt. The hypothesis is that with the sentinel surfaced as a
literal token the model can learn a CONDITIONAL policy (produce vs refuse)
rather than a register shift.

Inputs:
  data/dpo_v2.jsonl                      (57 third-party refuse pairs)
  data/dpo_author_to_principal_v05.jsonl (36 principal produce pairs)

Output:
  data/dpo_v06.jsonl (93 pairs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import (  # noqa: E402
    PLAIN_SYSTEM,
    PROMPTED_SYSTEM,
    reader_sentinel,
)
from src.items import load_items  # noqa: E402


def _body_for(system_text: str) -> str:
    """Pick the canonical PLAIN/PROMPTED body matching the existing pair."""
    s = (system_text or "").strip()
    prompted_start = PROMPTED_SYSTEM.strip()[:40]
    plain_start = PLAIN_SYSTEM.strip()[:40]
    if s.startswith(prompted_start[:30]) and "Important instructions" in s:
        return PROMPTED_SYSTEM
    if s.startswith(plain_start[:30]):
        return PLAIN_SYSTEM
    # Fallback: length heuristic
    return PROMPTED_SYSTEM if len(s) > 400 else PLAIN_SYSTEM


def main() -> int:
    items = {i.id: i for i in load_items(str(ROOT / "items" / "v0"))}

    inputs = [
        ROOT / "data" / "dpo_v2.jsonl",
        ROOT / "data" / "dpo_author_to_principal_v05.jsonl",
    ]
    out = ROOT / "data" / "dpo_v06.jsonl"

    n_written = 0
    n_principal = 0
    n_third = 0
    with out.open("w") as fo:
        for p in inputs:
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                item = items.get(row["item_id"])
                if item is None:
                    continue
                body = _body_for(row.get("system", ""))
                sentinel = reader_sentinel(item)
                new_system = sentinel + "\n" + body
                row["system"] = new_system
                row.setdefault("v06_sentinel", "PRINCIPAL" if item.reader_is_principal else "THIRD_PARTY")
                fo.write(json.dumps(row) + "\n")
                n_written += 1
                if item.reader_is_principal:
                    n_principal += 1
                else:
                    n_third += 1

    print(f"[v06] wrote {n_written} pairs -> {out}")
    print(f"[v06]   PRINCIPAL sentinel: {n_principal}")
    print(f"[v06]   THIRD_PARTY sentinel: {n_third}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
