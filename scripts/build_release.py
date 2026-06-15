"""Assemble a clean public release tree under release/ from the verified
in-repo artifacts, WITHOUT mutating the originals or any run data.

Fixes the three release-hygiene issues flagged in review:
  * excludes items/v0_5 (a superseded dev-intermediate set, 51 files,
    referenced by no eval script -- it is NOT part of the 75-item benchmark);
  * resolves the pb-capit-authority-01 id collision: items/v0 and
    items/v0_75 each contain a *different* item under that id. In the release
    the held-out copy is renamed to a unique id so the union is keyable by id;
  * lays out train/ vs heldout/ so the 50/25 split is explicit.

The 75-item benchmark = items/v0 (50, train) + items/v0_75 (25, held-out).

Usage:
    python3 scripts/build_release.py            # build ./release
    python3 scripts/build_release.py --out /tmp/principal-loyalty-release
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Held-out item whose id collides with a different train item; give the
# held-out copy a unique id in the release.
COLLISION_OLD = "pb-capit-authority-01"
COLLISION_NEW = "pb-capit-authority-ho-01"

CODE = [
    "scripts/recompute_all.py",
    "scripts/paired_seed_test.py",
    "scripts/per_arm_xvendor_wilcoxon.py",
    "scripts/audit_trajectories.py",
    "scripts/make_figs_arxiv.py",
    "scripts/onpolicy_distill_iter.py",
    "scripts/pertoken_kl_collect.py",
    "src/items.py",
    "src/scorer.py",
    "src/vendors.py",
]


def copy_items(src: Path, dst: Path, rename: dict[str, str] | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.json")):
        obj = json.loads(p.read_text())
        new_id = (rename or {}).get(obj.get("id"))
        out_name = p.name
        if new_id:
            obj["id"] = new_id
            out_name = f"{new_id}.json"
        (dst / out_name).write_text(json.dumps(obj, indent=2))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "release"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n_train = copy_items(ROOT / "items" / "v0", out / "items" / "train")
    n_held = copy_items(ROOT / "items" / "v0_75", out / "items" / "heldout",
                        rename={COLLISION_OLD: COLLISION_NEW})

    # de-dup / collision check on the union
    ids = {}
    dupes = []
    for split in ("train", "heldout"):
        for p in (out / "items" / split).glob("*.json"):
            i = json.loads(p.read_text())["id"]
            if i in ids:
                dupes.append(i)
            ids[i] = split

    (out / "code").mkdir(parents=True, exist_ok=True)
    copied_code = []
    for rel in CODE:
        src = ROOT / rel
        if src.exists():
            dst = out / "code" / Path(rel).name
            shutil.copy2(src, dst)
            copied_code.append(Path(rel).name)

    for paper in ("paper_arxiv.tex", "paper_arxiv.bib"):
        if (ROOT / paper).exists():
            shutil.copy2(ROOT / paper, out / paper)

    manifest = {
        "benchmark": "PrincipalBench",
        "n_items_total": n_train + n_held,
        "train_items": n_train,
        "heldout_items": n_held,
        "excluded": ["items/v0_5 (superseded dev-intermediate, not part of release)"],
        "collision_fix": f"{COLLISION_OLD} (held-out) -> {COLLISION_NEW}",
        "duplicate_ids_after_fix": dupes,
        "code": copied_code,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    assert n_train == 50, f"expected 50 train items, got {n_train}"
    assert n_held == 25, f"expected 25 held-out items, got {n_held}"
    assert not dupes, f"id collisions remain: {dupes}"
    print(f"\nClean release written to {out}  (50 train + 25 held-out, no collisions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
