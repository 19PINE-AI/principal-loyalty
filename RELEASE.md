# PrincipalBench — release & reproducibility

The paper (`paper_arxiv.tex`) references the repository
`github.com/19PINE-AI/principal-loyalty`. That repository now exists under the
**19PINE-AI** organization and is currently **private**; make it public (or run
`scripts/build_release.py` and push the clean tree to a public mirror) before
the paper is submitted, or the reproducibility statement is false. Steps below.

## What the release contains

The 75-item benchmark = `items/v0` (50 training) + `items/v0_75` (25 held-out).
`items/v0_5` is a superseded development-intermediate set (51 files, referenced
by no eval script) and is **not** part of the release.

Build the clean public tree with:

```
python3 scripts/build_release.py        # writes ./release
```

This produces `release/` with:
- `items/train/` (50) and `items/heldout/` (25), the 6-cell-balanced held-out split;
- a fix for the one id collision (`pb-capit-authority-01` denotes a *different*
  item in train vs held-out; the held-out copy is renamed `pb-capit-authority-ho-01`);
- `code/` with the scoring, distillation, and analysis scripts;
- `MANIFEST.json`.

## Reproducing every number in the paper

All reported counts and aggregates regenerate from the raw `runs/*/scored.jsonl`
trajectories — there is no longer any hardcoded figure value:

```
python3 scripts/recompute_all.py            # 49 checks; 13/13 split-grid rows
python3 scripts/paired_seed_test.py ...     # multi-seed paired Wilcoxon (harm gain p=0.0114, etc.)
python3 scripts/per_arm_xvendor_wilcoxon.py # cluster split p-values (1.8e-6 / 2.2e-7 / 5.9e-7)
python3 scripts/make_figs_arxiv.py          # all figures, computed live from runs/
```

Conventions (see `scripts/recompute_all.py` docstring): counts are on the
36-item core (×3 arms = 108 cells); the integrity gate is a **run-level**
pass/fail (≤5% error rows, no zero-agent-turn), not a per-row filter; per-arm
table columns are seed-1, aggregates are n=5 mean ± sample-sd.

## Creating the public repo (author action — not automatable here)

```
python3 scripts/build_release.py
# the repo already exists as 19PINE-AI/principal-loyalty (private); make it public with:
gh repo edit 19PINE-AI/principal-loyalty --visibility public --accept-visibility-change-consequences
# copy release/ (+ figures/, paper PDF, and any checkpoints/trajectories you choose to host) into the repo, then:
git -C <repo> add . && git -C <repo> commit -m "Initial PrincipalBench release" && git -C <repo> push
```

If you also publish raw trajectories, update the held-out collision id there too:
`grep -rl pb-capit-authority-01 <heldout-trajectories> | xargs sed -i 's/pb-capit-authority-01/pb-capit-authority-ho-01/g'`
(the paper's reported numbers are computed per-run and are unaffected by this rename).
