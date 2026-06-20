# Reproducing the paper

Every reported count and aggregate in `paper_arxiv.tex` regenerates from the raw
`runs/*/scored.jsonl` trajectories — there are no hardcoded figure values.

## Scope conventions

These must match the paper (see the `scripts/recompute_all.py` docstring):

- **The "36-item core"** is the union of `data/verl_train.parquet` (31) and
  `data/verl_val.parquet` (5); **108 cells = 36 items × 3 arms**.
- Counts subset to the core item-ids and de-duplicate `(item_id, arm)`, keeping
  the first occurrence (some runs contain duplicate rows).
- `harm = harm.harm_fire`; `leak = leak_rate > 0`;
  `bound = harm.leaked_private_bound`; `mi = harm.missed_instruction`
  (the over-refusal / missed-instruction axis).
- **Audit:** a row whose `early_end_reason` contains `"error"`, or that has 0
  agent turns, is dropped before scoring. (`counterparty_end` is a normal,
  benign ending.)
- Per-arm table columns are **seed-1**; aggregates are **n = 5 mean ± sample-sd**.
- Multi-seed mean = (sum of per-cell fires across the N seed dirs) / N.

> **Why the full file has more rows than the paper counts.** A full
> `scored.jsonl` has 141–146 rows; the paper counts are on the 36-item core ×3
> arms. A naive recompute over the whole file looks inflated but isn't — always
> scope to the core.

## The four reproduction commands

```bash
python3 scripts/recompute_all.py            # 49 checks; 13/13 split-grid rows
python3 scripts/paired_seed_test.py ...     # multi-seed paired Wilcoxon (harm gain p=0.0114, …)
python3 scripts/per_arm_xvendor_wilcoxon.py # cluster split p-values (1.8e-6 / 2.2e-7 / 5.9e-7)
python3 scripts/make_figs_arxiv.py          # all figures, computed live from runs/
```

`scripts/recompute_all.py --json out.json` writes a machine-readable
reconciliation report you can diff against the paper.

## Regenerating trajectories from scratch

`runs/` (150+ GB) and `outputs/` are git-ignored, so a fresh clone has no
trajectories. To rebuild them, run the harness over the items for each subject
and arm. The orchestration scripts under `scripts/run_*.sh` and
`scripts/multi_rollout_eval.py` do this; for example a multi-seed grid eval:

```bash
python3 scripts/multi_rollout_eval.py \
    --base runs/<subject>_grid --seeds 5 \
    --counterparty claude-sonnet --parallel 4
```

### Integrity caveats (from hard-won experience)

- **Always check `errs == 0` and that turns are not all `1`** before trusting
  any `scored.jsonl` number — provider auth (401) outages produced silent
  single-turn truncations that pass naive parsing.
- **Never `audit || true`.** On a shared GPU, contention can kill the local
  vLLM server mid-eval; fail fast, restart vLLM, and re-run with `--parallel 1`.
- **Local 32B-AWQ** vLLM needs `--enforce-eager` and
  `VLLM_ENGINE_READY_TIMEOUT_S=1800` (CUDA-graph capture exceeds the 600s
  default).

## Statistical tests

- `scripts/paired_seed_test.py` — multi-seed paired Wilcoxon on harm/leak gains.
- `scripts/per_arm_xvendor_wilcoxon.py` — cross-vendor cluster split p-values.
- `scripts/heldout_per_arm_wilcoxon.py` — the same on the held-out split.
- `scripts/dual_judge_kappa.py` — inter-judge agreement (Cohen's κ).

`scipy` is required for these (`pip install scipy`).
