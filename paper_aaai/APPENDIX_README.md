# Anonymous code and data appendix

This archive supports checklist items 3.3 and 4.3--4.5 for the anonymous
submission. It contains the released PrincipalBench items, the canonical
training inputs needed by the reported methods, and the smallest practical
selection of source files for data construction, evaluation, training,
statistical analysis, and figure regeneration.

## Contents

- "items/train" and "items/heldout": the 50 training and 25 held-out benchmark
  items described in PrincipalBench: the measurement instrument (sec:bench).
- "data": the clean teacher traces and SFT+DPO base datasets, the 31/5 DAPO
  split, and the canonical iteration-1 top-K teacher distributions used in
  Improving principal loyalty through model post-training (sec:ptkl).
- "src": the item loader, multi-turn harness, prompt conditions, leak probe,
  harm scorer, integrity-related logic, reward proxy, and provider adapters.
- "scripts/preprocessing": dataset and teacher-signal construction.
- "scripts/training": per-token-KL and DAPO training and adapter merging.
- "scripts/evaluation": trajectory generation, scoring, and integrity audit.
- "scripts/analysis": paired tests, judge agreement, paper-number
  reconciliation, and figure generation.

## Scope and limitations

The archive intentionally excludes API credentials, model weights, local
training outputs, and raw evaluation trajectories. The raw trajectories are
more than 150 GB and are not present in the source repository. The included
code can regenerate trajectories when the required model access and provider
credentials are supplied.

The dependency file records the available lower bounds for the benchmark and
analysis environment. Exact versions of several training frameworks and full
hardware details remain unavailable and are listed in the submission's
reproducibility handoff.

The checked-in Agent currently applies reader sentinels to every prompt arm
and adds a heuristic LoyaltyState to the scaffolded arm. This differs from the
arm description in the submitted paper and must be reconciled before this
archive can be described as an exact reproduction of the reported prompt-arm
experiments.

## Representative workflow

1. Install the dependencies in "requirements.txt" and the training frameworks
   imported by the selected training scripts.
2. Run "scripts/evaluation/run_traj_only.py" to generate benchmark
   trajectories.
3. Run "scripts/evaluation/score_only.py" and
   "scripts/evaluation/audit_trajectories.py".
4. Use "scripts/preprocessing/pertoken_kl_collect.py" followed by
   "scripts/training/train_pertoken_kl.py" for the per-token-KL intervention.
5. Use the scripts under "scripts/analysis" for paired tests and aggregation.

The shell orchestration scripts retain repository-relative paths and are
included as exact records of the reported configurations. Local paths must be
adapted to the reviewer's environment.
