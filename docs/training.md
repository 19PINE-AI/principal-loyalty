# Training recipes

We study two ways to instill principal loyalty in open-weight students: the
**per-token-KL distillation** recipe (mechanism M2, the strongest open-weight
result we measure) and **DAPO** single-knob RL baselines (which fail to cross
the leak/over-refusal frontier).

## Datasets (`data/`)

| File pattern | Purpose |
|---|---|
| `data/onpolicy_iter*_trajectories.jsonl` | On-policy student rollouts per iteration. |
| `data/onpolicy_iter*_sft*.jsonl` | SFT targets distilled from the teacher per iteration. |
| `data/onpolicy_iter*_dpo.jsonl` | DPO preference pairs per iteration. |
| `data/dpo_*.jsonl` | Off-policy DPO datasets (versioned `v0…v4`). |
| `data/verl_train.parquet` / `verl_val.parquet` | The 31/5 split that defines the 36-item core for DAPO via verl. |
| `data/verl_*_mistral.parquet` | Regenerable from the base parquet (git-ignored). |

## M2 — per-token-KL distillation

The canonical recipe transfers a **prompted Qwen3-32B teacher** (scaffolded arm,
extended reasoning) into **8B Qwen3 / Llama-3.1** students via on-policy,
**per-token** KL distillation.

Pipeline:

```bash
# 1. collect teacher per-token logprobs on student rollouts
python3 scripts/pertoken_kl_collect.py ...

# 2. run a distillation iteration (on-policy: rollout → distill → repeat)
python3 scripts/onpolicy_distill_iter.py ...

# orchestrated, idempotent multi-iteration runs:
bash scripts/run_iter4_pertoken_kl.sh
bash scripts/run_iter5_pertoken_kl.sh
```

The best 8B student (tier "D10" canonical per-token KL with the Qwen3-32B
teacher) reaches **harm 33 / leak 13** — the strongest open-weight recipe in
the paper.

### Gotchas

- **Masked-KL NaN trap.** A `0 * inf` in the masked KL produces `NaN`; use
  `-1e4` as the masking sentinel, not `-inf`.
- **vLLM + Qwen tokenizer.** A `tokenizer_config.json` with
  `extra_special_tokens=[...]` (a list) breaks vLLM and must be removed.
- **Qwen3-32B-AWQ** needs `--enforce-eager` and
  `VLLM_ENGINE_READY_TIMEOUT_S=1800`.

## DAPO baselines (verl)

DAPO is the single-knob RL baseline. It moves *along* the same frontier as the
scaffold and distillation — it does not break the structural floor.

```bash
bash scripts/run_dapo.sh                    # canonical run
bash scripts/run_dapo_from_pertoken_kl.sh   # DAPO initialized from the distilled student
bash scripts/run_dapo_variants_multiseed.sh # v2 / leak-only / v3 variants, n=5 seeds
```

The fast proxy reward used during online rollouts is `src/reward.py` (see
[architecture.md](architecture.md#reward-proxy-training-only)).

### Single-GPU note

Single-H200 DAPO on Qwen3-8B requires a **LoRA actor** — a full-parameter actor
plus reference plus rollout engine OOMs on one card.

## End-to-end

`scripts/run_full_pipeline.sh` chains the Llama-3.1-8B pipeline (SFT → DPO →
eval → DAPO → eval) and the DAPO-variant multi-seed sweep, then the paired
Wilcoxon comparison. Every sub-script is idempotent and skips completed stages.
