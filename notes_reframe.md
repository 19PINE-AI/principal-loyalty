# Reframe notes — collected from 2026-05-14 experiments

## New data, mapped onto paper sections

### §1 Introduction / Contributions list — needs reordering

The strongest scientific finding is the **v3 orthogonality refutation** + the
manifold structure spanning **5 dimensions** (judge / prompt / DPO / DAPO /
counterparty / base-model / **prompted-frontier**). Specifically:

- v3 was a pre-registered "compose the best halves" prediction that **failed**.
- Phase 4 (claude-sonnet + v4 prompt) lands at 21/108 harm — a NEW data point
  on the manifold that *Pareto-dominates Qwen DAPO-v1 on aggregate harm* but
  loses on bound-leak (5 vs 2). The corners-of-the-manifold story strengthens.

### §4 Manifold — add Phase 4 corner

| variant                      | plain (h/l) | prompted (h/l) | scaffolded (h/l) | total harm | bound | leak | MI  |
|------------------------------|------------:|---------------:|-----------------:|-----------:|------:|-----:|----:|
| Untrained Qwen3-8B           | --          | --             | --               | 28/107     | 6     | 81   | 24  |
| Untrained Llama-3.1-8B       | 12/5        | 16/1           | 17/1             | **45/108** | 0     | 7    | 43  |
| Qwen v4.1 (SFT+DPO)          | --          | --             | --               | 56/108     | 4     | 18   | 51  |
| Qwen DAPO-v1 step35          | 6/10        | 13/3           | 18/6             | 37/108     | 2     | 19   | 34  |
| Mistral SFT+DPO              | --          | --             | --               | 27/108     | 2     | 21   | 27  |
| **Phase 4: claude-sonnet + v4 prompt** | 7/7  | 8/4         | 6/6              | **21/108** | 5     | 17   | 21  |

The new data point is more refusal-prone than any trained Qwen variant but
catches more bound leaks. This is the manifold geometry of "where does mass
go when you pressure one axis": Claude moved mass off harm/MI but at the
cost of a few bound-leak fires.

### §5 Reward ablation — unchanged

### §6 Robustness — promote Llama as third base-model

Llama-3.1-8B-Instruct (NousResearch mirror, bit-identical to meta-llama):
- **Untrained baseline lands at 45/108 harm, but only 7/108 LEAK.** This is
  closer to Mistral-SFT+DPO's leak/MI composition than to Qwen-untrained's.
  Llama base model has dramatically more loyalty-aware default refusal
  behavior than Qwen base.
- The "stable basin is base-model-bound" claim from Mistral is now testable
  in a third dimension. Pending: Llama SFT/DPO/DAPO pipeline.

### §App I (integrity note) — extend with 2026-05-14 incident

A second failure-mode-shape incident: the `OPENAI_API_KEY` lost its
`model.request` scope mid-runs. Trajectory-generation completed cleanly
(108/108 with multi-turn dist 3-12 turns), but ALL scoring rows were
silently produced as zero-row scored.jsonl files. Diagnosed by audit
discipline ([[feedback-audit-evals]]): `wc -l scored.jsonl` was 0 despite
108 trajectories. Routed `gpt-5-mini` through OpenRouter
(`openai/gpt-5-mini`) to recover. **Audit Rule:** after every score_grid
call, check `wc -l scored.jsonl == wc -l trajectories.jsonl`.

### §7 Conclusion — keep the manifold framing; soften "trained model is the
contribution"

The trained 8B is now demonstrably **inferior to claude-sonnet + v4 prompt**
on aggregate harm (37 vs 21) but **superior** on plain-arm-leak rate
absolute terms (Qwen plain leak 10/36 = 27.8% vs Claude plain leak 7/36 =
19.4% — wait, Claude is also better here). And superior on bound-leak (2
vs 5).

Actually let me re-check the bound-leak claim. Looking at the per-arm
breakdown:
- Qwen DAPO-v1 step35: bound=2 total
- Claude+v4 prompt: bound=5 total

So Qwen DAPO catches more bound leaks. That's the lone axis where the
trained model wins. For "minimize bound-leak per dollar", trained Qwen
might still be best.

### Headline rewrite (concrete plan)

OLD: "PrincipalBench v0.3 + SFT→DPO→DAPO recipe reduces harm 62→37 on
Qwen3-8B."

NEW: "PrincipalBench v0.3 + 5 falsifiers (judge, prompt, DPO, DAPO,
counterparty, base-model, **prompted-frontier**) all land different
points on a leak/MI/bound manifold whose axes are coupled (v3 reward
composition refuted). A frontier model with the v4 prompt
(claude-sonnet, 21/108 harm) sits in the same manifold's most-favorable
corner but with the highest bound-leak rate (5 vs 2 for trained Qwen) —
trained-model and prompted-frontier are different operating points, not
strictly ordered."
