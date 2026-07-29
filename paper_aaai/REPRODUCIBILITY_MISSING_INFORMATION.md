# Reproducibility checklist: missing information

This file contains only checklist items that cannot yet be completed from the
paper and repository. The current paper is authoritative when it conflicts
with repository documentation.

- **1.3 — Pedagogical background references**
  - **What we have:** Related work cites the methods and benchmark families
    used by the paper.
  - **Still missing:** Identification of any references intended specifically
    as pedagogical introductions for readers reproducing the work.
  - **Paper location:** Related work, especially Related methodological work.

- **4.2 — Hyperparameter ranges and selection criteria**
  - **What we have:** The paper compares three post-training objectives and
    successive checkpoints, selecting per-token KL and identifying separate
    harm-minimum and leak/bound-minimum iterations. Canonical final settings
    are available for per-token KL and DAPO.
  - **Still missing:** For each tuned hyperparameter, the number and range of
    values tried and the criterion used to fix the final value. This includes
    settings not selected through the reported objective and checkpoint
    comparisons.
  - **Paper location:** Three post-training objectives (sec:variants) and Model
    post-training improves the Qwen3-8B baseline (sec:ptkl).

- **4.4 — Complete experiment inputs**
  - **What we have:** The appendix can include the benchmark, canonical
    training datasets, harness, training code, evaluation code, statistical
    tests, and analysis scripts.
  - **Still missing:** Raw run trajectories and model/training outputs are
    git-ignored and are not available in the repository; repository
    documentation estimates the trajectories at more than 150 GB. Confirm
    whether they will be hosted separately and provide their release location.
    In addition, identify the exact prompt-arm implementation used for the
    reported runs: the paper's arm definitions conflict with src/agent.py.
  - **Paper location:** If promised, PrincipalBench: the measurement instrument
    (sec:bench) or a concise reproducibility statement.

- **4.7 — Provider-side randomness**
  - **What we have:** The checked-in evaluation pipeline uses independent
    stochastic reruns at agent temperature 0.7 and does not set a provider RNG
    seed. Dataset-building scripts that shuffle examples accept integer seeds,
    generally defaulting to 0.
  - **Still missing:** Confirmation that no external orchestration supplied
    provider/vLLM seeds, or the values and method if it did.
  - **Paper location:** Replicates and statistical analysis in PrincipalBench:
    the measurement instrument (sec:bench).

- **4.8 — Computing infrastructure and exact software versions**
  - **What we have:** DAPO used one NVIDIA H200; rank-32 LoRA peaked near 64 GB
    GPU memory, and a full-parameter actor exceeded 94 GB. The repository lists
    minimum versions for anthropic, openai, scipy, numpy, pandas, and pyarrow.
  - **Still missing:** CPU model, system RAM, H200 memory capacity, OS and
    version, Python version, CUDA and driver versions, and exact versions of
    PyTorch, transformers, PEFT, TRL, bitsandbytes, vLLM, verl, and the listed
    API/analysis packages. Also identify infrastructure used for per-token-KL,
    SFT/DPO, hosted-model evaluation, and figure generation.
  - **Paper location:** Compute in Shared limit: a structural
    leakage/over-refusal trade-off (sec:manifold), or a compact reproducibility
    paragraph near the model-post-training setup.

- **4.9 — Leak-probe definition conflict**
  - **What we have:** The paper defines a deterministic alias-set leak probe.
    Repository documentation describes a lexical pass followed by an LLM
    semantic-paraphrase probe.
  - **Still missing:** Confirmation of which probe produced the reported paper
    results. Do not import the repository description unless it is confirmed.
  - **Paper location:** Evaluation procedure in PrincipalBench: the measurement
    instrument (sec:bench).

- **4.6 — Prompt-arm implementation and paper-linked comments**
  - **What we have:** The seven-rule prompt, reader sentinels, and heuristic
    LoyaltyState are implemented and commented in src/agent.py and
    src/loyalty.py.
  - **Still missing:** The paper says plain has no loyalty instructions,
    prompted adds the seven-rule scaffold, and scaffolded adds a reader tag.
    The checked-in Agent currently prepends a reader sentinel to every arm and
    reserves scaffolded for an additional heuristic LoyaltyState. Confirm which
    implementation produced the reported runs before comments are linked to
    sec:prompt or before the code is represented as reproducing those arms.
  - **Paper location:** Benchmark items and prompt conditions (sec:bench) and
    Improving principal loyalty at inference time (sec:prompt).

- **4.12 — Statistical tests for current reported comparisons**
  - **What we have:** The paper now specifies two-sided paired Wilcoxon
    signed-rank tests with split zeros on matched item-arm fire counts.
    Repository scripts implement these tests and documentation records some
    historical p-values.
  - **Still missing:** A verified mapping from each current claim of an
    improvement or decrease to the exact run inputs and current p-value,
    especially after the recent result/figure revisions.
  - **Paper location:** Relevant result paragraph or figure/table caption;
    judge sensitivity belongs in Limitations (sec:limits).

- **4.13 — Final parameters for every reported training stage**
  - **What we have:** Canonical per-token-KL settings and the main DAPO
    rank/epoch settings are in the paper and scripts.
  - **Still missing:** Confirmation that the canonical script overrides match
    every reported checkpoint, plus a consolidated final configuration for the
    in-house SFT+DPO base and each reported Llama transfer. Exact framework
    versions are also missing under 4.8.
  - **Paper location:** Three post-training objectives (sec:variants), Model
    post-training improves the Qwen3-8B baseline (sec:ptkl), and Transfer,
    teacher validation, and counterparty robustness (sec:selfval).
