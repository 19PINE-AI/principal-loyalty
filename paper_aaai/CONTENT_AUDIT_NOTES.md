# Content audit notes

This file tracks issues that should not be resolved by inference alone. Remove an
entry only after its provenance or intended interpretation has been confirmed.

## Unresolved

- **Leak-probe implementation wording:** both main-paper versions call the probe
  deterministic, but the original appendix says it uses curated lexical aliases
  *and an LLM judge for paraphrases*. Confirm whether the reported `leak` signal is
  purely deterministic, hybrid, or two separately combined signals before revising
  the methods description.

- **Figure 5, per-token-KL iteration 2 p-value:** the plotted `p=.012` differs from
  the documented matched-seed result, `p=0.0436` (`n=4`), reported in the original
  paper, `scripts/make_figs_arxiv.py`, `scripts/build_website_data.py`, and the
  website Wilcoxon data. The plotted label is hard-coded in the original and local
  plotting scripts, but it may come from a different test or aggregation. Identify
  its provenance before changing it to `p=.044*`.
- **Meaning intended by Table 3's bold values:** bold appears in selective-model
  cells even though the caption says it marks the over-refusing cluster. The values
  appear to mark
  favorable/highlighted cells, but neither the original caption nor nearby prose
  defines that convention. Do not invent a meaning; either remove those bold marks
  or confirm the intended convention.
- **Control 4 outcome detail:** the original provides no numerical result beyond
  `beats neither source reward`. Additional numbers require experiment records or
  author confirmation; they cannot be recovered from the paper text alone.
- **Training-only provenance of the 50+ scaffold-development trajectories:** the
  chronology strongly implies training/development data, but the original does not
  explicitly state this in the scaffold paragraph. Confirm before adding the claim.
