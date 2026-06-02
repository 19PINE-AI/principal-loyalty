import PaperFigure from '../lib/PaperFigure.jsx'

const FIGURES = [
  { src: 'arxiv_fig0_problem.png', label: 'Figure 0',
    caption: 'The multi-party loyalty problem. The agent maintains two parallel channels — a back-and-forth with the principal P, and a separate conversation with a counterparty C whose interests may conflict with P\'s. The default "help the current speaker" objective fails along four conversational axes (bottom panel).' },
  { src: 'arxiv_fig0b_cells.png', label: 'Figure 0b',
    caption: 'The six failure cells. Five cells (red) are different ways the agent fails the principal; the sixth (blue, sanity) is a cooperative item where over-refusal is the only failure — included so that "refuse everything" is not a winning strategy.' },
  { src: 'arxiv_fig1_manifold.png', label: 'Figure 1',
    caption: 'The leak / missed-instruction floor. Each marker is one variant on the 36-item × 3-arm grid; x is leak rate, y is missed-instruction rate, label is harm/108. The jointly favorable lower-left corner is empty. The prompted Claude teacher and the per-token-KL 8B student land on the same frontier at different operating points.' },
  { src: 'arxiv_fig7_xsubj.png', label: 'Figure 7',
    caption: 'The calibrated/over-refuse split (13 subjects, multi-seed n=5). Nine subjects cluster at ≤ 19.5% harm; GLM-4.6 is intermediate; three over-refuse at ≥ 53.6%. Error bars are ± 1σ across 5 eval seeds. The ~34-point gap is intrinsic (present at the no-prompt arm).' },
  { src: 'arxiv_fig8_heldout_xsubj.png', label: 'Figure 8',
    caption: 'Held-out items confirm the split is not item-specific. On 24 items authored after training was frozen, calibrated subjects stay ≤ 24% and over-refuse subjects ≥ 76%; GPT-5 amplifies to 93%.' },
  { src: 'arxiv_fig6_variants.png', label: 'Figure 6',
    caption: 'Distillation variant ladder (Qwen3-8B). Per-token KL is the only variant whose harm improvement against the SFT+DPO base ("v4.1") is significant at n=5 paired Wilcoxon (p = 0.011); per-turn SFT and the DAPO-style RL baseline are indistinguishable from seed noise.' },
  { src: 'arxiv_fig2_kiter.png', label: 'Figure 2',
    caption: 'Qwen3-8B K-iteration trajectory. Iteration 1 is the harm-minimum stopping point, iteration 2 is the leak/bound-minimum, iterations 3–4 regress, iteration 5 swings back. The policy circles a leak/harm trade-off rather than crossing it.' },
  { src: 'arxiv_fig9_llama_kiter.png', label: 'Figure 9',
    caption: 'Llama-3.1-8B K-iteration trajectory. Monotone descent to iteration 3 and a plateau at iteration 4 — the shape is base-model-dependent, but in both families each iteration trades one axis for another.' },
  { src: 'arxiv_fig3_wilcoxon.png', label: 'Figure 3',
    caption: 'Multi-seed paired Wilcoxon vs the SFT+DPO base. Both per-token-KL stopping points (iteration 1, the harm-minimum; iteration 2, the leak/bound-minimum) reach p < 0.05 on harm. Error bars are ± 1σ across seeds.' },
  { src: 'arxiv_fig4_teacher.png', label: 'Figure 4',
    caption: 'Teacher self-validation. The open Qwen3-32B-AWQ teacher matches Claude-Sonnet on harm and missed-instruction but leaks much more (21/31 vs 6/36). The student inherits a harm-low, leak-tolerant profile.' },
  { src: 'arxiv_fig5_robustness.png', label: 'Figure 5',
    caption: 'Counterparty swap (left) and held-out generalization (right). Per-token KL has the lowest training harm but the largest train-to-held-out gap; leak transfers across counterparties more cleanly than harm.' },
]

export default function Figures() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <h1 className="text-3xl font-bold serif mb-2">Paper figures</h1>
      <p className="text-ink/70 mb-8 max-w-3xl">
        All eleven figures from the paper, rendered at full resolution.
        Click any figure to zoom. Most also appear inline on the relevant section pages with an interactive version below.
      </p>

      <div className="space-y-10">
        {FIGURES.map(f => (
          <PaperFigure key={f.src} {...f} />
        ))}
      </div>
    </div>
  )
}
