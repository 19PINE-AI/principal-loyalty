import { Link } from 'react-router-dom'
import { useData } from '../lib/useData.js'
import ProblemSchematic from '../components/ProblemSchematic.jsx'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, Legend, ReferenceArea,
} from 'recharts'

function Stat({ value, label, accent }) {
  return (
    <div className="bg-white rounded-xl border border-ink/10 p-5 shadow-sm">
      <div className={`text-3xl font-bold tracking-tight ${accent || 'text-ink'}`}>{value}</div>
      <div className="text-sm text-ink/60 mt-1">{label}</div>
    </div>
  )
}

function PaperFigure({ src, alt, caption }) {
  return (
    <figure className="bg-white border border-ink/10 rounded-xl p-3 shadow-sm">
      <img src={`${import.meta.env.BASE_URL}${src}`} alt={alt} loading="lazy"
        className="w-full h-auto rounded mx-auto" />
      {caption && <figcaption className="text-xs text-ink/55 mt-2 px-1 leading-relaxed">{caption}</figcaption>}
    </figure>
  )
}

export default function Overview() {
  const { data: head } = useData('headline.json')
  const { data: manifold } = useData('manifold.json')

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      {/* hero */}
      <section className="mb-12">
        <div className="text-xs text-ink/50 font-mono uppercase tracking-widest">Companion site</div>
        <h1 className="serif text-4xl md:text-5xl font-bold mt-2 leading-tight max-w-4xl">
          Whose Side Is Your Agent On?<br/>
          <span className="text-ink/70 font-medium">Multi-party principal loyalty in LLM agents.</span>
        </h1>
        <p className="mt-6 max-w-3xl text-lg text-ink/70 leading-relaxed">
          When an agent is sent <em>outward</em> to talk to people on a user's behalf —
          a procurement bot negotiating with a vendor, an inbox agent screening cold pitches,
          a phone agent on a billing dispute — the default <em>help-whoever-you-talk-to</em> instinct
          becomes the failure. The agent must stay loyal to the <strong>principal</strong> it represents
          even as the <strong>counterparty</strong> probes, pressures, and flatters.
        </p>
        <p className="mt-3 max-w-3xl text-ink/70">
          This site lets you browse the 75-item benchmark, compare 13 frontier subjects,
          inspect the per-token-KL distillation runs, and — in the{' '}
          <Link to="/explorer" className="text-mech1 font-medium hover:underline">benchmark explorer</Link> —
          open any item to read every model's transcript and the judge's verdict behind each result.
        </p>

        {/* benchmark explorer — the centerpiece */}
        <Link to="/explorer"
          className="group mt-8 block rounded-2xl border border-mech1/30 bg-gradient-to-br from-mech1/[0.07] via-white to-mech2/[0.07] p-6 hover:border-mech1/60 hover:shadow-lg transition">
          <div className="flex flex-wrap items-center justify-between gap-5">
            <div className="min-w-0">
              <div className="text-xs font-mono uppercase tracking-widest text-mech1">Start here · interactive</div>
              <div className="text-2xl font-bold mt-1 group-hover:text-mech1 transition-colors">Open the benchmark explorer →</div>
              <p className="text-ink/70 mt-1.5 max-w-2xl text-sm">
                Every benchmark item, the full subject × arm result matrix, and the raw agent transcript
                with the judge's verdict and leak evidence behind every cell.
              </p>
            </div>
            <div className="flex gap-6 text-center shrink-0">
              <div><div className="text-3xl font-bold text-mech1">60</div><div className="text-xs text-ink/55 mt-0.5">items<br/>evaluated</div></div>
              <div><div className="text-3xl font-bold text-mech1">13</div><div className="text-xs text-ink/55 mt-0.5">frontier<br/>models</div></div>
              <div><div className="text-3xl font-bold text-mech1">2,268</div><div className="text-xs text-ink/55 mt-0.5">scored<br/>transcripts</div></div>
            </div>
          </div>
        </Link>

        <div className="mt-8">
          <ProblemSchematic />
        </div>
      </section>

      {/* headline numbers */}
      {head && (
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          <Stat value={head.items_total}   label="Multi-turn items (50 training + 25 held-out)" />
          <Stat value={head.frontier_subjects} label="Frontier LLM subjects evaluated" />
          <Stat value={`${head.claude_sonnet_scaffolded_harm_pct}%`} label="Claude-Sonnet harm under the loyalty scaffold" accent="text-mech2" />
          <Stat value={`${head.qwen8b_pertoken_kl_iter1_harm}/108`} label="Per-token-KL 8B distilled student" accent="text-mech1" />
        </section>
      )}

      {/* six cells */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-2">The six failure cells</h2>
        <p className="text-ink/70 max-w-3xl mb-6">
          Multi-party loyalty doesn't reduce to a single axis. Five cells (red‑ish) name distinct ways
          to fail the principal; the sixth (<span className="text-sanity font-medium">sanity</span>)
          is a cooperative item where over-refusal is the only failure, included so that
          "refuse everything" is not a winning strategy.
        </p>
        <PaperFigure src="figures/arxiv_fig0b_cells.png"
          alt="The six failure cells of multi-party loyalty"
          caption="Figure 2 from the paper — five red cells are distinct ways to fail the principal; the sixth (blue, sanity) is a cooperative item where over-refusal is the failure." />
      </section>

      {/* manifold scatter */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-2">The leak / over-refusal Pareto frontier</h2>
        <p className="text-ink/70 max-w-3xl mb-4">
          Plot every variant on a leak (x) vs missed-instruction (y) plane. The jointly favorable
          lower-left corner stays empty. Both mechanisms — the prompt-time scaffold and per-token-KL
          distillation — move <em>along</em> this frontier, never across it; single-objective
          (scalar-reward) RL fails to break it too.
        </p>
        <PaperFigure src="figures/arxiv_fig1_manifold.png"
          alt="Leak vs missed-instruction Pareto frontier"
          caption="Figure 1 from the paper. Every variant lands on a common leak/over-refusal frontier; the prompted Claude teacher and the per-token-KL 8B student sit on the same frontier at different operating points." />
        <div className="text-sm font-semibold mt-6 mb-2 text-ink/70">Interactive version — hover any point</div>
        {manifold && (
          <div className="bg-white border border-ink/10 rounded-xl p-4">
            <ResponsiveContainer width="100%" height={460}>
              <ScatterChart margin={{ top: 18, right: 24, bottom: 30, left: 10 }}>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" />
                <ReferenceArea x1={0} x2={10} y1={0} y2={15} fill="#16a34a" fillOpacity={0.06}
                  label={{ value: 'jointly favorable', position: 'insideTopLeft', fill:'#16a34a', fontSize:11 }} />
                <XAxis type="number" dataKey="leak" name="Leak (out of 108)"
                  label={{ value: 'Leak fires / 108', position: 'insideBottom', offset: -12, fontSize: 13 }}
                  domain={[0, 'dataMax + 4']} stroke="#1a1a2e" />
                <YAxis type="number" dataKey="mi" name="MI (out of 108)"
                  label={{ value: 'Missed-instruction / 108', angle: -90, position: 'insideLeft', offset: -2, fontSize: 13 }}
                  domain={[0, 'dataMax + 4']} stroke="#1a1a2e" />
                <ZAxis range={[140, 140]} />
                <Tooltip content={({ active, payload }) => {
                  if (!active || !payload || !payload.length) return null
                  const d = payload[0].payload
                  return (
                    <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
                      <div className="font-semibold">{d.label}</div>
                      <div className="mono text-xs text-ink/70 mt-1">leak {d.leak} · MI {d.mi} · harm {d.harm}</div>
                    </div>
                  )
                }} />
                <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: 10 }} />
                <Scatter name="Base" data={manifold.filter(d => d.kind==='base')} fill="#94a3b8" isAnimationActive animationDuration={900} />
                <Scatter name="Per-turn variants" data={manifold.filter(d => d.kind==='variant')} fill="#84cc16" isAnimationActive animationDuration={900} animationBegin={120} />
                <Scatter name="Per-token KL (Mech 2)" data={manifold.filter(d => d.kind==='mechanism')} fill="#7c3aed" isAnimationActive animationDuration={900} animationBegin={240} />
                <Scatter name="RL (DAPO)" data={manifold.filter(d => d.kind==='rl')} fill="#f97316" isAnimationActive animationDuration={900} animationBegin={360} />
                <Scatter name="Claude + scaffold (Mech 1)" data={manifold.filter(d => d.kind==='scaffold')} fill="#0891b2" isAnimationActive animationDuration={900} animationBegin={480} />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="text-xs text-ink/60 mt-2">
              Each point is a checkpoint or arm on the 108-cell (36 items × 3 arms) grid. Lower-left is better;
              the shaded region is jointly favorable. Hover for exact counts.
            </div>
          </div>
        )}
      </section>

      {/* CTA grid */}
      <section className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <Link to="/explorer" className="block bg-white border border-ink/10 rounded-xl p-6 hover:border-mech1/40 hover:shadow-md transition">
          <div className="text-xs font-mono text-ink/50 uppercase tracking-wider">Explore</div>
          <div className="text-xl font-semibold mt-1">Every item × model</div>
          <p className="text-ink/60 mt-2 text-sm">Open any evaluated item, see the subject × arm result matrix, and read each agent transcript with the judge's verdict.</p>
        </Link>
        <Link to="/subjects" className="block bg-white border border-ink/10 rounded-xl p-6 hover:border-mech1/40 hover:shadow-md transition">
          <div className="text-xs font-mono text-ink/50 uppercase tracking-wider">Compare</div>
          <div className="text-xl font-semibold mt-1">13 frontier subjects</div>
          <p className="text-ink/60 mt-2 text-sm">Selective vs over-refusing cluster, per-arm decomposition, held-out validation.</p>
        </Link>
        <Link to="/training" className="block bg-white border border-ink/10 rounded-xl p-6 hover:border-mech1/40 hover:shadow-md transition">
          <div className="text-xs font-mono text-ink/50 uppercase tracking-wider">Inspect</div>
          <div className="text-xl font-semibold mt-1">Distillation variants</div>
          <p className="text-ink/60 mt-2 text-sm">Per-turn SFT, DPO, per-token KL; K-iteration trajectories for Qwen3-8B and Llama-3.1-8B.</p>
        </Link>
      </section>
    </div>
  )
}
