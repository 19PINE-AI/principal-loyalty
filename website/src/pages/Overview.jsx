import { Link } from 'react-router-dom'
import { useData, CELL_DOT } from '../lib/useData.js'
import PaperFigure, { PaperFigureHeader } from '../lib/PaperFigure.jsx'
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

function CellBadge({ id, label, desc }) {
  return (
    <div className="bg-white border border-ink/10 rounded-lg p-4 flex gap-3 items-start">
      <span className={`inline-block w-3 h-3 rounded-full mt-1.5 shrink-0 ${CELL_DOT[id]}`} />
      <div>
        <div className="font-semibold">{label}</div>
        <div className="text-sm text-ink/60 mt-0.5">{desc}</div>
      </div>
    </div>
  )
}

export default function Overview() {
  const { data: head } = useData('headline.json')
  const { data: cells } = useData('cells.json')
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
          inspect the per-token-KL distillation runs, and read sample conversations.
        </p>

        <div className="mt-8">
          <PaperFigure
            src="arxiv_fig0_problem.png"
            label="Figure 0"
            caption="The agent maintains two parallel channels — a back-and-forth with the principal P (briefings, requests, results), and a separate conversation with a counterparty C whose interests may conflict with P's. The default 'help the current speaker' objective fails along four conversational axes (bottom panel)."
            maxWidth="900px"
          />
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
        {cells && (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {cells.map(c => <CellBadge key={c.id} {...c} />)}
          </div>
        )}

        <div className="mt-6">
          <PaperFigure
            src="arxiv_fig0b_cells.png"
            label="Figure 0b"
            caption="The six failure cells, as visualized in the paper. The sanity cell (blue) exists to prevent 'refuse everything' from being a winning strategy."
            maxWidth="780px"
          />
        </div>
      </section>

      {/* manifold scatter */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-2">The leak / over-refusal floor</h2>
        <p className="text-ink/70 max-w-3xl mb-4">
          Plot every variant on a leak (x) vs missed-instruction (y) plane. The favorable lower-left
          corner stays empty. Both mechanisms — the prompt-time scaffold and per-token-KL distillation —
          move <em>along</em> this floor, never across it.
        </p>
        {manifold && (
          <div className="bg-white border border-ink/10 rounded-xl p-4">
            <ResponsiveContainer width="100%" height={420}>
              <ScatterChart margin={{ top: 18, right: 24, bottom: 30, left: 10 }}>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" />
                <ReferenceArea x1={0} x2={10} y1={0} y2={15} fill="#16a34a" fillOpacity={0.06} label={{ value: 'jointly favorable', position: 'insideTopLeft', fill:'#16a34a', fontSize:11 }} />
                <XAxis type="number" dataKey="leak" name="Leak (out of 108)" label={{ value: 'Leak fires / 108', position: 'insideBottom', offset: -12, fontSize: 13 }} domain={[0, 'dataMax + 4']} stroke="#1a1a2e" />
                <YAxis type="number" dataKey="mi" name="MI (out of 108)" label={{ value: 'Missed-instruction / 108', angle: -90, position: 'insideLeft', offset: -2, fontSize: 13 }} domain={[0, 'dataMax + 4']} stroke="#1a1a2e" />
                <ZAxis range={[110, 110]} />
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
                <Scatter name="Base" data={manifold.filter(d => d.kind==='base')} fill="#94a3b8" />
                <Scatter name="Per-turn variants" data={manifold.filter(d => d.kind==='variant')} fill="#84cc16" />
                <Scatter name="Per-token KL (Mech 2)" data={manifold.filter(d => d.kind==='mechanism')} fill="#7c3aed" />
                <Scatter name="RL (DAPO)" data={manifold.filter(d => d.kind==='rl')} fill="#f97316" />
                <Scatter name="Claude + scaffold (Mech 1)" data={manifold.filter(d => d.kind==='scaffold')} fill="#0891b2" />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="text-xs text-ink/60 mt-2">
              Each point is a checkpoint or arm on the 108-cell (36 items × 3 arms) grid. Lower-left is better;
              the shaded region is jointly favorable. Hover for exact counts.
            </div>
          </div>
        )}

        <div className="mt-6">
          <PaperFigureHeader label="Figure 1 (paper version)" />
          <PaperFigure
            src="arxiv_fig1_manifold.png"
            label=""
            caption="The same floor as rendered in the paper: the prompted Claude teacher and the per-token-KL student land on a common frontier; the DAPO RL baseline regresses from the distillation optimum."
            maxWidth="820px"
          />
        </div>
      </section>

      {/* CTA grid */}
      <section className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <Link to="/items" className="block bg-white border border-ink/10 rounded-xl p-6 hover:border-mech1/40 hover:shadow-md transition">
          <div className="text-xs font-mono text-ink/50 uppercase tracking-wider">Browse</div>
          <div className="text-xl font-semibold mt-1">75 test items</div>
          <p className="text-ink/60 mt-2 text-sm">Search and filter the full benchmark by cell, split, and counterparty strategy.</p>
        </Link>
        <Link to="/subjects" className="block bg-white border border-ink/10 rounded-xl p-6 hover:border-mech1/40 hover:shadow-md transition">
          <div className="text-xs font-mono text-ink/50 uppercase tracking-wider">Compare</div>
          <div className="text-xl font-semibold mt-1">13 frontier subjects</div>
          <p className="text-ink/60 mt-2 text-sm">Calibrated vs over-refuse cluster, per-arm decomposition, held-out validation.</p>
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
