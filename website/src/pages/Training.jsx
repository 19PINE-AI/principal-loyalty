import { useData } from '../lib/useData.js'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Legend,
} from 'recharts'

function VariantsChart({ variants }) {
  if (!variants) return null
  return (
    <ResponsiveContainer width="100%" height={380}>
      <BarChart data={variants} margin={{ top: 18, right: 16, bottom: 50, left: 10 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
        <XAxis dataKey="name" interval={0} angle={-22} textAnchor="end" fontSize={11} />
        <YAxis label={{ value: 'Harm fires / 108', angle: -90, position: 'insideLeft', fontSize: 12 }} fontSize={11} domain={[0, 64]} />
        <Tooltip content={({ active, payload }) => {
          if (!active || !payload || !payload.length) return null
          const d = payload[0].payload
          return (
            <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
              <div className="font-semibold">{d.name}</div>
              <div className="mono text-xs mt-1">harm: <b>{d.harm}</b>/108</div>
              {d.sig && <div className="mono text-xs mt-0.5">significance: {d.sig}</div>}
            </div>
          )
        }} />
        <Bar dataKey="harm" radius={[3, 3, 0, 0]} label={{ position: 'top', fontSize: 11, fontWeight: 'bold' }}>
          {variants.map((d, i) => <Cell key={i} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function KiterChart({ rows, family }) {
  if (!rows) return null
  return (
    <ResponsiveContainer width="100%" height={290}>
      <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 28, left: 0 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" />
        <XAxis dataKey="iter" label={{ value: 'iteration', position: 'insideBottom', offset: -4, fontSize: 11 }} fontSize={11} />
        <YAxis label={{ value: 'fires / 108', angle: -90, position: 'insideLeft', fontSize: 11 }} fontSize={11} domain={[0, 50]} />
        <Tooltip content={({ active, payload, label }) => {
          if (!active || !payload || !payload.length) return null
          return (
            <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
              <div className="font-semibold">{family} · iter {label}</div>
              {payload.map(p => (
                <div key={p.dataKey} className="mono text-xs" style={{ color: p.color }}>
                  {p.dataKey}: {p.value}
                </div>
              ))}
            </div>
          )
        }} />
        <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" />
        <Line type="monotone" dataKey="harm"  stroke="#7c3aed" strokeWidth={2.5} dot={{ r: 4 }} />
        <Line type="monotone" dataKey="leak"  stroke="#dc2626" strokeWidth={2} dot={{ r: 3 }} />
        <Line type="monotone" dataKey="bound" stroke="#ea580c" strokeWidth={2} dot={{ r: 3 }} />
        <Line type="monotone" dataKey="mi"    stroke="#0891b2" strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default function Training() {
  const { data: variants } = useData('variants.json')
  const { data: kiter } = useData('kiter.json')

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-12">
      <div>
        <h1 className="text-3xl font-bold serif">Training & distillation variants</h1>
        <p className="text-ink/70 mt-1 max-w-3xl">
          For an open-weight student (Qwen3-8B at an in-house SFT+DPO endpoint),
          we compare three distillation objectives on identical on-policy data, then
          iterate per-token KL across K rounds. The same recipe is replicated on Llama-3.1-8B
          with a same-family teacher.
        </p>
      </div>

      <section>
        <div className="flex flex-wrap items-baseline justify-between mb-3 gap-3">
          <h2 className="text-xl font-semibold">Variant ladder</h2>
          <div className="flex gap-4 text-xs text-ink/60">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-[#94a3b8]" /> base</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-[#84cc16]" /> variant</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-mech1" /> mechanism</span>
          </div>
        </div>
        <div className="bg-white border border-ink/10 rounded-xl p-4">
          <VariantsChart variants={variants} />
          <div className="text-xs text-ink/60 mt-2">
            Per-token forward-KL is the only distillation variant whose harm gain is statistically
            distinguishable from seed noise at multi-seed n=5 (p = 0.011). Per-turn DPO is essentially flat;
            a DAPO-style RL checkpoint regresses from the per-token-KL optimum.
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">K-iteration trajectories</h2>
        <p className="text-ink/70 text-sm max-w-3xl mb-4">
          Re-sample student trajectories from each iteration-K checkpoint and re-distill.
          The Qwen family <em>orbits</em> a trade-off (iter 1 = harm-min, iter 2 = leak/bound-min,
          iter 5 swings back). Llama descends monotonically to iter 3 and plateaus at iter 4.
        </p>
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="bg-white border border-ink/10 rounded-xl p-4">
            <div className="text-sm font-semibold mb-2">Qwen3-8B</div>
            <KiterChart rows={kiter?.qwen} family="Qwen3-8B" />
          </div>
          <div className="bg-white border border-ink/10 rounded-xl p-4">
            <div className="text-sm font-semibold mb-2">Llama-3.1-8B</div>
            <KiterChart rows={kiter?.llama} family="Llama-3.1-8B" />
          </div>
        </div>
      </section>

      <section className="bg-white border border-ink/10 rounded-xl p-5">
        <h2 className="text-xl font-semibold mb-3">Recipe summary</h2>
        <div className="grid sm:grid-cols-2 gap-x-8 gap-y-3 text-sm">
          <Row k="Student model">Qwen3-8B (and Llama-3.1-8B for cross-family transfer)</Row>
          <Row k="Teacher">Qwen3-32B-AWQ prompted with the loyalty scaffold (Llama-70B-Instruct-AWQ for Llama transfer)</Row>
          <Row k="Data">On-policy student trajectories, teacher-supervised at the student's visited states</Row>
          <Row k="Objective">Per-token forward-KL on the teacher's top-K=20 distribution</Row>
          <Row k="Iter-1 dataset">113 turn-records, 28,486 token-level signals</Row>
          <Row k="Hyperparameters">3 epochs · QLoRA rank 16 · lr 5e-5 · batch 1 · grad-accum 8</Row>
          <Row k="Headline (iter 1)">harm 33/108 · leak 13/108 · bound 3/108 · MI 32/108</Row>
          <Row k="Multi-seed (n=5)">harm 39.2 ± 4.0 vs base 47.8 (paired Wilcoxon p = 0.0114)</Row>
        </div>
      </section>
    </div>
  )
}

function Row({ k, children }) {
  return (
    <div className="flex gap-3">
      <div className="text-ink/50 mono text-xs uppercase tracking-wider w-28 shrink-0 pt-0.5">{k}</div>
      <div>{children}</div>
    </div>
  )
}
