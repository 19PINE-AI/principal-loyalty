import { useMemo, useState } from 'react'
import { useData, CLUSTER_COLORS, ARM_LABELS, CELL_DOT } from '../lib/useData.js'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend, ReferenceLine,
} from 'recharts'

const ARM_LIST = ['plain', 'prompted', 'scaffolded']

function ClusterLegend() {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-ink/70">
      {Object.entries(CLUSTER_COLORS).map(([k, v]) => (
        <div key={k} className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: v }} />
          {k}
        </div>
      ))}
    </div>
  )
}

function MultiSeedChart({ subjects }) {
  if (!subjects) return null
  const data = subjects.map(s => ({ ...s, fill: CLUSTER_COLORS[s.cluster] }))
  return (
    <ResponsiveContainer width="100%" height={420}>
      <BarChart data={data} margin={{ top: 16, right: 24, bottom: 60, left: 10 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
        <XAxis dataKey="display" interval={0} angle={-32} textAnchor="end" stroke="#1a1a2e" fontSize={11} />
        <YAxis stroke="#1a1a2e" label={{ value: 'Harm rate (%)', angle: -90, position: 'insideLeft', fontSize: 12 }} domain={[0, 80]} />
        <Tooltip cursor={{ fill: 'rgba(0,0,0,0.04)' }} content={({ active, payload }) => {
          if (!active || !payload || !payload.length) return null
          const d = payload[0].payload
          return (
            <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
              <div className="font-semibold">{d.display}</div>
              <div className="mono text-xs text-ink/70">cluster: {d.cluster}</div>
              <div className="mono text-xs mt-1">multi-seed mean: <b>{d.mean.toFixed(1)}%</b> ± {d.sd.toFixed(1)}</div>
              <div className="mono text-xs mt-1">single seed: plain {d.plain}% · prompted {d.prompted}% · scaffolded {d.scaffolded}%</div>
            </div>
          )
        }} />
        <ReferenceLine y={20} stroke="#16a34a" strokeDasharray="4 4" label={{ value: 'calibrated ceiling 20%', fontSize: 10, fill: '#16a34a', position: 'left' }} />
        <ReferenceLine y={53.6} stroke="#dc2626" strokeDasharray="4 4" label={{ value: 'over-refuse floor 53.6%', fontSize: 10, fill: '#dc2626', position: 'left' }} />
        <Bar dataKey="mean" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function ArmChart({ subjects, arm }) {
  if (!subjects) return null
  const data = subjects.map(s => ({ ...s, val: s[arm], fill: CLUSTER_COLORS[s.cluster] }))
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 4, right: 16, bottom: 50, left: 10 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
        <XAxis dataKey="display" interval={0} angle={-30} textAnchor="end" fontSize={9} />
        <YAxis fontSize={11} domain={[0, 100]} label={{ value: '%', angle: -90, position: 'insideLeft', fontSize: 11 }} />
        <Tooltip content={({ active, payload }) => {
          if (!active || !payload || !payload.length) return null
          const d = payload[0].payload
          return <div className="bg-white px-2 py-1 rounded shadow border border-ink/10 text-xs"><b>{d.display}</b><br/>{d.val}%</div>
        }} />
        <Bar dataKey="val" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function HeldoutChart({ held }) {
  if (!held || !held.length) return null
  const data = held.map(s => ({ ...s, fill: CLUSTER_COLORS[s.cluster] }))
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} margin={{ top: 4, right: 16, bottom: 60, left: 10 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
        <XAxis dataKey="display" interval={0} angle={-30} textAnchor="end" fontSize={10} />
        <YAxis fontSize={11} domain={[0, 100]} label={{ value: 'Held-out harm (%)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
        <Tooltip content={({ active, payload }) => {
          if (!active || !payload || !payload.length) return null
          const d = payload[0].payload
          return (
            <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
              <div className="font-semibold">{d.display}</div>
              <div className="mono text-xs">held-out harm: <b>{d.harm_pct}%</b> ({d.harm}/{d.n})</div>
              <div className="mono text-xs">leak {d.leak} · bound {d.bound} · MI {d.mi}</div>
            </div>
          )
        }} />
        <Bar dataKey="harm_pct" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function SubjectDetail({ subjectKey, subject_arms }) {
  if (!subject_arms || !subjectKey) return null
  const d = subject_arms[subjectKey]
  if (!d) return <div className="text-sm text-ink/50">No per-cell data available for this subject (off-grid in §4).</div>
  return (
    <div className="space-y-4">
      <div className="grid sm:grid-cols-3 gap-2 text-sm">
        {ARM_LIST.map(a => {
          const x = d.per_arm[a]
          if (!x) return null
          const pct = (100 * x.harm / Math.max(x.n, 1)).toFixed(0)
          return (
            <div key={a} className="bg-stone-50 rounded p-3">
              <div className="text-xs font-mono text-ink/50 uppercase">{a}</div>
              <div className="font-bold text-2xl mt-0.5">{pct}%</div>
              <div className="text-xs text-ink/60 mono">harm {x.harm}/{x.n} · leak {x.leak} · MI {x.mi}</div>
            </div>
          )
        })}
      </div>

      <div>
        <div className="text-xs uppercase font-semibold tracking-wider text-ink/50 mb-2">Per-cell harm fires</div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
          {Object.entries(d.per_cell).map(([cell, x]) => (
            <div key={cell} className="bg-white border border-ink/10 rounded p-2 flex items-center gap-2">
              <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${CELL_DOT[cell] || 'bg-ink/30'}`} />
              <span className="mono text-ink/70">{cell}</span>
              <span className="ml-auto font-mono"><b>{x.harm}</b>/{x.n}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function Subjects() {
  const { data: subjects } = useData('subjects.json')
  const { data: subject_arms } = useData('subject_arms.json')
  const { data: held } = useData('subject_held.json')
  const [pickedArm, setPickedArm] = useState('all')
  const [detailKey, setDetailKey] = useState(null)

  const sortedSubjects = useMemo(() => {
    if (!subjects) return null
    return [...subjects].sort((a, b) => a.mean - b.mean)
  }, [subjects])

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-12">
      <div>
        <h1 className="text-3xl font-bold serif">Frontier subjects</h1>
        <p className="text-ink/70 mt-1 max-w-3xl">
          Thirteen frontier LLMs evaluated under three system-prompt arms on the 36-item core,
          with multi-seed n=5. The cluster split is sharp: nine calibrated subjects below 20% harm,
          three over-refuse subjects above 53% — driven by missed-instruction, not leakage.
        </p>
      </div>

      <section>
        <div className="flex flex-wrap items-baseline justify-between mb-3 gap-3">
          <h2 className="text-xl font-semibold">Calibrated vs over-refuse split</h2>
          <ClusterLegend />
        </div>
        <div className="bg-white border border-ink/10 rounded-xl p-4">
          <MultiSeedChart subjects={sortedSubjects} />
          <div className="text-xs text-ink/60 mt-2">
            Multi-seed (n=5) mean harm, paired evaluation seeds, 36-item core. Calibrated cluster (green)
            holds ≤ 20%; over-refuse cluster (red) is pegged on missed-instruction.
            Per-arm paired Wilcoxon vs cluster mean is significant at every arm
            (plain p = 1.8e-6, prompted p = 2.2e-7, scaffolded p = 5.9e-7).
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Per-arm breakdown</h2>
        <div className="grid lg:grid-cols-3 gap-4">
          {ARM_LIST.map(arm => (
            <div key={arm} className="bg-white border border-ink/10 rounded-xl p-3">
              <div className="text-sm font-semibold mb-1">{ARM_LABELS[arm]}</div>
              <ArmChart subjects={sortedSubjects} arm={arm} />
            </div>
          ))}
        </div>
        <p className="text-xs text-ink/60 mt-2">
          The over-refuse cluster fires 55–72% harm at the no-instruction <code className="mono">plain</code> arm,
          before any loyalty scaffold is added — the split is intrinsic to post-training, not prompt-induced.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Held-out validation</h2>
        <div className="bg-white border border-ink/10 rounded-xl p-4">
          <HeldoutChart held={held} />
          <div className="text-xs text-ink/60 mt-2">
            On 24 items authored after training was frozen, calibrated subjects stay ≤ 24% and over-refuse
            subjects ≥ 76%; GPT-5 amplifies from 71% to 93%. Per-arm test remains significant (p ≤ 1.8e-5).
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Drill into a subject</h2>
        <div className="flex flex-wrap gap-2 mb-4">
          {subjects && subjects.map(s => (
            <button key={s.key} onClick={() => setDetailKey(s.key)}
              className={`px-3 py-1.5 text-sm rounded-md border transition ${
                detailKey === s.key
                  ? 'bg-ink text-white border-ink'
                  : 'border-ink/15 text-ink/70 hover:border-ink/40 bg-white'
              }`}>
              <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ background: CLUSTER_COLORS[s.cluster] }} />
              {s.display}
            </button>
          ))}
        </div>
        {detailKey && (
          <div className="bg-white border border-ink/10 rounded-xl p-4">
            <SubjectDetail subjectKey={detailKey} subject_arms={subject_arms} />
          </div>
        )}
      </section>
    </div>
  )
}
