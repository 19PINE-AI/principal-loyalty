import { useData } from '../lib/useData.js'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ErrorBar, Cell, Legend,
} from 'recharts'

/**
 * Interactive recreation of arxiv_fig3_wilcoxon. Multi-seed paired Wilcoxon
 * comparing per-token-KL iter 1 vs SFT+DPO base on harm/leak/bound/MI, with
 * error bars across seeds and p-value labels above each pair.
 */
export default function WilcoxonChart() {
  const { data } = useData('wilcoxon.json')
  if (!data) return null

  const rows = data.iter1.metrics.map(m => ({
    metric: m.label,
    Base:   m.base,
    KLi1:   m.kl,
    kl_sd:  m.kl_sd,
    p:      m.p,
    sig:    m.p < 0.05 ? '*' : '',
  }))

  return (
    <div className="bg-white border border-ink/10 rounded-xl p-4">
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={rows} margin={{ top: 30, right: 24, bottom: 32, left: 6 }} barCategoryGap="22%">
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
          <XAxis dataKey="metric" fontSize={12}
            label={{ value: 'metric', position: 'insideBottom', offset: -10, fontSize: 11 }} />
          <YAxis fontSize={11} domain={[0, 55]}
            label={{ value: 'Mean fires per seed (out of 108)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <Tooltip
            cursor={{ fill: 'rgba(0,0,0,0.04)' }}
            content={({ active, payload, label }) => {
              if (!active || !payload || !payload.length) return null
              const d = payload[0].payload
              const delta = (d.Base - d.KLi1).toFixed(1)
              return (
                <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
                  <div className="font-semibold">{label}</div>
                  <div className="mono text-xs mt-1 text-ink/60">base: {d.Base.toFixed(1)}</div>
                  <div className="mono text-xs text-mech1">KL i1: {d.KLi1.toFixed(1)} ± {d.kl_sd.toFixed(1)}</div>
                  <div className="mono text-xs mt-1">Δ = {delta} · paired Wilcoxon <b>p = {d.p}</b>{d.sig}</div>
                </div>
              )
            }} />
          <Legend wrapperStyle={{ fontSize: 12, paddingBottom: 6 }} verticalAlign="top" iconType="rect" />
          <Bar dataKey="Base" fill="#94a3b8" radius={[3,3,0,0]}
            isAnimationActive animationDuration={900} animationEasing="ease-out" />
          <Bar dataKey="KLi1" name="Per-token KL i1" fill="#7c3aed" radius={[3,3,0,0]}
            isAnimationActive animationDuration={900} animationEasing="ease-out" animationBegin={150}>
            <ErrorBar dataKey="kl_sd" width={6} stroke="#1a1a2e" strokeWidth={1.5} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 -mt-2 px-4 mb-2">
        {rows.map(r => (
          <div key={r.metric} className="text-center">
            <span className={`text-xs mono ${r.p < 0.05 ? 'text-mech1 font-bold' : 'text-ink/50'}`}>
              p = {r.p}{r.sig}
            </span>
          </div>
        ))}
      </div>
      <div className="text-xs text-ink/60 mt-1">
        Paired Wilcoxon on per-cell fire counts across <b>n = 5</b> independent evaluation seeds.
        Both stopping points clear p &lt; 0.05 on harm (iter 1: p = 0.0114; iter 2 separately: p = 0.0436).
        Hover any bar for exact deltas.
      </div>
    </div>
  )
}
