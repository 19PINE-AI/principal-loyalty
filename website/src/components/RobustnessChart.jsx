import { useData } from '../lib/useData.js'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts'

/**
 * Interactive recreation of arxiv_fig5_robustness: counterparty swap + held-out
 * generalization. Two side-by-side panels.
 *
 * Left panel — swap the counterparty model on the per-token-KL iter1 checkpoint:
 *   Claude-Sonnet → GPT-5 → Gemini-3-flash. The leak-axis gain transfers more
 *   cleanly across counterparties than the harm-axis gain.
 *
 * Right panel — train vs held-out harm for each recipe. Per-token KL has the
 * lowest training harm but the largest train-to-held-out gap.
 */
export default function RobustnessChart() {
  const { data } = useData('robustness.json')
  if (!data) return null

  return (
    <div className="grid lg:grid-cols-2 gap-4">
      {/* Counterparty swap */}
      <div className="bg-white border border-ink/10 rounded-xl p-4">
        <div className="text-sm font-semibold mb-2">Counterparty swap (KL iter 1)</div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.counterparty} margin={{ top: 16, right: 16, bottom: 24, left: 0 }} barCategoryGap="28%">
            <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
            <XAxis dataKey="counterparty" fontSize={10} angle={-12} textAnchor="end" height={48} />
            <YAxis fontSize={11} domain={[0, 56]} label={{ value: 'fires / 108', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip
              cursor={{ fill: 'rgba(0,0,0,0.04)' }}
              content={({ active, payload, label }) => {
                if (!active || !payload || !payload.length) return null
                const d = payload[0].payload
                return (
                  <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
                    <div className="font-semibold">{label}</div>
                    <div className="mono text-xs">harm: <b>{d.harm}</b>/108</div>
                    <div className="mono text-xs text-leak">leak: <b>{d.leak}</b>/108</div>
                  </div>
                )
              }} />
            <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" iconType="rect" />
            <Bar dataKey="harm" name="harm" fill="#7c3aed" radius={[3,3,0,0]} isAnimationActive animationDuration={800}>
              {data.counterparty.map((d, i) => <Cell key={i} fill="#7c3aed" />)}
            </Bar>
            <Bar dataKey="leak" name="leak" fill="#dc2626" radius={[3,3,0,0]} isAnimationActive animationDuration={800} animationBegin={100} />
          </BarChart>
        </ResponsiveContainer>
        <div className="text-xs text-ink/60 mt-1">
          Sweep counterparty model on the same checkpoint: leak rises modestly (13→14→20)
          but harm doubles on Gemini-3-flash (33→49). The leak gain transfers more cleanly
          than the harm gain across counterparties.
        </div>
      </div>

      {/* Held-out gap */}
      <div className="bg-white border border-ink/10 rounded-xl p-4">
        <div className="text-sm font-semibold mb-2">Train vs held-out (harm %)</div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.heldout} margin={{ top: 16, right: 16, bottom: 24, left: 0 }} barCategoryGap="28%">
            <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
            <XAxis dataKey="recipe" fontSize={10} angle={-12} textAnchor="end" height={48} />
            <YAxis fontSize={11} domain={[0, 50]} unit="%" label={{ value: 'harm %', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip
              cursor={{ fill: 'rgba(0,0,0,0.04)' }}
              content={({ active, payload, label }) => {
                if (!active || !payload || !payload.length) return null
                const d = payload[0].payload
                const gap = (d.heldout - d.training).toFixed(1)
                return (
                  <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
                    <div className="font-semibold">{label}</div>
                    <div className="mono text-xs">training:  <b>{d.training}%</b></div>
                    <div className="mono text-xs">held-out: <b>{d.heldout}%</b></div>
                    <div className="mono text-xs mt-1 text-leak">gap = {gap} points</div>
                  </div>
                )
              }} />
            <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" iconType="rect" />
            <Bar dataKey="training" name="training" fill="#94a3b8" radius={[3,3,0,0]} isAnimationActive animationDuration={800} />
            <Bar dataKey="heldout"  name="held-out" fill="#dc2626" radius={[3,3,0,0]} isAnimationActive animationDuration={800} animationBegin={100} />
          </BarChart>
        </ResponsiveContainer>
        <div className="text-xs text-ink/60 mt-1">
          Per-token KL has the lowest training harm but the largest train-to-held-out gap.
          Per-turn SFT (iter 2) nearly closes the gap — the gap appears to be a feature of the
          per-token-KL objective rather than of distillation in general.
        </div>
      </div>
    </div>
  )
}
