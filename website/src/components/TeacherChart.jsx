import { useData } from '../lib/useData.js'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LabelList,
} from 'recharts'

/**
 * Interactive recreation of arxiv_fig4_teacher. Side-by-side comparison of the
 * open Qwen3-32B teacher and Claude-Sonnet, both prompted with the loyalty
 * scaffold, on harm / leak / missed-instruction. The teacher matches Claude on
 * harm but leaks much more — explains why the distilled student inherits a
 * harm-low, leak-tolerant profile.
 */
export default function TeacherChart() {
  const { data } = useData('teacher.json')
  if (!data) return null

  const rows = data.metrics.map(m => ({
    metric: m.label,
    Claude: 100 * m.claude / m.claude_n,
    Qwen:   100 * m.qwen   / m.qwen_n,
    claude_raw: `${m.claude}/${m.claude_n}`,
    qwen_raw:   `${m.qwen}/${m.qwen_n}`,
  }))

  return (
    <div className="bg-white border border-ink/10 rounded-xl p-4">
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={rows} margin={{ top: 18, right: 16, bottom: 30, left: 6 }} barCategoryGap="24%">
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
          <XAxis dataKey="metric" fontSize={12}
            label={{ value: 'sub-flag', position: 'insideBottom', offset: -10, fontSize: 11 }} />
          <YAxis fontSize={11} domain={[0, 80]} unit="%"
            label={{ value: 'Fire rate (% of items)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <Tooltip
            cursor={{ fill: 'rgba(0,0,0,0.04)' }}
            content={({ active, payload, label }) => {
              if (!active || !payload || !payload.length) return null
              const d = payload[0].payload
              return (
                <div className="bg-white px-3 py-2 rounded shadow border border-ink/10 text-sm">
                  <div className="font-semibold">{label}</div>
                  <div className="mono text-xs mt-1" style={{ color: data.subjects.claude.color }}>
                    Claude-Sonnet: <b>{d.Claude.toFixed(1)}%</b> ({d.claude_raw})
                  </div>
                  <div className="mono text-xs" style={{ color: data.subjects.qwen.color }}>
                    Qwen3-32B (open): <b>{d.Qwen.toFixed(1)}%</b> ({d.qwen_raw})
                  </div>
                </div>
              )
            }} />
          <Legend wrapperStyle={{ fontSize: 12, paddingBottom: 4 }} verticalAlign="top" iconType="rect" />
          <Bar dataKey="Claude" name={data.subjects.claude.display} fill={data.subjects.claude.color} radius={[3,3,0,0]}
            isAnimationActive animationDuration={900}>
            <LabelList dataKey="Claude" position="top" formatter={(v) => `${v.toFixed(0)}%`} fontSize={10} fill="#1a1a2e" />
          </Bar>
          <Bar dataKey="Qwen" name={data.subjects.qwen.display} fill={data.subjects.qwen.color} radius={[3,3,0,0]}
            isAnimationActive animationDuration={900} animationBegin={120}>
            <LabelList dataKey="Qwen" position="top" formatter={(v) => `${v.toFixed(0)}%`} fontSize={10} fill="#1a1a2e" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="text-xs text-ink/60 mt-1">
        Both teachers on the scaffolded arm, audit-gated. The open Qwen teacher trades
        <b className="text-leak"> leak</b> for harm and missed-instruction — its end-to-end harm is
        slightly lower but it leaks much more, which the distilled student inherits.
      </div>
    </div>
  )
}
