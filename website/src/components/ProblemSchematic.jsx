import { useState } from 'react'

/**
 * Interactive recreation of arxiv_fig0_problem: a three-party schematic with
 * principal (P), agent (A), and counterparty (C). Two parallel channels run
 * back and forth; the P-A channel carries private state (lock icon), the A-C
 * channel is adversarial. Click any failure axis at the bottom to highlight
 * the parts of the diagram that fail it.
 */

const AXES = [
  { id: 'leak',     label: 'Leakage',      desc: 'C extracts a withheld fact from S_P.',                   highlight: ['lock', 'a2c'] },
  { id: 'bound',    label: 'Bound breach', desc: 'A names or accepts a price inside P\'s private floor.',   highlight: ['lock', 'a2c'] },
  { id: 'capit',    label: 'Capitulation', desc: 'A folds on P\'s stated position under pressure from C.',  highlight: ['c2a', 'a2c'] },
  { id: 'refuse',   label: 'Over-refuse',  desc: 'A blanket-refuses C\'s legitimate, P-authorized asks.',   highlight: ['c2a'] },
]

const Channel = ({ from, to, color, label, dashed, highlight, delay }) => (
  <g>
    <path
      d={`M ${from.x},${from.y} C ${(from.x+to.x)/2},${from.y-30} ${(from.x+to.x)/2},${to.y-30} ${to.x},${to.y}`}
      fill="none"
      stroke={highlight ? color : `${color}66`}
      strokeWidth={highlight ? 2.5 : 1.5}
      strokeDasharray={dashed ? '6 4' : 'none'}
      markerEnd={`url(#arrow-${color.replace('#','')})`}
      style={{ transition: 'stroke 0.3s, stroke-width 0.3s' }}
    >
      {highlight && <animate attributeName="stroke-dashoffset" from="20" to="0" dur="0.8s" repeatCount="indefinite" />}
    </path>
    <text x={(from.x+to.x)/2} y={(from.y+to.y)/2 - 26} textAnchor="middle" fontSize="10.5"
          className="mono" fill={highlight ? color : '#64748b'}>{label}</text>
  </g>
)

const Party = ({ x, y, w=120, h=68, label, sub, fill, stroke, badge }) => (
  <g>
    <rect x={x-w/2} y={y-h/2} width={w} height={h} rx="10" fill={fill} stroke={stroke} strokeWidth="1.8" />
    <text x={x} y={y-6} textAnchor="middle" fontSize="18" fontWeight="700" fill={stroke}>{label}</text>
    <text x={x} y={y+14} textAnchor="middle" fontSize="11" fill="#475569">{sub}</text>
    {badge && (
      <g>
        <circle cx={x+w/2 - 12} cy={y - h/2 + 12} r="10" fill="#fef3c7" stroke="#d97706" strokeWidth="1.5" />
        <text x={x+w/2 - 12} y={y - h/2 + 16} textAnchor="middle" fontSize="11">🔒</text>
      </g>
    )}
  </g>
)

export default function ProblemSchematic() {
  const [axis, setAxis] = useState(null)
  const hi = axis ? AXES.find(a => a.id === axis).highlight : []
  const isHi = id => hi.includes(id)

  return (
    <div className="bg-white border border-ink/10 rounded-xl p-4">
      <div className="text-center mb-1">
        <div className="text-sm uppercase tracking-wider text-ink/50 font-semibold">The setup</div>
        <div className="text-ink/70 text-sm mt-1 max-w-xl mx-auto">
          The agent runs two parallel channels: a private back-and-forth with the principal
          and an adversarial conversation with the counterparty. Click a failure axis to highlight
          where it lives in the diagram.
        </div>
      </div>

      <svg viewBox="0 98 720 162" className="w-full max-w-2xl mx-auto block">
        <defs>
          {['1a1a2e','dc2626','7c3aed','0891b2','94a3b8'].map(c => (
            <marker key={c} id={`arrow-#${c}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={`#${c}`} />
            </marker>
          ))}
        </defs>

        {/* Channels — principal-agent */}
        <Channel from={{x:130,y:130}} to={{x:360,y:130}} color="#7c3aed" label="briefings · follow-ups" highlight={isHi('p2a') || (!axis)} />
        <Channel from={{x:360,y:175}} to={{x:130,y:175}} color="#7c3aed" label="results · status"        highlight={isHi('p2a') || (!axis)} />

        {/* Channels — agent-counterparty */}
        <Channel from={{x:360,y:130}} to={{x:590,y:130}} color="#dc2626" label="public negotiation" highlight={isHi('a2c')} />
        <Channel from={{x:590,y:175}} to={{x:360,y:175}} color="#dc2626" label="probes · pressure"  highlight={isHi('c2a')} />

        {/* Parties */}
        <Party x={130} y={152} label="P" sub="Principal" fill="#f5f3ff" stroke="#7c3aed" badge={isHi('lock') || true} />
        <Party x={360} y={152} label="A" sub="Agent" fill="#fafafa" stroke="#1a1a2e" />
        <Party x={590} y={152} label="C" sub="Counterparty" fill="#fef2f2" stroke="#dc2626" />

        {/* Private state hint */}
        <g>
          <text x={130} y={232} textAnchor="middle" fontSize="11" className="mono" fill="#6b21a8">S_P (private)</text>
          <text x={130} y={246} textAnchor="middle" fontSize="10" fill="#6b21a8" opacity="0.7">facts · bounds · positions</text>
        </g>
        <g>
          <text x={590} y={232} textAnchor="middle" fontSize="11" className="mono" fill="#991b1b">adversarial</text>
          <text x={590} y={246} textAnchor="middle" fontSize="10" fill="#991b1b" opacity="0.7">probes · flatters · pressures</text>
        </g>
      </svg>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2 max-w-3xl mx-auto">
        {AXES.map(a => (
          <button
            key={a.id}
            onClick={() => setAxis(axis === a.id ? null : a.id)}
            className={`p-3 rounded-lg text-left border transition ${
              axis === a.id
                ? 'border-leak bg-leak/8'
                : 'border-ink/10 hover:border-ink/30 bg-white'
            }`}>
            <div className="text-xs font-semibold mono uppercase tracking-wider text-ink/60 mb-0.5">{a.label}</div>
            <div className="text-xs text-ink/70 leading-snug">{a.desc}</div>
          </button>
        ))}
      </div>
    </div>
  )
}
