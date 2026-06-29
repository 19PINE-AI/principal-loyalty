/**
 * Three-party setup: the principal (P) briefs the agent (A) over a private
 * channel and hands it secret state; the agent negotiates with an adversarial
 * counterparty (C) over a separate channel. The concrete failure modes are
 * shown in the six-cell example table beneath this figure.
 */

const Channel = ({ from, to, color, label, animated }) => (
  <g>
    <path
      d={`M ${from.x},${from.y} C ${(from.x + to.x) / 2},${from.y - 30} ${(from.x + to.x) / 2},${to.y - 30} ${to.x},${to.y}`}
      fill="none" stroke={color} strokeWidth={2}
      markerEnd={`url(#arrow-${color.replace('#', '')})`}>
      {animated && <animate attributeName="stroke-dashoffset" from="20" to="0" dur="1s" repeatCount="indefinite" />}
    </path>
    <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 26} textAnchor="middle" fontSize="10.5"
      className="mono" fill="#64748b">{label}</text>
  </g>
)

const Party = ({ x, y, w = 124, h = 70, label, sub, fill, stroke, badge }) => (
  <g>
    <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx="10" fill={fill} stroke={stroke} strokeWidth="1.8" />
    <text x={x} y={y - 4} textAnchor="middle" fontSize="17" fontWeight="700" fill={stroke}>{label}</text>
    <text x={x} y={y + 15} textAnchor="middle" fontSize="11.5" fill="#475569">{sub}</text>
    {badge && (
      <g>
        <circle cx={x + w / 2 - 12} cy={y - h / 2 + 12} r="10" fill="#fef3c7" stroke="#d97706" strokeWidth="1.5" />
        <text x={x + w / 2 - 12} y={y - h / 2 + 16} textAnchor="middle" fontSize="11">🔒</text>
      </g>
    )}
  </g>
)

export default function ProblemSchematic() {
  return (
    <div className="bg-white border border-ink/10 rounded-xl p-4">
      <div className="text-center mb-1">
        <div className="text-sm uppercase tracking-wider text-ink/50 font-semibold">The setup</div>
        <div className="text-ink/70 text-sm mt-1 max-w-xl mx-auto">
          The agent works two channels at once: a <span className="text-mech1 font-medium">private</span> line to the
          principal it represents, and an <span className="text-leak font-medium">adversarial</span> line to a
          counterparty who wants what the principal is protecting.
        </div>
      </div>

      <svg viewBox="0 80 720 176" className="w-full max-w-2xl mx-auto block">
        <defs>
          {['7c3aed', 'dc2626'].map(c => (
            <marker key={c} id={`arrow-#${c}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={`#${c}`} />
            </marker>
          ))}
        </defs>

        {/* principal <-> agent (private) */}
        <Channel from={{ x: 132, y: 130 }} to={{ x: 358, y: 130 }} color="#7c3aed" label="briefings · follow-ups" />
        <Channel from={{ x: 358, y: 175 }} to={{ x: 132, y: 175 }} color="#7c3aed" label="results · status" />

        {/* agent <-> counterparty (adversarial) */}
        <Channel from={{ x: 362, y: 130 }} to={{ x: 588, y: 130 }} color="#dc2626" label="public negotiation" />
        <Channel from={{ x: 588, y: 175 }} to={{ x: 362, y: 175 }} color="#dc2626" label="probes · pressure" animated />

        <Party x={130} y={152} label="P" sub="Principal" fill="#f5f3ff" stroke="#7c3aed" badge />
        <Party x={360} y={152} label="A" sub="Agent" fill="#fafafa" stroke="#1a1a2e" />
        <Party x={590} y={152} label="C" sub="Counterparty" fill="#fef2f2" stroke="#dc2626" />

        <g>
          <text x={130} y={232} textAnchor="middle" fontSize="11" fontWeight="600" fill="#6b21a8">private to the principal</text>
          <text x={130} y={247} textAnchor="middle" fontSize="10" fill="#6b21a8" opacity="0.75">secret facts · price limits · positions</text>
        </g>
        <g>
          <text x={590} y={232} textAnchor="middle" fontSize="11" fontWeight="600" fill="#991b1b">the counterparty</text>
          <text x={590} y={247} textAnchor="middle" fontSize="10" fill="#991b1b" opacity="0.75">probes · flatters · pressures</text>
        </g>
      </svg>

      <div className="mt-2 text-xs text-ink/65 max-w-2xl mx-auto bg-stone-50 border border-ink/10 rounded-lg px-3 py-2 leading-relaxed">
        <span className="font-semibold text-ink/75">Running example.</span> You're selling a car (the principal).
        You brief the agent with a secret floor — <span className="mono">"don't go below $12,000"</span> — and the
        buyer (the counterparty) keeps pressing the agent for that number. The six ways the agent can fail are below.
      </div>
    </div>
  )
}
