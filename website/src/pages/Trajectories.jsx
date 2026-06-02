import { useEffect, useMemo, useState } from 'react'
import { useData, CELL_DOT } from '../lib/useData.js'

function FlagBadge({ on, color, label }) {
  return (
    <span className={`px-2 py-0.5 text-xs rounded-md mono ${on ? color : 'bg-stone-100 text-ink/40 line-through'}`}>{label}</span>
  )
}

function Turn({ t }) {
  const role = t.speaker
  const isAgent = role === 'agent'
  const isCounter = role === 'counterparty'
  const isPrincipal = role === 'principal'
  return (
    <div className={`flex ${isAgent ? 'justify-start' : 'justify-end'} mb-3`}>
      <div className={`max-w-[78%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
        isAgent ? 'bg-white border border-ink/15 text-ink' :
        isPrincipal ? 'bg-sanity/10 border border-sanity/30 text-ink' :
        'bg-leak/8 border border-leak/30 text-ink'
      }`}>
        <div className="text-xs font-semibold mono mb-1 opacity-60">
          {isAgent ? 'AGENT' : isCounter ? 'COUNTERPARTY' : 'PRINCIPAL'} · turn {t.turn}
        </div>
        {t.text}
      </div>
    </div>
  )
}

function TrajectoryView({ data }) {
  if (!data) return null
  return (
    <div className="bg-stone-100 border border-ink/10 rounded-xl p-4">
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className={`inline-block w-2.5 h-2.5 rounded-full ${CELL_DOT[data.cell]}`} />
        <span className="text-sm font-semibold">{data.item_title}</span>
        <span className="text-xs mono text-ink/50">{data.item_id} · {data.subject} · {data.arm}</span>
      </div>

      {data.score && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          <FlagBadge on={data.score.harm_fire} color="bg-leak text-white" label="harm" />
          <FlagBadge on={data.score.leak_rate > 0} color="bg-leak text-white" label={`leak ${data.score.leak_rate > 0 ? `(${Math.round(data.score.leak_rate*100)}%)` : ''}`} />
          <FlagBadge on={data.score.leaked_private_bound} color="bg-bound text-white" label="bound" />
          <FlagBadge on={data.score.missed_instruction} color="bg-moder text-white" label="missed-instruction" />
        </div>
      )}

      <div>
        {data.turns.map((t, i) => <Turn key={i} t={t} />)}
      </div>

      {data.score?.notes && (
        <div className="mt-4 bg-white border border-ink/10 rounded p-3 text-xs">
          <div className="font-semibold text-ink/60 mb-1">Judge notes</div>
          <div className="italic text-ink/80">{data.score.notes}</div>
        </div>
      )}
    </div>
  )
}

export default function Trajectories() {
  const { data: index } = useData('trajectories_index.json')
  const [selected, setSelected] = useState(null)
  const [traj, setTraj] = useState(null)

  useEffect(() => {
    if (!selected) { setTraj(null); return }
    fetch(`${import.meta.env.BASE_URL}data/trajectories/${selected}`)
      .then(r => r.json()).then(setTraj)
  }, [selected])

  // group by cell for the picker
  const byCell = useMemo(() => {
    if (!index) return {}
    const g = {}
    for (const e of index) (g[e.cell] = g[e.cell] || []).push(e)
    return g
  }, [index])

  // initial selection
  useEffect(() => {
    if (index && !selected && index.length) setSelected(index[0].file)
  }, [index, selected])

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-6">
        <h1 className="text-3xl font-bold serif">Sample conversations</h1>
        <p className="text-ink/70 mt-1 max-w-3xl">
          Curated trajectories — one item per cell × three subjects × three arms. Each conversation
          shows what the counterparty asked and how the agent responded; harm flags from the judge
          are shown above each transcript.
        </p>
      </div>

      <div className="grid lg:grid-cols-[260px_1fr] gap-6">
        <aside className="bg-white border border-ink/10 rounded-xl p-3 max-h-[80vh] overflow-y-auto self-start sticky top-16">
          {Object.entries(byCell).map(([cell, entries]) => (
            <div key={cell} className="mb-3">
              <div className="flex items-center gap-2 mb-1.5">
                <span className={`inline-block w-2 h-2 rounded-full ${CELL_DOT[cell]}`} />
                <span className="text-xs font-semibold uppercase tracking-wider text-ink/60">{cell}</span>
              </div>
              <div className="space-y-0.5">
                {entries.map(e => (
                  <button key={e.file} onClick={() => setSelected(e.file)}
                    className={`w-full text-left px-2 py-1 text-xs rounded transition flex items-center gap-2 ${
                      selected === e.file ? 'bg-ink text-white' : 'hover:bg-stone-100 text-ink/70'
                    }`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${e.harm_fire ? 'bg-leak' : 'bg-author'}`} />
                    <span className="mono truncate">{e.subject} · {e.arm}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </aside>

        <div>
          {traj ? <TrajectoryView data={traj} /> : <div className="text-sm text-ink/50">Loading…</div>}
        </div>
      </div>
    </div>
  )
}
