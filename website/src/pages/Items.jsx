import { useMemo, useState } from 'react'
import { useData, CELL_DOT } from '../lib/useData.js'

const CELLS = ['leakage','capitulation','posture','authoring','moderation','sanity']

function Pill({ active, children, onClick, color }) {
  return (
    <button onClick={onClick} className={`px-2.5 py-1 text-xs rounded-full border transition ${
      active ? 'bg-ink text-white border-ink' : 'border-ink/15 text-ink/70 hover:border-ink/40'
    }`}>
      {color && <span className={`inline-block w-2 h-2 rounded-full mr-1.5 align-middle ${color}`} />}
      {children}
    </button>
  )
}

function ItemCard({ item, onClick }) {
  return (
    <button onClick={onClick} className="text-left bg-white border border-ink/10 rounded-lg p-4 hover:border-mech1/40 hover:shadow-md transition">
      <div className="flex items-center gap-2 mb-1">
        <span className={`inline-block w-2 h-2 rounded-full ${CELL_DOT[item.cell]}`} />
        <span className="text-xs mono text-ink/50">{item.cell}{item.subcell ? ` · ${item.subcell}` : ''}</span>
        <span className={`text-xs mono ml-auto ${item.split === 'held-out' ? 'text-mech1' : 'text-ink/40'}`}>{item.split}</span>
      </div>
      <div className="font-medium text-sm leading-snug mb-1">{item.title}</div>
      <div className="text-xs mono text-ink/40">{item.id}</div>
    </button>
  )
}

function ItemDetail({ item, onClose }) {
  if (!item) return null
  return (
    <div className="fixed inset-0 bg-ink/40 z-50 flex items-stretch justify-center sm:items-center sm:p-6" onClick={onClose}>
      <div className="bg-white max-w-3xl w-full sm:rounded-xl overflow-y-auto max-h-full p-6" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className={`inline-block w-2.5 h-2.5 rounded-full ${CELL_DOT[item.cell]}`} />
            <span className="text-sm mono text-ink/60">{item.cell}{item.subcell ? ` · ${item.subcell}` : ''}</span>
            <span className={`text-xs mono ml-2 ${item.split === 'held-out' ? 'text-mech1' : 'text-ink/40'}`}>{item.split}</span>
          </div>
          <button onClick={onClose} className="text-ink/40 hover:text-ink">✕</button>
        </div>
        <h2 className="text-xl font-bold mb-1">{item.title}</h2>
        <div className="text-xs mono text-ink/50 mb-4">{item.id}</div>

        <Section title="Principal briefing">
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{item.principal_briefing}</p>
        </Section>

        {item.private_facts.length > 0 && (
          <Section title={`Private facts (${item.private_facts.length})`}>
            <div className="space-y-2">
              {item.private_facts.map(pf => (
                <div key={pf.name} className="border border-ink/10 rounded p-3 text-sm">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="mono font-semibold">{pf.name}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      pf.sharing === 'withhold' ? 'bg-leak/10 text-leak' :
                      pf.sharing === 'on_request' ? 'bg-post/10 text-post' :
                      'bg-ink/5 text-ink/60'
                    }`}>{pf.sharing}</span>
                    {pf.category && <span className="text-xs text-ink/50 mono">{pf.category}</span>}
                  </div>
                  <div className="text-ink/80">{pf.value}</div>
                  {pf.aliases.length > 0 && (
                    <div className="text-xs text-ink/50 mt-1 mono">aliases: {pf.aliases.join(', ')}</div>
                  )}
                </div>
              ))}
            </div>
          </Section>
        )}

        {item.stated_positions && item.stated_positions.length > 0 && (
          <Section title="Stated positions">
            <ul className="text-sm list-disc pl-5">
              {item.stated_positions.map(p => (
                <li key={p.name || p.statement}>
                  <span className="mono text-xs text-ink/50">{p.name || ''}</span>{' '}
                  {p.statement}{p.must_hold && <span className="text-xs text-leak ml-2">must hold</span>}
                </li>
              ))}
            </ul>
          </Section>
        )}

        {item.private_bounds && item.private_bounds.length > 0 && (
          <Section title="Private bounds">
            <ul className="text-sm list-disc pl-5">
              {item.private_bounds.map((b, i) => (
                <li key={i} className="mono text-xs">{JSON.stringify(b)}</li>
              ))}
            </ul>
          </Section>
        )}

        <Section title="Counterparty">
          <div className="text-sm">
            <div className="text-xs text-ink/50 uppercase tracking-wide font-semibold">Strategy</div>
            <div className="mono text-sm mb-2">{item.counterparty_strategy}</div>
            <div className="text-xs text-ink/50 uppercase tracking-wide font-semibold">Persona</div>
            <div className="whitespace-pre-wrap mb-2">{item.counterparty_persona}</div>
            <div className="text-xs text-ink/50 uppercase tracking-wide font-semibold">Opening move</div>
            <div className="italic border-l-2 border-ink/20 pl-3 mt-1">{item.counterparty_opening}</div>
          </div>
        </Section>

        <div className="text-xs text-ink/40 mt-4 mono">max_turns: {item.max_turns} · posture_pressure: {String(item.posture_pressure)}</div>
      </div>
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="mb-4">
      <div className="text-xs uppercase tracking-wider text-ink/50 font-semibold mb-2">{title}</div>
      {children}
    </div>
  )
}

export default function Items() {
  const { data: items } = useData('items.json')
  const [cell, setCell] = useState('all')
  const [split, setSplit] = useState('all')
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(null)

  const filtered = useMemo(() => {
    if (!items) return []
    return items.filter(it => {
      if (cell !== 'all' && it.cell !== cell) return false
      if (split !== 'all' && it.split !== split) return false
      if (query) {
        const q = query.toLowerCase()
        return (
          it.title.toLowerCase().includes(q) ||
          it.id.toLowerCase().includes(q) ||
          it.principal_briefing.toLowerCase().includes(q) ||
          (it.counterparty_persona || '').toLowerCase().includes(q)
        )
      }
      return true
    })
  }, [items, cell, split, query])

  const cellCounts = useMemo(() => {
    if (!items) return {}
    const c = {}
    for (const it of items) c[it.cell] = (c[it.cell] || 0) + 1
    return c
  }, [items])

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-6">
        <h1 className="text-3xl font-bold serif">Test items</h1>
        <p className="text-ink/70 mt-1">All {items?.length || 75} multi-turn benchmark items. Click any card to see the full briefing, private facts, bounds, and counterparty setup.</p>
      </div>

      <div className="bg-white border border-ink/10 rounded-xl p-4 mb-6 sticky top-14 z-20">
        <input
          type="text"
          placeholder="Search title, briefing, or counterparty..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          className="w-full px-3 py-2 border border-ink/15 rounded-md text-sm focus:outline-none focus:border-mech1"
        />
        <div className="mt-3 flex flex-wrap gap-1.5 items-center">
          <span className="text-xs text-ink/50 uppercase tracking-wider font-semibold mr-2">Cell:</span>
          <Pill active={cell === 'all'} onClick={() => setCell('all')}>All ({items?.length || 0})</Pill>
          {CELLS.map(c => (
            <Pill key={c} active={cell === c} onClick={() => setCell(c)} color={CELL_DOT[c]}>
              {c} ({cellCounts[c] || 0})
            </Pill>
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5 items-center">
          <span className="text-xs text-ink/50 uppercase tracking-wider font-semibold mr-2">Split:</span>
          <Pill active={split === 'all'} onClick={() => setSplit('all')}>All</Pill>
          <Pill active={split === 'training'} onClick={() => setSplit('training')}>Training (50)</Pill>
          <Pill active={split === 'held-out'} onClick={() => setSplit('held-out')}>Held-out (25)</Pill>
        </div>
      </div>

      <div className="text-sm text-ink/60 mb-3 mono">{filtered.length} item{filtered.length === 1 ? '' : 's'}</div>
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map(it => <ItemCard key={it.id} item={it} onClick={() => setSelected(it)} />)}
      </div>

      <ItemDetail item={selected} onClose={() => setSelected(null)} />
    </div>
  )
}
