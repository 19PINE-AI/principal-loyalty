import { useEffect, useMemo, useRef, useState } from 'react'
import { useData, CELL_DOT, CELL_COLORS, ARM_LABELS } from '../lib/useData.js'

const ARMS = ['plain', 'prompted', 'scaffolded']
const ARM_SHORT = { plain: 'Plain', prompted: 'Prompted', scaffolded: 'Scaffolded' }
const CLUSTER_ORDER = { calibrated: 0, intermediate: 1, 'over-refuse': 2 }
const CLUSTER_DOT = { calibrated: 'bg-calibrated', intermediate: 'bg-intermediate', 'over-refuse': 'bg-overrefuse' }

// ---------- small pieces ----------
function Turn({ t }) {
  const isAgent = t.speaker === 'agent'
  const isPrincipal = t.speaker === 'principal'
  return (
    <div className={`flex ${isAgent ? 'justify-start' : 'justify-end'} mb-2.5`}>
      <div className={`max-w-[82%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed whitespace-pre-wrap ${
        isAgent ? 'bg-white border border-ink/15 text-ink'
          : isPrincipal ? 'bg-sanity/10 border border-sanity/30 text-ink'
          : 'bg-leak/[0.07] border border-leak/30 text-ink'
      }`}>
        <div className="text-[10px] font-semibold mono mb-1 opacity-50 tracking-wider">
          {isAgent ? 'AGENT' : isPrincipal ? 'PRINCIPAL' : 'COUNTERPARTY'}
        </div>
        {t.text || <span className="italic opacity-40">(empty turn)</span>}
      </div>
    </div>
  )
}

function Flag({ on, label, color }) {
  return (
    <span className={`px-1.5 py-0.5 text-[11px] rounded mono ${on ? color : 'bg-stone-100 text-ink/35 line-through'}`}>{label}</span>
  )
}

// One subject×arm cell in the matrix
function MatrixCell({ run, active, onClick }) {
  if (!run) return <td className="px-1 py-1 text-center text-ink/20">·</td>
  const harm = run.harm_fire
  const dots = [
    run.leak_rate > 0 && 'L',
    run.missed_instruction && 'M',
    run.leaked_private_bound && 'B',
  ].filter(Boolean)
  return (
    <td className="px-1 py-1">
      <button
        onClick={onClick}
        title={`${run.display} · ${run.arm} — ${harm ? 'harm' : 'clean'}${run.scored ? '' : ' (unscored)'}`}
        className={`w-full min-w-[58px] rounded-md px-2 py-1.5 text-xs font-medium transition border ${
          active ? 'ring-2 ring-ink ring-offset-1' : ''
        } ${
          !run.scored ? 'bg-stone-100 border-stone-200 text-ink/40'
            : harm ? 'bg-leak/15 border-leak/40 text-leak hover:bg-leak/25'
            : 'bg-author/15 border-author/40 text-author hover:bg-author/25'
        }`}>
        <span className="font-bold">{!run.scored ? '?' : harm ? '✗' : '✓'}</span>
        {dots.length > 0 && <span className="ml-1 opacity-70 mono">{dots.join('')}</span>}
      </button>
    </td>
  )
}

function ResultsMatrix({ runs, selKey, onSelect }) {
  // group runs by subject (preserving cluster order from the sorted data)
  const subjects = useMemo(() => {
    const seen = new Map()
    for (const r of runs) {
      if (!seen.has(r.subject)) seen.set(r.subject, { subject: r.subject, display: r.display, cluster: r.cluster, byArm: {} })
      seen.get(r.subject).byArm[r.arm] = r
    }
    return [...seen.values()].sort((a, b) =>
      (CLUSTER_ORDER[a.cluster] - CLUSTER_ORDER[b.cluster]) || a.display.localeCompare(b.display))
  }, [runs])

  return (
    <div className="overflow-x-auto">
      <table className="text-sm border-separate border-spacing-0">
        <thead>
          <tr>
            <th className="text-left px-2 py-1.5 text-xs font-semibold text-ink/50">Model</th>
            {ARMS.map(a => (
              <th key={a} className="px-1 py-1.5 text-xs font-semibold text-ink/60 text-center">{ARM_SHORT[a]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {subjects.map(s => (
            <tr key={s.subject}>
              <td className="px-2 py-1 whitespace-nowrap">
                <span className={`inline-block w-2 h-2 rounded-full mr-2 align-middle ${CLUSTER_DOT[s.cluster]}`} />
                <span className="text-ink/80">{s.display}</span>
              </td>
              {ARMS.map(a => {
                const run = s.byArm[a]
                const key = run ? `${s.subject}__${a}` : null
                return <MatrixCell key={a} run={run} active={key && key === selKey}
                  onClick={() => onSelect(s.subject, a)} />
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Judgment({ run }) {
  return (
    <div className="bg-white border border-ink/10 rounded-lg p-3 mb-3">
      <div className="flex flex-wrap items-center gap-1.5 mb-2">
        <span className="text-xs font-semibold text-ink/60 mr-1">Judge verdict:</span>
        <Flag on={run.harm_fire} label="harm" color="bg-leak text-white" />
        <Flag on={run.leak_rate > 0} label={`leak${run.n_withhold ? ` ${run.n_leaked}/${run.n_withhold}` : ''}`} color="bg-leak text-white" />
        <Flag on={run.missed_instruction} label="missed-instruction" color="bg-moder text-white" />
        <Flag on={run.leaked_private_bound} label="bound" color="bg-bound text-white" />
        <Flag on={run.fabrication} label="fabrication" color="bg-ink text-white" />
        <Flag on={run.against_principal} label="against-principal" color="bg-ink text-white" />
        {!run.scored && <span className="text-[11px] text-ink/40 italic">unscored trajectory</span>}
      </div>
      {run.notes && <div className="text-xs italic text-ink/75 leading-relaxed">{run.notes}</div>}
      {run.leaks?.some(l => l.evidence) && (
        <div className="mt-2 space-y-1">
          {run.leaks.filter(l => l.evidence).map((l, i) => (
            <div key={i} className="text-[11px] text-ink/60">
              <span className={`mono font-semibold ${l.leaked ? 'text-leak' : 'text-author'}`}>{l.leaked ? '✗' : '✓'} {l.fact}</span>
              <span className="ml-1.5 text-ink/50">{l.evidence}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TaskCard({ doc }) {
  const fact = doc.private_facts || []
  return (
    <div className="bg-white border border-ink/10 rounded-xl p-4 mb-5">
      <div className="text-xs font-semibold uppercase tracking-wider text-ink/50 mb-2">Principal briefing</div>
      <p className="text-sm text-ink/85 leading-relaxed whitespace-pre-wrap">{doc.principal_briefing}</p>

      {fact.length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink/50 mb-1">Withheld private facts</div>
          <ul className="text-sm text-ink/80 space-y-0.5">
            {fact.map((f, i) => (
              <li key={i}><span className="mono text-leak">{f.name}</span>{f.value ? <span className="text-ink/55"> = {String(f.value)}</span> : null}</li>
            ))}
          </ul>
        </div>
      )}
      {doc.private_bounds?.length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink/50 mb-1">Private bounds</div>
          <ul className="text-sm text-ink/80 space-y-0.5">
            {doc.private_bounds.map((b, i) => (
              <li key={i}><span className="mono text-bound">{b.name}</span> <span className="text-ink/55">{b.value} ({b.kind})</span></li>
            ))}
          </ul>
        </div>
      )}
      {doc.counterparty_opening && (
        <div className="mt-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink/50 mb-1">Counterparty opening move</div>
          <p className="text-sm italic text-ink/75 leading-relaxed">“{doc.counterparty_opening}”</p>
          {doc.counterparty_persona && <p className="text-xs text-ink/50 mt-1.5">{doc.counterparty_persona}</p>}
        </div>
      )}
    </div>
  )
}

// ---------- main page ----------
export default function Explorer() {
  const { data: index } = useData('explorer_index.json')
  const [selId, setSelId] = useState(null)
  const [doc, setDoc] = useState(null)
  const [loadingDoc, setLoadingDoc] = useState(false)
  const [cellKey, setCellKey] = useState(null) // `${subject}__${arm}`
  const [q, setQ] = useState('')
  const [cellFilter, setCellFilter] = useState('all')
  const [runsOnly, setRunsOnly] = useState(true)
  const detailRef = useRef(null)

  // initial selection: first item with runs
  useEffect(() => {
    if (index && !selId) {
      const first = index.find(e => e.n_runs) || index[0]
      if (first) setSelId(first.id)
    }
  }, [index, selId])

  // load the per-item document
  useEffect(() => {
    if (!selId) return
    setLoadingDoc(true); setDoc(null); setCellKey(null)
    let alive = true
    fetch(`${import.meta.env.BASE_URL}data/explorer/${selId}.json`)
      .then(r => r.json())
      .then(d => {
        if (!alive) return
        setDoc(d)
        // auto-select the first harm run (most interesting), else first run
        const harm = d.runs.find(r => r.harm_fire) || d.runs[0]
        if (harm) setCellKey(`${harm.subject}__${harm.arm}`)
      })
      .finally(() => alive && setLoadingDoc(false))
    return () => { alive = false }
  }, [selId])

  const cells = useMemo(() => {
    if (!index) return []
    return [...new Set(index.map(e => e.cell))]
  }, [index])

  const filtered = useMemo(() => {
    if (!index) return []
    const needle = q.trim().toLowerCase()
    return index.filter(e =>
      (cellFilter === 'all' || e.cell === cellFilter) &&
      (!runsOnly || e.n_runs > 0) &&
      (!needle || e.id.toLowerCase().includes(needle) || e.title.toLowerCase().includes(needle))
    )
  }, [index, q, cellFilter, runsOnly])

  const grouped = useMemo(() => {
    const g = {}
    for (const e of filtered) (g[e.cell] = g[e.cell] || []).push(e)
    return g
  }, [filtered])

  const selRun = useMemo(() => {
    if (!doc || !cellKey) return null
    return doc.runs.find(r => `${r.subject}__${r.arm}` === cellKey) || null
  }, [doc, cellKey])

  const selectCell = (subject, arm) => {
    setCellKey(`${subject}__${arm}`)
    if (detailRef.current) detailRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  return (
    <div className="max-w-[1500px] mx-auto px-6 py-8">
      <div className="mb-5">
        <h1 className="text-3xl font-bold serif">PrincipalBench</h1>
        <p className="text-ink/70 mt-1 max-w-3xl">
          Browse every PrincipalBench item, then inspect how each of the 13 evaluated models behaved
          under all three prompt arms — the result matrix, the judge's verdict and leak evidence, and the
          full agent/counterparty transcript behind every cell.
        </p>
      </div>

      <div className="grid lg:grid-cols-[300px_1fr] gap-6 items-start">
        {/* sidebar: item picker */}
        <aside className="bg-white border border-ink/10 rounded-xl p-3 lg:sticky lg:top-16 lg:max-h-[85vh] overflow-y-auto">
          <input
            value={q} onChange={e => setQ(e.target.value)}
            placeholder="Search items…"
            className="w-full px-3 py-2 text-sm border border-ink/15 rounded-lg mb-2 focus:outline-none focus:ring-2 focus:ring-ink/20"
          />
          <div className="flex items-center gap-2 mb-2">
            <select value={cellFilter} onChange={e => setCellFilter(e.target.value)}
              className="flex-1 text-xs border border-ink/15 rounded-md px-2 py-1.5 bg-white">
              <option value="all">All cells</option>
              {cells.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <label className="flex items-center gap-1 text-xs text-ink/60 whitespace-nowrap">
              <input type="checkbox" checked={runsOnly} onChange={e => setRunsOnly(e.target.checked)} />
              has runs
            </label>
          </div>
          <div className="text-[11px] text-ink/40 mb-2 px-1">{filtered.length} items</div>

          {Object.entries(grouped).map(([cell, items]) => (
            <div key={cell} className="mb-3">
              <div className="flex items-center gap-2 mb-1 px-1">
                <span className={`inline-block w-2 h-2 rounded-full ${CELL_DOT[cell]}`} />
                <span className="text-xs font-semibold uppercase tracking-wider text-ink/55">{cell}</span>
              </div>
              <div className="space-y-0.5">
                {items.map(e => (
                  <button key={e.id} onClick={() => setSelId(e.id)}
                    className={`w-full text-left px-2 py-1.5 rounded-md transition ${
                      selId === e.id ? 'bg-ink text-white' : 'hover:bg-stone-100'
                    }`}>
                    <div className="text-xs font-medium leading-snug">{e.title}</div>
                    <div className={`text-[10px] mono mt-0.5 flex items-center gap-1.5 ${selId === e.id ? 'text-white/60' : 'text-ink/45'}`}>
                      <span>{e.id}</span>
                      {e.n_runs > 0
                        ? <span>· {e.n_runs} runs · {e.harm_rate}% harm</span>
                        : <span className="italic">· task only</span>}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </aside>

        {/* main detail */}
        <div>
          {!doc && <div className="text-sm text-ink/50 py-20 text-center">{loadingDoc ? 'Loading item…' : 'Select an item.'}</div>}
          {doc && (
            <>
              <div className="flex flex-wrap items-center gap-2 mb-3">
                <span className={`text-xs px-2 py-0.5 rounded-md ${CELL_COLORS[doc.cell]}`}>{doc.cell}</span>
                <span className="text-xs px-2 py-0.5 rounded-md bg-stone-200 text-ink/70">{doc.split}</span>
                <h2 className="text-xl font-bold">{doc.title}</h2>
                <span className="text-xs mono text-ink/45">{doc.id}</span>
              </div>

              <TaskCard doc={doc} />

              {doc.runs.length === 0 ? (
                <div className="bg-stone-100 border border-ink/10 rounded-xl p-6 text-sm text-ink/55">
                  No model runs for this item — it is part of the released task set but outside the
                  evaluated 36-item core / 24-item held-out subsets.
                </div>
              ) : (
                <div className="grid xl:grid-cols-[minmax(0,420px)_1fr] gap-5">
                  <div>
                    <div className="text-sm font-semibold mb-2">Results matrix
                      <span className="font-normal text-ink/50 text-xs"> — {doc.runs.length} runs · click a cell</span>
                    </div>
                    <div className="bg-stone-50 border border-ink/10 rounded-xl p-3">
                      <ResultsMatrix runs={doc.runs} selKey={cellKey} onSelect={selectCell} />
                      <div className="mt-3 pt-2 border-t border-ink/10 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-ink/55">
                        <span><span className="text-author font-bold">✓</span> clean</span>
                        <span><span className="text-leak font-bold">✗</span> harm</span>
                        <span className="mono">L</span> leak
                        <span className="mono">M</span> missed-instruction
                        <span className="mono">B</span> bound
                      </div>
                    </div>
                  </div>

                  {/* transcript + judgment */}
                  <div ref={detailRef}>
                    {selRun ? (
                      <>
                        <div className="flex flex-wrap items-center gap-2 mb-2">
                          <span className={`inline-block w-2 h-2 rounded-full ${CLUSTER_DOT[selRun.cluster]}`} />
                          <span className="font-semibold text-sm">{selRun.display}</span>
                          <span className="text-xs mono text-ink/50">{ARM_LABELS[selRun.arm] || selRun.arm}</span>
                          {selRun.early_end_reason && <span className="text-[11px] text-ink/40">· {selRun.early_end_reason}</span>}
                        </div>
                        <Judgment run={selRun} />
                        <div className="bg-stone-100 border border-ink/10 rounded-xl p-3 max-h-[70vh] overflow-y-auto">
                          {selRun.turns.length
                            ? selRun.turns.map((t, i) => <Turn key={i} t={t} />)
                            : <div className="text-sm text-ink/40 italic">No transcript recorded.</div>}
                        </div>
                      </>
                    ) : <div className="text-sm text-ink/50">Select a cell in the matrix.</div>}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
