import { useEffect, useState } from 'react'

const cache = new Map()

export function useData(name) {
  const [data, setData] = useState(cache.get(name) || null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    if (cache.has(name)) return
    let alive = true
    fetch(`${import.meta.env.BASE_URL}data/${name}`)
      .then(r => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
        return r.json()
      })
      .then(d => { if (alive) { cache.set(name, d); setData(d) } })
      .catch(e => { if (alive) setErr(e) })
    return () => { alive = false }
  }, [name])

  return { data, err, loading: !data && !err }
}

export const CELL_COLORS = {
  leakage:     'bg-leak text-white',
  capitulation:'bg-capit text-white',
  posture:     'bg-post text-ink',
  authoring:   'bg-author text-white',
  moderation:  'bg-moder text-white',
  sanity:      'bg-sanity text-white',
}

export const CELL_DOT = {
  leakage:     'bg-leak',
  capitulation:'bg-capit',
  posture:     'bg-post',
  authoring:   'bg-author',
  moderation:  'bg-moder',
  sanity:      'bg-sanity',
}

export const CLUSTER_COLORS = {
  calibrated:  '#16a34a',
  intermediate:'#ca8a04',
  'over-refuse':'#dc2626',
}

export const ARM_LABELS = {
  plain: 'Plain (no instructions)',
  prompted: 'Prompted (loyalty scaffold)',
  scaffolded: 'Scaffolded (+ reader sentinel)',
}
