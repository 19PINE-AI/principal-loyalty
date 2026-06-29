# Principal Loyalty companion site

A static React app that lets you browse the 75 benchmark items, compare the 13 frontier models, inspect the per-token-KL training runs, and read sample agent/counterparty conversations from the paper *Whose Side Is Your Agent On? Multi-Party Principal Loyalty in LLM Agents*.

## Stack

- **Vite 5 + React 19** — static build, no server runtime
- **Tailwind CSS 3** — utility-first styling
- **Recharts** — scatter, bar, and line charts
- **React Router** (hash-router) — works on any static host without server-side rewrites

## Local dev

```bash
cd website
npm install
npm run dev    # http://localhost:5173
```

## Build

```bash
npm run build  # outputs to website/dist/
npm run preview
```

The build uses `base: './'` so the `dist/` folder can be served from any subpath (GitHub Pages, `file://`, S3, etc.).

## Refreshing data

The site reads JSON from `public/data/`. Re-extract from the canonical paper/runs:

```bash
# from the project root
python3 scripts/build_website_data.py
```

This rebuilds:

- `items.json`, `items_held.json` — 75 test items
- `subjects.json`, `subject_arms.json`, `subject_held.json` — 13-subject grid
- `variants.json` — distillation variant ladder
- `kiter.json` — K-iteration trajectories (Qwen and Llama)
- `cells.json`, `manifold.json`, `headline.json` — taxonomy and hero numbers
- `trajectories/*.json` + `trajectories_index.json` — curated sample conversations

The script reads `items/v0`, `items/v0_75`, `runs/phase4_promptv4_*/scored.jsonl`, and `runs/phase4_promptv4_*/trajectories.jsonl` from the parent repo.

## Pages

- **Overview** (`/`) — six failure cells, headline numbers, leak/MI floor scatter
- **Principal Bench** (`/explorer`) — searchable catalog of every benchmark item plus the 13-model × 3-arm result matrix and the full agent transcript + judge verdict (with leak evidence) behind every cell
- **Training & variants** (`/training`) — distillation variant ladder and K-iteration line charts

Principal Bench reads `explorer_index.json` and per-item `explorer/<id>.json` files (item meta + every model×arm transcript, judge sub-flags, and leak evidence), emitted by `scripts/build_website_data.py`.

## Deploy

Anything that serves a static folder works. Some quick recipes:

**GitHub Pages (project site):**
```bash
npm run build
# commit dist/ to a gh-pages branch and push, or use a workflow.
```

**Vercel / Netlify / Cloudflare Pages:**
Point the build command at `npm run build` and publish `dist/`.

**Plain HTTP server:**
```bash
cd dist && python3 -m http.server 8000
```
