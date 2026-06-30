# Companion website

`website/` is a static React app (titled **Principal Loyalty**) that lets you
browse the 75 benchmark items, inspect how the 13 frontier models behave under
each prompt arm (the **PrincipalBench** explorer), explore the per-token-KL
training runs, and read sample agent/counterparty conversations from the paper.

## Stack

- **Vite 5 + React 19** — static build, no server runtime
- **Tailwind CSS 3** — styling
- **Recharts** — scatter / bar / line charts
- **React Router** (hash router) — works on any static host without rewrites

## Local development

```bash
cd website
npm install
npm run dev        # http://localhost:5173
```

## Build

```bash
npm run build      # outputs to website/dist/
npm run preview
```

The build uses `base: './'`, so `dist/` can be served from any subpath
(GitHub Pages, `file://`, S3, …).

## Refreshing data

The site's data is generated from the run outputs by
`scripts/build_website_data.py`, which writes the JSON the app consumes.
Re-run it after regenerating `runs/` to refresh the charts and conversation
samples. See `website/README.md` for the data-refresh details.
