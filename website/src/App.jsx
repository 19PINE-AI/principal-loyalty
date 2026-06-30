import { Routes, Route, NavLink, Link } from 'react-router-dom'
import Overview from './pages/Overview.jsx'
import Explorer from './pages/Explorer.jsx'
import Training from './pages/Training.jsx'

const NAV = [
  { to: '/',          label: 'Overview' },
  { to: '/explorer',  label: 'PrincipalBench' },
  { to: '/training',  label: 'Training & variants' },
]

function Nav() {
  const cls = ({ isActive }) =>
    `px-3 py-2 text-sm font-medium rounded-md transition-colors ${
      isActive ? 'bg-ink text-white' : 'text-ink/70 hover:bg-ink/5 hover:text-ink'
    }`
  return (
    <header className="bg-white border-b border-ink/10 sticky top-0 z-30 backdrop-blur">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between gap-6 flex-wrap">
        <Link to="/" className="flex items-baseline gap-3 shrink-0">
          <span className="font-bold text-lg tracking-tight">Principal Loyalty</span>
          <span className="text-xs text-ink/50 mono hidden sm:inline">multi-party loyalty</span>
        </Link>
        <nav className="flex flex-wrap gap-1">
          {NAV.map(n => <NavLink key={n.to} to={n.to} end={n.to === '/'} className={cls}>{n.label}</NavLink>)}
        </nav>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="mt-20 border-t border-ink/10 bg-white">
      <div className="max-w-7xl mx-auto px-6 py-8 text-sm text-ink/60 flex flex-wrap gap-6 justify-between">
        <div>
          <div className="font-semibold text-ink/80">Principal Loyalty — Companion site</div>
          <div className="mt-1">Whose Side Is Your Agent On? Multi-Party Principal Loyalty in LLM Agents.</div>
        </div>
        <div className="text-right">
          <a href="https://github.com/19PINE-AI/principal-loyalty" target="_blank" rel="noreferrer" className="hover:text-ink">GitHub</a>
          <span className="mx-2">·</span>
          <a href="https://arxiv.org/abs/2606.30383" target="_blank" rel="noreferrer" className="hover:text-ink">Paper (arXiv)</a>
        </div>
      </div>
    </footer>
  )
}

function App() {
  return (
    <div className="min-h-full flex flex-col bg-stone-50">
      <Nav />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/training" element={<Training />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App
