import React from 'react'
import ClaimSubmission from './components/ClaimSubmission'
import ClaimAnalysisPage from './components/ClaimAnalysisPage'
import './App.css'

function App() {
  const [path, setPath] = React.useState(window.location.pathname)
  React.useEffect(() => { const update = () => setPath(window.location.pathname); window.addEventListener('popstate', update); return () => window.removeEventListener('popstate', update) }, [])
  if (path.includes('/analysis')) return <main className="app-shell"><ClaimAnalysisPage /></main>
  return (
    <main className="app-shell">
      <header className="app-header"><div className="mark" aria-hidden="true">C</div><div><p className="brand">ClaimShield</p><p className="small-line">Temporal Cross-Modal Guardrail Framework</p><p className="tiny-line">AI-assisted insurance claim security prototype</p></div><div className="status-strip"><span>● Backend Connected</span><span>● ML Guard Active</span><span>● OCR Ready</span><span>● Groq Connected</span></div></header>
      <ClaimSubmission />
    </main>
  )
}

export default App
