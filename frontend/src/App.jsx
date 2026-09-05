import ClaimSubmission from './components/ClaimSubmission'
import './App.css'

function App() {
  return (
    <main className="app-shell">
      <header className="app-header"><div className="mark" aria-hidden="true">C</div><div><p className="brand">ClaimShield</p><p className="small-line">Temporal Cross-Modal Guardrail Framework</p><p className="tiny-line">AI-assisted insurance claim security prototype</p></div><div className="status-strip"><span>● Backend Connected</span><span>● ML Guard Active</span><span>● OCR Ready</span><span>● Groq Connected</span></div></header>
      <ClaimSubmission />
    </main>
  )
}

export default App
