import { useEffect, useState } from 'react'
import { AnalysisResults } from './ClaimSubmission'

export default function ClaimAnalysisPage() {
  const [result, setResult] = useState(() => JSON.parse(sessionStorage.getItem('claimshield-analysis') || 'null'))
  useEffect(() => { if (!result) setResult(JSON.parse(sessionStorage.getItem('claimshield-analysis') || 'null')) }, [result])
  if (!result) return <section className="submission-card"><h2>Analysis not found.</h2><a href="/">Return to Claim Submission</a></section>
  return <><button type="button" onClick={() => { window.history.pushState({}, '', '/'); window.dispatchEvent(new PopStateEvent('popstate')) }}>← New Claim</button><AnalysisResults result={result} /></>
}
