import { useState } from 'react'

export default function CouncilModal({ data, onClose }) {
  const [expandedOutput, setExpandedOutput] = useState(null)
  const { phaseName, councilName, traces, artifacts, iterations } = data

  // Group traces by model
  const tracesByModel = {}
  traces.forEach(trace => {
    const model = trace.data?.chosen_model || trace.data?.models_used?.[0] || 'unknown'
    if (!tracesByModel[model]) {
      tracesByModel[model] = []
    }
    tracesByModel[model].push(trace)
  })

  // Calculate agreement score (mock based on trace data)
  const successCount = traces.filter(t => t.data?.success !== false).length
  const agreementScore = traces.length > 0 ? Math.round((successCount / traces.length) * 100) : 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="glass-strong rounded-2xl w-full max-w-4xl max-h-[85vh] overflow-hidden animate-slide-up flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-dt-border flex items-start justify-between shrink-0">
          <div>
            <h2 className="text-xl font-display font-bold text-gradient">{phaseName}</h2>
            <p className="text-dt-text-dim text-sm mt-1">
              Council: <span className="text-dt-accent">{councilName}</span>
              {iterations > 0 && <span className="ml-3">{iterations} iterations</span>}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-dt-surface-light transition-colors"
          >
            <svg className="w-5 h-5 text-dt-text-dim" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Agreement score */}
          <div className="glass rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-dt-text">Agreement Score</h3>
              <span className={`text-2xl font-display font-bold ${
                agreementScore >= 80 ? 'text-dt-success' :
                agreementScore >= 60 ? 'text-dt-warning' : 'text-dt-error'
              }`}>
                {agreementScore}%
              </span>
            </div>
            <div className="w-full bg-dt-surface-light rounded-full h-2 overflow-hidden">
              <div
                className={`h-full transition-all ${
                  agreementScore >= 80 ? 'bg-dt-success' :
                  agreementScore >= 60 ? 'bg-dt-warning' : 'bg-dt-error'
                }`}
                style={{ width: `${agreementScore}%` }}
              />
            </div>
          </div>

          {/* Model outputs table */}
          <div>
            <h3 className="text-sm font-medium text-dt-text mb-3">Model Outputs</h3>
            <div className="space-y-3">
              {Object.entries(tracesByModel).map(([model, modelTraces]) => (
                <div key={model} className="glass rounded-lg overflow-hidden">
                  <div className="p-4 flex items-center justify-between bg-dt-surface-light/50">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full bg-dt-accent" />
                      <span className="font-mono text-sm text-dt-text">{model}</span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-dt-text-dim">
                      <span>{modelTraces.length} trace(s)</span>
                      {modelTraces[0]?.data?.duration_s && (
                        <span>{modelTraces[0].data.duration_s.toFixed(2)}s</span>
                      )}
                    </div>
                  </div>
                  
                  {modelTraces.map((trace, idx) => (
                    <div key={idx} className="border-t border-dt-border">
                      <button
                        onClick={() => setExpandedOutput(
                          expandedOutput === `${model}-${idx}` ? null : `${model}-${idx}`
                        )}
                        className="w-full p-3 flex items-center justify-between hover:bg-dt-surface-light/30 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            trace.data?.status === 'completed' || trace.data?.success
                              ? 'bg-dt-success/20 text-dt-success'
                              : trace.data?.status === 'failed' || trace.data?.error
                                ? 'bg-dt-error/20 text-dt-error'
                                : 'bg-dt-warning/20 text-dt-warning'
                          }`}>
                            {trace.data?.status || (trace.data?.success ? 'success' : 'unknown')}
                          </span>
                          <span className="text-xs text-dt-text-dim font-mono">
                            {new Date(trace.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        <svg
                          className={`w-4 h-4 text-dt-text-dim transition-transform ${
                            expandedOutput === `${model}-${idx}` ? 'rotate-180' : ''
                          }`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                      
                      {expandedOutput === `${model}-${idx}` && (
                        <div className="p-4 bg-dt-bg/50 border-t border-dt-border">
                          <pre className="text-xs text-dt-text-dim font-mono whitespace-pre-wrap overflow-x-auto max-h-60">
                            {JSON.stringify(trace.data, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ))}
              
              {Object.keys(tracesByModel).length === 0 && (
                <div className="text-center py-8 text-dt-text-dim">
                  No trace data available
                </div>
              )}
            </div>
          </div>

          {/* Artifacts */}
          {Object.keys(artifacts).length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-dt-text mb-3">Phase Artifacts</h3>
              <div className="space-y-2">
                {Object.entries(artifacts).map(([name, content]) => (
                  <div key={name} className="glass rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-mono text-dt-accent">{name}</span>
                      <span className="text-xs text-dt-text-dim">
                        {content.length} chars
                      </span>
                    </div>
                    <pre className="text-xs text-dt-text-dim font-mono whitespace-pre-wrap max-h-32 overflow-y-auto">
                      {content.slice(0, 500)}{content.length > 500 ? '...' : ''}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-dt-border shrink-0">
          <button
            onClick={onClose}
            className="w-full py-3 rounded-lg bg-dt-surface-light hover:bg-dt-border transition-colors text-dt-text font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

