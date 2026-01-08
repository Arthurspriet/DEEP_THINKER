import { useState } from 'react'

/**
 * Shows real-time step execution details from SSE events.
 */
export default function StepExecutionPanel({ stepData }) {
  const [expanded, setExpanded] = useState(false)
  
  if (!stepData) {
    return (
      <div className="glass rounded-lg p-4 animate-pulse">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-dt-border" />
          <span className="text-sm text-dt-text-dim">Waiting for step execution...</span>
        </div>
      </div>
    )
  }
  
  const { step_name, step_type, model_used, duration_s, status, attempts, output_preview, error, pivot_suggestion } = stepData
  
  const statusConfig = {
    completed: { color: 'text-dt-success', bg: 'bg-dt-success/20', icon: '✓', pulse: false },
    running: { color: 'text-dt-accent', bg: 'bg-dt-accent/20', icon: '●', pulse: true },
    failed: { color: 'text-dt-error', bg: 'bg-dt-error/20', icon: '✕', pulse: false },
    pending: { color: 'text-dt-text-dim', bg: 'bg-dt-surface-light', icon: '○', pulse: false },
  }
  
  const config = statusConfig[status] || statusConfig.pending
  
  return (
    <div className={`glass rounded-lg overflow-hidden transition-all ${config.pulse ? 'status-running' : ''}`}>
      {/* Header */}
      <div className="p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${config.bg} ${config.pulse ? 'animate-pulse' : ''}`}>
            <span className={`text-xs ${config.color}`}>{config.icon}</span>
          </div>
          <div>
            <h4 className="text-sm font-medium text-dt-text">{step_name}</h4>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-xs text-dt-text-dim">{step_type}</span>
              {model_used && (
                <>
                  <span className="text-dt-border">•</span>
                  <span className="text-xs font-mono text-dt-accent">{model_used}</span>
                </>
              )}
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {duration_s != null && (
            <span className="text-xs text-dt-text-dim font-mono">
              {duration_s.toFixed(1)}s
            </span>
          )}
          {attempts > 1 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-dt-warning/20 text-dt-warning">
              {attempts} attempts
            </span>
          )}
          <span className={`text-xs px-2 py-0.5 rounded-full ${config.bg} ${config.color}`}>
            {status}
          </span>
        </div>
      </div>
      
      {/* Expandable output preview */}
      {(output_preview || error || pivot_suggestion) && (
        <div className="border-t border-dt-border">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full p-3 flex items-center justify-between hover:bg-dt-surface-light/30 transition-colors"
          >
            <span className="text-xs text-dt-text-dim">
              {error ? 'Error details' : pivot_suggestion ? 'Pivot suggested' : 'Output preview'}
            </span>
            <svg
              className={`w-4 h-4 text-dt-text-dim transition-transform ${expanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {expanded && (
            <div className="p-4 bg-dt-bg/50">
              {error && (
                <div className="mb-3 p-3 rounded bg-dt-error/10 border border-dt-error/30">
                  <p className="text-xs text-dt-error font-mono">{error}</p>
                </div>
              )}
              {pivot_suggestion && (
                <div className="mb-3 p-3 rounded bg-dt-warning/10 border border-dt-warning/30">
                  <p className="text-xs text-dt-warning">
                    <span className="font-medium">Pivot:</span> {pivot_suggestion}
                  </p>
                </div>
              )}
              {output_preview && (
                <pre className="text-xs text-dt-text-dim font-mono whitespace-pre-wrap max-h-40 overflow-y-auto">
                  {output_preview}
                </pre>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

