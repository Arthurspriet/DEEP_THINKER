import { useState } from 'react'

/**
 * Shows council consensus mechanism results from SSE events.
 */
export default function ConsensusGauge({ consensusData }) {
  const [expanded, setExpanded] = useState(false)
  
  if (!consensusData) {
    return null
  }
  
  const { council_name, mechanism, agreement_score, model_outputs, final_decision } = consensusData
  
  const scorePercent = (agreement_score || 0) * 100
  const scoreColor = scorePercent >= 80 ? 'dt-success' : scorePercent >= 60 ? 'dt-warning' : 'dt-error'
  
  // Mechanism display names
  const mechanismLabels = {
    voting: 'Voting',
    weighted_blend: 'Weighted Blend',
    critique_exchange: 'Critique Exchange',
    semantic_distance: 'Semantic Distance',
  }
  
  return (
    <div className="glass rounded-lg overflow-hidden">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h4 className="text-sm font-medium text-dt-text flex items-center gap-2">
              <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              Consensus Result
            </h4>
            <p className="text-xs text-dt-text-dim mt-0.5">{council_name}</p>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-full bg-dt-surface-light text-dt-text-dim">
            {mechanismLabels[mechanism] || mechanism}
          </span>
        </div>
        
        {/* Agreement Score Gauge */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-dt-text-dim">Agreement Score</span>
            <span className={`text-lg font-display font-bold text-${scoreColor}`}>
              {scorePercent.toFixed(0)}%
            </span>
          </div>
          <div className="h-2 bg-dt-surface-light rounded-full overflow-hidden">
            <div
              className={`h-full bg-${scoreColor} rounded-full transition-all duration-500`}
              style={{ width: `${scorePercent}%` }}
            />
          </div>
        </div>
        
        {/* Model Outputs Summary */}
        {model_outputs && Object.keys(model_outputs).length > 0 && (
          <div className="space-y-2">
            <span className="text-xs text-dt-text-dim">Model Outputs</span>
            <div className="flex flex-wrap gap-2">
              {Object.entries(model_outputs).map(([model, output]) => (
                <div
                  key={model}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs ${
                    output.success
                      ? 'bg-dt-success/10 text-dt-success border border-dt-success/30'
                      : 'bg-dt-error/10 text-dt-error border border-dt-error/30'
                  }`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${output.success ? 'bg-dt-success' : 'bg-dt-error'}`} />
                  <span className="font-mono">{model}</span>
                  {output.duration_s && (
                    <span className="opacity-70">{output.duration_s.toFixed(1)}s</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      {/* Expandable Final Decision */}
      {final_decision && (
        <div className="border-t border-dt-border">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full p-3 flex items-center justify-between hover:bg-dt-surface-light/30 transition-colors"
          >
            <span className="text-xs text-dt-text-dim">Final Decision</span>
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
              <pre className="text-xs text-dt-text-dim font-mono whitespace-pre-wrap">
                {final_decision}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

