import { useState } from 'react'

/**
 * Shows multi-view analysis comparing Optimist vs Skeptic perspectives.
 */
export default function MultiViewAnalysis({ multiViewData }) {
  const [expanded, setExpanded] = useState(false)
  
  if (!multiViewData) {
    return null
  }
  
  const {
    agreement,
    optimist_confidence,
    skeptic_confidence,
    optimist_opportunities,
    skeptic_risks,
    high_agreement,
    confidence_gap
  } = multiViewData
  
  const agreementPercent = (agreement || 0) * 100
  const agreementColor = agreementPercent >= 80 ? 'dt-success' : agreementPercent >= 50 ? 'dt-warning' : 'dt-error'
  
  return (
    <div className="glass rounded-lg overflow-hidden">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-dt-text flex items-center gap-2">
            <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Multi-View Analysis
          </h4>
          {high_agreement ? (
            <span className="text-xs px-2 py-0.5 rounded-full bg-dt-success/10 text-dt-success border border-dt-success/30">
              High Agreement
            </span>
          ) : (
            <span className="text-xs px-2 py-0.5 rounded-full bg-dt-warning/10 text-dt-warning border border-dt-warning/30">
              Perspectives Differ
            </span>
          )}
        </div>
        
        {/* Agreement Score Gauge */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-dt-text-dim">Agreement Score</span>
            <span className={`text-lg font-display font-bold text-${agreementColor}`}>
              {agreementPercent.toFixed(0)}%
            </span>
          </div>
          <div className="h-2 bg-dt-surface-light rounded-full overflow-hidden">
            <div
              className={`h-full bg-${agreementColor} rounded-full transition-all duration-500`}
              style={{ width: `${agreementPercent}%` }}
            />
          </div>
        </div>
        
        {high_agreement ? (
          /* High Agreement - Unified View */
          <div className="space-y-2">
            <div className="text-xs text-dt-text-dim mb-2">
              ✅ High Agreement - Unified Summary
            </div>
            <div className="text-xs text-dt-text">
              <span className="text-dt-text-dim">Combined confidence: </span>
              <span className="font-mono">{((optimist_confidence + skeptic_confidence) / 2).toFixed(2)}</span>
            </div>
            {(optimist_opportunities?.length > 0 || skeptic_risks?.length > 0) && (
              <div className="pt-2 border-t border-dt-border">
                <span className="text-xs text-dt-text-dim">Key Points:</span>
                <div className="mt-1 space-y-1">
                  {optimist_opportunities?.slice(0, 2).map((opp, idx) => (
                    <div key={idx} className="text-xs text-dt-success flex items-start gap-1">
                      <span>+</span>
                      <span>{opp}</span>
                    </div>
                  ))}
                  {skeptic_risks?.slice(0, 2).map((risk, idx) => (
                    <div key={idx} className="text-xs text-dt-error flex items-start gap-1">
                      <span>-</span>
                      <span>{risk}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Low Agreement - Side by Side */
          <div className="space-y-3">
            <div className="text-xs text-dt-text-dim mb-2">
              ⚔️ Perspectives Differ - Detailed View
            </div>
            
            {/* Optimist Perspective */}
            <div className="p-2 rounded bg-dt-success/5 border border-dt-success/20">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-dt-success">🌟 OPTIMIST</span>
                <span className="text-xs text-dt-text-dim">(Confidence: {optimist_confidence?.toFixed(2) || 'N/A'})</span>
              </div>
              {optimist_opportunities?.length > 0 ? (
                <div className="space-y-1">
                  {optimist_opportunities.slice(0, 3).map((opp, idx) => (
                    <div key={idx} className="text-xs text-dt-success flex items-start gap-1">
                      <span>+</span>
                      <span>{opp}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-dt-text-dim">No opportunities listed</div>
              )}
            </div>
            
            {/* Skeptic Perspective */}
            <div className="p-2 rounded bg-dt-error/5 border border-dt-error/20">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-dt-error">🔍 SKEPTIC</span>
                <span className="text-xs text-dt-text-dim">(Confidence: {skeptic_confidence?.toFixed(2) || 'N/A'})</span>
              </div>
              {skeptic_risks?.length > 0 ? (
                <div className="space-y-1">
                  {skeptic_risks.slice(0, 3).map((risk, idx) => (
                    <div key={idx} className="text-xs text-dt-error flex items-start gap-1">
                      <span>-</span>
                      <span>{risk}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-dt-text-dim">No risks listed</div>
              )}
            </div>
            
            {/* Confidence Gap */}
            {confidence_gap != null && (
              <div className="pt-2 border-t border-dt-border">
                <div className="text-xs text-dt-text-dim">
                  <span>Confidence gap: </span>
                  <span className="font-mono text-dt-warning">{confidence_gap.toFixed(2)}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Expandable detailed view */}
      {(optimist_opportunities?.length > 3 || skeptic_risks?.length > 3) && (
        <div className="border-t border-dt-border">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full p-3 flex items-center justify-between hover:bg-dt-surface-light/30 transition-colors"
          >
            <span className="text-xs text-dt-text-dim">View All Points</span>
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
            <div className="p-4 bg-dt-bg/50 space-y-3">
              {optimist_opportunities?.length > 3 && (
                <div>
                  <div className="text-xs font-medium text-dt-success mb-1">All Opportunities:</div>
                  <div className="space-y-1">
                    {optimist_opportunities.slice(3).map((opp, idx) => (
                      <div key={idx} className="text-xs text-dt-success flex items-start gap-1">
                        <span>+</span>
                        <span>{opp}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {skeptic_risks?.length > 3 && (
                <div>
                  <div className="text-xs font-medium text-dt-error mb-1">All Risks:</div>
                  <div className="space-y-1">
                    {skeptic_risks.slice(3).map((risk, idx) => (
                      <div key={idx} className="text-xs text-dt-error flex items-start gap-1">
                        <span>-</span>
                        <span>{risk}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

