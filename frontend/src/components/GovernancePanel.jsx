import { useState } from 'react'

/**
 * GovernancePanel - Shows governance verdicts, violations, and retry status
 * Displays epistemic risk, violation details, and governance decisions
 */
export default function GovernancePanel({ governanceData, className = '' }) {
  const [expanded, setExpanded] = useState(false)
  
  if (!governanceData) {
    return null
  }
  
  const {
    phase_name,
    verdict,
    violations,
    violation_details,
    retry_count,
    max_retries,
    retry_reason,
    force_web_search,
    epistemic_risk_score
  } = governanceData
  
  // Verdict styling
  const getVerdictConfig = (v) => {
    switch (v?.toUpperCase()) {
      case 'PASS':
        return {
          bg: 'bg-emerald-500/10',
          border: 'border-emerald-500/50',
          text: 'text-emerald-400',
          icon: '✓',
          glow: 'shadow-emerald-500/20'
        }
      case 'BLOCK':
        return {
          bg: 'bg-amber-500/10',
          border: 'border-amber-500/50',
          text: 'text-amber-400',
          icon: '⛔',
          glow: 'shadow-amber-500/20'
        }
      case 'WARN':
        return {
          bg: 'bg-yellow-500/10',
          border: 'border-yellow-500/50',
          text: 'text-yellow-400',
          icon: '⚠',
          glow: 'shadow-yellow-500/20'
        }
      default:
        return {
          bg: 'bg-slate-500/10',
          border: 'border-slate-500/50',
          text: 'text-slate-400',
          icon: '○',
          glow: ''
        }
    }
  }
  
  const config = getVerdictConfig(verdict)
  const riskPercent = (epistemic_risk_score || 0) * 100
  const hasViolations = violation_details && violation_details.length > 0
  
  // Risk color based on score
  const getRiskColor = (score) => {
    if (score >= 0.7) return 'text-red-400'
    if (score >= 0.5) return 'text-amber-400'
    if (score >= 0.3) return 'text-yellow-400'
    return 'text-emerald-400'
  }
  
  return (
    <div className={`rounded-xl border ${config.border} ${config.bg} shadow-lg ${config.glow} overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`text-lg ${config.text}`}>{config.icon}</span>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Governance
            </h3>
            {phase_name && (
              <span className="text-xs text-white/40">{phase_name}</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${config.bg} ${config.text} border ${config.border}`}>
            {verdict || 'PENDING'}
          </span>
        </div>
      </div>
      
      {/* Stats Row */}
      <div className="px-4 py-3 grid grid-cols-3 gap-4 border-b border-white/5">
        {/* Violations */}
        <div className="text-center">
          <div className={`text-xl font-mono font-bold ${violations > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
            {violations || 0}
          </div>
          <div className="text-xs text-white/40 uppercase tracking-wide">Violations</div>
        </div>
        
        {/* Retry */}
        <div className="text-center">
          <div className="text-xl font-mono font-bold text-white/70">
            {retry_count || 0}<span className="text-white/30">/{max_retries || 3}</span>
          </div>
          <div className="text-xs text-white/40 uppercase tracking-wide">Retries</div>
        </div>
        
        {/* Risk Score */}
        <div className="text-center">
          <div className={`text-xl font-mono font-bold ${getRiskColor(epistemic_risk_score || 0)}`}>
            {riskPercent.toFixed(0)}%
          </div>
          <div className="text-xs text-white/40 uppercase tracking-wide">Risk</div>
        </div>
      </div>
      
      {/* Risk Bar */}
      <div className="px-4 py-2 border-b border-white/5">
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-500 rounded-full ${
              riskPercent >= 70 ? 'bg-red-500' :
              riskPercent >= 50 ? 'bg-amber-500' :
              riskPercent >= 30 ? 'bg-yellow-500' :
              'bg-emerald-500'
            }`}
            style={{ width: `${Math.min(riskPercent, 100)}%` }}
          />
        </div>
      </div>
      
      {/* Retry Reason */}
      {retry_reason && (
        <div className="px-4 py-2 border-b border-white/5 flex items-center gap-2">
          <span className="text-xs text-white/40">Retry Reason:</span>
          <span className="text-xs font-mono text-amber-400/80">{retry_reason}</span>
        </div>
      )}
      
      {/* Force Web Search Badge */}
      {force_web_search && (
        <div className="px-4 py-2 border-b border-white/5">
          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Web Search Required
          </span>
        </div>
      )}
      
      {/* Violations List */}
      {hasViolations && (
        <div className="px-4 py-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-between text-left hover:bg-white/5 rounded-lg px-2 py-1.5 transition-colors -mx-2"
          >
            <span className="text-xs text-white/50 font-medium">
              Violation Details ({violation_details.length})
            </span>
            <svg
              className={`w-4 h-4 text-white/30 transition-transform ${expanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {expanded && (
            <div className="mt-2 space-y-1.5">
              {violation_details.map((v, idx) => (
                <div 
                  key={idx}
                  className="flex items-center justify-between px-2 py-1.5 rounded-lg bg-white/5"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-amber-400/60 text-xs">•</span>
                    <span className="text-xs text-white/60 font-mono">
                      {v.type || v}
                    </span>
                  </div>
                  {v.severity && (
                    <span className={`text-xs font-mono ${
                      v.severity >= 0.7 ? 'text-red-400' :
                      v.severity >= 0.5 ? 'text-amber-400' :
                      'text-yellow-400'
                    }`}>
                      {(v.severity * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

