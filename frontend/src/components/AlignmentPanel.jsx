import { useState, useMemo } from 'react'

/**
 * AlignmentPanel - Shows alignment control layer status with trajectory visualization
 * Displays alignment score, drift warnings, and corrective actions
 */
export default function AlignmentPanel({ 
  alignmentData,
  sseAlignment,
  sseCorrections = [],
  className = '' 
}) {
  const [expanded, setExpanded] = useState(false)
  
  // Merge REST data with SSE updates
  const mergedData = useMemo(() => {
    if (!alignmentData && !sseAlignment) {
      return null
    }
    
    // Start with REST data
    const base = alignmentData || {
      enabled: false,
      trajectory: [],
      actions_taken: [],
      summary: { status: 'no_data' }
    }
    
    // If we have SSE updates, merge them
    if (sseAlignment) {
      // Add SSE update as latest trajectory point if newer
      const trajectory = [...(base.trajectory || [])]
      const lastPoint = trajectory[trajectory.length - 1]
      
      if (!lastPoint || sseAlignment.phase_name !== lastPoint.phase_name) {
        trajectory.push({
          t: trajectory.length,
          phase_name: sseAlignment.phase_name,
          alignment_score: sseAlignment.alignment_score,
          drift_delta: sseAlignment.drift_delta,
          cusum_neg: sseAlignment.cusum_neg,
          warning: sseAlignment.warning,
          triggered: sseAlignment.correction,
          timestamp: new Date().toISOString(),
        })
      }
      
      return {
        ...base,
        trajectory,
        summary: {
          ...base.summary,
          current_alignment: sseAlignment.alignment_score,
          status: sseAlignment.status,
        }
      }
    }
    
    return base
  }, [alignmentData, sseAlignment])
  
  // Merge corrections from REST and SSE
  const allCorrections = useMemo(() => {
    const restCorrections = alignmentData?.actions_taken || []
    const combined = [...restCorrections]
    
    // Add any new SSE corrections not in REST data
    sseCorrections.forEach(c => {
      const exists = combined.some(
        r => r.phase_name === c.phase_name && r.action === c.action
      )
      if (!exists) {
        combined.push(c)
      }
    })
    
    return combined.slice(-10) // Keep last 10
  }, [alignmentData, sseCorrections])
  
  // Handle disabled or no data states
  if (!mergedData || !mergedData.enabled) {
    return (
      <div className={`rounded-xl border border-slate-500/30 bg-slate-500/5 overflow-hidden ${className}`}>
        <div className="px-4 py-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-slate-500/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Alignment
            </h3>
            <span className="text-xs text-white/40">Monitoring disabled</span>
          </div>
        </div>
      </div>
    )
  }
  
  const { trajectory, summary } = mergedData
  const currentScore = summary?.current_alignment || 0
  const scorePercent = (currentScore * 100).toFixed(0)
  const status = summary?.status || 'no_data'
  
  // Status styling
  const getStatusConfig = (s) => {
    switch (s) {
      case 'healthy':
        return {
          bg: 'bg-emerald-500/10',
          border: 'border-emerald-500/50',
          text: 'text-emerald-400',
          icon: '✓',
          label: 'Healthy',
          glow: 'shadow-emerald-500/20'
        }
      case 'warning':
        return {
          bg: 'bg-yellow-500/10',
          border: 'border-yellow-500/50',
          text: 'text-yellow-400',
          icon: '⚠',
          label: 'Warning',
          glow: 'shadow-yellow-500/20'
        }
      case 'correction':
        return {
          bg: 'bg-amber-500/10',
          border: 'border-amber-500/50',
          text: 'text-amber-400',
          icon: '🔄',
          label: 'Correcting',
          glow: 'shadow-amber-500/20'
        }
      default:
        return {
          bg: 'bg-slate-500/10',
          border: 'border-slate-500/50',
          text: 'text-slate-400',
          icon: '○',
          label: 'No Data',
          glow: ''
        }
    }
  }
  
  const config = getStatusConfig(status)
  
  // Color for alignment score
  const getScoreColor = (score) => {
    if (score >= 70) return 'emerald'
    if (score >= 50) return 'yellow'
    if (score >= 30) return 'amber'
    return 'red'
  }
  
  const scoreColor = getScoreColor(parseFloat(scorePercent))
  
  // Build sparkline data from trajectory
  const sparklinePoints = useMemo(() => {
    if (!trajectory || trajectory.length === 0) return []
    
    const maxPoints = 20
    const points = trajectory.slice(-maxPoints)
    const width = 100
    const height = 24
    
    if (points.length < 2) return []
    
    const xStep = width / (points.length - 1)
    
    return points.map((p, i) => ({
      x: i * xStep,
      y: height - (p.alignment_score * height),
      warning: p.warning,
      triggered: p.triggered,
      phase: p.phase_name,
      score: p.alignment_score,
    }))
  }, [trajectory])
  
  // Generate SVG path for sparkline
  const sparklinePath = useMemo(() => {
    if (sparklinePoints.length < 2) return ''
    return sparklinePoints
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
      .join(' ')
  }, [sparklinePoints])
  
  return (
    <div className={`rounded-xl border ${config.border} ${config.bg} shadow-lg ${config.glow} overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg ${config.bg} flex items-center justify-center`}>
            <svg className={`w-4 h-4 ${config.text}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Alignment
            </h3>
            <span className="text-xs text-white/40">Goal Drift Monitor</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${config.bg} ${config.text} border ${config.border}`}>
            {config.label}
          </span>
        </div>
      </div>
      
      {/* Stats Row */}
      <div className="px-4 py-3 grid grid-cols-3 gap-4 border-b border-white/5">
        {/* Score */}
        <div className="text-center">
          <div className={`text-xl font-mono font-bold text-${scoreColor}-400`}>
            {scorePercent}%
          </div>
          <div className="text-xs text-white/40 uppercase tracking-wide">Aligned</div>
        </div>
        
        {/* Triggers */}
        <div className="text-center">
          <div className={`text-xl font-mono font-bold ${(summary?.trigger_count || 0) > 0 ? 'text-amber-400' : 'text-white/70'}`}>
            {summary?.trigger_count || 0}
          </div>
          <div className="text-xs text-white/40 uppercase tracking-wide">Triggers</div>
        </div>
        
        {/* Actions */}
        <div className="text-center">
          <div className={`text-xl font-mono font-bold ${(summary?.actions_count || 0) > 0 ? 'text-cyan-400' : 'text-white/70'}`}>
            {summary?.actions_count || 0}
          </div>
          <div className="text-xs text-white/40 uppercase tracking-wide">Actions</div>
        </div>
      </div>
      
      {/* Alignment Bar */}
      <div className="px-4 py-2 border-b border-white/5">
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-500 rounded-full bg-gradient-to-r ${
              parseFloat(scorePercent) >= 70 ? 'from-emerald-500 to-emerald-400' :
              parseFloat(scorePercent) >= 50 ? 'from-yellow-500 to-yellow-400' :
              parseFloat(scorePercent) >= 30 ? 'from-amber-500 to-amber-400' :
              'from-red-500 to-red-400'
            }`}
            style={{ width: `${Math.min(parseFloat(scorePercent), 100)}%` }}
          />
        </div>
      </div>
      
      {/* Sparkline Trajectory */}
      {sparklinePoints.length >= 2 && (
        <div className="px-4 py-3 border-b border-white/5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-white/40">Trajectory</span>
            <span className="text-xs text-white/30 font-mono">{trajectory.length} phases</span>
          </div>
          <svg 
            viewBox="0 0 100 24" 
            className="w-full h-6"
            preserveAspectRatio="none"
          >
            {/* Warning threshold line */}
            <line 
              x1="0" y1={24 - 0.6 * 24} 
              x2="100" y2={24 - 0.6 * 24}
              stroke="rgba(234, 179, 8, 0.3)"
              strokeWidth="0.5"
              strokeDasharray="2,2"
            />
            {/* Correction threshold line */}
            <line 
              x1="0" y1={24 - 0.4 * 24} 
              x2="100" y2={24 - 0.4 * 24}
              stroke="rgba(239, 68, 68, 0.3)"
              strokeWidth="0.5"
              strokeDasharray="2,2"
            />
            {/* Trajectory line */}
            <path 
              d={sparklinePath}
              fill="none"
              stroke="url(#alignmentGradient)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {/* Gradient definition */}
            <defs>
              <linearGradient id="alignmentGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="rgba(34, 197, 94, 0.8)" />
                <stop offset="50%" stopColor="rgba(234, 179, 8, 0.8)" />
                <stop offset="100%" stopColor="rgba(239, 68, 68, 0.8)" />
              </linearGradient>
            </defs>
            {/* Phase markers */}
            {sparklinePoints.map((p, i) => (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={p.triggered ? 2 : (p.warning ? 1.5 : 1)}
                fill={p.triggered ? '#f59e0b' : (p.warning ? '#eab308' : '#22c55e')}
                className={p.triggered ? 'animate-pulse' : ''}
              />
            ))}
          </svg>
        </div>
      )}
      
      {/* Corrections List */}
      {allCorrections.length > 0 && (
        <div className="px-4 py-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-between text-left hover:bg-white/5 rounded-lg px-2 py-1.5 transition-colors -mx-2"
          >
            <span className="text-xs text-white/50 font-medium">
              Corrections ({allCorrections.length})
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
              {allCorrections.slice(-5).reverse().map((c, idx) => (
                <div 
                  key={idx}
                  className="flex items-center justify-between px-2 py-1.5 rounded-lg bg-white/5"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400/60 text-xs">🔄</span>
                    <div>
                      <span className="text-xs font-mono text-white/60">
                        {c.action?.replace('_', ' ') || c.action}
                      </span>
                      <span className="text-xs text-white/30 ml-2">
                        @ {c.phase_name || c.phase}
                      </span>
                    </div>
                  </div>
                  {c.alignment_score !== undefined && (
                    <span className="text-xs font-mono text-amber-400/80">
                      {(c.alignment_score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* North Star (if available) */}
      {mergedData.north_star?.intent_summary && (
        <div className="px-4 py-2 border-t border-white/5">
          <div className="text-xs text-white/30 mb-1">Goal</div>
          <p className="text-xs text-white/50 line-clamp-2">
            {mergedData.north_star.intent_summary}
          </p>
        </div>
      )}
    </div>
  )
}

