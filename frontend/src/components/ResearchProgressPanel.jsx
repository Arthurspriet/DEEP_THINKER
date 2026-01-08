import { useState } from 'react'

/**
 * ResearchProgressPanel - Shows research iteration progress with visual gauges
 * Displays completeness, context delta, web searches, and continuation status
 */
export default function ResearchProgressPanel({ researchData, className = '' }) {
  const [showDetails, setShowDetails] = useState(false)
  
  if (!researchData) {
    return null
  }
  
  const {
    phase_name,
    iteration,
    completeness,
    completeness_percent,
    should_continue,
    context_delta,
    web_searches_performed,
    key_points_count,
    gaps_count,
    confidence_score
  } = researchData
  
  const percent = completeness_percent || (completeness || 0) * 100
  const confPercent = (confidence_score || 0) * 100
  
  // Completeness color
  const getCompletenessColor = (p) => {
    if (p >= 80) return 'from-emerald-500 to-emerald-400'
    if (p >= 60) return 'from-cyan-500 to-cyan-400'
    if (p >= 40) return 'from-yellow-500 to-yellow-400'
    return 'from-amber-500 to-amber-400'
  }
  
  return (
    <div className={`rounded-xl border border-cyan-500/30 bg-cyan-500/5 shadow-lg shadow-cyan-500/10 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Research Progress
            </h3>
            {phase_name && (
              <span className="text-xs text-white/40">{phase_name}</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-md text-xs font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            Iteration {iteration || 0}
          </span>
        </div>
      </div>
      
      {/* Completeness Gauge */}
      <div className="px-4 py-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-white/50 uppercase tracking-wide">Completeness</span>
          <span className={`text-lg font-mono font-bold ${
            percent >= 80 ? 'text-emerald-400' :
            percent >= 60 ? 'text-cyan-400' :
            percent >= 40 ? 'text-yellow-400' :
            'text-amber-400'
          }`}>
            {percent.toFixed(0)}%
          </span>
        </div>
        <div className="h-3 bg-white/5 rounded-full overflow-hidden">
          <div 
            className={`h-full bg-gradient-to-r ${getCompletenessColor(percent)} rounded-full transition-all duration-700 ease-out`}
            style={{ width: `${Math.min(percent, 100)}%` }}
          />
        </div>
      </div>
      
      {/* Stats Grid */}
      <div className="px-4 pb-3 grid grid-cols-4 gap-3">
        {/* Key Points */}
        <div className="text-center p-2 rounded-lg bg-white/5">
          <div className="text-lg font-mono font-bold text-cyan-400">
            {key_points_count || 0}
          </div>
          <div className="text-[10px] text-white/40 uppercase">Points</div>
        </div>
        
        {/* Gaps */}
        <div className="text-center p-2 rounded-lg bg-white/5">
          <div className={`text-lg font-mono font-bold ${gaps_count > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
            {gaps_count || 0}
          </div>
          <div className="text-[10px] text-white/40 uppercase">Gaps</div>
        </div>
        
        {/* Web Searches */}
        <div className="text-center p-2 rounded-lg bg-white/5">
          <div className="text-lg font-mono font-bold text-white/70">
            {web_searches_performed || 0}
          </div>
          <div className="text-[10px] text-white/40 uppercase">Searches</div>
        </div>
        
        {/* Confidence */}
        <div className="text-center p-2 rounded-lg bg-white/5">
          <div className={`text-lg font-mono font-bold ${
            confPercent >= 70 ? 'text-emerald-400' :
            confPercent >= 50 ? 'text-yellow-400' :
            'text-amber-400'
          }`}>
            {confPercent.toFixed(0)}%
          </div>
          <div className="text-[10px] text-white/40 uppercase">Conf</div>
        </div>
      </div>
      
      {/* Continue Status */}
      <div className="px-4 py-2 border-t border-white/5 flex items-center justify-between">
        <span className="text-xs text-white/40">Continue Research:</span>
        <span className={`flex items-center gap-1.5 text-xs font-medium ${
          should_continue ? 'text-cyan-400' : 'text-emerald-400'
        }`}>
          {should_continue ? (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              In Progress
            </>
          ) : (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Complete
            </>
          )}
        </span>
      </div>
      
      {/* Context Delta */}
      {context_delta && (
        <div className="px-4 py-2 border-t border-white/5">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full flex items-center justify-between text-left hover:bg-white/5 rounded-lg px-2 py-1.5 transition-colors -mx-2"
          >
            <span className="text-xs text-white/50">Context Delta</span>
            <svg
              className={`w-4 h-4 text-white/30 transition-transform ${showDetails ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {showDetails && (
            <div className="mt-2 px-2 py-2 rounded-lg bg-white/5">
              <p className="text-xs text-white/60 font-mono leading-relaxed">
                {context_delta}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

