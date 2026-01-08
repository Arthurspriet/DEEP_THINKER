/**
 * SupervisorPanel - Shows supervisor model selection decisions
 * Displays models, downgrade status, GPU pressure, and fallback options
 */
export default function SupervisorPanel({ supervisorData, className = '' }) {
  if (!supervisorData) {
    return null
  }
  
  const {
    phase_name,
    models,
    decision_type,
    downgraded,
    downgrade_reason,
    fallback_models,
    gpu_pressure,
    estimated_vram_mb,
    wait_for_capacity,
    phase_importance,
    temperature,
    parallelism
  } = supervisorData
  
  // GPU pressure styling
  const getPressureConfig = (pressure) => {
    switch (pressure?.toLowerCase()) {
      case 'low':
        return { color: 'emerald', label: 'LOW', icon: '▽' }
      case 'medium':
        return { color: 'yellow', label: 'MEDIUM', icon: '◇' }
      case 'high':
        return { color: 'red', label: 'HIGH', icon: '△' }
      default:
        return { color: 'slate', label: 'NORMAL', icon: '○' }
    }
  }
  
  const pressureConfig = getPressureConfig(gpu_pressure)
  const importancePercent = (phase_importance || 0.5) * 100
  
  return (
    <div className={`rounded-xl border ${downgraded ? 'border-amber-500/50 bg-amber-500/5' : 'border-indigo-500/30 bg-indigo-500/5'} shadow-lg ${downgraded ? 'shadow-amber-500/10' : 'shadow-indigo-500/10'} overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg ${downgraded ? 'bg-amber-500/20' : 'bg-indigo-500/20'} flex items-center justify-center`}>
            <svg className={`w-4 h-4 ${downgraded ? 'text-amber-400' : 'text-indigo-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Supervisor
            </h3>
            {phase_name && (
              <span className="text-xs text-white/40">{phase_name}</span>
            )}
          </div>
        </div>
        
        {/* Status Badge */}
        {downgraded ? (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">
            <span>⚡</span>
            DOWNGRADED
          </span>
        ) : (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
            <span>✓</span>
            OPTIMAL
          </span>
        )}
      </div>
      
      {/* Selected Models */}
      <div className="px-4 py-3 border-b border-white/5">
        <div className="text-xs text-white/40 uppercase tracking-wide mb-2">Selected Models</div>
        <div className="flex flex-wrap gap-2">
          {(models || []).map((model, idx) => (
            <span
              key={idx}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-mono ${
                downgraded 
                  ? 'bg-amber-500/10 text-amber-300 border border-amber-500/20' 
                  : 'bg-indigo-500/10 text-indigo-300 border border-indigo-500/20'
              }`}
            >
              {model}
            </span>
          ))}
        </div>
      </div>
      
      {/* Stats Row */}
      <div className="px-4 py-3 grid grid-cols-4 gap-3 border-b border-white/5">
        {/* GPU Pressure */}
        <div className="text-center">
          <div className={`text-lg font-bold text-${pressureConfig.color}-400`}>
            {pressureConfig.icon}
          </div>
          <div className="text-[10px] text-white/40 uppercase">{pressureConfig.label}</div>
        </div>
        
        {/* VRAM Estimate */}
        <div className="text-center">
          <div className="text-lg font-mono font-bold text-white/70">
            {estimated_vram_mb ? (estimated_vram_mb / 1024).toFixed(1) : '0'}
          </div>
          <div className="text-[10px] text-white/40 uppercase">GB VRAM</div>
        </div>
        
        {/* Temperature */}
        <div className="text-center">
          <div className="text-lg font-mono font-bold text-white/70">
            {(temperature || 0.7).toFixed(1)}
          </div>
          <div className="text-[10px] text-white/40 uppercase">Temp</div>
        </div>
        
        {/* Parallelism */}
        <div className="text-center">
          <div className="text-lg font-mono font-bold text-white/70">
            {parallelism || 1}x
          </div>
          <div className="text-[10px] text-white/40 uppercase">Parallel</div>
        </div>
      </div>
      
      {/* Phase Importance */}
      <div className="px-4 py-3 border-b border-white/5">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-white/40">Phase Importance</span>
          <span className="text-xs font-mono text-white/60">{importancePercent.toFixed(0)}%</span>
        </div>
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(importancePercent, 100)}%` }}
          />
        </div>
      </div>
      
      {/* Downgrade Reason */}
      {downgraded && downgrade_reason && (
        <div className="px-4 py-2 border-b border-white/5 flex items-start gap-2">
          <span className="text-amber-400 text-xs mt-0.5">⚠</span>
          <div>
            <span className="text-xs text-white/40">Downgrade Reason:</span>
            <p className="text-xs text-amber-400/80 mt-0.5">{downgrade_reason}</p>
          </div>
        </div>
      )}
      
      {/* Wait for Capacity */}
      {wait_for_capacity && (
        <div className="px-4 py-2 border-b border-white/5">
          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            Waiting for GPU Capacity
          </span>
        </div>
      )}
      
      {/* Fallback Models */}
      {fallback_models && fallback_models.length > 0 && (
        <div className="px-4 py-3">
          <div className="text-xs text-white/40 uppercase tracking-wide mb-2">Fallback Options</div>
          <div className="flex flex-wrap gap-1.5">
            {fallback_models.map((model, idx) => (
              <span
                key={idx}
                className="px-2 py-1 rounded text-[11px] font-mono bg-white/5 text-white/40 border border-white/10"
              >
                {model}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

