import { useState } from 'react'

/**
 * PhaseErrorsPanel - Shows phase failures with error details
 * Displays error context, suggestions, and retry information
 */
export default function PhaseErrorsPanel({ errorData, errorHistory = [], className = '' }) {
  const [expanded, setExpanded] = useState(true)
  const [selectedError, setSelectedError] = useState(null)
  
  // Use current error or fall back to history
  const currentError = errorData || (errorHistory.length > 0 ? errorHistory[errorHistory.length - 1] : null)
  
  if (!currentError && errorHistory.length === 0) {
    return null
  }
  
  const {
    phase_name,
    phase_index,
    error_type,
    error_message,
    retry_available,
    suggestions,
    context
  } = currentError || {}
  
  // Combine current with history for display
  const allErrors = errorHistory.length > 0 ? errorHistory : (currentError ? [currentError] : [])
  
  const getErrorTypeConfig = (type) => {
    const t = type?.toLowerCase() || ''
    if (t.includes('governance')) return { icon: '⛔', color: 'amber' }
    if (t.includes('timeout')) return { icon: '⏱', color: 'yellow' }
    if (t.includes('validation')) return { icon: '✗', color: 'red' }
    if (t.includes('resource')) return { icon: '📊', color: 'violet' }
    return { icon: '⚠', color: 'red' }
  }
  
  const config = getErrorTypeConfig(error_type)
  
  return (
    <div className={`rounded-xl border border-red-500/50 bg-red-500/5 shadow-lg shadow-red-500/10 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center">
            <span className="text-lg">{config.icon}</span>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-red-400 tracking-wide uppercase">
              Phase Error
            </h3>
            {phase_name && (
              <span className="text-xs text-white/40">{phase_name}</span>
            )}
          </div>
        </div>
        
        {/* Retry Badge */}
        {retry_available ? (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            Retry Available
          </span>
        ) : (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-red-500/10 text-red-400 border border-red-500/30">
            Failed
          </span>
        )}
      </div>
      
      {/* Error Type & Index */}
      <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-white/40">Type:</span>
          <span className={`px-2 py-0.5 rounded text-xs font-mono bg-${config.color}-500/10 text-${config.color}-400 border border-${config.color}-500/30`}>
            {error_type || 'Unknown'}
          </span>
        </div>
        {phase_index !== undefined && (
          <span className="text-xs text-white/40">
            Phase #{phase_index + 1}
          </span>
        )}
      </div>
      
      {/* Error Message */}
      <div className="px-4 py-3 border-b border-white/5">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center justify-between text-left"
        >
          <span className="text-xs text-white/50 font-medium">Error Message</span>
          <svg
            className={`w-4 h-4 text-white/30 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        {expanded && error_message && (
          <div className="mt-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
            <p className="text-xs text-red-300 font-mono leading-relaxed whitespace-pre-wrap">
              {error_message}
            </p>
          </div>
        )}
      </div>
      
      {/* Context Details */}
      {context && Object.keys(context).length > 0 && (
        <div className="px-4 py-3 border-b border-white/5">
          <div className="text-xs text-white/40 uppercase tracking-wide mb-2">Context</div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(context).slice(0, 6).map(([key, value]) => (
              <div key={key} className="px-2 py-1.5 rounded-lg bg-white/5">
                <div className="text-[10px] text-white/30 uppercase">{key}</div>
                <div className="text-xs text-white/60 font-mono truncate" title={String(value)}>
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Suggestions */}
      {suggestions && suggestions.length > 0 && (
        <div className="px-4 py-3 border-b border-white/5">
          <div className="text-xs text-white/40 uppercase tracking-wide mb-2">Suggestions</div>
          <div className="space-y-1.5">
            {suggestions.map((suggestion, idx) => (
              <div 
                key={idx}
                className="flex items-start gap-2 px-3 py-2 rounded-lg bg-emerald-500/5 border-l-2 border-emerald-500/30"
              >
                <span className="text-emerald-400 text-xs mt-0.5">→</span>
                <p className="text-xs text-white/70 leading-relaxed">{suggestion}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Error History */}
      {allErrors.length > 1 && (
        <div className="px-4 py-3">
          <div className="text-xs text-white/40 uppercase tracking-wide mb-2">
            Error History ({allErrors.length})
          </div>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {allErrors.slice(0, -1).reverse().map((err, idx) => {
              const errConfig = getErrorTypeConfig(err.error_type)
              const isSelected = selectedError === idx
              
              return (
                <button
                  key={idx}
                  onClick={() => setSelectedError(isSelected ? null : idx)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left transition-colors ${
                    isSelected ? 'bg-white/10' : 'bg-white/5 hover:bg-white/10'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{errConfig.icon}</span>
                    <span className="text-xs text-white/60">{err.phase_name || `Phase #${err.phase_index + 1}`}</span>
                  </div>
                  <span className={`text-xs font-mono text-${errConfig.color}-400`}>
                    {err.error_type || 'Error'}
                  </span>
                </button>
              )
            })}
          </div>
          
          {/* Selected Error Details */}
          {selectedError !== null && allErrors[allErrors.length - 2 - selectedError] && (
            <div className="mt-2 p-3 rounded-lg bg-white/5">
              <p className="text-xs text-white/50 font-mono leading-relaxed">
                {allErrors[allErrors.length - 2 - selectedError].error_message || 'No message'}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

