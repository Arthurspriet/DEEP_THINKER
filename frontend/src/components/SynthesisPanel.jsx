import { useState } from 'react'

/**
 * SynthesisPanel - Shows synthesis iterations with content preview
 * Displays iteration history, quality scores, and final synthesis
 */
export default function SynthesisPanel({ synthesisData, synthesisHistory = [], className = '' }) {
  const [selectedIteration, setSelectedIteration] = useState(null)
  const [expanded, setExpanded] = useState(false)
  
  // Use current data or fall back to history
  const currentSynthesis = synthesisData || (synthesisHistory.length > 0 ? synthesisHistory[synthesisHistory.length - 1] : null)
  
  if (!currentSynthesis && synthesisHistory.length === 0) {
    return null
  }
  
  const {
    iteration,
    content_preview,
    quality_score,
    word_count,
    sections_count,
    is_final
  } = currentSynthesis || {}
  
  const qualityPercent = (quality_score || 0) * 10 // Assuming 0-10 scale
  
  // Combine current with history for display
  const allIterations = synthesisHistory.length > 0 ? synthesisHistory : (currentSynthesis ? [currentSynthesis] : [])
  
  const getQualityColor = (score) => {
    if (score >= 8) return 'emerald'
    if (score >= 6) return 'cyan'
    if (score >= 4) return 'yellow'
    return 'amber'
  }
  
  const color = getQualityColor(quality_score || 0)
  
  return (
    <div className={`rounded-xl border border-teal-500/30 bg-teal-500/5 shadow-lg shadow-teal-500/10 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-teal-500/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Synthesis
            </h3>
            <span className="text-xs text-white/40">Report Generation</span>
          </div>
        </div>
        
        {/* Status Badge */}
        {is_final ? (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <span>✓</span>
            FINAL
          </span>
        ) : (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono bg-teal-500/10 text-teal-400 border border-teal-500/30">
            Iteration {iteration || allIterations.length}
          </span>
        )}
      </div>
      
      {/* Quality & Stats */}
      <div className="px-4 py-3 border-b border-white/5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-white/50 uppercase tracking-wide">Quality Score</span>
          <span className={`text-lg font-mono font-bold text-${color}-400`}>
            {quality_score?.toFixed(1) || '0.0'}/10
          </span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div 
            className={`h-full bg-gradient-to-r from-${color}-500 to-${color}-400 rounded-full transition-all duration-500`}
            style={{ width: `${Math.min(qualityPercent * 10, 100)}%` }}
          />
        </div>
        
        {/* Stats Row */}
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/5">
          <div className="flex items-center gap-4">
            <div className="text-center">
              <div className="text-sm font-mono font-bold text-white/70">{word_count || 0}</div>
              <div className="text-[10px] text-white/40 uppercase">Words</div>
            </div>
            <div className="text-center">
              <div className="text-sm font-mono font-bold text-white/70">{sections_count || 0}</div>
              <div className="text-[10px] text-white/40 uppercase">Sections</div>
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-mono font-bold text-teal-400">{allIterations.length}</div>
            <div className="text-[10px] text-white/40 uppercase">Iterations</div>
          </div>
        </div>
      </div>
      
      {/* Iteration History */}
      {allIterations.length > 1 && (
        <div className="px-4 py-2 border-b border-white/5">
          <div className="text-xs text-white/40 uppercase tracking-wide mb-2">Iteration History</div>
          <div className="flex gap-1.5 overflow-x-auto pb-1">
            {allIterations.map((iter, idx) => {
              const iterQuality = iter.quality_score || 0
              const iterColor = getQualityColor(iterQuality)
              const isSelected = selectedIteration === idx
              
              return (
                <button
                  key={idx}
                  onClick={() => setSelectedIteration(isSelected ? null : idx)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-mono whitespace-nowrap transition-colors ${
                    isSelected
                      ? `bg-${iterColor}-500/20 text-${iterColor}-400 border border-${iterColor}-500/30`
                      : 'bg-white/5 text-white/50 hover:bg-white/10 border border-transparent'
                  }`}
                >
                  #{idx + 1}
                  <span className={`ml-1 text-${iterColor}-400`}>
                    {iterQuality.toFixed(1)}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      )}
      
      {/* Content Preview */}
      {(content_preview || (selectedIteration !== null && allIterations[selectedIteration]?.content_preview)) && (
        <div className="px-4 py-3">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-between text-left hover:bg-white/5 rounded-lg px-2 py-1.5 transition-colors -mx-2"
          >
            <span className="text-xs text-white/50 font-medium">
              {selectedIteration !== null ? `Iteration #${selectedIteration + 1} Preview` : 'Current Preview'}
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
            <div className="mt-2 p-3 rounded-lg bg-white/5 max-h-48 overflow-y-auto">
              <p className="text-xs text-white/60 leading-relaxed whitespace-pre-wrap font-mono">
                {selectedIteration !== null 
                  ? allIterations[selectedIteration]?.content_preview 
                  : content_preview
                }
              </p>
            </div>
          )}
        </div>
      )}
      
      {/* Final Badge */}
      {is_final && (
        <div className="px-4 py-3 border-t border-white/5 bg-emerald-500/5">
          <div className="flex items-center gap-2 text-emerald-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-xs font-medium">Synthesis Complete - Report Ready</span>
          </div>
        </div>
      )}
    </div>
  )
}

