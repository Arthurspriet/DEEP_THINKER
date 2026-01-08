import { useState } from 'react'

/**
 * DeepAnalysisPanel - Shows scenarios, stress tests, and tradeoff analysis
 * Collapsible sections for dense analysis information
 */
export default function DeepAnalysisPanel({ analysisData, className = '' }) {
  const [activeSection, setActiveSection] = useState('scenarios')
  
  if (!analysisData) {
    return null
  }
  
  const {
    scenarios,
    scenarios_count,
    stress_tests,
    stress_tests_count,
    tradeoffs,
    tradeoffs_count,
    robustness_score,
    failure_modes,
    failure_modes_count,
    recommendations
  } = analysisData
  
  const robustnessPercent = (robustness_score || 0) * 10 // Assuming 0-10 scale
  
  const sections = [
    { id: 'scenarios', label: 'Scenarios', count: scenarios_count || scenarios?.length || 0, icon: '🎯' },
    { id: 'stress', label: 'Stress Tests', count: stress_tests_count || stress_tests?.length || 0, icon: '⚡' },
    { id: 'tradeoffs', label: 'Tradeoffs', count: tradeoffs_count || tradeoffs?.length || 0, icon: '⚖️' },
    { id: 'failures', label: 'Failure Modes', count: failure_modes_count || failure_modes?.length || 0, icon: '⚠️' },
  ]
  
  const renderListItem = (item, idx, type) => {
    if (typeof item === 'string') {
      return (
        <div key={idx} className="px-3 py-2 rounded-lg bg-white/5 border-l-2 border-rose-500/30">
          <p className="text-xs text-white/70 leading-relaxed">{item}</p>
        </div>
      )
    }
    
    // Object with name/description
    return (
      <div key={idx} className="px-3 py-2 rounded-lg bg-white/5 border-l-2 border-rose-500/30">
        {item.name && (
          <div className="text-xs font-medium text-white/80 mb-1">{item.name}</div>
        )}
        {item.description && (
          <p className="text-xs text-white/60 leading-relaxed">{item.description}</p>
        )}
        {item.probability && (
          <span className="inline-block mt-1 text-[10px] text-rose-400 font-mono">
            {(item.probability * 100).toFixed(0)}% probability
          </span>
        )}
        {item.timeline && (
          <span className="inline-block mt-1 ml-2 text-[10px] text-white/40">
            {item.timeline}
          </span>
        )}
        {item.severity && (
          <span className={`inline-block mt-1 ml-2 text-[10px] ${
            item.severity === 'critical' ? 'text-red-400' :
            item.severity === 'high' ? 'text-amber-400' :
            item.severity === 'medium' ? 'text-yellow-400' :
            'text-white/40'
          }`}>
            {item.severity}
          </span>
        )}
      </div>
    )
  }
  
  return (
    <div className={`rounded-xl border border-rose-500/30 bg-rose-500/5 shadow-lg shadow-rose-500/10 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-rose-500/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Deep Analysis
            </h3>
            <span className="text-xs text-white/40">Scenarios & Stress Tests</span>
          </div>
        </div>
        
        {/* Robustness Score */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-white/40">Robustness:</span>
          <span className={`px-2 py-1 rounded-md text-xs font-mono font-bold ${
            robustness_score >= 7 ? 'bg-emerald-500/10 text-emerald-400' :
            robustness_score >= 5 ? 'bg-yellow-500/10 text-yellow-400' :
            'bg-red-500/10 text-red-400'
          }`}>
            {robustness_score?.toFixed(1) || '0.0'}/10
          </span>
        </div>
      </div>
      
      {/* Section Tabs */}
      <div className="px-4 py-2 border-b border-white/5 flex gap-1 overflow-x-auto">
        {sections.map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
              activeSection === section.id
                ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                : 'bg-white/5 text-white/50 hover:bg-white/10 border border-transparent'
            }`}
          >
            <span className="mr-1">{section.icon}</span>
            {section.label}
            {section.count > 0 && (
              <span className="ml-1.5 px-1.5 py-0.5 rounded bg-white/10 text-[10px]">
                {section.count}
              </span>
            )}
          </button>
        ))}
      </div>
      
      {/* Content */}
      <div className="px-4 py-3 max-h-64 overflow-y-auto">
        {/* Scenarios */}
        {activeSection === 'scenarios' && (
          <div className="space-y-2">
            {scenarios && scenarios.length > 0 ? (
              scenarios.map((scenario, idx) => renderListItem(scenario, idx, 'scenario'))
            ) : (
              <p className="text-xs text-white/30 text-center py-4">No scenarios analyzed</p>
            )}
          </div>
        )}
        
        {/* Stress Tests */}
        {activeSection === 'stress' && (
          <div className="space-y-2">
            {stress_tests && stress_tests.length > 0 ? (
              stress_tests.map((test, idx) => renderListItem(test, idx, 'stress'))
            ) : (
              <p className="text-xs text-white/30 text-center py-4">No stress tests performed</p>
            )}
          </div>
        )}
        
        {/* Tradeoffs */}
        {activeSection === 'tradeoffs' && (
          <div className="space-y-2">
            {tradeoffs && tradeoffs.length > 0 ? (
              tradeoffs.map((tradeoff, idx) => renderListItem(tradeoff, idx, 'tradeoff'))
            ) : (
              <p className="text-xs text-white/30 text-center py-4">No tradeoffs identified</p>
            )}
          </div>
        )}
        
        {/* Failure Modes */}
        {activeSection === 'failures' && (
          <div className="space-y-2">
            {failure_modes && failure_modes.length > 0 ? (
              failure_modes.map((mode, idx) => renderListItem(mode, idx, 'failure'))
            ) : (
              <p className="text-xs text-white/30 text-center py-4">No failure modes detected</p>
            )}
          </div>
        )}
      </div>
      
      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="px-4 py-3 border-t border-white/5">
          <div className="text-xs text-white/40 uppercase tracking-wide mb-2">Recommendations</div>
          <div className="space-y-1.5">
            {recommendations.slice(0, 3).map((rec, idx) => (
              <div key={idx} className="flex items-start gap-2">
                <span className="text-emerald-400 text-xs mt-0.5">→</span>
                <p className="text-xs text-white/60 leading-relaxed">{rec}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

