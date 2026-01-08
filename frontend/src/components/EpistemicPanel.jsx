import { useState } from 'react'

/**
 * EpistemicPanel - Shows epistemic state with questions, evidence, and focus areas
 * Tracks knowledge gaps, claims, and grounding score
 */
export default function EpistemicPanel({ epistemicData, className = '' }) {
  const [activeTab, setActiveTab] = useState('questions')
  
  if (!epistemicData) {
    return null
  }
  
  const {
    phase_name,
    unresolved_questions,
    unresolved_count,
    evidence_requests,
    evidence_count,
    focus_areas,
    focus_count,
    claims_count,
    verified_claims,
    grounding_score
  } = epistemicData
  
  const groundingPercent = (grounding_score || 0) * 100
  const verifiedPercent = claims_count > 0 ? (verified_claims / claims_count) * 100 : 0
  
  const tabs = [
    { id: 'questions', label: 'Questions', count: unresolved_count || unresolved_questions?.length || 0, icon: '❓' },
    { id: 'evidence', label: 'Evidence', count: evidence_count || evidence_requests?.length || 0, icon: '📋' },
    { id: 'focus', label: 'Focus Areas', count: focus_count || focus_areas?.length || 0, icon: '🎯' },
  ]
  
  const getGroundingColor = (score) => {
    if (score >= 70) return 'emerald'
    if (score >= 50) return 'yellow'
    if (score >= 30) return 'amber'
    return 'red'
  }
  
  const color = getGroundingColor(groundingPercent)
  
  return (
    <div className={`rounded-xl border border-orange-500/30 bg-orange-500/5 shadow-lg shadow-orange-500/10 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-orange-500/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              Epistemic State
            </h3>
            {phase_name && (
              <span className="text-xs text-white/40">{phase_name}</span>
            )}
          </div>
        </div>
        
        {/* Grounding Score */}
        <div className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-${color}-500/10 text-${color}-400 border border-${color}-500/30`}>
          {groundingPercent.toFixed(0)}% Grounded
        </div>
      </div>
      
      {/* Claims Stats */}
      <div className="px-4 py-3 border-b border-white/5 grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className="text-lg font-mono font-bold text-white/70">{claims_count || 0}</div>
          <div className="text-[10px] text-white/40 uppercase">Claims</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-mono font-bold text-emerald-400">{verified_claims || 0}</div>
          <div className="text-[10px] text-white/40 uppercase">Verified</div>
        </div>
        <div className="text-center">
          <div className={`text-lg font-mono font-bold text-${color}-400`}>{verifiedPercent.toFixed(0)}%</div>
          <div className="text-[10px] text-white/40 uppercase">Rate</div>
        </div>
      </div>
      
      {/* Grounding Bar */}
      <div className="px-4 py-2 border-b border-white/5">
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div 
            className={`h-full bg-gradient-to-r from-${color}-500 to-${color}-400 rounded-full transition-all duration-500`}
            style={{ width: `${Math.min(groundingPercent, 100)}%` }}
          />
        </div>
      </div>
      
      {/* Tabs */}
      <div className="px-4 py-2 border-b border-white/5 flex gap-1">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                : 'bg-white/5 text-white/50 hover:bg-white/10 border border-transparent'
            }`}
          >
            <span className="mr-1">{tab.icon}</span>
            {tab.count > 0 && (
              <span className="ml-1 px-1.5 py-0.5 rounded bg-white/10 text-[10px]">
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>
      
      {/* Content */}
      <div className="px-4 py-3 max-h-48 overflow-y-auto">
        {/* Questions */}
        {activeTab === 'questions' && (
          <div className="space-y-2">
            {unresolved_questions && unresolved_questions.length > 0 ? (
              unresolved_questions.map((q, idx) => (
                <div 
                  key={idx}
                  className="flex items-start gap-2 px-3 py-2 rounded-lg bg-white/5 border-l-2 border-orange-500/30"
                >
                  <span className="text-orange-400 text-xs mt-0.5">?</span>
                  <p className="text-xs text-white/70 leading-relaxed">{q}</p>
                </div>
              ))
            ) : (
              <p className="text-xs text-white/30 text-center py-4">No unresolved questions</p>
            )}
          </div>
        )}
        
        {/* Evidence */}
        {activeTab === 'evidence' && (
          <div className="space-y-2">
            {evidence_requests && evidence_requests.length > 0 ? (
              evidence_requests.map((e, idx) => (
                <div 
                  key={idx}
                  className="flex items-start gap-2 px-3 py-2 rounded-lg bg-white/5 border-l-2 border-cyan-500/30"
                >
                  <span className="text-cyan-400 text-xs mt-0.5">→</span>
                  <p className="text-xs text-white/70 leading-relaxed">{e}</p>
                </div>
              ))
            ) : (
              <p className="text-xs text-white/30 text-center py-4">No evidence requests</p>
            )}
          </div>
        )}
        
        {/* Focus Areas */}
        {activeTab === 'focus' && (
          <div className="space-y-2">
            {focus_areas && focus_areas.length > 0 ? (
              focus_areas.map((f, idx) => (
                <div 
                  key={idx}
                  className="flex items-start gap-2 px-3 py-2 rounded-lg bg-white/5 border-l-2 border-violet-500/30"
                >
                  <span className="text-violet-400 text-xs mt-0.5">◎</span>
                  <p className="text-xs text-white/70 leading-relaxed">{f}</p>
                </div>
              ))
            ) : (
              <p className="text-xs text-white/30 text-center py-4">No focus areas defined</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

