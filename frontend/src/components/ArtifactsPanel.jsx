import { useState } from 'react'

const TABS = [
  { id: 'current', label: 'Current Phase' },
  { id: 'all', label: 'All Artifacts' },
]

export default function ArtifactsPanel({
  artifacts,
  phases,
  currentPhase,
  selectedPhase,
  onPhaseSelect,
}) {
  const [activeTab, setActiveTab] = useState('current')
  const [expandedArtifact, setExpandedArtifact] = useState(null)

  // Filter artifacts based on tab
  const filteredArtifacts = activeTab === 'current'
    ? artifacts.filter(a => a.phase === currentPhase || a.phase === selectedPhase)
    : artifacts

  // Group artifacts by phase
  const artifactsByPhase = {}
  filteredArtifacts.forEach(artifact => {
    if (!artifactsByPhase[artifact.phase]) {
      artifactsByPhase[artifact.phase] = []
    }
    artifactsByPhase[artifact.phase].push(artifact)
  })

  const getTypeIcon = (type) => {
    switch (type) {
      case 'code':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
        )
      case 'document':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        )
      case 'data':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
          </svg>
        )
      default:
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
        )
    }
  }

  const getTypeColor = (type) => {
    switch (type) {
      case 'code': return 'text-dt-accent'
      case 'document': return 'text-dt-success'
      case 'data': return 'text-dt-warning'
      case 'final': return 'text-gradient'
      default: return 'text-dt-text-dim'
    }
  }

  return (
    <div className="glass rounded-xl overflow-hidden animate-fade-in flex flex-col h-[400px]">
      {/* Header */}
      <div className="p-4 border-b border-dt-border shrink-0">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-dt-text flex items-center gap-2">
            <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
            Artifacts
          </h3>
          <span className="text-xs text-dt-text-dim">
            {filteredArtifacts.length} item{filteredArtifacts.length !== 1 ? 's' : ''}
          </span>
        </div>

        {/* Tabs */}
        <div className="flex gap-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-dt-accent text-dt-bg'
                  : 'bg-dt-surface-light text-dt-text-dim hover:bg-dt-border'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {Object.keys(artifactsByPhase).length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-dt-text-dim">
            <svg className="w-12 h-12 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
            <p className="text-sm">No artifacts yet</p>
          </div>
        ) : (
          Object.entries(artifactsByPhase).map(([phase, phaseArtifacts]) => (
            <div key={phase}>
              <h4 className="text-xs font-medium text-dt-text-dim mb-2 uppercase tracking-wider">
                {phase}
              </h4>
              <div className="space-y-2">
                {phaseArtifacts.map((artifact, idx) => (
                  <div
                    key={idx}
                    className="glass rounded-lg overflow-hidden"
                  >
                    <button
                      onClick={() => setExpandedArtifact(
                        expandedArtifact === `${phase}-${idx}` ? null : `${phase}-${idx}`
                      )}
                      className="w-full p-3 flex items-center justify-between hover:bg-dt-surface-light/50 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span className={getTypeColor(artifact.type)}>
                          {getTypeIcon(artifact.type)}
                        </span>
                        <span className="text-sm font-mono text-dt-text truncate max-w-[200px]">
                          {artifact.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-dt-text-dim">
                          {(artifact.content?.length || 0).toLocaleString()} chars
                        </span>
                        <svg
                          className={`w-4 h-4 text-dt-text-dim transition-transform ${
                            expandedArtifact === `${phase}-${idx}` ? 'rotate-180' : ''
                          }`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </div>
                    </button>
                    
                    {expandedArtifact === `${phase}-${idx}` && (
                      <div className="p-3 bg-dt-bg/50 border-t border-dt-border">
                        <pre className="text-xs text-dt-text-dim font-mono whitespace-pre-wrap overflow-x-auto max-h-48">
                          {artifact.content || 'No content'}
                        </pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

