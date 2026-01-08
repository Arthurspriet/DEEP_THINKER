const STATUS_STYLES = {
  pending: 'bg-dt-surface-light border-dt-border text-dt-text-dim',
  running: 'bg-dt-accent/10 border-dt-accent text-dt-accent status-running',
  completed: 'bg-dt-success/10 border-dt-success text-dt-success',
  failed: 'bg-dt-error/10 border-dt-error text-dt-error',
  skipped: 'bg-dt-surface-light border-dt-border text-dt-text-dim opacity-50',
}

const STATUS_ICONS = {
  pending: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <circle cx="12" cy="12" r="10" strokeWidth={2} />
    </svg>
  ),
  running: (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  ),
  completed: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  failed: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  skipped: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M13 5l7 7-7 7M5 5l7 7-7 7" />
    </svg>
  ),
}

export default function PhasesTimeline({ phases, currentPhaseIndex, onPhaseClick }) {
  return (
    <div className="glass rounded-xl p-4 animate-fade-in">
      <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin">
        {phases.map((phase, index) => {
          const isCurrent = index === currentPhaseIndex
          const statusStyle = STATUS_STYLES[phase.status] || STATUS_STYLES.pending
          const icon = STATUS_ICONS[phase.status] || STATUS_ICONS.pending

          return (
            <div key={index} className="flex items-center shrink-0">
              {/* Phase box */}
              <button
                onClick={() => onPhaseClick?.(phase.name)}
                className={`
                  px-4 py-2.5 rounded-lg border-2 transition-all duration-300
                  flex items-center gap-2 min-w-[120px]
                  ${statusStyle}
                  ${isCurrent ? 'ring-2 ring-dt-accent/50 ring-offset-2 ring-offset-dt-bg' : ''}
                  hover:scale-105
                `}
              >
                {icon}
                <span className="text-sm font-medium truncate">{phase.name}</span>
              </button>

              {/* Connector line */}
              {index < phases.length - 1 && (
                <div className="w-8 h-0.5 mx-1 relative overflow-hidden">
                  <div className={`absolute inset-0 ${
                    phase.status === 'completed' 
                      ? 'bg-dt-success' 
                      : phase.status === 'running'
                        ? 'bg-gradient-to-r from-dt-accent to-transparent flow-line animate-flow'
                        : 'bg-dt-border'
                  }`} />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

