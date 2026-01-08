const STATUS_CONFIG = {
  running: { color: 'text-dt-success', bg: 'bg-dt-success/20', border: 'border-dt-success/40', label: 'Running' },
  pending: { color: 'text-dt-warning', bg: 'bg-dt-warning/20', border: 'border-dt-warning/40', label: 'Pending' },
  completed: { color: 'text-dt-accent', bg: 'bg-dt-accent/20', border: 'border-dt-accent/40', label: 'Completed' },
  failed: { color: 'text-dt-error', bg: 'bg-dt-error/20', border: 'border-dt-error/40', label: 'Failed' },
  aborted: { color: 'text-dt-error', bg: 'bg-dt-error/20', border: 'border-dt-error/40', label: 'Aborted' },
  expired: { color: 'text-dt-text-dim', bg: 'bg-dt-text-dim/20', border: 'border-dt-text-dim/40', label: 'Expired' },
}

export default function MissionHeader({
  title,
  status,
  remainingMinutes,
  isConnected,
  onAbort,
  onResume,
  onBack,
}) {
  const statusConfig = STATUS_CONFIG[status] || STATUS_CONFIG.pending
  const isTerminal = ['completed', 'failed', 'aborted', 'expired'].includes(status)
  const canResume = status === 'pending' || status === 'paused'
  const canAbort = status === 'running' || status === 'pending'

  return (
    <div className="glass rounded-xl p-6 animate-fade-in">
      <div className="flex items-start gap-6">
        {/* Back button */}
        <button
          onClick={onBack}
          className="p-2 rounded-lg hover:bg-dt-surface-light transition-colors shrink-0"
          aria-label="Back to missions"
        >
          <svg className="w-5 h-5 text-dt-text-dim" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <span className={`px-3 py-1 rounded-full text-xs font-medium border ${statusConfig.bg} ${statusConfig.color} ${statusConfig.border}`}>
              {statusConfig.label}
            </span>
            
            {/* SSE connection indicator */}
            {!isTerminal && (
              <div className={`flex items-center gap-1.5 text-xs ${isConnected ? 'text-dt-success' : 'text-dt-warning'}`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-dt-success' : 'bg-dt-warning'} ${isConnected ? 'animate-pulse' : ''}`} />
                {isConnected ? 'Live' : 'Reconnecting...'}
              </div>
            )}
          </div>

          <h1 className="text-xl font-medium text-dt-text line-clamp-2">
            {title}
          </h1>
        </div>

        {/* Time and controls */}
        <div className="flex items-center gap-4 shrink-0">
          {/* Time remaining */}
          {!isTerminal && (
            <div className="text-right">
              <div className="text-2xl font-display font-bold text-gradient">
                {remainingMinutes > 0 ? (
                  <>
                    {Math.floor(remainingMinutes)}
                    <span className="text-lg">m</span>
                  </>
                ) : (
                  <span className="text-dt-warning">Overtime</span>
                )}
              </div>
              <p className="text-xs text-dt-text-dim">remaining</p>
            </div>
          )}

          {/* Control buttons */}
          <div className="flex gap-2">
            {canResume && (
              <button
                onClick={onResume}
                className="p-2.5 rounded-lg bg-dt-success/20 text-dt-success hover:bg-dt-success/30 transition-colors"
                title="Resume mission"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
            )}
            
            {canAbort && (
              <button
                onClick={onAbort}
                className="p-2.5 rounded-lg bg-dt-error/20 text-dt-error hover:bg-dt-error/30 transition-colors"
                title="Abort mission"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

