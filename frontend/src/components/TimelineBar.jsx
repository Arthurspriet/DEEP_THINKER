export default function TimelineBar({
  elapsedMinutes,
  totalMinutes,
  progress,
  status,
}) {
  const isComplete = status === 'completed'
  const isFailed = status === 'failed' || status === 'aborted'
  const isOvertime = elapsedMinutes > totalMinutes

  const formatTime = (minutes) => {
    if (minutes < 60) return `${Math.floor(minutes)}m`
    const hours = Math.floor(minutes / 60)
    const mins = Math.floor(minutes % 60)
    return `${hours}h ${mins}m`
  }

  return (
    <div className="glass rounded-xl p-4 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm font-medium text-dt-text">Mission Timeline</span>
        </div>
        
        <div className="flex items-center gap-4 text-sm">
          <span className="text-dt-text-dim">
            Elapsed: <span className="text-dt-text font-mono">{formatTime(elapsedMinutes)}</span>
          </span>
          <span className="text-dt-text-dim">
            Total: <span className="text-dt-text font-mono">{formatTime(totalMinutes)}</span>
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="relative">
        <div className="w-full h-3 bg-dt-surface-light rounded-full overflow-hidden">
          {/* Progress fill */}
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isComplete
                ? 'bg-gradient-to-r from-dt-success to-dt-accent'
                : isFailed
                  ? 'bg-dt-error'
                  : isOvertime
                    ? 'bg-dt-warning'
                    : 'bg-gradient-to-r from-dt-accent to-dt-success'
            }`}
            style={{ width: `${Math.min(100, progress)}%` }}
          />
          
          {/* Animated flow for running missions */}
          {status === 'running' && !isOvertime && (
            <div
              className="absolute top-0 left-0 h-full rounded-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-flow"
              style={{ width: `${Math.min(100, progress)}%` }}
            />
          )}
        </div>

        {/* Progress markers */}
        <div className="absolute top-0 left-0 right-0 h-3 flex items-center">
          {[25, 50, 75].map((marker) => (
            <div
              key={marker}
              className="absolute w-0.5 h-2 bg-dt-border/50"
              style={{ left: `${marker}%` }}
            />
          ))}
        </div>
      </div>

      {/* Status labels */}
      <div className="flex items-center justify-between mt-2 text-xs">
        <span className="text-dt-text-dim">Start</span>
        
        <div className="flex items-center gap-1">
          {isComplete && (
            <>
              <svg className="w-4 h-4 text-dt-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-dt-success">Completed</span>
            </>
          )}
          {isFailed && (
            <>
              <svg className="w-4 h-4 text-dt-error" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              <span className="text-dt-error">{status === 'aborted' ? 'Aborted' : 'Failed'}</span>
            </>
          )}
          {status === 'running' && !isOvertime && (
            <span className="text-dt-accent">
              {Math.round(progress)}% complete
            </span>
          )}
          {status === 'running' && isOvertime && (
            <span className="text-dt-warning">
              Overtime ({formatTime(elapsedMinutes - totalMinutes)} over)
            </span>
          )}
        </div>
        
        <span className="text-dt-text-dim">Deadline</span>
      </div>
    </div>
  )
}

