/**
 * Shows phase metrics with visual bar charts for difficulty, uncertainty, progress, novelty, and confidence.
 */
export default function PhaseMetrics({ metricsData }) {
  if (!metricsData) {
    return null
  }
  
  const {
    phase_name,
    difficulty_score,
    uncertainty_score,
    progress_score,
    novelty_score,
    confidence_score
  } = metricsData
  
  const metrics = [
    {
      label: 'Difficulty',
      value: difficulty_score || 0,
      color: 'dt-warning',
      icon: (
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      )
    },
    {
      label: 'Uncertainty',
      value: uncertainty_score || 0,
      color: (uncertainty_score || 0) > 0.7 ? 'dt-error' : 'dt-warning',
      icon: (
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    {
      label: 'Progress',
      value: progress_score || 0,
      color: 'dt-success',
      icon: (
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    {
      label: 'Novelty',
      value: novelty_score || 0,
      color: 'dt-accent',
      icon: (
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      label: 'Confidence',
      value: confidence_score || 0,
      color: 'dt-accent',
      icon: (
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      )
    }
  ]
  
  return (
    <div className="glass rounded-lg p-4">
      <h4 className="text-sm font-medium text-dt-text flex items-center gap-2 mb-4">
        <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        Phase Metrics
        {phase_name && (
          <span className="text-xs text-dt-text-dim font-normal">({phase_name})</span>
        )}
      </h4>
      
      <div className="space-y-3">
        {metrics.map((metric) => {
          const percent = (metric.value || 0) * 100
          return (
            <div key={metric.label}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-dt-text-dim flex items-center gap-1.5">
                  <span className={`text-${metric.color}`}>
                    {metric.icon}
                  </span>
                  {metric.label}
                </span>
                <span className={`text-xs font-mono text-${metric.color}`}>
                  {metric.value.toFixed(2)}
                </span>
              </div>
              <div className="h-2 bg-dt-surface-light rounded-full overflow-hidden">
                <div
                  className={`h-full bg-${metric.color} rounded-full transition-all duration-500`}
                  style={{ width: `${percent}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

