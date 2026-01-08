/**
 * Shows GPU and time resource utilization from SSE events.
 */
export default function ResourceBar({ resourceData }) {
  if (!resourceData) {
    return null
  }
  
  const {
    gpu_available,
    gpu_total,
    gpu_utilization,
    vram_used_mb,
    vram_total_mb,
    vram_utilization,
    time_remaining,
    total_time,
    time_utilization,
    active_models
  } = resourceData
  
  const getUtilColor = (percent) => {
    if (percent >= 90) return 'dt-error'
    if (percent >= 70) return 'dt-warning'
    return 'dt-success'
  }
  
  const formatMemory = (mb) => {
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)}GB`
    return `${mb}MB`
  }
  
  return (
    <div className="glass rounded-lg p-4">
      <h4 className="text-sm font-medium text-dt-text flex items-center gap-2 mb-4">
        <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        Resources
      </h4>
      
      <div className="space-y-4">
        {/* Time Usage */}
        {total_time != null && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-dt-text-dim flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Time Budget
              </span>
              <span className="text-xs font-mono text-dt-text">
                {(time_remaining || 0).toFixed(1)}m / {total_time.toFixed(0)}m
              </span>
            </div>
            <div className="h-2 bg-dt-surface-light rounded-full overflow-hidden">
              <div
                className={`h-full bg-${getUtilColor(time_utilization || 0)} rounded-full transition-all`}
                style={{ width: `${time_utilization || 0}%` }}
              />
            </div>
          </div>
        )}
        
        {/* GPU Usage */}
        {gpu_total != null && gpu_total > 0 && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-dt-text-dim flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                GPU ({gpu_available}/{gpu_total} available)
              </span>
              <span className={`text-xs font-mono text-${getUtilColor(gpu_utilization || 0)}`}>
                {(gpu_utilization || 0).toFixed(0)}%
              </span>
            </div>
            <div className="h-2 bg-dt-surface-light rounded-full overflow-hidden">
              <div
                className={`h-full bg-${getUtilColor(gpu_utilization || 0)} rounded-full transition-all`}
                style={{ width: `${gpu_utilization || 0}%` }}
              />
            </div>
          </div>
        )}
        
        {/* VRAM Usage */}
        {vram_total_mb != null && vram_total_mb > 0 && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-dt-text-dim flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                VRAM
              </span>
              <span className="text-xs font-mono text-dt-text">
                {formatMemory(vram_used_mb || 0)} / {formatMemory(vram_total_mb)}
              </span>
            </div>
            <div className="h-2 bg-dt-surface-light rounded-full overflow-hidden">
              <div
                className={`h-full bg-${getUtilColor(vram_utilization || 0)} rounded-full transition-all`}
                style={{ width: `${vram_utilization || 0}%` }}
              />
            </div>
          </div>
        )}
        
        {/* Active Models */}
        {active_models?.length > 0 && (
          <div className="pt-2 border-t border-dt-border">
            <span className="text-xs text-dt-text-dim">Active Models</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {active_models.map((model, idx) => (
                <span
                  key={idx}
                  className="text-xs font-mono px-1.5 py-0.5 rounded bg-dt-accent/10 text-dt-accent animate-pulse"
                >
                  {model}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

