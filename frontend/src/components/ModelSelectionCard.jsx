/**
 * Shows model selection decision with reasoning from SSE events.
 */
export default function ModelSelectionCard({ selectionData }) {
  if (!selectionData) {
    return null
  }
  
  const {
    phase_name,
    models,
    reason,
    downgraded,
    wait_for_capacity,
    fallback_models,
    phase_importance,
    estimated_vram,
    time_remaining,
    total_time,
    time_percent,
    gpu_stats
  } = selectionData
  
  const statusBadge = () => {
    if (downgraded) {
      return (
        <span className="text-xs px-2 py-0.5 rounded-full bg-dt-warning/20 text-dt-warning flex items-center gap-1">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          Downgraded
        </span>
      )
    }
    if (wait_for_capacity) {
      return (
        <span className="text-xs px-2 py-0.5 rounded-full bg-dt-accent/20 text-dt-accent flex items-center gap-1">
          <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Waiting for GPU
        </span>
      )
    }
    return (
      <span className="text-xs px-2 py-0.5 rounded-full bg-dt-success/20 text-dt-success flex items-center gap-1">
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
        Optimal
      </span>
    )
  }
  
  return (
    <div className="glass rounded-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="text-sm font-medium text-dt-text flex items-center gap-2">
            <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            Model Selection
          </h4>
          <p className="text-xs text-dt-text-dim mt-0.5">{phase_name}</p>
        </div>
        {statusBadge()}
      </div>
      
      {/* Selected Models */}
      <div className="mb-3">
        <div className="flex flex-wrap gap-2">
          {models?.map((model, idx) => (
            <span
              key={idx}
              className="text-xs font-mono px-2 py-1 rounded bg-dt-accent/10 text-dt-accent border border-dt-accent/30"
            >
              {model}
            </span>
          ))}
        </div>
      </div>
      
      {/* Reasoning */}
      {reason && (
        <div className="mb-3 p-2 rounded bg-dt-surface-light/50">
          <p className="text-xs text-dt-text-dim">{reason}</p>
        </div>
      )}
      
      {/* Metrics */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <span className="text-dt-text-dim">Phase Importance</span>
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 h-1.5 bg-dt-surface-light rounded-full overflow-hidden">
              <div
                className="h-full bg-dt-accent rounded-full transition-all"
                style={{ width: `${(phase_importance || 0) * 100}%` }}
              />
            </div>
            <span className="text-dt-text font-mono">{((phase_importance || 0) * 100).toFixed(0)}%</span>
          </div>
        </div>
        
        <div>
          <span className="text-dt-text-dim">Time Budget</span>
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 h-1.5 bg-dt-surface-light rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  (time_percent || 0) > 80 ? 'bg-dt-error' : 
                  (time_percent || 0) > 50 ? 'bg-dt-warning' : 'bg-dt-success'
                }`}
                style={{ width: `${time_percent || 0}%` }}
              />
            </div>
            <span className="text-dt-text font-mono">{(time_remaining || 0).toFixed(1)}m</span>
          </div>
        </div>
      </div>
      
      {/* GPU Stats */}
      {gpu_stats && (gpu_stats.available_gpus != null || gpu_stats.vram_used_mb != null) && (
        <div className="mt-3 pt-3 border-t border-dt-border grid grid-cols-2 gap-3 text-xs">
          {gpu_stats.available_gpus != null && (
            <div>
              <span className="text-dt-text-dim">GPUs Available</span>
              <p className="text-dt-text font-mono mt-0.5">{gpu_stats.available_gpus}</p>
            </div>
          )}
          {gpu_stats.vram_used_mb != null && gpu_stats.vram_total_mb != null && (
            <div>
              <span className="text-dt-text-dim">VRAM Usage</span>
              <p className="text-dt-text font-mono mt-0.5">
                {Math.round(gpu_stats.vram_used_mb / 1024)}GB / {Math.round(gpu_stats.vram_total_mb / 1024)}GB
              </p>
            </div>
          )}
        </div>
      )}
      
      {/* Fallback Models */}
      {fallback_models?.length > 0 && (
        <div className="mt-3 pt-3 border-t border-dt-border">
          <span className="text-xs text-dt-text-dim">Fallback models:</span>
          <div className="flex flex-wrap gap-1 mt-1">
            {fallback_models.map((model, idx) => (
              <span
                key={idx}
                className="text-xs font-mono px-1.5 py-0.5 rounded bg-dt-surface-light text-dt-text-dim"
              >
                {model}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

