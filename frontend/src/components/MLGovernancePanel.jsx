import { useState } from 'react'

/**
 * MLGovernancePanel - Shows ML predictor health and monitoring
 * Displays system health, per-predictor status, and advisory readiness
 */
export default function MLGovernancePanel({ mlData, className = '' }) {
  const [expanded, setExpanded] = useState(false)
  
  if (!mlData) {
    return null
  }
  
  const {
    system_health,
    system_health_percent,
    predictor_status,
    advisory_readiness,
    advisory_ready,
    alerts,
    alerts_count
  } = mlData
  
  const healthPercent = system_health_percent || (system_health || 0) * 100
  const advisoryPercent = (advisory_readiness || 0) * 100
  
  // Health status config
  const getHealthConfig = (percent) => {
    if (percent >= 70) return { label: 'HEALTHY', color: 'emerald', icon: '✓' }
    if (percent >= 40) return { label: 'WARNING', color: 'yellow', icon: '⚠' }
    return { label: 'CRITICAL', color: 'red', icon: '✗' }
  }
  
  const healthConfig = getHealthConfig(healthPercent)
  const hasAlerts = alerts_count > 0 || (alerts && alerts.length > 0)
  const predictors = predictor_status ? Object.entries(predictor_status) : []
  
  return (
    <div className={`rounded-xl border border-violet-500/30 bg-violet-500/5 shadow-lg shadow-violet-500/10 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white/90 tracking-wide uppercase">
              ML Governance
            </h3>
            <span className="text-xs text-white/40">Predictor Health</span>
          </div>
        </div>
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-${healthConfig.color}-500/10 text-${healthConfig.color}-400 border border-${healthConfig.color}-500/30`}>
          <span>{healthConfig.icon}</span>
          <span>{healthConfig.label}</span>
        </div>
      </div>
      
      {/* System Health Gauge */}
      <div className="px-4 py-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-white/50 uppercase tracking-wide">System Health</span>
          <span className={`text-lg font-mono font-bold text-${healthConfig.color}-400`}>
            {healthPercent.toFixed(0)}%
          </span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              healthPercent >= 70 ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' :
              healthPercent >= 40 ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
              'bg-gradient-to-r from-red-500 to-red-400'
            }`}
            style={{ width: `${Math.min(healthPercent, 100)}%` }}
          />
        </div>
      </div>
      
      {/* Predictor Status List */}
      {predictors.length > 0 && (
        <div className="px-4 pb-3 space-y-2">
          <span className="text-xs text-white/40 uppercase tracking-wide">Predictors</span>
          {predictors.map(([name, status]) => {
            const metrics = status.metrics || {}
            const predHealth = status.health_score || 0.5
            const isHealthy = status.status === 'healthy'
            const isWarning = status.status === 'warning'
            const fallbackRate = (metrics.fallback_rate || 0) * 100
            const predCount = metrics.prediction_count || 0
            const avgConf = metrics.avg_confidence || 0
            
            return (
              <div key={name} className="flex items-center justify-between p-2 rounded-lg bg-white/5">
                <div className="flex items-center gap-2">
                  <span className={`text-sm ${
                    isHealthy ? 'text-emerald-400' :
                    isWarning ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {isHealthy ? '✓' : isWarning ? '⚠' : '✗'}
                  </span>
                  <span className="text-xs font-mono text-white/70">{name}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-white/40">
                  <span>{predCount} pred</span>
                  <span className={fallbackRate > 50 ? 'text-amber-400' : ''}>
                    {fallbackRate.toFixed(0)}% fb
                  </span>
                  <span className="font-mono text-white/50">
                    {avgConf.toFixed(2)}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
      
      {/* Advisory Readiness */}
      <div className="px-4 py-3 border-t border-white/5">
        <div className="flex items-center justify-between">
          <span className="text-xs text-white/50">Advisory Readiness</span>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-white/60">
              {advisoryPercent.toFixed(0)}%
            </span>
            <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
              advisory_ready 
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
            }`}>
              {advisory_ready ? 'Ready' : 'Not Ready'}
            </span>
          </div>
        </div>
      </div>
      
      {/* Alerts */}
      {hasAlerts && (
        <div className="px-4 py-2 border-t border-white/5">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-between text-left hover:bg-white/5 rounded-lg px-2 py-1.5 transition-colors -mx-2"
          >
            <span className="text-xs text-amber-400 font-medium flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              Alerts ({alerts_count || alerts?.length || 0})
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
          
          {expanded && alerts && alerts.length > 0 && (
            <div className="mt-2 space-y-1.5">
              {alerts.slice(0, 5).map((alert, idx) => {
                const severity = alert.severity || 0
                const severityColor = severity > 0.8 ? 'red' : severity > 0.5 ? 'yellow' : 'emerald'
                
                return (
                  <div 
                    key={idx}
                    className="flex items-center justify-between px-2 py-1.5 rounded-lg bg-white/5"
                  >
                    <div className="flex items-center gap-2">
                      <span className={`text-${severityColor}-400 text-xs`}>
                        {severity > 0.8 ? '🔴' : severity > 0.5 ? '🟡' : '🟢'}
                      </span>
                      <span className="text-xs text-white/60">
                        {alert.drift_type || 'Alert'} 
                        {alert.predictor_name && (
                          <span className="text-white/30 ml-1">({alert.predictor_name})</span>
                        )}
                      </span>
                    </div>
                    <span className={`text-xs font-mono text-${severityColor}-400`}>
                      {(severity * 100).toFixed(0)}%
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
      
      {/* No Alerts */}
      {!hasAlerts && (
        <div className="px-4 py-2 border-t border-white/5 flex items-center justify-between">
          <span className="text-xs text-white/40">Alerts</span>
          <span className="text-xs text-emerald-400 flex items-center gap-1">
            <span>✓</span> None
          </span>
        </div>
      )}
    </div>
  )
}

