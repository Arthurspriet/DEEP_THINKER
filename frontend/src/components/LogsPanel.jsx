import { useEffect, useRef } from 'react'

export default function LogsPanel({ logs }) {
  const containerRef = useRef(null)
  const autoScrollRef = useRef(true)

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScrollRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs])

  // Track if user has scrolled up
  const handleScroll = () => {
    if (!containerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 50
  }

  const parseLogLevel = (log) => {
    if (log.includes('ERROR') || log.includes('error')) return 'error'
    if (log.includes('WARNING') || log.includes('warn')) return 'warning'
    if (log.includes('SUCCESS') || log.includes('success') || log.includes('✓')) return 'success'
    return 'info'
  }

  const getLevelStyle = (level) => {
    switch (level) {
      case 'error': return 'text-dt-error'
      case 'warning': return 'text-dt-warning'
      case 'success': return 'text-dt-success'
      default: return 'text-dt-text-dim'
    }
  }

  return (
    <div className="glass rounded-xl overflow-hidden animate-fade-in flex flex-col h-[400px]">
      {/* Header */}
      <div className="p-4 border-b border-dt-border flex items-center justify-between shrink-0">
        <h3 className="text-sm font-medium text-dt-text flex items-center gap-2">
          <svg className="w-4 h-4 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          Execution Logs
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-dt-text-dim">
            {logs.length} entries
          </span>
          {autoScrollRef.current && (
            <span className="text-xs text-dt-success flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-dt-success animate-pulse" />
              Live
            </span>
          )}
        </div>
      </div>

      {/* Logs */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-1"
      >
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-dt-text-dim">
            <svg className="w-12 h-12 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <p className="text-sm">Waiting for logs...</p>
          </div>
        ) : (
          logs.map((log, idx) => {
            const level = parseLogLevel(log)
            const levelStyle = getLevelStyle(level)
            
            // Extract timestamp if present
            const timestampMatch = log.match(/^\[([^\]]+)\]/)
            const timestamp = timestampMatch ? timestampMatch[1] : null
            const message = timestampMatch ? log.slice(timestampMatch[0].length).trim() : log
            
            return (
              <div
                key={idx}
                className={`flex gap-2 py-1 px-2 rounded hover:bg-dt-surface-light/50 transition-colors ${levelStyle}`}
              >
                {timestamp && (
                  <span className="text-dt-text-dim/50 shrink-0 w-[180px]">
                    {formatTimestamp(timestamp)}
                  </span>
                )}
                <span className="break-all">{message}</span>
              </div>
            )
          })
        )}
      </div>

      {/* Scroll to bottom button */}
      {!autoScrollRef.current && logs.length > 0 && (
        <button
          onClick={() => {
            if (containerRef.current) {
              containerRef.current.scrollTop = containerRef.current.scrollHeight
              autoScrollRef.current = true
            }
          }}
          className="absolute bottom-16 right-4 p-2 rounded-lg bg-dt-accent text-dt-bg shadow-lg hover:bg-dt-accent/90 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </button>
      )}
    </div>
  )
}

function formatTimestamp(timestamp) {
  try {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
  } catch {
    return timestamp.slice(0, 8)
  }
}

