import { useEffect, useRef, useState, useCallback, useMemo } from 'react'

/**
 * All SSE event types supported by DeepThinker
 */
export const SSE_EVENT_TYPES = {
  // Connection
  CONNECTED: 'connected',
  
  // Phase lifecycle
  PHASE_STARTED: 'phase_started',
  PHASE_COMPLETED: 'phase_completed',
  PHASE_ERROR: 'phase_error',
  
  // Council execution
  COUNCIL_STARTED: 'council_started',
  COUNCIL_COMPLETED: 'council_completed',
  
  // Artifacts
  ARTIFACT_GENERATED: 'artifact_generated',
  
  // Logging
  LOG_ADDED: 'log_added',
  
  // Mission lifecycle
  MISSION_COMPLETED: 'mission_completed',
  
  // Meta-cognition
  META_UPDATE: 'meta_update',
  
  // Execution details
  STEP_EXECUTION: 'step_execution',
  MODEL_SELECTION: 'model_selection',
  CONSENSUS_RESULT: 'consensus_result',
  RESOURCE_UPDATE: 'resource_update',
  MODEL_EXECUTION: 'model_execution',
  MULTI_VIEW_ANALYSIS: 'multi_view_analysis',
  PHASE_METRICS: 'phase_metrics',
  
  // NEW: Governance & Epistemic
  GOVERNANCE_UPDATE: 'governance_update',
  RESEARCH_PROGRESS: 'research_progress',
  ML_GOVERNANCE: 'ml_governance',
  SYNTHESIS_ITERATION: 'synthesis_iteration',
  EPISTEMIC_UPDATE: 'epistemic_update',
  DEEP_ANALYSIS_UPDATE: 'deep_analysis_update',
  SUPERVISOR_DECISION: 'supervisor_decision',
  
  // NEW: Alignment Control Layer
  ALIGNMENT_UPDATE: 'alignment_update',
  ALIGNMENT_WARNING: 'alignment_warning',
  ALIGNMENT_CORRECTION: 'alignment_correction',
}

/**
 * Custom hook for SSE connections with automatic reconnection
 * Enhanced to track all new event types for the frontend overhaul
 * 
 * @param {string} url - SSE endpoint URL
 * @param {Object} options - Hook options
 */
export function useSSE(url, options = {}) {
  const {
    onMessage,
    onOpen,
    onError,
    onEvent = {},
    enabled = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState(null)
  const [events, setEvents] = useState([])
  
  // Track latest state for each event type
  const [latestByType, setLatestByType] = useState({})
  
  // Track history for specific event types
  const [eventHistory, setEventHistory] = useState({
    synthesis: [],
    errors: [],
    research: [],
    governance: [],
    alignment: [],
    alignmentCorrections: [],
  })
  
  const eventSourceRef = useRef(null)
  const reconnectAttempts = useRef(0)
  const reconnectTimeout = useRef(null)

  const connect = useCallback(() => {
    if (!enabled || !url) return

    try {
      const eventSource = new EventSource(url)
      eventSourceRef.current = eventSource

      eventSource.onopen = () => {
        setIsConnected(true)
        setError(null)
        reconnectAttempts.current = 0
        onOpen?.()
      }

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          const eventType = data.type
          const eventData = data.data

          // Store event in general list
          setEvents((prev) => [...prev.slice(-99), { ...data, receivedAt: new Date() }])

          // Update latest by type
          setLatestByType((prev) => ({
            ...prev,
            [eventType]: eventData
          }))

          // Track history for specific types
          if (eventType === SSE_EVENT_TYPES.SYNTHESIS_ITERATION) {
            setEventHistory((prev) => ({
              ...prev,
              synthesis: [...prev.synthesis.slice(-9), eventData]
            }))
          } else if (eventType === SSE_EVENT_TYPES.PHASE_ERROR) {
            setEventHistory((prev) => ({
              ...prev,
              errors: [...prev.errors.slice(-19), eventData]
            }))
          } else if (eventType === SSE_EVENT_TYPES.RESEARCH_PROGRESS) {
            setEventHistory((prev) => ({
              ...prev,
              research: [...prev.research.slice(-19), eventData]
            }))
          } else if (eventType === SSE_EVENT_TYPES.GOVERNANCE_UPDATE) {
            setEventHistory((prev) => ({
              ...prev,
              governance: [...prev.governance.slice(-19), eventData]
            }))
          } else if (eventType === SSE_EVENT_TYPES.ALIGNMENT_UPDATE) {
            setEventHistory((prev) => ({
              ...prev,
              alignment: [...prev.alignment.slice(-49), eventData]
            }))
          } else if (eventType === SSE_EVENT_TYPES.ALIGNMENT_CORRECTION) {
            setEventHistory((prev) => ({
              ...prev,
              alignmentCorrections: [...prev.alignmentCorrections.slice(-19), eventData]
            }))
          }

          // Call type-specific handler
          if (onEvent[eventType]) {
            onEvent[eventType](eventData, data.timestamp)
          }

          // Call general handler
          onMessage?.(data)
        } catch (e) {
          console.error('Failed to parse SSE message:', e)
        }
      }

      eventSource.onerror = (err) => {
        setIsConnected(false)
        setError(err)
        onError?.(err)

        // Close the connection
        eventSource.close()
        eventSourceRef.current = null

        // Attempt reconnection
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++
          reconnectTimeout.current = setTimeout(connect, reconnectInterval)
        }
      }
    } catch (err) {
      setError(err)
      onError?.(err)
    }
  }, [url, enabled, onMessage, onOpen, onError, onEvent, reconnectInterval, maxReconnectAttempts])

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    setIsConnected(false)
  }, [])

  const clearEvents = useCallback(() => {
    setEvents([])
    setLatestByType({})
    setEventHistory({
      synthesis: [],
      errors: [],
      research: [],
      governance: [],
      alignment: [],
      alignmentCorrections: [],
    })
  }, [])

  // Get latest event of a specific type
  const getLatest = useCallback((eventType) => {
    return latestByType[eventType] || null
  }, [latestByType])

  // Get all events of a specific type
  const getEventsByType = useCallback((eventType) => {
    return events.filter(e => e.type === eventType).map(e => e.data)
  }, [events])

  useEffect(() => {
    connect()
    return disconnect
  }, [connect, disconnect])

  // Memoized derived state for common queries
  const derivedState = useMemo(() => ({
    // Governance
    governance: latestByType[SSE_EVENT_TYPES.GOVERNANCE_UPDATE] || null,
    governanceHistory: eventHistory.governance,
    
    // Research
    research: latestByType[SSE_EVENT_TYPES.RESEARCH_PROGRESS] || null,
    researchHistory: eventHistory.research,
    
    // ML Governance
    mlGovernance: latestByType[SSE_EVENT_TYPES.ML_GOVERNANCE] || null,
    
    // Supervisor
    supervisor: latestByType[SSE_EVENT_TYPES.SUPERVISOR_DECISION] || null,
    modelSelection: latestByType[SSE_EVENT_TYPES.MODEL_SELECTION] || null,
    
    // Deep Analysis
    deepAnalysis: latestByType[SSE_EVENT_TYPES.DEEP_ANALYSIS_UPDATE] || null,
    
    // Synthesis
    synthesis: latestByType[SSE_EVENT_TYPES.SYNTHESIS_ITERATION] || null,
    synthesisHistory: eventHistory.synthesis,
    
    // Epistemic
    epistemic: latestByType[SSE_EVENT_TYPES.EPISTEMIC_UPDATE] || null,
    
    // Errors
    phaseError: latestByType[SSE_EVENT_TYPES.PHASE_ERROR] || null,
    errorHistory: eventHistory.errors,
    
    // Execution (existing)
    stepExecution: latestByType[SSE_EVENT_TYPES.STEP_EXECUTION] || null,
    consensusResult: latestByType[SSE_EVENT_TYPES.CONSENSUS_RESULT] || null,
    resourceUpdate: latestByType[SSE_EVENT_TYPES.RESOURCE_UPDATE] || null,
    multiViewAnalysis: latestByType[SSE_EVENT_TYPES.MULTI_VIEW_ANALYSIS] || null,
    phaseMetrics: latestByType[SSE_EVENT_TYPES.PHASE_METRICS] || null,
    modelExecution: latestByType[SSE_EVENT_TYPES.MODEL_EXECUTION] || null,
    
    // Alignment Control Layer
    alignment: latestByType[SSE_EVENT_TYPES.ALIGNMENT_UPDATE] || null,
    alignmentHistory: eventHistory.alignment,
    alignmentWarning: latestByType[SSE_EVENT_TYPES.ALIGNMENT_WARNING] || null,
    alignmentCorrections: eventHistory.alignmentCorrections,
  }), [latestByType, eventHistory])

  return {
    // Connection state
    isConnected,
    error,
    
    // Events
    events,
    latestByType,
    eventHistory,
    
    // Actions
    clearEvents,
    reconnect: connect,
    disconnect,
    
    // Helpers
    getLatest,
    getEventsByType,
    
    // Derived state for easy access
    ...derivedState,
  }
}

/**
 * Hook specifically for mission-related SSE events
 * Provides structured access to all panel data
 */
export function useMissionSSE(missionId, isActive = true) {
  const url = isActive && missionId ? `/api/missions/${missionId}/events` : null
  
  const sse = useSSE(url, {
    enabled: isActive,
  })
  
  return sse
}

export default useSSE
