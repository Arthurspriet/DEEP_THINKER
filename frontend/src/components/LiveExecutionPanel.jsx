import { useState, useEffect, useRef } from 'react'
import StepExecutionPanel from './StepExecutionPanel'
import ModelSelectionCard from './ModelSelectionCard'
import ConsensusGauge from './ConsensusGauge'
import ResourceBar from './ResourceBar'
import MultiViewAnalysis from './MultiViewAnalysis'
import PhaseMetrics from './PhaseMetrics'

/**
 * Container for live execution panels that visualize SSE events.
 * Shows real-time step execution, model selection, consensus, and resource data.
 */
export default function LiveExecutionPanel({ sseEvents }) {
  const [latestStep, setLatestStep] = useState(null)
  const [latestModelSelection, setLatestModelSelection] = useState(null)
  const [latestConsensus, setLatestConsensus] = useState(null)
  const [latestResource, setLatestResource] = useState(null)
  const [latestMultiView, setLatestMultiView] = useState(null)
  const [latestPhaseMetrics, setLatestPhaseMetrics] = useState(null)
  const [modelExecutions, setModelExecutions] = useState([])
  const [isMinimized, setIsMinimized] = useState(false)
  
  // Process SSE events
  useEffect(() => {
    if (!sseEvents || sseEvents.length === 0) return
    
    const latest = sseEvents[sseEvents.length - 1]
    
    switch (latest.type) {
      case 'step_execution':
        setLatestStep(latest.data)
        break
      case 'model_selection':
        setLatestModelSelection(latest.data)
        break
      case 'consensus_result':
        setLatestConsensus(latest.data)
        break
      case 'resource_update':
        setLatestResource(latest.data)
        break
      case 'multi_view_analysis':
        setLatestMultiView(latest.data)
        break
      case 'phase_metrics':
        setLatestPhaseMetrics(latest.data)
        break
      case 'model_execution':
        setModelExecutions(prev => [...prev.slice(-9), latest.data])
        break
    }
  }, [sseEvents])
  
  // Check if we have any data to show
  const hasData = latestStep || latestModelSelection || latestConsensus || latestResource || latestMultiView || latestPhaseMetrics || modelExecutions.length > 0
  
  if (!hasData) {
    return null
  }
  
  return (
    <div className="glass rounded-xl overflow-hidden animate-fade-in">
      {/* Header */}
      <div 
        className="p-4 border-b border-dt-border flex items-center justify-between cursor-pointer hover:bg-dt-surface-light/30 transition-colors"
        onClick={() => setIsMinimized(!isMinimized)}
      >
        <h3 className="text-sm font-medium text-dt-text flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-dt-accent opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-dt-accent"></span>
          </span>
          Live Execution
        </h3>
        <div className="flex items-center gap-3">
          {latestStep?.status === 'running' && (
            <span className="text-xs text-dt-accent animate-pulse">
              {latestStep.step_name}
            </span>
          )}
          <svg
            className={`w-4 h-4 text-dt-text-dim transition-transform ${isMinimized ? '' : 'rotate-180'}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
      
      {/* Content */}
      {!isMinimized && (
        <div className="p-4 space-y-4">
          {/* Top row: Step + Model Selection */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <StepExecutionPanel stepData={latestStep} />
            <ModelSelectionCard selectionData={latestModelSelection} />
          </div>
          
          {/* Bottom row: Consensus + Resources */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ConsensusGauge consensusData={latestConsensus} />
            <ResourceBar resourceData={latestResource} />
          </div>
          
          {/* Multi-View Analysis and Phase Metrics */}
          {(latestMultiView || latestPhaseMetrics) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-4 border-t border-dt-border">
              <MultiViewAnalysis multiViewData={latestMultiView} />
              <PhaseMetrics metricsData={latestPhaseMetrics} />
            </div>
          )}
          
          {/* Model Execution History */}
          {modelExecutions.length > 0 && (
            <div className="pt-4 border-t border-dt-border">
              <h4 className="text-xs font-medium text-dt-text-dim mb-2 flex items-center gap-2">
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Recent Model Calls
              </h4>
              <div className="flex flex-wrap gap-2">
                {modelExecutions.map((exec, idx) => (
                  <div
                    key={idx}
                    className={`text-xs px-2 py-1 rounded flex items-center gap-1.5 ${
                      exec.success
                        ? 'bg-dt-success/10 text-dt-success border border-dt-success/20'
                        : 'bg-dt-error/10 text-dt-error border border-dt-error/20'
                    }`}
                    title={exec.output_preview || exec.error}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${exec.success ? 'bg-dt-success' : 'bg-dt-error'}`} />
                    <span className="font-mono">{exec.model_name}</span>
                    {exec.duration_s && (
                      <span className="opacity-70">{exec.duration_s.toFixed(1)}s</span>
                    )}
                    {exec.tokens_out && (
                      <span className="opacity-50">{exec.tokens_out}t</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

