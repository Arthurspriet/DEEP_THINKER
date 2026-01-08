import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getMission, abortMission, resumeMission, getMissionArtifacts, getMissionLogs, getMissionAlignment } from '../api/missions'
import { getAgentTraces } from '../api/agents'
import { useMissionSSE, SSE_EVENT_TYPES } from '../hooks/useSSE'

// Existing components
import MissionHeader from '../components/MissionHeader'
import PhasesTimeline from '../components/PhasesTimeline'
import MissionFactory from '../components/MissionFactory'
import LiveExecutionPanel from '../components/LiveExecutionPanel'
import ArtifactsPanel from '../components/ArtifactsPanel'
import LogsPanel from '../components/LogsPanel'
import TimelineBar from '../components/TimelineBar'
import CouncilModal from '../components/CouncilModal'
import LoadingSpinner from '../components/LoadingSpinner'

// NEW panels
import GovernancePanel from '../components/GovernancePanel'
import ResearchProgressPanel from '../components/ResearchProgressPanel'
import MLGovernancePanel from '../components/MLGovernancePanel'
import SupervisorPanel from '../components/SupervisorPanel'
import DeepAnalysisPanel from '../components/DeepAnalysisPanel'
import SynthesisPanel from '../components/SynthesisPanel'
import EpistemicPanel from '../components/EpistemicPanel'
import PhaseErrorsPanel from '../components/PhaseErrorsPanel'
import AlignmentPanel from '../components/AlignmentPanel'

export default function MissionView() {
  const { id } = useParams()
  const navigate = useNavigate()
  
  const [mission, setMission] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [artifacts, setArtifacts] = useState([])
  const [logs, setLogs] = useState([])
  const [selectedPhase, setSelectedPhase] = useState(null)
  const [councilModalData, setCouncilModalData] = useState(null)
  const [activeSection, setActiveSection] = useState('execution') // execution, analysis, synthesis
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true)
  const [alignmentData, setAlignmentData] = useState(null)

  // Load mission data
  const loadMission = useCallback(async () => {
    try {
      const data = await getMission(id)
      setMission(data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [id])

  // Load artifacts
  const loadArtifacts = useCallback(async () => {
    try {
      const data = await getMissionArtifacts(id)
      setArtifacts(data)
    } catch (err) {
      console.error('Failed to load artifacts:', err)
    }
  }, [id])

  // Load logs
  const loadLogs = useCallback(async () => {
    try {
      const data = await getMissionLogs(id, 200)
      setLogs(data.logs || [])
    } catch (err) {
      console.error('Failed to load logs:', err)
    }
  }, [id])

  // Load alignment data
  const loadAlignment = useCallback(async () => {
    try {
      const data = await getMissionAlignment(id)
      setAlignmentData(data)
    } catch (err) {
      console.error('Failed to load alignment:', err)
      // Alignment errors are non-fatal - just log them
    }
  }, [id])

  // SSE connection with enhanced hook
  const isActive = mission && !mission.status?.match(/completed|failed|aborted|expired/)
  const sse = useMissionSSE(id, isActive)
  
  // Handle SSE events for mission state updates
  useEffect(() => {
    if (!sse.events.length) return
    
    const latest = sse.events[sse.events.length - 1]
    
    if (latest.type === 'phase_started' || latest.type === 'phase_completed') {
      loadMission()
      loadAlignment() // Refresh alignment on phase changes
    } else if (latest.type === 'artifact_generated') {
      loadArtifacts()
    } else if (latest.type === 'log_added') {
      setLogs(prev => [...prev, `[${latest.timestamp}] ${latest.data.message}`])
    } else if (latest.type === 'mission_completed') {
      loadMission()
      loadArtifacts()
      loadAlignment()
    }
  }, [sse.events, loadMission, loadArtifacts, loadAlignment])

  // Initial load
  useEffect(() => {
    loadMission()
    loadArtifacts()
    loadLogs()
    loadAlignment()
  }, [loadMission, loadArtifacts, loadLogs, loadAlignment])

  // Periodic refresh for non-terminal missions
  useEffect(() => {
    if (!mission || mission.status?.match(/completed|failed|aborted|expired/)) return
    
    const interval = setInterval(() => {
      loadMission()
      loadLogs()
    }, 5000)
    
    return () => clearInterval(interval)
  }, [mission, loadMission, loadLogs])

  // Handle abort
  const handleAbort = async () => {
    if (!confirm('Are you sure you want to abort this mission?')) return
    try {
      await abortMission(id)
      loadMission()
    } catch (err) {
      console.error('Failed to abort:', err)
    }
  }

  // Handle resume
  const handleResume = async () => {
    try {
      await resumeMission(id)
      loadMission()
    } catch (err) {
      console.error('Failed to resume:', err)
    }
  }

  // Handle council details click
  const handleCouncilDetails = async (phaseName, councilName) => {
    try {
      const traces = await getAgentTraces(councilName, 50, id)
      const phase = mission.phases.find(p => p.name === phaseName)
      
      setCouncilModalData({
        phaseName,
        councilName,
        traces,
        artifacts: phase?.artifacts || {},
        iterations: phase?.iterations || 0,
      })
    } catch (err) {
      console.error('Failed to load council details:', err)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="glass rounded-xl p-8 text-center">
          <p className="text-dt-error mb-4">{error}</p>
          <button
            onClick={() => navigate('/missions')}
            className="px-4 py-2 rounded-lg bg-dt-surface-light text-dt-text hover:bg-dt-border transition-colors"
          >
            Back to Missions
          </button>
        </div>
      </div>
    )
  }

  if (!mission) return null

  const currentPhase = mission.phases[mission.current_phase_index]
  const totalTime = mission.constraints?.time_budget_minutes || 60
  const elapsedTime = totalTime - mission.remaining_minutes
  const progressPercent = Math.min(100, (elapsedTime / totalTime) * 100)
  const isRunning = mission.status === 'running'

  // Section tabs for the main content area
  const sectionTabs = [
    { id: 'execution', label: 'Execution', icon: '⚡' },
    { id: 'analysis', label: 'Analysis', icon: '🔬' },
    { id: 'synthesis', label: 'Synthesis', icon: '📄' },
  ]

  return (
    <div className="min-h-screen bg-dt-bg">
      {/* Background grid pattern */}
      <div className="fixed inset-0 bg-grid opacity-30 pointer-events-none" />
      
      <div className="relative">
        {/* Main container with sidebar layout */}
        <div className="flex">
          {/* Main content area */}
          <div className={`flex-1 transition-all duration-300 ${rightSidebarOpen ? 'mr-80' : ''}`}>
            <div className="max-w-[1400px] mx-auto px-6 py-6 space-y-6">
              {/* Header */}
              <MissionHeader
                title={mission.objective}
                status={mission.status}
                remainingMinutes={mission.remaining_minutes}
                isConnected={sse.isConnected}
                onAbort={handleAbort}
                onResume={handleResume}
                onBack={() => navigate('/missions')}
              />

              {/* Phases Timeline */}
              <PhasesTimeline
                phases={mission.phases}
                currentPhaseIndex={mission.current_phase_index}
                onPhaseClick={setSelectedPhase}
              />

              {/* Mission Factory (conveyor visualization) */}
              <MissionFactory
                phases={mission.phases}
                currentPhaseIndex={mission.current_phase_index}
                onPhaseDetails={handleCouncilDetails}
              />

              {/* Section Tabs */}
              <div className="flex items-center gap-2 border-b border-dt-border pb-4">
                {sectionTabs.map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveSection(tab.id)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      activeSection === tab.id
                        ? 'bg-dt-accent/10 text-dt-accent border border-dt-accent/30'
                        : 'bg-dt-surface-light text-dt-text-dim hover:bg-dt-surface-lighter border border-transparent'
                    }`}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* EXECUTION Section */}
              {activeSection === 'execution' && (
                <div className="space-y-6 animate-fade-in">
                  {/* Live Execution Panel */}
                  {isRunning && (
                    <LiveExecutionPanel sseEvents={sse.events} />
                  )}

                  {/* Supervisor + Research Row */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <SupervisorPanel 
                      supervisorData={sse.supervisor || sse.modelSelection}
                    />
                    <ResearchProgressPanel 
                      researchData={sse.research}
                    />
                  </div>

                  {/* Phase Errors */}
                  {(sse.phaseError || sse.errorHistory?.length > 0) && (
                    <PhaseErrorsPanel 
                      errorData={sse.phaseError}
                      errorHistory={sse.errorHistory}
                    />
                  )}

                  {/* Artifacts & Logs */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <ArtifactsPanel
                      artifacts={artifacts}
                      phases={mission.phases}
                      currentPhase={currentPhase?.name}
                      selectedPhase={selectedPhase}
                      onPhaseSelect={setSelectedPhase}
                    />
                    <LogsPanel logs={logs} />
                  </div>
                </div>
              )}

              {/* ANALYSIS Section */}
              {activeSection === 'analysis' && (
                <div className="space-y-6 animate-fade-in">
                  {/* Deep Analysis + Epistemic Row */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <DeepAnalysisPanel 
                      analysisData={sse.deepAnalysis}
                    />
                    <EpistemicPanel 
                      epistemicData={sse.epistemic}
                    />
                  </div>

                  {/* Multi-View if available from LiveExecution */}
                  {sse.multiViewAnalysis && (
                    <div className="glass rounded-xl p-6">
                      <h3 className="text-sm font-semibold text-white/90 uppercase tracking-wide mb-4">
                        Multi-View Analysis
                      </h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                          <div className="text-emerald-400 text-xs uppercase tracking-wide mb-2">Optimist</div>
                          <div className="text-white/70 text-sm">
                            Confidence: {((sse.multiViewAnalysis.optimist_confidence || 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="p-4 rounded-lg bg-red-500/5 border border-red-500/20">
                          <div className="text-red-400 text-xs uppercase tracking-wide mb-2">Skeptic</div>
                          <div className="text-white/70 text-sm">
                            Confidence: {((sse.multiViewAnalysis.skeptic_confidence || 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* SYNTHESIS Section */}
              {activeSection === 'synthesis' && (
                <div className="space-y-6 animate-fade-in">
                  <SynthesisPanel 
                    synthesisData={sse.synthesis}
                    synthesisHistory={sse.synthesisHistory}
                  />

                  {/* Final Artifacts */}
                  <ArtifactsPanel
                    artifacts={artifacts.filter(a => a.type === 'final' || a.name?.includes('report'))}
                    phases={mission.phases}
                    currentPhase={currentPhase?.name}
                    selectedPhase={selectedPhase}
                    onPhaseSelect={setSelectedPhase}
                  />
                </div>
              )}

              {/* Timeline Bar */}
              <TimelineBar
                elapsedMinutes={elapsedTime}
                totalMinutes={totalTime}
                progress={progressPercent}
                status={mission.status}
              />
            </div>
          </div>

          {/* Right Sidebar - Governance & ML */}
          <div 
            className={`fixed right-0 top-0 h-full w-80 bg-dt-bg-elevated border-l border-dt-border overflow-y-auto transition-transform duration-300 ${
              rightSidebarOpen ? 'translate-x-0' : 'translate-x-full'
            }`}
          >
            {/* Sidebar Toggle */}
            <button
              onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
              className="absolute -left-10 top-20 w-10 h-10 bg-dt-surface-light border border-dt-border rounded-l-lg flex items-center justify-center text-dt-text-dim hover:text-dt-accent transition-colors"
            >
              <svg 
                className={`w-5 h-5 transition-transform ${rightSidebarOpen ? '' : 'rotate-180'}`} 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>

            <div className="p-4 space-y-4">
              {/* Sidebar Header */}
              <div className="flex items-center justify-between pb-3 border-b border-dt-border">
                <h2 className="text-sm font-semibold text-white/90 uppercase tracking-wide">
                  System Status
                </h2>
                <div className={`w-2 h-2 rounded-full ${sse.isConnected ? 'bg-emerald-400' : 'bg-red-400'}`} />
              </div>

              {/* Alignment Panel */}
              <AlignmentPanel 
                alignmentData={alignmentData}
                sseAlignment={sse.alignment}
                sseCorrections={sse.alignmentCorrections}
              />

              {/* Governance Panel */}
              <GovernancePanel 
                governanceData={sse.governance}
              />

              {/* ML Governance Panel */}
              <MLGovernancePanel 
                mlData={sse.mlGovernance}
              />

              {/* Resource Stats */}
              {sse.resourceUpdate && (
                <div className="glass rounded-xl p-4">
                  <h3 className="text-xs text-white/50 uppercase tracking-wide mb-3">Resources</h3>
                  <div className="space-y-3">
                    {/* GPU */}
                    {sse.resourceUpdate.gpu_total > 0 && (
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/40">GPU</span>
                          <span className="text-white/60 font-mono">
                            {sse.resourceUpdate.gpu_available}/{sse.resourceUpdate.gpu_total}
                          </span>
                        </div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-violet-500 to-indigo-500 rounded-full"
                            style={{ width: `${sse.resourceUpdate.gpu_utilization || 0}%` }}
                          />
                        </div>
                      </div>
                    )}
                    
                    {/* VRAM */}
                    {sse.resourceUpdate.vram_total_mb > 0 && (
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/40">VRAM</span>
                          <span className="text-white/60 font-mono">
                            {(sse.resourceUpdate.vram_used_mb / 1024).toFixed(1)}GB
                          </span>
                        </div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-cyan-500 to-teal-500 rounded-full"
                            style={{ width: `${sse.resourceUpdate.vram_utilization || 0}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Time */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-white/40">Time</span>
                        <span className="text-white/60 font-mono">
                          {mission.remaining_minutes?.toFixed(0)}m left
                        </span>
                      </div>
                      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-amber-500 to-orange-500 rounded-full"
                          style={{ width: `${progressPercent}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Active Models */}
              {sse.resourceUpdate?.active_models?.length > 0 && (
                <div className="glass rounded-xl p-4">
                  <h3 className="text-xs text-white/50 uppercase tracking-wide mb-2">Active Models</h3>
                  <div className="flex flex-wrap gap-1.5">
                    {sse.resourceUpdate.active_models.map((model, idx) => (
                      <span 
                        key={idx}
                        className="px-2 py-1 rounded text-xs font-mono bg-dt-accent/10 text-dt-accent border border-dt-accent/20 animate-pulse"
                      >
                        {model}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Connection Status */}
              <div className="glass rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-white/40">SSE Connection</span>
                  <span className={`text-xs font-medium ${sse.isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
                    {sse.isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                {!sse.isConnected && sse.error && (
                  <button
                    onClick={() => sse.reconnect()}
                    className="mt-2 w-full py-1.5 rounded-lg text-xs bg-dt-accent/10 text-dt-accent hover:bg-dt-accent/20 transition-colors"
                  >
                    Reconnect
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Council Modal */}
      {councilModalData && (
        <CouncilModal
          data={councilModalData}
          onClose={() => setCouncilModalData(null)}
        />
      )}
    </div>
  )
}
