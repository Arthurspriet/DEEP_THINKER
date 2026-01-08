import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getMissions, createMission } from '../api/missions'
import LoadingSpinner from '../components/LoadingSpinner'

const STATUS_COLORS = {
  running: 'bg-dt-success/20 text-dt-success border-dt-success/40',
  pending: 'bg-dt-warning/20 text-dt-warning border-dt-warning/40',
  completed: 'bg-dt-accent/20 text-dt-accent border-dt-accent/40',
  failed: 'bg-dt-error/20 text-dt-error border-dt-error/40',
  aborted: 'bg-dt-error/20 text-dt-error border-dt-error/40',
  expired: 'bg-dt-text-dim/20 text-dt-text-dim border-dt-text-dim/40',
}

function MissionCard({ mission, onClick }) {
  const statusClass = STATUS_COLORS[mission.status] || STATUS_COLORS.pending

  return (
    <div
      onClick={onClick}
      className="glass rounded-xl p-6 cursor-pointer hover:border-dt-accent/50 transition-all duration-300 group animate-fade-in"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`px-3 py-1 rounded-full text-xs font-medium border ${statusClass}`}>
          {mission.status}
        </div>
        <span className="text-dt-text-dim text-xs font-mono">
          {new Date(mission.created_at).toLocaleDateString()}
        </span>
      </div>

      <h3 className="text-lg font-medium text-dt-text mb-2 line-clamp-2 group-hover:text-dt-accent transition-colors">
        {mission.objective}
      </h3>

      <div className="mt-4 space-y-3">
        {/* Progress bar */}
        <div className="w-full bg-dt-surface-light rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-dt-accent to-dt-success transition-all duration-500"
            style={{ width: `${mission.progress_percent}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-dt-text-dim">
            {mission.completed_phases}/{mission.total_phases} phases
          </span>
          <span className="text-dt-text-dim font-mono">
            {mission.remaining_minutes > 0
              ? `${Math.floor(mission.remaining_minutes)}m remaining`
              : 'Completed'}
          </span>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-end">
        <button className="text-dt-accent text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
          Open Mission
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  )
}

function CreateMissionModal({ isOpen, onClose, onCreate }) {
  const [objective, setObjective] = useState('')
  const [timeBudget, setTimeBudget] = useState(60)
  const [allowInternet, setAllowInternet] = useState(true)
  const [allowCodeExecution, setAllowCodeExecution] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!objective.trim()) return

    setIsSubmitting(true)
    try {
      await onCreate({
        objective: objective.trim(),
        time_budget_minutes: timeBudget,
        allow_internet: allowInternet,
        allow_code_execution: allowCodeExecution,
      })
      setObjective('')
      onClose()
    } catch (error) {
      console.error('Failed to create mission:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="glass-strong rounded-2xl w-full max-w-lg p-6 animate-slide-up">
        <h2 className="text-xl font-display font-bold text-gradient mb-6">New Mission</h2>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-dt-text-dim mb-2">
              Mission Objective
            </label>
            <textarea
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              placeholder="Describe what you want DeepThinker to accomplish..."
              className="w-full bg-dt-surface-light border border-dt-border rounded-lg px-4 py-3 text-dt-text placeholder-dt-text-dim/50 focus:outline-none focus:border-dt-accent resize-none"
              rows={4}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-dt-text-dim mb-2">
              Time Budget: {timeBudget} minutes
            </label>
            <input
              type="range"
              min={10}
              max={480}
              step={10}
              value={timeBudget}
              onChange={(e) => setTimeBudget(Number(e.target.value))}
              className="w-full accent-dt-accent"
            />
            <div className="flex justify-between text-xs text-dt-text-dim mt-1">
              <span>10m</span>
              <span>8h</span>
            </div>
          </div>

          <div className="flex gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={allowInternet}
                onChange={(e) => setAllowInternet(e.target.checked)}
                className="w-4 h-4 rounded accent-dt-accent"
              />
              <span className="text-sm text-dt-text">Allow Web Research</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={allowCodeExecution}
                onChange={(e) => setAllowCodeExecution(e.target.checked)}
                className="w-4 h-4 rounded accent-dt-accent"
              />
              <span className="text-sm text-dt-text">Allow Code Execution</span>
            </label>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-3 rounded-lg border border-dt-border text-dt-text-dim hover:bg-dt-surface-light transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !objective.trim()}
              className="flex-1 px-4 py-3 rounded-lg bg-gradient-to-r from-dt-accent to-dt-success text-dt-bg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-dt-accent/20 transition-all"
            >
              {isSubmitting ? 'Creating...' : 'Start Mission'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function MissionList() {
  const navigate = useNavigate()
  const [missions, setMissions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [filter, setFilter] = useState('all')

  const loadMissions = async () => {
    try {
      setLoading(true)
      const data = await getMissions(filter === 'all' ? null : filter)
      setMissions(data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadMissions()
    // Refresh every 10 seconds
    const interval = setInterval(loadMissions, 10000)
    return () => clearInterval(interval)
  }, [filter])

  const handleCreateMission = async (missionData) => {
    const newMission = await createMission(missionData)
    navigate(`/missions/${newMission.mission_id}`)
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-display font-bold text-gradient">Missions</h1>
          <p className="text-dt-text-dim mt-1">
            {missions.length} mission{missions.length !== 1 ? 's' : ''}
          </p>
        </div>

        <button
          onClick={() => setShowCreateModal(true)}
          className="px-5 py-3 rounded-xl bg-gradient-to-r from-dt-accent to-dt-success text-dt-bg font-medium flex items-center gap-2 hover:shadow-lg hover:shadow-dt-accent/20 transition-all"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Mission
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {['all', 'running', 'pending', 'completed', 'failed'].map((status) => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              filter === status
                ? 'bg-dt-accent text-dt-bg'
                : 'bg-dt-surface-light text-dt-text-dim hover:bg-dt-border'
            }`}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      {loading && missions.length === 0 ? (
        <div className="flex items-center justify-center py-20">
          <LoadingSpinner size="lg" />
        </div>
      ) : error ? (
        <div className="glass rounded-xl p-8 text-center">
          <p className="text-dt-error mb-4">{error}</p>
          <button
            onClick={loadMissions}
            className="px-4 py-2 rounded-lg bg-dt-surface-light text-dt-text hover:bg-dt-border transition-colors"
          >
            Retry
          </button>
        </div>
      ) : missions.length === 0 ? (
        <div className="glass rounded-xl p-12 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-dt-surface-light flex items-center justify-center">
            <svg className="w-8 h-8 text-dt-text-dim" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-dt-text mb-2">No missions yet</h3>
          <p className="text-dt-text-dim mb-6">Start your first mission to see it here</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-5 py-3 rounded-xl bg-gradient-to-r from-dt-accent to-dt-success text-dt-bg font-medium hover:shadow-lg hover:shadow-dt-accent/20 transition-all"
          >
            Create Mission
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {missions.map((mission) => (
            <MissionCard
              key={mission.mission_id}
              mission={mission}
              onClick={() => navigate(`/missions/${mission.mission_id}`)}
            />
          ))}
        </div>
      )}

      {/* Create Modal */}
      <CreateMissionModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreate={handleCreateMission}
      />
    </div>
  )
}

