import CouncilStack from './CouncilStack'

const STATUS_CONFIG = {
  pending: {
    bg: 'bg-dt-surface-light',
    border: 'border-dt-border',
    glow: '',
  },
  running: {
    bg: 'bg-dt-accent/10',
    border: 'border-dt-accent',
    glow: 'shadow-lg shadow-dt-accent/30 status-running',
  },
  completed: {
    bg: 'bg-dt-success/10',
    border: 'border-dt-success',
    glow: 'shadow-lg shadow-dt-success/20',
  },
  failed: {
    bg: 'bg-dt-error/10',
    border: 'border-dt-error',
    glow: 'shadow-lg shadow-dt-error/20',
  },
  skipped: {
    bg: 'bg-dt-surface-light',
    border: 'border-dt-border',
    glow: 'opacity-50',
  },
}

// Mapping of phase types to council models
const PHASE_COUNCILS = {
  research: ['gemma3:12b', 'llama3.1:8b'],
  recon: ['gemma3:12b', 'llama3.1:8b'],
  plan: ['cogito:14b', 'qwen2.5:14b'],
  design: ['cogito:14b', 'qwen2.5:14b'],
  code: ['deepseek-r1:8b', 'qwen2.5-coder:7b'],
  implement: ['deepseek-r1:8b', 'qwen2.5-coder:7b'],
  eval: ['gemma3:27b', 'llama3.1:70b'],
  review: ['gemma3:27b', 'llama3.1:70b'],
  test: ['mistral:instruct', 'llama3.2:3b'],
  simulation: ['mistral:instruct', 'llama3.2:3b'],
  synthesis: ['cogito:14b', 'gemma3:27b'],
}

function getCouncilModels(phaseName) {
  const name = phaseName.toLowerCase()
  for (const [key, models] of Object.entries(PHASE_COUNCILS)) {
    if (name.includes(key)) return models
  }
  return ['cogito:14b'] // Default
}

export default function PhaseStation({ phase, isActive, phaseIndex, onDetailsClick }) {
  const config = STATUS_CONFIG[phase.status] || STATUS_CONFIG.pending
  const models = getCouncilModels(phase.name)

  return (
    <div className="flex flex-col items-center">
      {/* Station box */}
      <div
        className={`
          w-36 rounded-xl border-2 p-4 transition-all duration-300
          ${config.bg} ${config.border} ${config.glow}
          ${isActive ? 'scale-105' : 'hover:scale-102'}
        `}
      >
        {/* Phase name */}
        <div className="text-center mb-3">
          <h4 className="text-sm font-medium text-dt-text truncate" title={phase.name}>
            {phase.name}
          </h4>
          {phase.status === 'running' && (
            <span className="text-xs text-dt-accent">Processing...</span>
          )}
          {phase.status === 'completed' && phase.duration_seconds && (
            <span className="text-xs text-dt-text-dim">
              {phase.duration_seconds.toFixed(1)}s
            </span>
          )}
        </div>

        {/* Iterations badge */}
        {phase.iterations > 0 && (
          <div className="flex justify-center mb-3">
            <span className="px-2 py-0.5 rounded-full text-xs bg-dt-surface border border-dt-border text-dt-text-dim">
              {phase.iterations} iter
            </span>
          </div>
        )}

        {/* Details button */}
        <button
          onClick={onDetailsClick}
          className="w-full py-1.5 rounded-lg text-xs font-medium bg-dt-surface hover:bg-dt-border transition-colors text-dt-text-dim hover:text-dt-accent flex items-center justify-center gap-1"
        >
          details
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>

      {/* Council stack */}
      <div className="mt-3">
        <CouncilStack models={models} isActive={phase.status === 'running'} />
      </div>
    </div>
  )
}

