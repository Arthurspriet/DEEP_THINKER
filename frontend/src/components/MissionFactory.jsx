import PhaseStation from './PhaseStation'

export default function MissionFactory({ phases, currentPhaseIndex, onPhaseDetails }) {
  return (
    <div className="glass rounded-xl p-6 animate-fade-in overflow-hidden">
      <h3 className="text-sm font-medium text-dt-text-dim mb-4 flex items-center gap-2">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        Mission Factory Line
      </h3>

      {/* Horizontal scrollable factory */}
      <div className="overflow-x-auto pb-4 -mx-2 px-2">
        <div className="flex items-stretch gap-4 min-w-min">
          {/* Input station */}
          <div className="flex flex-col items-center justify-center shrink-0">
            <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-dt-accent/20 to-dt-success/20 border-2 border-dashed border-dt-accent/50 flex items-center justify-center">
              <svg className="w-8 h-8 text-dt-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <span className="text-xs text-dt-text-dim mt-2">Objective</span>
          </div>

          {/* Connector */}
          <div className="flex items-center shrink-0">
            <div className="w-8 h-1 bg-gradient-to-r from-dt-accent to-dt-border rounded-full" />
            <svg className="w-4 h-4 text-dt-accent -ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>

          {/* Phase stations */}
          {phases.map((phase, index) => (
            <div key={index} className="flex items-center shrink-0">
              <PhaseStation
                phase={phase}
                isActive={index === currentPhaseIndex}
                phaseIndex={index}
                onDetailsClick={() => onPhaseDetails?.(phase.name, guessCouncil(phase.name))}
              />

              {/* Connector */}
              {index < phases.length - 1 && (
                <div className="flex items-center mx-2">
                  <div className={`w-12 h-1 rounded-full transition-colors ${
                    phase.status === 'completed'
                      ? 'bg-dt-success'
                      : phase.status === 'running'
                        ? 'bg-gradient-to-r from-dt-accent to-dt-border animate-pulse'
                        : 'bg-dt-border'
                  }`} />
                  <svg className={`w-4 h-4 -ml-1 ${
                    phase.status === 'completed' ? 'text-dt-success' : 'text-dt-border'
                  }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              )}
            </div>
          ))}

          {/* Connector to output */}
          <div className="flex items-center shrink-0">
            <div className="w-8 h-1 bg-gradient-to-r from-dt-border to-dt-success rounded-full" />
            <svg className="w-4 h-4 text-dt-success -ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>

          {/* Output station */}
          <div className="flex flex-col items-center justify-center shrink-0">
            <div className={`w-16 h-16 rounded-xl flex items-center justify-center transition-all ${
              phases.every(p => p.status === 'completed')
                ? 'bg-gradient-to-br from-dt-success/30 to-dt-accent/30 border-2 border-dt-success'
                : 'bg-dt-surface-light border-2 border-dashed border-dt-border'
            }`}>
              <svg className={`w-8 h-8 ${
                phases.every(p => p.status === 'completed') ? 'text-dt-success' : 'text-dt-text-dim'
              }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="text-xs text-dt-text-dim mt-2">Output</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// Guess which council handles a phase based on name
function guessCouncil(phaseName) {
  const name = phaseName.toLowerCase()
  if (name.includes('research') || name.includes('recon') || name.includes('gather')) return 'researcher'
  if (name.includes('plan') || name.includes('design') || name.includes('architect')) return 'planner'
  if (name.includes('code') || name.includes('implement') || name.includes('build')) return 'coder'
  if (name.includes('eval') || name.includes('review') || name.includes('quality')) return 'evaluator'
  if (name.includes('test') || name.includes('simul') || name.includes('valid')) return 'simulator'
  return 'planner'
}

