# Meta Module

The meta module implements the **Meta-Cognition Engine** for self-reflection, hypothesis management, and reasoning supervision.

## Core Concepts

### What is Meta-Cognition?
Meta-cognition is "thinking about thinking":
- Reflecting on what was learned
- Forming and testing hypotheses
- Debating different perspectives
- Revising plans based on insights
- Monitoring reasoning depth and quality

### Why Meta-Cognition?
For long-running missions (1h+), meta-cognition provides:
- Deeper reasoning beyond surface-level outputs
- Self-correction when approaches aren't working
- Balanced analysis through internal debate
- Quality assurance via depth contracts

## Components

### 1. Meta Controller
Orchestrates all meta-cognition components.

**File**: `meta_controller.py`

### 2. Reflection Engine
Reflects on phase outputs to extract insights.

**File**: `reflection.py`

### 3. Hypothesis Manager
Manages hypotheses throughout the mission.

**File**: `hypotheses.py`

### 4. Debate Engine
Runs internal debates between perspectives.

**File**: `debate.py`

### 5. Plan Reviser
Revises mission plan based on insights.

**File**: `plan_revision.py`

### 6. Reasoning Supervisor
Monitors depth contracts and reasoning metrics.

**File**: `supervisor.py`

### 7. Self Diagnosis
Self-assessment of mission health.

**File**: `self_diagnosis.py`

### 8. Depth Evaluator
Evaluates reasoning depth.

**File**: `depth_evaluator.py`

## Meta-Cognition Flow

```
Phase Completes
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. REFLECTION                                               │
│    - What did we learn?                                     │
│    - What worked? What didn't?                              │
│    - What assumptions were validated/invalidated?           │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. HYPOTHESIS UPDATE                                        │
│    - Update hypothesis confidence                           │
│    - Add new hypotheses from reflection                     │
│    - Mark invalidated hypotheses                            │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. INTERNAL DEBATE                                          │
│    - Optimist vs Skeptic on key hypotheses                  │
│    - Synthesize balanced perspective                        │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. PLAN REVISION                                            │
│    - Should we change approach?                             │
│    - Are we on track for the objective?                     │
│    - Update phase priorities                                │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. SUPERVISOR CHECK                                         │
│    - Are we meeting depth contracts?                        │
│    - Is reasoning quality sufficient?                       │
│    - Should we trigger deepening?                           │
└─────────────────────────────────────────────────────────────┘
      ↓
Next Phase (with updated context)
```

## Key Structures

```python
@dataclass
class Hypothesis:
    id: str
    statement: str
    confidence: float      # 0.0-1.0
    evidence_for: List[str]
    evidence_against: List[str]
    status: str            # "active", "validated", "invalidated"

@dataclass
class DepthContract:
    min_evidence_pieces: int
    min_perspectives: int
    required_challenges: int
    
@dataclass
class PhaseMetrics:
    reasoning_depth: float
    evidence_count: int
    perspective_count: int
    loop_detected: bool
```

## Reasoning Supervisor

The supervisor enforces quality standards:

```python
@dataclass
class DeepeningPlan:
    trigger_reason: str           # Why deepening needed
    recommended_actions: List[str]
    priority_areas: List[str]
    
@dataclass
class LoopDetection:
    is_looping: bool
    similarity_score: float       # How similar to previous outputs
    recommendation: str           # "continue" | "break" | "pivot"
```

## Usage

```python
from deepthinker.meta import MetaController

# Initialize
meta = MetaController(
    model_pool=model_pool,
    enable_multiview=True,
    enable_supervisor=True
)

# After phase completion
insights = meta.process_phase_completion(
    phase_output=phase_artifacts,
    mission_state=state,
    phase_name="deep_analysis"
)

# Check if deepening needed
if insights.deepening_plan:
    # Apply deepening actions
    pass

# Get updated hypotheses
for h in insights.hypotheses:
    print(f"{h.statement}: {h.confidence:.2f}")
```

## Multi-View Integration

When enabled, triggers Optimist and Skeptic councils:

```
Phase Output
      ↓
┌─────────────┐     ┌─────────────┐
│  Optimist   │     │  Skeptic    │
│  Council    │     │  Council    │
│             │     │             │
│ Strengths   │     │ Weaknesses  │
│ Opportunities│    │ Risks       │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               ↓
        Weighted Blend
               ↓
        Balanced Output
```


