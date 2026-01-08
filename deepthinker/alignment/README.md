# Alignment Module

The alignment module implements the **Alignment Control Layer** for keeping missions on track with user objectives.

## Core Concepts

### What is Alignment?
Alignment ensures the mission stays true to:
- The original user objective
- User preferences and constraints
- Expected quality standards
- Ethical guidelines

### Soft Control Philosophy
The alignment layer uses **soft corrective pressure**:
- Never hard-stops a mission
- Escalates gradually based on consecutive triggers
- Actions are suggestions, not commands
- Integrates with existing mission controls

## Components

### 1. Alignment Controller
PID-like controller that applies corrective pressure.

**File**: `controller.py`

### 2. Drift Detector
Detects when mission is drifting from objective.

**File**: `drift.py`

### 3. Alignment Evaluator
LLM-based assessment of alignment.

**File**: `evaluator.py`

### 4. Learning System
Learns from alignment corrections over time.

**File**: `learning.py`

### 5. Persistence
Save/load alignment state.

**File**: `persist.py`

### 6. Models
Data structures for alignment.

**File**: `models.py`

## Escalation Ladder

Corrections escalate gradually:

| Level | Trigger | Action |
|-------|---------|--------|
| 1 | First drift detected | `REANCHOR_INTERNAL` - Remind councils of objective |
| 2 | Second consecutive | `INCREASE_SKEPTIC_WEIGHT` - More critical analysis |
| 3 | Third consecutive | `SWITCH_DEEPEN_MODE_TO_EVIDENCE` - Require more evidence |
| 4 | Fourth+ | `PRUNE_OR_PARK_FOCUS_AREAS` - Reduce scope |
| 5 | Fifth+ or severe | `TRIGGER_USER_DRIFT_CONFIRMATION` - Ask user |

## Key Structures

```python
@dataclass
class AlignmentPoint:
    objective_similarity: float  # 0-1, how aligned with objective
    grounding_score: float       # 0-1, evidence-based reasoning
    consistency_score: float     # 0-1, internal consistency
    timestamp: datetime
    phase_name: str

@dataclass
class AlignmentTrajectory:
    points: List[AlignmentPoint]
    trend: str                   # "improving", "stable", "declining"
    velocity: float              # Rate of change
    
@dataclass
class AlignmentAction:
    action_type: str             # From escalation ladder
    priority: int                # 1-5
    rationale: str
    suggested_prompt: str        # What to tell councils
```

## Alignment Flow

```
Phase Output
      ↓
┌─────────────────────────────────────┐
│  Compute Alignment Point            │
│  - Objective similarity             │
│  - Grounding score                  │
│  - Consistency score                │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Update Trajectory                  │
│  - Add new point                    │
│  - Compute trend                    │
│  - Detect drift velocity            │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Controller Decision                │
│  - Check thresholds                 │
│  - Consider consecutive triggers    │
│  - Select appropriate action        │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Apply Actions                      │
│  - Update council prompts           │
│  - Adjust weights                   │
│  - Log for learning                 │
└─────────────────────────────────────┘
```

## Usage

```python
from deepthinker.alignment import AlignmentController, AlignmentPoint

# Initialize controller
controller = AlignmentController()

# Compute alignment after phase
point = AlignmentPoint(
    objective_similarity=0.75,
    grounding_score=0.80,
    consistency_score=0.85,
    phase_name="analysis"
)

# Get corrective actions
actions = controller.decide(
    point=point,
    trajectory=trajectory,
    assessment=optional_llm_assessment
)

# Apply actions
for action in actions:
    if action.action_type == "REANCHOR_INTERNAL":
        # Add objective reminder to prompts
        pass
    elif action.action_type == "INCREASE_SKEPTIC_WEIGHT":
        # Boost skeptic council influence
        pass
```

## Drift Detection

```
Original Objective: "Analyze market trends"
                          ↓
                    ┌─────────────┐
  Phase 1: "Market overview" ────→ High alignment (0.9)
                    └─────────────┘
                          ↓
                    ┌─────────────┐
  Phase 2: "Economic factors" ───→ Good alignment (0.8)
                    └─────────────┘
                          ↓
                    ┌─────────────┐
  Phase 3: "Political history" ──→ DRIFT DETECTED (0.5)
                    └─────────────┘
                          ↓
              Trigger REANCHOR_INTERNAL
```

## Learning Integration

The alignment layer learns from corrections:

```python
# Log correction for learning
alignment_learning.log_correction(
    mission_id=state.mission_id,
    phase=phase_name,
    action=action,
    outcome=improvement_after_action
)

# Model improves over time
# - Better drift detection
# - More appropriate action selection
# - Faster convergence
```

