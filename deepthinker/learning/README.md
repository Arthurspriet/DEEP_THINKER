# Learning Module

The learning module provides **online learning capabilities** for continuous improvement of DeepThinker's decision-making.

## Core Concepts

### What is Online Learning?
Unlike batch training, online learning updates models incrementally:
- Learn from each mission execution
- Adapt to changing patterns
- No need to retrain from scratch

### Learning Goals
- Better council/model selection
- More accurate quality predictions
- Smarter escalation decisions
- Optimal iteration counts

## Components

### 1. Stop/Escalate Predictor
Predicts when to stop iteration or escalate to human.

**File**: `stop_escalate_predictor.py`

### 2. Policy Features
Feature extraction for policy learning.

**File**: `policy_features.py`

### 3. Configuration
Learning hyperparameters.

**File**: `config.py`

## Learning Targets

| Target | Description | Model Type |
|--------|-------------|------------|
| `stop_iteration` | When to stop refining | Binary classifier |
| `escalate_to_user` | When to ask for help | Binary classifier |
| `quality_prediction` | Expected final quality | Regressor |
| `optimal_rounds` | Best number of iterations | Multi-class |

## Key Structures

```python
@dataclass
class PolicyFeatures:
    # Phase context
    phase_index: int
    time_remaining_pct: float
    iterations_completed: int
    
    # Quality trajectory
    current_quality: float
    quality_improvement_rate: float
    quality_plateau_detected: bool
    
    # Cost context
    tokens_used: int
    token_budget_remaining_pct: float
    
    # Alignment
    alignment_score: float
    drift_velocity: float

@dataclass
class StopEscalateDecision:
    should_stop: bool
    should_escalate: bool
    confidence: float
    rationale: str
```

## Stop/Escalate Logic

```
Current State
      ↓
┌─────────────────────────────────────┐
│  Extract Policy Features            │
│  - Quality trajectory               │
│  - Time/token budgets               │
│  - Alignment scores                 │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Predictor Inference                │
│  - P(stop) = probability to stop    │
│  - P(escalate) = need human help    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Decision                           │
│  If P(stop) > threshold → Stop      │
│  If P(escalate) > threshold → Ask   │
│  Else → Continue                    │
└─────────────────────────────────────┘
```

## Usage

```python
from deepthinker.learning import StopEscalatePredictor, PolicyFeatures

# Initialize predictor
predictor = StopEscalatePredictor(model_path="kb/models/stop_predictor.joblib")

# Extract features
features = PolicyFeatures(
    phase_index=2,
    time_remaining_pct=0.4,
    iterations_completed=3,
    current_quality=0.75,
    quality_improvement_rate=0.02,  # Slowing down
    quality_plateau_detected=True,
    tokens_used=15000,
    token_budget_remaining_pct=0.3,
    alignment_score=0.85,
    drift_velocity=0.01
)

# Get decision
decision = predictor.predict(features)

if decision.should_stop:
    print(f"Stopping: {decision.rationale}")
elif decision.should_escalate:
    print(f"Escalating: {decision.rationale}")
else:
    print("Continuing iteration...")
```

## Training Flow

```
Mission Execution
      ↓
┌─────────────────────────────────────┐
│  Collect Training Examples          │
│  - Features at decision point       │
│  - Actual outcome (good/bad stop)   │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Incremental Update                 │
│  - Update model weights             │
│  - Adjust thresholds                │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Persist Updated Model              │
└─────────────────────────────────────┘
```

## Integration with Rewards

Learning uses reward signals as training labels:

```python
from deepthinker.rewards import RewardSignal

# After stop decision
actual_outcome = RewardSignal.from_phase_outcome(...)
reward = actual_outcome.compute_reward()

# Was it a good decision to stop?
was_good_stop = reward > threshold

# Update predictor
predictor.update(
    features=features,
    decision=decision,
    outcome_reward=reward,
    was_correct=was_good_stop
)
```

## Model Storage

```
kb/
└── models/
    ├── stop_predictor.joblib      # Stop/escalate model
    ├── quality_predictor.joblib   # Quality forecast
    └── training_history.jsonl     # Training examples
```

