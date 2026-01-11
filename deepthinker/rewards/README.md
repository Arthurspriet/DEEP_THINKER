# Rewards Module

The rewards module provides **unified reward signals** for training and optimizing DeepThinker's learning components.

## Core Concepts

### What are Reward Signals?
Reward signals quantify how well a decision or action performed:
- Positive rewards for good outcomes
- Negative rewards (penalties) for bad outcomes
- Used by bandits, routing, and alignment learning

### Design Principles
- **Versioned**: Schema version for reproducibility
- **Deterministic**: Same inputs always produce same outputs
- **Auditable**: Raw components and normalization metadata preserved
- **Clamped**: Hard limits on penalties for safety

## Components

### 1. Reward Signal
Core reward computation.

**File**: `reward_signal.py`

### 2. Reward Weights
Configurable weights for reward components.

**File**: `reward_weights.py`

### 3. Reward Store
Persistence for reward history.

**File**: `reward_store.py`

### 4. Configuration
Reward normalization and clamping settings.

**File**: `config.py`

## Reward Components

| Component | Range | Description |
|-----------|-------|-------------|
| `score_delta` | -1 to +1 | Quality score improvement |
| `consistency_delta` | -1 to +1 | Consistency improvement |
| `grounding_delta` | -1 to +1 | Evidence grounding improvement |
| `cost_penalty` | 0 to -1 | Token/compute cost penalty |
| `alignment_penalty` | 0 to -1 | Alignment drift penalty |
| `time_penalty` | 0 to -1 | Time budget overrun penalty |

## Reward Formula

```
reward = w_score * score_delta
       + w_consistency * consistency_delta
       + w_grounding * grounding_delta
       - w_cost * clamp(cost_penalty)
       - w_alignment * clamp(alignment_penalty)
       - w_time * clamp(time_penalty)

Final reward clamped to [-1, +1]
```

## Key Structures

```python
@dataclass
class RewardSignal:
    # Versioning
    reward_version: str = "1.0.0"
    
    # Raw components (pre-normalization)
    components_raw: Dict[str, float]
    normalization_meta: Dict[str, Any]
    
    # Normalized components
    score_delta: float
    consistency_delta: float
    grounding_delta: float
    cost_penalty: float
    alignment_penalty: float
    time_penalty: float
    
    # Clamped values
    cost_penalty_clamped: float
    alignment_penalty_clamped: float
    time_penalty_clamped: float
    
    # Final
    reward: float  # in [-1, +1]

@dataclass
class RewardWeights:
    score_weight: float = 0.3
    consistency_weight: float = 0.2
    grounding_weight: float = 0.2
    cost_weight: float = 0.1
    alignment_weight: float = 0.1
    time_weight: float = 0.1
```

## Usage

```python
from deepthinker.rewards import RewardSignal, RewardWeights

# Compute reward from phase outcome
signal = RewardSignal.from_phase_outcome(
    score_delta=0.1,           # Quality improved by 0.1
    consistency_delta=0.05,    # Consistency improved slightly
    cost_tokens=5000,          # Tokens used
    cost_latency_ms=2000,      # Time taken
    alignment_events=1,        # One alignment trigger
    time_budget_used_pct=0.4,  # Used 40% of time budget
)

# Get final reward
reward = signal.compute_reward()
print(f"Reward: {reward:.3f}")  # e.g., 0.15

# Access components for debugging
print(f"Score delta: {signal.score_delta:.3f}")
print(f"Cost penalty: {signal.cost_penalty_clamped:.3f}")
```

## Normalization

Raw values are normalized before combination:

```python
# Token cost normalization (log scale)
cost_norm = min(1.0, log(1 + tokens) / log(1 + max_tokens))

# Time normalization (linear scale)
time_norm = min(1.0, time_used / time_budget)

# Score delta (already in reasonable range)
score_norm = clamp(score_delta, -1, 1)
```

## Clamping

Penalties are clamped to prevent runaway negative rewards:

```python
# Maximum penalty contribution
MAX_COST_PENALTY = 0.3
MAX_ALIGNMENT_PENALTY = 0.4
MAX_TIME_PENALTY = 0.3

cost_clamped = min(cost_penalty, MAX_COST_PENALTY)
```

## Integration with Learning

```
Phase Execution
      ↓
┌─────────────────────────────────┐
│  Collect Outcome Metrics        │
│  - Quality scores               │
│  - Token usage                  │
│  - Time usage                   │
│  - Alignment events             │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  Compute Reward Signal          │
│  - Normalize components         │
│  - Apply weights                │
│  - Clamp penalties              │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  Update Learning Components     │
│  - Bandit arm updates           │
│  - Router model training        │
│  - Alignment learning           │
└─────────────────────────────────┘
      ↓
Store Reward History
```

## Reward Store

Persist rewards for analysis and training:

```python
from deepthinker.rewards import RewardStore

store = RewardStore(path="kb/rewards")

# Store reward
store.add(
    mission_id=state.mission_id,
    phase=phase_name,
    decision_context=context,
    signal=signal
)

# Retrieve for training
history = store.get_history(
    decision_class="model_selection",
    min_date=datetime.now() - timedelta(days=7)
)
```




