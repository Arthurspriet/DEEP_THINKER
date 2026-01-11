# Bandits Module

The bandits module implements **contextual multi-armed bandits** for exploration-exploitation decisions.

## Core Concepts

### What is a Multi-Armed Bandit?
A multi-armed bandit is a reinforcement learning approach for making decisions under uncertainty:
- **Arms**: Available choices (e.g., which model to use, which council to activate)
- **Rewards**: Feedback on how good a choice was
- **Exploration**: Trying new options to learn
- **Exploitation**: Using known good options

### Why Bandits in DeepThinker?
DeepThinker uses bandits to learn optimal decisions over time:
- Which model works best for which task types?
- Which council configuration produces better results?
- How many iterations are typically needed?

## Components

### 1. Generalized Bandit
Core bandit implementation with UCB and Thompson Sampling.

**File**: `generalized_bandit.py`

Features:
- Schema versioning for state validation
- Arms signature hash for migration detection
- Freeze mode for read-only operation
- Minimum trials before exploitation

### 2. Bandit Registry
Central registry for managing multiple bandits.

**File**: `bandit_registry.py`

### 3. Contextual Features
Feature extraction for contextual bandits.

**File**: `contextual_features.py`

### 4. Configuration
Bandit hyperparameters and settings.

**File**: `config.py`

## Algorithms

### Upper Confidence Bound (UCB)
Optimistic approach - assumes uncertain arms might be good.

```
score = mean_reward + c * sqrt(log(total_trials) / arm_trials)
```

### Thompson Sampling
Bayesian approach - samples from posterior distribution.

```
sample ~ Beta(successes + 1, failures + 1)
select arm with highest sample
```

## Key Structures

```python
@dataclass
class BanditArm:
    name: str           # Arm identifier
    trials: int         # Times selected
    successes: float    # Cumulative reward
    last_reward: float  # Most recent reward
    
@dataclass
class BanditSchema:
    schema_version: str      # For state validation
    arms_signature_hash: str # Detect arm changes
    decision_class: str      # What decision this bandit makes
    algorithm: str           # "ucb" or "thompson"
```

## Usage

```python
from deepthinker.bandits import GeneralizedBandit, BanditConfig

# Create bandit for model selection
config = BanditConfig(
    algorithm="ucb",
    exploration_constant=2.0,
    min_trials_before_exploit=5
)

bandit = GeneralizedBandit(
    arms=["small_model", "medium_model", "large_model"],
    decision_class="model_selection",
    config=config
)

# Select an arm
context = {"task_complexity": 0.7, "time_pressure": 0.3}
selected_arm = bandit.select(context)

# Update with reward
bandit.update(selected_arm, reward=0.85)

# Persist state
bandit.save("kb/models/model_bandit.json")
```

## Integration Flow

```
Decision Point
      ↓
Extract Context Features
      ↓
Bandit.select(context)
      ↓
Execute with Selected Arm
      ↓
Observe Outcome
      ↓
Compute Reward
      ↓
Bandit.update(arm, reward)
      ↓
Persist State
```

## Decision Classes

| Decision Class | Arms | Used For |
|---------------|------|----------|
| `model_selection` | Model names | Choosing LLM |
| `council_set` | Council configs | Workflow selection |
| `iteration_count` | 1, 2, 3, ... | How many rounds |
| `deepening_strategy` | Strategies | Analysis depth |




