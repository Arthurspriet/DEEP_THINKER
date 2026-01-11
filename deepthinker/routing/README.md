# Routing Module

The routing module provides **ML-based decision routing** for optimizing mission execution.

## Core Concepts

### What is ML Routing?
Instead of using fixed rules, ML routing uses learned models to make decisions:
- Which councils should handle this task?
- What model tier is appropriate?
- How many rounds are needed?

### Advisory Pattern
The ML Router acts as an **ADVISOR**, not a commander:
- Outputs decision + confidence + rationale
- Orchestrator applies only if no constraint violation
- Human-readable explanations for transparency

## Components

### 1. ML Router
Main router that provides recommendations.

**File**: `ml_router.py`

### 2. Features
Feature extraction from mission context.

**File**: `features.py`

### 3. Bandit Integration
Bandit-based routing for exploration.

**File**: `bandit.py`

## Router Decisions

| Decision | Options | Based On |
|----------|---------|----------|
| `council_set` | standard, deep, fast | Task complexity, time budget |
| `model_tier` | SMALL, MEDIUM, LARGE | Task requirements, cost constraints |
| `num_rounds` | 1, 2, 3 | Quality requirements, time available |

## Key Structures

```python
@dataclass
class RoutingContext:
    objective: str
    time_budget_minutes: float
    phase_name: str
    prior_quality_score: float
    complexity_estimate: float
    
@dataclass
class RoutingDecision:
    council_set: str      # "standard", "deep", "fast"
    model_tier: str       # "SMALL", "MEDIUM", "LARGE"
    num_rounds: int       # 1-3
    confidence: float     # 0.0-1.0
    rationale: str        # Human-readable explanation
    features_used: Dict   # For debugging/audit
```

## Features Extracted

| Feature | Description | Range |
|---------|-------------|-------|
| `time_budget_norm` | Normalized time budget | 0-1 |
| `objective_length` | Length of objective text | 0-1 |
| `complexity_score` | Estimated task complexity | 0-1 |
| `phase_index` | Current phase position | 0-1 |
| `prior_quality` | Previous quality score | 0-1 |
| `code_execution` | Whether code exec enabled | 0/1 |
| `web_research` | Whether web research enabled | 0/1 |

## Usage

```python
from deepthinker.routing import MLRouter, RoutingContext

# Initialize router
router = MLRouter(model_path="kb/models/router.joblib")

# Create context
context = RoutingContext(
    objective="Build a REST API",
    time_budget_minutes=30,
    phase_name="deep_analysis",
    prior_quality_score=0.6,
    complexity_estimate=0.7
)

# Get recommendation
decision = router.advise(context)

print(f"Council: {decision.council_set}")
print(f"Model Tier: {decision.model_tier}")
print(f"Rounds: {decision.num_rounds}")
print(f"Confidence: {decision.confidence:.2f}")
print(f"Rationale: {decision.rationale}")
```

## Model Training

The router can use:
1. **sklearn models** (LogisticRegression, LightGBM)
2. **sklearn-free fallback** with JSON-stored weights

```python
# Training (if sklearn available)
router.train(training_data, labels)
router.save("kb/models/router.joblib")

# Fallback weights (sklearn-free)
router.save_weights("kb/models/router_weights.json")
```

## Integration Flow

```
Mission Context
      ↓
┌─────────────────────────────┐
│      Feature Extraction     │
│  (objective, time, phase)   │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│        ML Router            │
│   (model inference)         │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│    Routing Decision         │
│  + Confidence + Rationale   │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  Orchestrator Validation    │
│  (constraint checking)      │
└─────────────────────────────┘
      ↓
Apply Decision or Override
```




