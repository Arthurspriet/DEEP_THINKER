# Missions Module

The missions module is the **heart of DeepThinker**, orchestrating long-running, time-bounded autonomous tasks.

## Core Concepts

### Mission
A mission is a time-bounded execution of a complex objective. It has:
- **Objective**: What needs to be accomplished
- **Time Budget**: How long the mission can run
- **Constraints**: What tools/resources are allowed
- **Phases**: Sequential stages of execution

### Phases
Missions execute through structured phases:

1. **Reconnaissance** - Gather context, background info, resources
2. **Analysis & Planning** - Strategic planning and workflow design
3. **Deep Analysis** - In-depth investigation with evidence councils
4. **Synthesis & Report** - Consolidate findings into deliverables

### Effort Levels
Missions are categorized by complexity:

| Level | Duration | Typical Use |
|-------|----------|-------------|
| Quick | 5-15 min | Simple lookups, quick analysis |
| Standard | 15-60 min | Moderate research, coding tasks |
| Deep | 1-4 hours | Comprehensive analysis |
| Marathon | 4+ hours | Large projects |

## Key Files

| File | Purpose |
|------|---------|
| `mission_orchestrator.py` | Main orchestration logic, phase execution |
| `mission_types.py` | Data structures (MissionState, MissionPhase, MissionConstraints) |
| `mission_store.py` | Persistence layer for saving/loading missions |
| `mission_time_manager.py` | Time budget allocation and tracking |
| `mission_planner.py` | Initial phase planning from objectives |

## Usage

```python
from deepthinker.missions import MissionOrchestrator, MissionStore
from deepthinker.missions.mission_types import build_constraints_from_time_budget

# Create constraints
constraints = build_constraints_from_time_budget(
    time_budget_minutes=30,
    allow_code=True,
    allow_internet=True
)

# Create and run mission
orchestrator = MissionOrchestrator(...)
state = orchestrator.create_mission("Your objective", constraints)
final_state = orchestrator.run_until_complete_or_timeout(state.mission_id)
```

## State Machine

```
Pending → Running → [Reconnaissance → Analysis → Deep Analysis → Synthesis] → Completed
                 ↘ Expired (time out)
                 ↘ Aborted (user cancel)
                 ↘ Failed (unrecoverable error)
```

## Checkpointing

Missions automatically checkpoint after each phase, enabling:
- Resume after interruption
- State inspection during execution
- Post-mortem analysis


