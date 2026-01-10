# Councils Module

Councils are **specialized teams of AI agents** that collaborate to accomplish specific types of tasks.

## Core Concepts

### What is a Council?
A council is a group of agents with complementary skills that:
- Receive a task from the Mission Orchestrator
- Collaborate using consensus mechanisms
- Produce validated, high-quality outputs

### Council Types

| Council | Responsibility | Key Agents |
|---------|---------------|------------|
| **Planner** | Strategic planning, workflow design | Strategy, Requirements, Workflow |
| **Researcher** | Information gathering, web search | Search, RAG, Validator |
| **Coder** | Code generation and revision | Generator, Reviewer, Refactor |
| **Evaluator** | Quality assessment, feedback | Assessor, Scorer, Feedback |
| **Simulation** | Testing and scenario analysis | Scenario Builder, Test Runner |

### Multi-View Councils
For balanced analysis, specialized councils provide different perspectives:
- **Optimist Council**: Focuses on strengths, opportunities, best-case scenarios
- **Skeptic Council**: Focuses on risks, weaknesses, potential failures

## Directory Structure

```
councils/
├── base_council.py           # Abstract base class for all councils
├── dynamic_council_factory.py # Runtime council creation
├── planner_council/
├── researcher_council/
├── coder_council/
├── evaluator_council/
├── simulation_council/
├── evidence_council/
├── explorer_council/
└── multi_view/               # Optimist/Skeptic councils
```

## Council Lifecycle

```
1. Receive Context    → Council gets objective + prior artifacts
2. Agent Deliberation → Multiple agents propose solutions
3. Consensus Building → Critique exchange, voting, blending
4. Output Validation  → Schema validation, quality checks
5. Return Artifacts   → Structured outputs for next phase
```

## Implementing a Council

```python
from deepthinker.councils.base_council import BaseCouncil

class MyCouncil(BaseCouncil):
    def __init__(self, ollama_base_url: str):
        super().__init__(ollama_base_url)
        self.agents = self._create_agents()
    
    def execute(self, context: dict) -> dict:
        # 1. Distribute work to agents
        # 2. Collect proposals
        # 3. Build consensus
        # 4. Return artifacts
        pass
```

## Consensus Integration

Councils use mechanisms from `deepthinker/consensus/`:
- **Critique Exchange**: Agents critique each other's proposals
- **Voting**: Democratic selection of best approach
- **Weighted Blend**: Combine multiple outputs with weights


