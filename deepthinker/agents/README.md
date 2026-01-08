# Agents Module

Agents are **individual AI workers** that perform specific tasks within councils.

## Core Concepts

### What is an Agent?
An agent is a single AI instance with:
- **Role**: What it specializes in (coder, evaluator, etc.)
- **Persona**: Behavioral guidelines and expertise
- **Model**: Which LLM powers it
- **Tools**: What capabilities it has access to

### Agent vs Council
- **Agent**: Individual worker, single perspective
- **Council**: Team of agents, collective intelligence

## Agent Types

| Agent | Role | Typical Model |
|-------|------|---------------|
| **Planner** | Strategic thinking, workflow design | cogito:14b |
| **Researcher** | Web search, information gathering | gemma3:12b |
| **Coder** | Code generation and revision | deepseek-r1:8b |
| **Evaluator** | Quality assessment, feedback | gemma3:27b |
| **Simulator** | Testing, scenario analysis | mistral:instruct |
| **Executor** | Code execution, tool use | llama3.2:3b |

## Directory Structure

```
agents/
├── coder/
│   ├── __init__.py
│   └── coder_agent.py
├── evaluator/
│   ├── __init__.py
│   └── evaluator_agent.py
├── executor/
│   ├── __init__.py
│   └── executor_agent.py
├── planner/
│   ├── __init__.py
│   └── planner_agent.py
├── researcher/
│   ├── __init__.py
│   └── researcher_agent.py
└── simulator/
    ├── __init__.py
    └── simulator_agent.py
```

## Agent Lifecycle

```
1. Initialization  → Load model, configure persona
2. Receive Task    → Get objective + context from council
3. Generate Output → Call LLM with structured prompt
4. Return Result   → Structured output for consensus
```

## Creating an Agent

```python
from deepthinker.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, model_name: str, ollama_url: str):
        super().__init__(model_name, ollama_url)
        self.role = "specialist"
        self.persona = "You are an expert in..."
    
    def execute(self, task: str, context: dict) -> str:
        prompt = self._build_prompt(task, context)
        return self._call_llm(prompt)
```

## Model Configuration

Agents can use different models based on task requirements:

```python
from deepthinker.models import AgentModelConfig

config = AgentModelConfig(
    planner_model="cogito:14b",      # Complex reasoning
    coder_model="deepseek-r1:8b",    # Code generation
    evaluator_model="gemma3:27b",    # Quality assessment
    executor_model="llama3.2:3b"     # Fast execution
)
```

## Personas

Agent behavior is guided by persona definitions in `deepthinker/personas/`:
- `skeptic.md` - Critical analysis perspective
- `evidence_hunter.md` - Fact-finding focus
- And more...

