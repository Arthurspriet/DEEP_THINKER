# Arbiter Module

The Arbiter is the **final decision-maker** that reconciles outputs from multiple councils.

## Core Concepts

### What is the Arbiter?
When councils produce different or conflicting outputs, the Arbiter:
- Analyzes all proposals
- Identifies conflicts and agreements
- Makes the final authoritative decision
- Provides reasoning for the decision

### When is the Arbiter Used?
- Multi-view analysis (Optimist vs Skeptic)
- Council disagreements
- Quality threshold decisions
- Final synthesis of mission outputs

## Key Responsibilities

| Responsibility | Description |
|---------------|-------------|
| **Conflict Resolution** | Reconcile contradictory council outputs |
| **Quality Gate** | Decide if outputs meet standards |
| **Synthesis** | Combine partial outputs into coherent whole |
| **Attribution** | Explain why certain decisions were made |

## Arbiter Flow

```
Council A Output ──┐
                   │
Council B Output ──┼──→ Arbiter ──→ Final Decision
                   │         │
Council C Output ──┘         └──→ Decision Record
                                  (with reasoning)
```

## Key Files

| File | Purpose |
|------|---------|
| `arbiter.py` | Main Arbiter implementation |

## Usage

```python
from deepthinker.arbiter import Arbiter

arbiter = Arbiter(ollama_base_url="http://localhost:11434")

# Reconcile multiple outputs
decision = arbiter.reconcile(
    outputs=[
        {"source": "optimist", "content": "..."},
        {"source": "skeptic", "content": "..."}
    ],
    objective="Analyze market risks"
)

print(decision.final_output)
print(decision.reasoning)
```

## Decision Records

The Arbiter creates decision records that include:
- **Input**: What was received from councils
- **Analysis**: How conflicts were identified
- **Resolution**: What decision was made
- **Reasoning**: Why this decision was chosen
- **Confidence**: How certain the Arbiter is

## Integration with Consensus

The Arbiter works alongside consensus mechanisms:
1. **Consensus** tries to reach agreement
2. If consensus fails or needs validation → **Arbiter** decides
3. Arbiter decision is final and logged

