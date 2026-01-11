# Consensus Module

The consensus module provides **mechanisms for multiple agents to reach agreement** on complex decisions.

## Core Concepts

### Why Consensus?
When multiple AI agents work on the same problem, they may produce different solutions. Consensus mechanisms help:
- Combine diverse perspectives
- Filter out low-quality proposals
- Produce robust, well-reasoned outputs

## Mechanisms

### 1. Critique Exchange
Agents critique each other's proposals in structured rounds.

```
Round 1: Each agent proposes a solution
Round 2: Each agent critiques other proposals
Round 3: Synthesize critiques into final output
```

**File**: `critique_exchange.py`

### 2. Voting
Democratic selection where agents vote on the best proposal.

```
1. Collect all proposals
2. Each agent votes (can be weighted by confidence)
3. Winner selected, optionally blended with runner-up
```

**File**: `voting.py`

### 3. Weighted Blend
Combine multiple outputs using learned or configured weights.

```
final = w1 * output1 + w2 * output2 + w3 * output3
```

**File**: `weighted_blend.py`

### 4. Semantic Distance
Measure how different proposals are from each other.

```python
distance = semantic_distance(proposal_a, proposal_b)
# Used to detect convergence or identify outliers
```

**File**: `semantic_distance.py`

### 5. Policy Engine
Rules-based filtering and validation of proposals.

**File**: `policy_engine.py`

## Key Files

| File | Purpose |
|------|---------|
| `critique_exchange.py` | Multi-round critique protocol |
| `voting.py` | Proposal voting and selection |
| `weighted_blend.py` | Output blending with weights |
| `semantic_distance.py` | Embedding-based similarity |
| `policy_engine.py` | Rule-based validation |

## Usage Example

```python
from deepthinker.consensus import CritiqueExchange, VotingMechanism

# Critique exchange
exchange = CritiqueExchange(agents=[agent1, agent2, agent3])
result = exchange.run(objective="Design an API")

# Voting
voting = VotingMechanism(proposals=[p1, p2, p3])
winner = voting.select_best(voters=[agent1, agent2])
```

## When to Use Each

| Mechanism | Best For |
|-----------|----------|
| Critique Exchange | Complex decisions needing refinement |
| Voting | Clear alternatives, need quick decision |
| Weighted Blend | Combining complementary outputs |
| Semantic Distance | Detecting convergence/divergence |




