# Epistemics Module

The epistemics module manages **knowledge claims, validation, and contradiction detection**.

## Core Concepts

### Epistemic Tracking
Keeps track of what the system "knows" and how confident it is:
- **Claims**: Assertions made during analysis
- **Evidence**: Supporting information for claims
- **Contradictions**: Conflicts between claims
- **Confidence**: How certain we are about claims

### Why Epistemics?
In long-running missions, the system accumulates knowledge. Epistemics helps:
- Track what has been established as fact
- Detect when new information contradicts prior claims
- Maintain consistency across phases
- Support reasoning transparency

## Components

### 1. Claim Registry
Central store for all claims made during a mission.

**File**: `claim_registry.py`

### 2. Claim Graph
Graph structure showing relationships between claims.

**File**: `claim_graph.py`

### 3. Claim Validator
Validates claims against evidence and prior knowledge.

**File**: `claim_validator.py`

### 4. Contradiction Detector
Identifies conflicts between claims.

**File**: `contradiction_detector.py`

### 5. Focus Area Manager
Tracks which topics/areas are currently being investigated.

**File**: `focus_area_manager.py`

## Key Files

| File | Purpose |
|------|---------|
| `claim_registry.py` | Store and retrieve claims |
| `claim_graph.py` | Claim relationship graph |
| `claim_validator.py` | Validate claims with evidence |
| `contradiction_detector.py` | Find conflicting claims |
| `focus_area_manager.py` | Track investigation focus |

## Claim Structure

```python
@dataclass
class Claim:
    id: str
    content: str           # The assertion
    source: str            # Where it came from
    confidence: float      # 0.0 to 1.0
    evidence: List[str]    # Supporting information
    timestamp: datetime
    status: str            # proposed | validated | contradicted
```

## Claim Lifecycle

```
1. Claim Proposed    → Agent makes an assertion
2. Evidence Gathered → Supporting info collected
3. Validation        → Check against known facts
4. Contradiction Check → Compare with other claims
5. Status Update     → validated | contradicted | uncertain
```

## Usage

```python
from deepthinker.epistemics import ClaimRegistry, ContradictionDetector

# Register a claim
registry = ClaimRegistry()
claim = registry.add_claim(
    content="The market will grow 5% annually",
    source="researcher_council",
    confidence=0.7,
    evidence=["Report A", "Study B"]
)

# Check for contradictions
detector = ContradictionDetector(registry)
conflicts = detector.find_contradictions(claim)

if conflicts:
    # Handle contradictory information
    arbiter.resolve_contradiction(claim, conflicts)
```

## Integration with Councils

```
Council Output
      ↓
Extract Claims ──→ Claim Registry
      ↓
Contradiction Check
      ↓
┌─────────────────────────────────────┐
│ No Conflicts → Proceed normally     │
│ Conflicts Found → Arbiter resolves  │
└─────────────────────────────────────┘
```


