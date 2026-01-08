# Epistemic Control Tooling Layer

A comprehensive tooling layer for DeepThinker that reduces hallucinations, enforces web search execution, improves reasoning depth, protects local resources, and makes memory usage observable.

## Overview

The tooling layer provides **deterministic helper tools** that councils invoke, not autonomous agents. All tools are designed to be:
- **Composable**: Can be used together or independently
- **Testable**: Deterministic behavior with clear inputs/outputs
- **Observable**: All operations are logged and traceable

## Architecture

```
deepthinker/tooling/
├── epistemic/          # Anti-hallucination core
│   ├── claim_extractor.py
│   ├── confidence_estimator.py
│   └── citation_gate.py
├── search/            # Web search enforcement
│   ├── justification_generator.py
│   ├── budget_allocator.py
│   └── evidence_compressor.py
├── memory/            # Memory integrity
│   ├── provenance_tracker.py
│   └── retrieval_auditor.py
├── reasoning/         # Depth control
│   ├── depth_controller.py
│   └── roi_evaluator.py
├── resources/         # Resource management
│   ├── capability_router.py
│   └── tier_escalator.py
├── output/            # Output trust
│   ├── confidence_header.py
│   └── counterfactual_checker.py
├── debug/             # Debugging & meta-control
│   ├── trace_visualizer.py
│   └── disagreement_detector.py
├── schemas.py         # Data models
└── integration_helpers.py  # Easy integration API
```

## Phase 1: Epistemic Control

### ClaimExtractorTool
Extracts atomic, verifiable claims from council outputs using regex patterns and heuristics.

**Features:**
- Identifies factual claims, inferences, assumptions, and uncertainty markers
- Generates stable IDs for claims (hash-based)
- Preserves context around each claim

### ClaimConfidenceTool
Estimates confidence in claims using multiple signals:
- Council agreement level
- Memory presence (RAG store lookup)
- Linguistic uncertainty markers

**Threshold:** Claims below 0.6 confidence are considered "untrusted"

### CitationGate
Enforces citation requirements before final output.

**Hard Constraint:** No claim passes without:
- Trusted memory reference OR
- Web evidence OR
- Explicit uncertainty label

Raises `UnverifiedClaimError` if constraint violated.

## Phase 2: Web Search Enforcement

### SearchJustificationGenerator
Generates explicit justifications for web searches.

**Rule:** If justification exists → search MUST execute (no silent skip)

### SearchBudgetAllocator
Allocates search budget based on:
- Mission time remaining
- Claim criticality scores
- Prior searches (reuse detection)

### EvidenceCompressorTool
Converts raw web search pages into council-consumable format:
- Quoted snippets (preserve context)
- Bullet evidence points
- Source reliability scores

## Phase 3: Memory Integrity

### MemoryProvenanceTracker
Tracks provenance on all memory writes:
- `origin`: "web", "inference", or "human"
- `decay_rate`: Confidence decay per day
- `expiry_date`: When evidence becomes stale

**Extended EvidenceSchema** with provenance fields.

### MemoryRetrievalAuditor
Logs all memory retrievals:
- What was retrieved (claim IDs, evidence IDs)
- Why it was selected (query, similarity score)
- Whether it influenced output (trace to final answer)

## Phase 4: Reasoning Depth Control

### DepthController
Enforces max reasoning loops per phase and detects diminishing returns.

**Rule:** If no new signal → force synthesis

### PhaseROIEvaluator
Tracks per-phase metrics:
- Tokens spent
- Facts added (new claims verified)
- Decisions changed (plan revisions)

**ROI Formula:** `(facts_added + decisions_changed) / tokens_spent`

Aborts phases with zero ROI.

## Phase 5: Resource-Aware Execution

### CPUGPUCapabilityRouter
Routes tasks to appropriate compute:
- Summarization → CPU
- Embeddings → GPU (if available)
- Parsing → CPU

### ExecutionTierEscalator
Formalizes escalation ladder: SAFE → SEARCH → GPU → BROWSER → TRUSTED

Escalation requires:
- Justification (claim risk, expected value)
- Budget check (time remaining)

## Phase 6: Output Trust

### AnswerConfidenceHeader
Generates confidence metadata for final output:
- Overall confidence score
- Verified vs unverified claims count
- Assumptions count

### CounterfactualChecker
Lightweight pass: "What fails if this claim is false?"

Flags weak conclusions (high fragility).

## Phase 7: Debugging

### MissionTraceVisualizer
Produces structured JSON trace:
- Phase durations
- Searches executed vs skipped (with justifications)
- Memory usage (retrievals, writes)
- Hallucination flags (unverified claims)

### CouncilDisagreementDetector
Measures divergence between council outputs.

Requires explicit resolution or abstention.

## Usage

### Quick Start

```python
from deepthinker.tooling.integration_helpers import ToolingIntegrationHelper

# Initialize helper
helper = ToolingIntegrationHelper(
    memory_manager=memory_manager,
    gpu_available=True
)

# Process council output
claims, confidence_scores = helper.process_council_output(
    output=council_result.output,
    council_name="researcher",
    phase_name="research"
)

# Check citations
verified = helper.check_citations(
    claims=claims,
    confidence_scores=confidence_scores,
    memory_references=memory_refs,
    web_evidence=web_evidence
)

# Generate search plan
justifications, budget = helper.generate_search_plan(
    claims=claims,
    confidence_scores=confidence_scores,
    time_remaining_minutes=30.0
)
```

See `INTEGRATION_GUIDE.md` for detailed integration examples.

## Success Criteria

✅ Web searches execute when justified (no silent skips)  
✅ Hallucinated claims are verified, downgraded, or marked uncertain  
✅ GPU usage optimized (CPU for lightweight tasks)  
✅ Mission timelines align with budgets  
✅ Memory usage is explainable (audit logs available)

## Testing

All tools are designed to be testable with deterministic behavior. Unit tests should verify:
- Claim extraction accuracy
- Confidence estimation correctness
- Citation gate enforcement
- Budget allocation logic
- ROI calculation accuracy

## Future Enhancements

- Embedding-based claim similarity for better memory matching
- Machine learning models for confidence estimation
- Advanced counterfactual reasoning
- Real-time trace visualization dashboard

