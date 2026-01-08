# Tooling Layer Integration Guide

This document describes how to integrate the epistemic control tooling layer into DeepThinker.

## Quick Start

### 1. Claim Extraction and Confidence Estimation

```python
from deepthinker.tooling.epistemic import ClaimExtractorTool, ClaimConfidenceTool

# Extract claims from council output
extractor = ClaimExtractorTool()
claims = extractor.extract_from_council_output(
    output=council_result.output,
    council_name="researcher",
    phase_name="research"
)

# Estimate confidence
confidence_tool = ClaimConfidenceTool(memory_manager=memory_manager)
confidence_scores = confidence_tool.estimate_batch(claims)
```

### 2. Citation Gate Enforcement

```python
from deepthinker.tooling.epistemic import CitationGate

# Check claims before final output
gate = CitationGate(require_citations=True)
verified = gate.check_claims(
    claims=claims,
    confidence_scores=confidence_scores,
    memory_references=memory_refs,
    web_evidence=web_evidence
)
```

### 3. Search Justification and Budget

```python
from deepthinker.tooling.search import SearchJustificationGenerator, SearchBudgetAllocator

# Generate search justifications
justifier = SearchJustificationGenerator()
justifications = justifier.generate_justifications(claims, confidence_scores)

# Allocate budget
allocator = SearchBudgetAllocator()
budget = allocator.allocate_budget(
    time_remaining_minutes=time_remaining,
    claims=claims,
    justifications=justifications
)
```

### 4. Memory Provenance Tracking

```python
from deepthinker.tooling.memory import MemoryProvenanceTracker

tracker = MemoryProvenanceTracker()
evidence = tracker.track_evidence(
    evidence=evidence_schema,
    origin="web",
    decay_rate=0.01,
    expiry_days=90
)
```

### 5. Depth Control and ROI

```python
from deepthinker.tooling.reasoning import DepthController, PhaseROIEvaluator

# Check depth
depth_controller = DepthController(max_iterations_per_phase=5)
decision = depth_controller.check_depth(
    iteration_count=current_iteration,
    new_facts_detected=True
)

# Evaluate ROI
roi_evaluator = PhaseROIEvaluator()
roi_evaluator.start_phase("research")
roi_evaluator.record_tokens("research", tokens_spent)
metrics = roi_evaluator.evaluate_phase("research")
if metrics.should_abort:
    # Abort phase
    pass
```

## Integration Points

### Mission Orchestrator

Add to `MissionOrchestrator.run_phase()`:
- Depth control before each iteration
- ROI evaluation at phase end
- Citation gate before final arbiter call

### Councils

Add to `BaseCouncil.execute()`:
- Claim extraction on output
- Confidence estimation before consensus

### Memory Manager

Wrap `MemoryManager` methods:
- Provenance tracking on `add_evidence()`
- Retrieval auditing on `retrieve()`

### Arbiter

Add to `Arbiter.arbitrate()`:
- Citation gate enforcement
- Confidence header generation
- Counterfactual checking

### Researcher Council

Add to `ResearcherCouncil._perform_auto_searches()`:
- Search justification generation
- Evidence compression on results

