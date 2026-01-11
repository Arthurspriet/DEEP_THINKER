"""
CLI for Bio Priors.

Provides commands to validate, list, and demo bio patterns.

Usage:
    python -m deepthinker.bio_priors validate
    python -m deepthinker.bio_priors list
    python -m deepthinker.bio_priors run-demo
"""

import sys
import json
from pathlib import Path
from typing import Optional

from .loader import load_patterns, validate_all_patterns, get_patterns_summary, PATTERNS_DIR
from .config import BioPriorConfig
from .engine import BioPriorEngine
from .metrics import BioPriorContext, RECENT_WINDOW_STEPS


def cmd_validate(patterns_dir: Optional[Path] = None) -> int:
    """
    Validate all YAML pattern cards.
    
    Returns:
        Exit code (0 = success, 1 = validation errors)
    """
    if patterns_dir is None:
        patterns_dir = PATTERNS_DIR
    
    print(f"Validating patterns in: {patterns_dir}")
    print("-" * 60)
    
    all_valid, errors_by_file = validate_all_patterns(patterns_dir)
    
    if all_valid:
        # Load to get count
        patterns = load_patterns(patterns_dir, validate=False)
        print(f"SUCCESS: All {len(patterns)} patterns are valid.")
        return 0
    else:
        print("VALIDATION ERRORS:")
        print()
        for file_path, errors in errors_by_file.items():
            print(f"  {file_path}:")
            for error in errors:
                print(f"    - {error}")
            print()
        
        print(f"FAILED: {len(errors_by_file)} file(s) with errors.")
        return 1


def cmd_list(patterns_dir: Optional[Path] = None) -> int:
    """
    List all patterns with id, maturity, and weight.
    
    Returns:
        Exit code (0 = success)
    """
    if patterns_dir is None:
        patterns_dir = PATTERNS_DIR
    
    print(f"Patterns in: {patterns_dir}")
    print("-" * 60)
    print(f"{'ID':<35} {'Maturity':<10} {'Weight':<8}")
    print("-" * 60)
    
    summaries = get_patterns_summary(patterns_dir)
    
    if not summaries:
        print("No patterns found.")
        return 0
    
    for summary in sorted(summaries, key=lambda x: x["id"]):
        print(
            f"{summary['id']:<35} "
            f"{summary['maturity']:<10} "
            f"{summary['weight']:<8.2f}"
        )
    
    print("-" * 60)
    print(f"Total: {len(summaries)} patterns")
    
    return 0


def cmd_run_demo() -> int:
    """
    Run engine on a mocked context and print output.
    
    Returns:
        Exit code (0 = success)
    """
    print("Bio Prior Engine Demo")
    print("=" * 60)
    
    # Create config for demo (enabled, soft mode)
    config = BioPriorConfig(
        enabled=True,
        mode="soft",
        topk=3,
        max_pressure=1.0,
    )
    
    print(f"Config: mode={config.mode}, topk={config.topk}")
    print()
    
    # Create engine
    try:
        engine = BioPriorEngine(config)
        print(f"Loaded {len(engine.patterns)} patterns")
    except Exception as e:
        print(f"ERROR loading patterns: {e}")
        return 1
    
    # Create mock context simulating stagnation + high drift
    ctx = BioPriorContext(
        phase="research",
        step_index=5,
        time_remaining_s=300.0,
        evidence_new_count_recent=0,  # Stagnation!
        contradiction_rate=0.3,       # High contradiction
        uncertainty_trend=0.1,
        drift_score=0.4,              # High drift
        plan_branching_factor=2.0,
        last_step_evidence_delta=0,
        recent_window_steps=RECENT_WINDOW_STEPS,
    )
    
    print("Context:")
    print(f"  phase: {ctx.phase}")
    print(f"  step_index: {ctx.step_index}")
    print(f"  evidence_new_count_recent: {ctx.evidence_new_count_recent}")
    print(f"  contradiction_rate: {ctx.contradiction_rate}")
    print(f"  drift_score: {ctx.drift_score}")
    print(f"  has_stagnation_signal: {ctx.has_stagnation_signal}")
    print(f"  has_high_contradiction: {ctx.has_high_contradiction}")
    print(f"  has_high_drift: {ctx.has_high_drift}")
    print()
    
    # Evaluate
    output = engine.evaluate(ctx)
    
    print("Selected Patterns:")
    for p in output.selected_patterns:
        print(f"  - {p['id']} (score={p['score']:.2f}, weight={p['weight']:.2f})")
    print()
    
    print("Pressure Signals:")
    signals = output.signals
    print(f"  exploration_bias_delta: {signals.exploration_bias_delta:.3f}")
    print(f"  depth_budget_delta: {signals.depth_budget_delta}")
    print(f"  redundancy_check: {signals.redundancy_check}")
    print(f"  force_falsification_step: {signals.force_falsification_step}")
    print(f"  branch_pruning_suggested: {signals.branch_pruning_suggested}")
    print(f"  confidence_penalty_delta: {signals.confidence_penalty_delta:.3f}")
    print(f"  retrieval_diversify: {signals.retrieval_diversify}")
    print(f"  council_diversity_min: {signals.council_diversity_min}")
    print(f"  bounds_version: {signals.bounds_version}")
    print(f"  intent: {signals.intent}")
    print()
    
    print("Output Metadata:")
    print(f"  mode: {output.mode}")
    print(f"  applied: {output.applied}")
    print(f"  applied_fields: {output.applied_fields}")
    print()
    
    print("Trace (missing_metrics):")
    print(f"  {output.trace.get('missing_metrics', [])}")
    print(f"  recent_window_steps: {output.trace.get('recent_window_steps')}")
    print()
    
    print("Advisory Text:")
    print("-" * 40)
    print(output.advisory_text)
    print("-" * 40)
    
    # Verify determinism
    print()
    print("Determinism Check:")
    output2 = engine.evaluate(ctx)
    if output.selected_patterns == output2.selected_patterns:
        print("  PASS: Same patterns selected on re-evaluation")
    else:
        print("  FAIL: Different patterns selected!")
        return 1
    
    if output.signals == output2.signals:
        print("  PASS: Same signals on re-evaluation")
    else:
        print("  FAIL: Different signals!")
        return 1
    
    print()
    print("Demo complete.")
    return 0


def main(args: Optional[list] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]
    
    if not args:
        print("Usage: python -m deepthinker.bio_priors <command>")
        print()
        print("Commands:")
        print("  validate   Validate all YAML pattern cards")
        print("  list       List all patterns with id, maturity, weight")
        print("  run-demo   Run engine on mocked context")
        return 0
    
    command = args[0].lower()
    
    if command == "validate":
        return cmd_validate()
    elif command == "list":
        return cmd_list()
    elif command in ("run-demo", "demo"):
        return cmd_run_demo()
    else:
        print(f"Unknown command: {command}")
        print("Use 'validate', 'list', or 'run-demo'")
        return 1


if __name__ == "__main__":
    sys.exit(main())

