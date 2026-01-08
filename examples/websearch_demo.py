#!/usr/bin/env python3
"""
WebSearch Agent Demo

Demonstrates the web research phase that runs before code generation.
"""

from deepthinker.execution import (
    run_deepthinker_workflow,
    IterationConfig,
    ResearchConfig
)


def main():
    """Run a simple workflow with web research enabled."""
    
    # Create research config
    research_config = ResearchConfig(
        enabled=True,
        max_results=5,
        timeout=10
    )
    
    # Create iteration config
    iteration_config = IterationConfig(
        max_iterations=2,
        quality_threshold=7.0,
        enabled=True
    )
    
    # Define objective
    objective = "Create a Python function that calculates the Fibonacci sequence efficiently using memoization"
    
    print("🔍 Running DeepThinker with Web Research enabled...")
    print(f"Objective: {objective}")
    print(f"Research: {research_config.enabled} (max {research_config.max_results} results)")
    print()
    
    # Run workflow with research enabled
    result = run_deepthinker_workflow(
        objective=objective,
        model_name="deepseek-r1:8b",
        iteration_config=iteration_config,
        research_config=research_config,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Iterations: {result['iterations_completed']}")
    print(f"Quality Score: {result['quality_score']:.1f}/10")
    print(f"Status: {'✅ PASSED' if result['final_evaluation'].passed else '⚠️  NEEDS WORK'}")
    print()
    print("Final Code:")
    print("-"*60)
    print(result['final_code'])


def demo_disabled_research():
    """Run workflow with research disabled for comparison."""
    
    # Create research config with research disabled
    research_config = ResearchConfig(enabled=False)
    
    objective = "Create a simple binary search tree implementation"
    
    print("📝 Running DeepThinker WITHOUT web research...")
    print(f"Objective: {objective}")
    print(f"Research: Disabled")
    print()
    
    result = run_deepthinker_workflow(
        objective=objective,
        model_name="deepseek-r1:8b",
        iteration_config=IterationConfig(max_iterations=1),
        research_config=research_config,
        verbose=True
    )
    
    print(f"\n✅ Complete! Quality: {result['quality_score']:.1f}/10")


if __name__ == "__main__":
    # Demo with research enabled
    main()
    
    # Uncomment to compare with research disabled
    # print("\n\n" + "="*60)
    # demo_disabled_research()

