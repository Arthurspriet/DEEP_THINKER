#!/usr/bin/env python3
"""
Demo script showing LiteLLM monitoring in action.

This script demonstrates how to:
1. Enable monitoring
2. Run a simple workflow
3. View monitoring statistics
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure Ollama
os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

from deepthinker.models import enable_monitoring, print_monitoring_summary
from deepthinker.execution import run_deepthinker_workflow, IterationConfig


def main():
    """Run a simple demo with monitoring."""
    
    print("="*60)
    print("🎯 LiteLLM Monitoring Demo")
    print("="*60)
    print()
    
    # Step 1: Enable monitoring with verbose output
    print("📊 Step 1: Enabling LiteLLM monitoring...")
    enable_monitoring(
        log_dir="logs/demo",
        verbose=True,  # Show detailed monitoring output
        enable_console_output=False
    )
    print()
    
    # Step 2: Run a simple workflow
    print("🚀 Step 2: Running a simple code generation task...")
    print()
    
    result = run_deepthinker_workflow(
        objective="Create a simple function that calculates the factorial of a number",
        model_name="deepseek-r1:8b",
        iteration_config=IterationConfig(
            max_iterations=2,
            quality_threshold=7.0,
            enabled=True
        ),
        verbose=False  # Keep workflow output quiet to see monitoring clearly
    )
    
    print("\n" + "="*60)
    print("✅ Workflow Complete!")
    print("="*60)
    print(f"Iterations completed: {result['iterations_completed']}")
    print(f"Quality score: {result['quality_score']:.1f}/10")
    print()
    
    # Step 3: Display monitoring summary
    print("="*60)
    print("📈 Step 3: Monitoring Statistics")
    print("="*60)
    print_monitoring_summary()
    
    # Show the final code
    print("\n" + "="*60)
    print("📝 Generated Code")
    print("="*60)
    print(result['final_code'])
    print()
    
    print("="*60)
    print("✨ Demo Complete!")
    print("="*60)
    print("\nCheck the logs/demo/ directory for detailed monitoring logs.")
    print("Each log entry is in JSONL format for easy analysis.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

