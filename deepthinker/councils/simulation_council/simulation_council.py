"""
Simulation Council Implementation for DeepThinker 2.0.

Generates edge-case scenarios and stress tests using semantic distance
consensus to avoid hallucination overlap.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...models.council_model_config import SIMULATION_MODELS
from ...consensus.semantic_distance import SemanticDistanceConsensus


@dataclass
class SimulationContext:
    """Context for simulation council execution."""
    
    code: str
    objective: str
    simulation_config: Optional[Any] = None
    execution_results: Optional[Dict[str, Any]] = None
    focus_scenarios: Optional[List[str]] = None
    # Knowledge context from RAG retrieval
    knowledge_context: Optional[str] = None
    # Time-awareness fields for depth adjustment
    time_budget_seconds: Optional[float] = None
    time_remaining_seconds: Optional[float] = None
    
    @property
    def time_pressure(self) -> str:
        """
        Get time pressure level for prompt guidance.
        
        Returns:
            "high" - Limited time, be concise and focus on essentials
            "low" - Ample time, explore thoroughly
            "normal" - Balanced approach
        """
        if self.time_remaining_seconds is None:
            return "normal"
        if self.time_budget_seconds is None or self.time_budget_seconds <= 0:
            return "normal"
        ratio = self.time_remaining_seconds / self.time_budget_seconds
        if ratio < 0.3:
            return "high"
        elif ratio > 0.7:
            return "low"
        return "normal"


@dataclass
class ScenarioResult:
    """Result for a single simulation scenario."""
    
    name: str
    description: str
    expected_behavior: str
    potential_issues: List[str]
    severity: str  # "high", "medium", "low"


@dataclass
class SimulationFindings:
    """Structured simulation findings."""
    
    scenarios: List[ScenarioResult]
    edge_cases: List[str]
    stress_tests: List[str]
    robustness_score: float
    recommendations: List[str]
    raw_output: str
    
    @classmethod
    def from_text(cls, text: str) -> "SimulationFindings":
        """Parse simulation findings from text output."""
        scenarios = []
        edge_cases = []
        stress_tests = []
        recommendations = []
        
        lines = text.strip().split('\n')
        current_section = None
        current_scenario = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect sections
            if 'scenario' in line_lower and ':' in line:
                current_section = 'scenarios'
                name = line.split(':')[-1].strip() if ':' in line else "Scenario"
                current_scenario = {'name': name, 'description': '', 'issues': []}
            elif 'edge case' in line_lower:
                current_section = 'edge_cases'
            elif 'stress test' in line_lower:
                current_section = 'stress_tests'
            elif 'recommend' in line_lower:
                current_section = 'recommendations'
            elif line.strip().startswith(('-', '*', '•', '1', '2', '3', '4', '5')):
                content = line.strip().lstrip('-*•0123456789. ')
                
                if current_section == 'edge_cases':
                    edge_cases.append(content)
                elif current_section == 'stress_tests':
                    stress_tests.append(content)
                elif current_section == 'recommendations':
                    recommendations.append(content)
                elif current_section == 'scenarios' and current_scenario:
                    current_scenario['issues'].append(content)
        
        # Create scenario objects
        if current_scenario and current_scenario.get('name'):
            scenarios.append(ScenarioResult(
                name=current_scenario['name'],
                description=current_scenario.get('description', ''),
                expected_behavior='',
                potential_issues=current_scenario.get('issues', []),
                severity='medium'
            ))
        
        # Calculate robustness score based on issues found
        total_issues = len(edge_cases) + sum(len(s.potential_issues) for s in scenarios)
        robustness_score = max(0.0, 10.0 - total_issues * 0.5)
        
        return cls(
            scenarios=scenarios,
            edge_cases=edge_cases[:10],
            stress_tests=stress_tests[:5],
            robustness_score=robustness_score,
            recommendations=recommendations[:5],
            raw_output=text
        )


class SimulationCouncil(BaseCouncil):
    """
    Simulation council for edge-case and stress testing.
    
    Multiple simulation models generate test scenarios,
    with semantic distance consensus filtering out hallucinations.
    
    Supports dynamic configuration via CouncilDefinition for
    runtime model/temperature/persona selection.
    """
    
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        consensus_engine: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
        cognitive_spine: Optional[Any] = None,
        council_definition: Optional[Any] = None
    ):
        """
        Initialize simulation council.
        
        Args:
            model_pool: Custom model pool (defaults to SIMULATION_MODELS)
            consensus_engine: Custom consensus (defaults to SemanticDistanceConsensus)
                            If None and cognitive_spine provided, gets from spine
            ollama_base_url: Ollama server URL
            cognitive_spine: Optional CognitiveSpine for validation and consensus
            council_definition: Optional CouncilDefinition for dynamic configuration
        """
        # Use council_definition models if provided and no custom pool
        if model_pool is None:
            if council_definition is not None and council_definition.models:
                model_pool = ModelPool(
                    pool_config=council_definition.get_model_configs(),
                    base_url=ollama_base_url
                )
            else:
                model_pool = ModelPool(
                    pool_config=SIMULATION_MODELS,
                    base_url=ollama_base_url
                )
        
        # Get consensus from CognitiveSpine if not provided
        if consensus_engine is None:
            if cognitive_spine is not None:
                consensus_engine = cognitive_spine.get_consensus_engine(
                    "semantic_distance", "simulation_council"
                )
            else:
                consensus_engine = SemanticDistanceConsensus(
                    ollama_base_url=ollama_base_url
                )
        
        super().__init__(
            model_pool=model_pool,
            consensus_engine=consensus_engine,
            council_name="simulation_council",
            cognitive_spine=cognitive_spine,
            council_definition=council_definition
        )
        
        self._load_default_system_prompt()
    
    def _load_default_system_prompt(self) -> None:
        """Load the default system prompt for simulation council."""
        self._system_prompt = """You are part of a simulation council of testing experts.
Your role is to identify edge cases, stress tests, and potential failure modes.
Be thorough, creative, and focus on realistic scenarios.

You excel at:
- Identifying edge cases and boundary conditions
- Designing stress tests for performance limits
- Finding potential failure modes and error conditions
- Simulating real-world usage patterns
- Discovering unexpected interactions

Your simulation output should include:
- Specific test scenarios with expected behaviors
- Edge cases that could cause failures
- Stress test scenarios for performance
- Robustness assessment
- Recommendations for handling issues"""
    
    def build_prompt(
        self,
        simulation_context: SimulationContext
    ) -> str:
        """
        Build simulation prompt from context.
        
        Args:
            simulation_context: Context containing code and configuration
            
        Returns:
            Prompt string for council members
        """
        # Build execution results context
        exec_str = ""
        if simulation_context.execution_results:
            exec_str = f"""
## EXECUTION RESULTS
{simulation_context.execution_results}

Consider these results in your simulation analysis.
"""
        
        # Build focus scenarios
        focus_str = ""
        if simulation_context.focus_scenarios:
            focus_str = "\n\nFocus Scenarios:\n" + "\n".join(
                f"- {s}" for s in simulation_context.focus_scenarios
            )
        
        # Build knowledge context (from RAG retrieval)
        knowledge_str = ""
        if simulation_context.knowledge_context:
            knowledge_str = f"\n\n## RETRIEVED KNOWLEDGE (use as reference):\n{simulation_context.knowledge_context}"
        
        prompt = f"""Analyze the following code and generate comprehensive simulation scenarios:

## OBJECTIVE
{simulation_context.objective}

## CODE TO SIMULATE
```python
{simulation_context.code}
```
{exec_str}
{focus_str}
{knowledge_str}

## INSTRUCTIONS
Generate a thorough simulation analysis including:

### TEST SCENARIOS
For each scenario, describe:
- Scenario Name: Brief identifier
- Description: What is being tested
- Input Conditions: Specific inputs or state
- Expected Behavior: What should happen
- Potential Issues: What could go wrong

### EDGE CASES
List boundary conditions and edge cases to test:
- Empty inputs
- Maximum/minimum values
- Invalid inputs
- Concurrent operations
- Resource exhaustion

### STRESS TESTS
Describe stress testing scenarios:
- High volume inputs
- Long-running operations
- Memory pressure
- Error recovery

### ROBUSTNESS ASSESSMENT
Evaluate overall robustness (0-10) with justification.

### RECOMMENDATIONS
Provide specific recommendations for improving robustness.

Be creative but realistic in your scenarios."""

        # Add time-aware guidance based on time pressure
        time_guidance = self._get_time_guidance(simulation_context)
        if time_guidance:
            prompt += time_guidance

        return prompt
    
    def _get_time_guidance(self, context: SimulationContext) -> str:
        """
        Generate time-aware guidance for the prompt.
        
        Args:
            context: Simulation context with time pressure info
            
        Returns:
            Time guidance string to append to prompt
        """
        if not hasattr(context, 'time_pressure'):
            return ""
        
        pressure = context.time_pressure
        if pressure == "high":
            return """

## TIME CONSTRAINT
Limited time available for this phase. Adjust your approach:
- Focus on most critical scenarios only
- Prioritize high-risk edge cases
- Limit to 3-5 key test scenarios
- Skip low-priority stress tests"""
        elif pressure == "low":
            return """

## TIME AVAILABLE
Ample time available for thorough simulation:
- Generate comprehensive test scenarios
- Include unusual edge cases
- Design thorough stress tests
- Consider failure mode combinations"""
        return ""
    
    def postprocess(self, consensus_output: Any) -> SimulationFindings:
        """
        Postprocess consensus output into SimulationFindings.
        
        Args:
            consensus_output: Raw consensus output string
            
        Returns:
            Parsed SimulationFindings object
        """
        if not consensus_output:
            return SimulationFindings(
                scenarios=[],
                edge_cases=[],
                stress_tests=[],
                robustness_score=5.0,
                recommendations=[],
                raw_output=""
            )
        
        return SimulationFindings.from_text(str(consensus_output))
    
    def simulate(
        self,
        code: str,
        objective: str,
        execution_results: Optional[Dict[str, Any]] = None,
        focus_scenarios: Optional[List[str]] = None
    ) -> CouncilResult:
        """
        Convenience method to run simulation analysis.
        
        Args:
            code: Code to simulate
            objective: Original objective
            execution_results: Results from code execution
            focus_scenarios: Specific scenarios to focus on
            
        Returns:
            CouncilResult with SimulationFindings
        """
        simulation_context = SimulationContext(
            code=code,
            objective=objective,
            execution_results=execution_results,
            focus_scenarios=focus_scenarios
        )
        
        return self.execute(simulation_context)

