"""
Council-Agent Bridge for DeepThinker.

Provides a bridge that allows the workflow path to optionally use council
features (multi-model execution with consensus) while maintaining backward
compatibility with simple CrewAI agents.

The bridge:
- Wraps CrewAI agents in a council-like interface
- Enables optional multi-model execution with consensus
- Provides unified output format
- Supports gradual adoption of council benefits

Usage:
    from deepthinker.core.council_bridge import AgentBridge, create_bridged_agent
    
    # Create a bridged agent (defaults to simple mode)
    coder = create_bridged_agent(
        agent_type="coder",
        llm=my_llm,
        use_council=False  # Simple mode
    )
    
    # Or with council mode
    coder = create_bridged_agent(
        agent_type="coder",
        llm=my_llm,
        use_council=True,
        consensus_type="voting"
    )
    
    # Execute - same interface regardless of mode
    result = coder.execute(task_description="Generate a binary search tree")
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

@dataclass
class BridgeResult:
    """
    Unified result format from agent/council execution.
    
    Provides consistent interface regardless of execution mode.
    """
    
    output: str
    success: bool = True
    error: Optional[str] = None
    
    # Execution metadata
    execution_mode: str = "simple"  # "simple" or "council"
    model_name: Optional[str] = None
    models_used: List[str] = field(default_factory=list)
    
    # Consensus info (only for council mode)
    consensus_type: Optional[str] = None
    agreement_score: Optional[float] = None
    raw_outputs: Dict[str, str] = field(default_factory=dict)
    
    # Metrics
    duration_seconds: Optional[float] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "execution_mode": self.execution_mode,
            "model_name": self.model_name,
            "models_used": self.models_used,
            "consensus_type": self.consensus_type,
            "agreement_score": self.agreement_score,
            "duration_seconds": self.duration_seconds,
            "token_usage": self.token_usage,
        }


# =============================================================================
# Agent Bridge Interface
# =============================================================================

class AgentBridge(ABC):
    """
    Abstract bridge interface for agents.
    
    Provides unified interface for both simple agents and council-backed agents.
    """
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Get the agent type (coder, evaluator, etc.)."""
        pass
    
    @property
    @abstractmethod
    def execution_mode(self) -> str:
        """Get execution mode (simple or council)."""
        pass
    
    @abstractmethod
    def execute(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BridgeResult:
        """
        Execute a task.
        
        Args:
            task_description: Description of the task
            context: Optional context dictionary
            **kwargs: Additional arguments
            
        Returns:
            BridgeResult with execution output
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass


class SimpleAgentBridge(AgentBridge):
    """
    Bridge for simple CrewAI agents.
    
    Wraps a CrewAI agent to provide the unified bridge interface.
    """
    
    def __init__(
        self,
        agent_type: str,
        crewai_agent: Any,
        llm: Any,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize simple agent bridge.
        
        Args:
            agent_type: Type of agent (coder, evaluator, etc.)
            crewai_agent: CrewAI Agent instance
            llm: LLM instance
            system_prompt: Optional system prompt override
        """
        self._agent_type = agent_type
        self._agent = crewai_agent
        self._llm = llm
        self._system_prompt = system_prompt or self._get_default_prompt()
    
    @property
    def agent_type(self) -> str:
        return self._agent_type
    
    @property
    def execution_mode(self) -> str:
        return "simple"
    
    def execute(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BridgeResult:
        """Execute using CrewAI agent."""
        import time
        
        start_time = time.time()
        
        try:
            from crewai import Task, Crew, Process
            
            # Create task
            task = Task(
                description=task_description,
                expected_output="Complete output for the task",
                agent=self._agent
            )
            
            # Create mini-crew
            crew = Crew(
                agents=[self._agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            # Execute
            result = crew.kickoff()
            output = str(result)
            
            duration = time.time() - start_time
            
            return BridgeResult(
                output=output,
                success=True,
                execution_mode="simple",
                model_name=getattr(self._llm, 'model_name', None),
                models_used=[getattr(self._llm, 'model_name', 'unknown')],
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Simple agent execution failed: {e}")
            return BridgeResult(
                output="",
                success=False,
                error=str(e),
                execution_mode="simple",
                duration_seconds=time.time() - start_time
            )
    
    def get_system_prompt(self) -> str:
        return self._system_prompt
    
    def _get_default_prompt(self) -> str:
        """Get default system prompt based on agent type."""
        prompts = {
            "coder": "You are an expert Python code generator.",
            "evaluator": "You are a code quality evaluator.",
            "researcher": "You are a research specialist.",
            "planner": "You are a strategic planner.",
            "simulator": "You are a simulation specialist.",
            "executor": "You are a code execution specialist.",
        }
        return prompts.get(self._agent_type, "You are an AI assistant.")


class CouncilAgentBridge(AgentBridge):
    """
    Bridge that wraps agents in council execution.
    
    Enables multi-model execution with consensus for agents.
    """
    
    def __init__(
        self,
        agent_type: str,
        model_pool: Any,
        consensus_engine: Any,
        system_prompt: Optional[str] = None,
        council_name: Optional[str] = None
    ):
        """
        Initialize council agent bridge.
        
        Args:
            agent_type: Type of agent
            model_pool: ModelPool instance for multi-model execution
            consensus_engine: Consensus engine instance
            system_prompt: Optional system prompt override
            council_name: Optional council name for logging
        """
        self._agent_type = agent_type
        self._model_pool = model_pool
        self._consensus = consensus_engine
        self._system_prompt = system_prompt or self._get_default_prompt()
        self._council_name = council_name or f"{agent_type}_council"
    
    @property
    def agent_type(self) -> str:
        return self._agent_type
    
    @property
    def execution_mode(self) -> str:
        return "council"
    
    def execute(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BridgeResult:
        """Execute using council with consensus."""
        import time
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self._build_prompt(task_description, context)
            
            # Execute on all models
            raw_outputs = self._model_pool.run_all(
                prompt=prompt,
                system_prompt=self._system_prompt
            )
            
            # Filter successful outputs
            successful_outputs = {
                name: output 
                for name, output in raw_outputs.items()
                if output.success and output.output
            }
            
            if not successful_outputs:
                return BridgeResult(
                    output="",
                    success=False,
                    error="All models failed to produce output",
                    execution_mode="council",
                    models_used=list(raw_outputs.keys()),
                    duration_seconds=time.time() - start_time
                )
            
            # Apply consensus
            consensus_result = self._consensus.apply(successful_outputs)
            
            # Extract output
            output = self._extract_consensus_output(consensus_result)
            
            # Calculate agreement score
            agreement_score = self._calculate_agreement(
                successful_outputs, output
            )
            
            duration = time.time() - start_time
            
            return BridgeResult(
                output=output,
                success=True,
                execution_mode="council",
                models_used=list(successful_outputs.keys()),
                consensus_type=type(self._consensus).__name__,
                agreement_score=agreement_score,
                raw_outputs={
                    name: out.output[:500] 
                    for name, out in successful_outputs.items()
                },
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Council execution failed: {e}")
            return BridgeResult(
                output="",
                success=False,
                error=str(e),
                execution_mode="council",
                duration_seconds=time.time() - start_time
            )
    
    def get_system_prompt(self) -> str:
        return self._system_prompt
    
    def _build_prompt(
        self, 
        task_description: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the execution prompt."""
        prompt = f"## Task\n{task_description}\n"
        
        if context:
            prompt += "\n## Context\n"
            for key, value in context.items():
                prompt += f"### {key}\n{value}\n\n"
        
        return prompt
    
    def _extract_consensus_output(self, consensus_result: Any) -> str:
        """Extract output from consensus result."""
        if hasattr(consensus_result, 'winner'):
            return consensus_result.winner
        elif hasattr(consensus_result, 'blended_output'):
            return consensus_result.blended_output
        elif hasattr(consensus_result, 'final_output'):
            return consensus_result.final_output
        elif hasattr(consensus_result, 'selected_output'):
            return consensus_result.selected_output
        elif hasattr(consensus_result, 'output'):
            return str(consensus_result.output)
        else:
            return str(consensus_result)
    
    def _calculate_agreement(
        self,
        outputs: Dict[str, Any],
        consensus_output: str
    ) -> float:
        """Calculate agreement score (simplified)."""
        if len(outputs) <= 1:
            return 1.0
        
        # Simple length-based similarity
        consensus_len = len(consensus_output)
        similarities = []
        
        for name, output in outputs.items():
            out_text = output.output if hasattr(output, 'output') else str(output)
            out_len = len(out_text)
            
            # Length ratio as proxy for similarity
            if consensus_len > 0 and out_len > 0:
                ratio = min(consensus_len, out_len) / max(consensus_len, out_len)
                similarities.append(ratio)
        
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    
    def _get_default_prompt(self) -> str:
        """Get default system prompt based on agent type."""
        prompts = {
            "coder": """You are an expert Python code generator. Write clean, efficient, 
well-documented code that follows best practices.""",
            "evaluator": """You are a code quality evaluator. Assess code for correctness,
efficiency, readability, and adherence to best practices.""",
            "researcher": """You are a research specialist. Gather and synthesize information
from multiple sources to provide comprehensive analysis.""",
            "planner": """You are a strategic planner. Create detailed plans with clear
objectives, milestones, and success criteria.""",
            "simulator": """You are a simulation specialist. Design and analyze test scenarios
to validate system behavior.""",
            "executor": """You are a code execution specialist. Execute and validate code,
handling errors and edge cases appropriately.""",
        }
        return prompts.get(self._agent_type, "You are an AI assistant.")


# =============================================================================
# Bridge Factory
# =============================================================================

class BridgeFactory:
    """
    Factory for creating agent bridges.
    
    Provides consistent bridge creation with automatic council setup.
    """
    
    # Agent type to council mapping
    COUNCIL_CONFIGS = {
        "coder": {
            "models": [("deepseek-r1:8b", 0.2), ("codellama:13b", 0.3)],
            "consensus": "voting"
        },
        "evaluator": {
            "models": [("gemma3:27b", 0.3), ("mistral:instruct", 0.4)],
            "consensus": "voting"
        },
        "researcher": {
            "models": [("gemma3:12b", 0.4), ("llama3:8b", 0.5)],
            "consensus": "weighted_blend"
        },
        "planner": {
            "models": [("cogito:14b", 0.3), ("gemma3:27b", 0.3)],
            "consensus": "voting"
        },
        "simulator": {
            "models": [("mistral:instruct", 0.5), ("llama3:8b", 0.4)],
            "consensus": "voting"
        },
    }
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self._ollama_url = ollama_base_url
        self._model_pools: Dict[str, Any] = {}
        self._consensus_engines: Dict[str, Any] = {}
    
    def create_bridge(
        self,
        agent_type: str,
        use_council: bool = False,
        llm: Optional[Any] = None,
        crewai_agent: Optional[Any] = None,
        consensus_type: Optional[str] = None,
        models: Optional[List[tuple]] = None,
        system_prompt: Optional[str] = None
    ) -> AgentBridge:
        """
        Create an agent bridge.
        
        Args:
            agent_type: Type of agent (coder, evaluator, etc.)
            use_council: Whether to use council mode
            llm: LLM instance (for simple mode)
            crewai_agent: CrewAI agent (for simple mode)
            consensus_type: Consensus type for council mode
            models: Models for council mode [(name, temp), ...]
            system_prompt: Optional system prompt override
            
        Returns:
            AgentBridge instance
        """
        if use_council:
            return self._create_council_bridge(
                agent_type, consensus_type, models, system_prompt
            )
        else:
            return self._create_simple_bridge(
                agent_type, llm, crewai_agent, system_prompt
            )
    
    def _create_simple_bridge(
        self,
        agent_type: str,
        llm: Optional[Any],
        crewai_agent: Optional[Any],
        system_prompt: Optional[str]
    ) -> SimpleAgentBridge:
        """Create a simple agent bridge."""
        if crewai_agent is None:
            # Create default agent
            crewai_agent = self._create_default_agent(agent_type, llm)
        
        return SimpleAgentBridge(
            agent_type=agent_type,
            crewai_agent=crewai_agent,
            llm=llm,
            system_prompt=system_prompt
        )
    
    def _create_council_bridge(
        self,
        agent_type: str,
        consensus_type: Optional[str],
        models: Optional[List[tuple]],
        system_prompt: Optional[str]
    ) -> CouncilAgentBridge:
        """Create a council agent bridge."""
        # Get config for agent type
        config = self.COUNCIL_CONFIGS.get(agent_type, {})
        
        models = models or config.get("models", [("llama3:8b", 0.3)])
        consensus_type = consensus_type or config.get("consensus", "voting")
        
        # Get or create model pool
        pool_key = f"{agent_type}_{hash(tuple(models))}"
        if pool_key not in self._model_pools:
            self._model_pools[pool_key] = self._create_model_pool(models)
        
        # Get or create consensus engine
        if consensus_type not in self._consensus_engines:
            self._consensus_engines[consensus_type] = self._create_consensus(
                consensus_type
            )
        
        return CouncilAgentBridge(
            agent_type=agent_type,
            model_pool=self._model_pools[pool_key],
            consensus_engine=self._consensus_engines[consensus_type],
            system_prompt=system_prompt,
            council_name=f"{agent_type}_council"
        )
    
    def _create_default_agent(self, agent_type: str, llm: Any) -> Any:
        """Create a default CrewAI agent."""
        try:
            from crewai import Agent
            
            configs = {
                "coder": {
                    "role": "Expert Python Code Generator",
                    "goal": "Generate clean, efficient Python code",
                    "backstory": "Senior software engineer with 15+ years experience"
                },
                "evaluator": {
                    "role": "Code Quality Evaluator",
                    "goal": "Assess code quality and provide feedback",
                    "backstory": "Expert code reviewer focused on quality"
                },
                "researcher": {
                    "role": "Research Specialist",
                    "goal": "Gather and synthesize information",
                    "backstory": "Research expert with broad knowledge"
                },
                "planner": {
                    "role": "Strategic Planner",
                    "goal": "Create comprehensive plans",
                    "backstory": "Experienced project planner"
                },
                "simulator": {
                    "role": "Simulation Specialist",
                    "goal": "Design and run test scenarios",
                    "backstory": "Testing and simulation expert"
                },
            }
            
            config = configs.get(agent_type, {
                "role": "AI Assistant",
                "goal": "Complete the assigned task",
                "backstory": "Helpful AI assistant"
            })
            
            return Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                llm=llm,
                verbose=False,
                allow_delegation=False
            )
            
        except ImportError:
            logger.warning("CrewAI not available, returning None")
            return None
    
    def _create_model_pool(self, models: List[tuple]) -> Any:
        """Create a ModelPool instance."""
        try:
            from ..models.model_pool import ModelPool
            
            return ModelPool(
                ollama_base_url=self._ollama_url,
                pool_config=models
            )
        except ImportError:
            logger.warning("ModelPool not available")
            return None
    
    def _create_consensus(self, consensus_type: str) -> Any:
        """Create a consensus engine."""
        try:
            if consensus_type == "voting":
                from ..consensus.voting import VotingConsensus
                return VotingConsensus()
            elif consensus_type == "weighted_blend":
                from ..consensus.weighted_blend import WeightedBlendConsensus
                return WeightedBlendConsensus()
            elif consensus_type == "critique_exchange":
                from ..consensus.critique_exchange import CritiqueExchangeConsensus
                return CritiqueExchangeConsensus()
            else:
                # Default to voting
                from ..consensus.voting import VotingConsensus
                return VotingConsensus()
        except ImportError:
            logger.warning(f"Consensus engine {consensus_type} not available")
            return None


# =============================================================================
# Convenience Functions
# =============================================================================

# Global factory instance
_bridge_factory: Optional[BridgeFactory] = None


def get_bridge_factory(
    ollama_base_url: str = "http://localhost:11434"
) -> BridgeFactory:
    """Get or create the global bridge factory."""
    global _bridge_factory
    if _bridge_factory is None:
        _bridge_factory = BridgeFactory(ollama_base_url)
    return _bridge_factory


def create_bridged_agent(
    agent_type: str,
    use_council: bool = False,
    llm: Optional[Any] = None,
    crewai_agent: Optional[Any] = None,
    consensus_type: Optional[str] = None,
    models: Optional[List[tuple]] = None,
    system_prompt: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434"
) -> AgentBridge:
    """
    Convenience function to create a bridged agent.
    
    Args:
        agent_type: Type of agent (coder, evaluator, etc.)
        use_council: Whether to use council mode
        llm: LLM instance (for simple mode)
        crewai_agent: CrewAI agent (for simple mode)
        consensus_type: Consensus type for council mode
        models: Models for council mode
        system_prompt: Optional system prompt override
        ollama_base_url: Ollama API URL
        
    Returns:
        AgentBridge instance
    """
    factory = get_bridge_factory(ollama_base_url)
    return factory.create_bridge(
        agent_type=agent_type,
        use_council=use_council,
        llm=llm,
        crewai_agent=crewai_agent,
        consensus_type=consensus_type,
        models=models,
        system_prompt=system_prompt
    )




