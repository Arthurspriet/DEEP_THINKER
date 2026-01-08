"""
Critique Exchange Consensus for DeepThinker 2.0.

Models critique each other's outputs, then re-evaluate based on critiques
to produce a refined consensus output.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CritiqueResult:
    """Result of critique exchange consensus."""
    
    final_output: str
    original_outputs: Dict[str, str]
    critiques: Dict[str, List[Dict[str, str]]]
    refinement_round: int
    consensus_reached: bool


class CritiqueConsensus:
    """
    Critique-based consensus where models review and critique each other.
    
    Process:
    1. Collect initial outputs from all models
    2. Each model critiques other models' outputs
    3. Models refine their outputs based on critiques received
    4. Select the best refined output or synthesize
    """
    
    def __init__(
        self,
        critique_model: str = "gemma3:27b",
        critique_temperature: float = 0.4,
        refinement_model: str = "gemma3:27b",
        refinement_temperature: float = 0.3,
        max_rounds: int = 2,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize critique consensus.
        
        Args:
            critique_model: Model for generating critiques
            critique_temperature: Temperature for critique generation
            refinement_model: Model for refining outputs
            refinement_temperature: Temperature for refinement
            max_rounds: Maximum critique/refinement rounds
            ollama_base_url: Ollama server URL
        """
        self.critique_model = critique_model
        self.critique_temperature = critique_temperature
        self.refinement_model = refinement_model
        self.refinement_temperature = refinement_temperature
        self.max_rounds = max_rounds
        self.ollama_base_url = ollama_base_url
    
    def _generate_critique(
        self,
        output_to_critique: str,
        critic_perspective: str,
        task_context: Optional[str] = None
    ) -> str:
        """
        Generate a critique of an output.
        
        Args:
            output_to_critique: The output being critiqued
            critic_perspective: Perspective/role of the critic
            task_context: Optional context about the task
            
        Returns:
            Critique text
        """
        prompt = f"""You are a critical reviewer analyzing an AI-generated output.

{f"Task Context: {task_context}" if task_context else ""}

Critic Perspective: {critic_perspective}

## Output to Critique:
{output_to_critique}

## Instructions:
Provide a constructive critique focusing on:
1. Correctness - Are there any errors or inaccuracies?
2. Completeness - Is anything important missing?
3. Clarity - Is the output clear and well-structured?
4. Quality - Could any aspects be improved?

Be specific and actionable. Focus on substantive issues, not minor stylistic preferences.

## Critique:"""

        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage
            
            llm = ChatOllama(
                model=self.critique_model,
                base_url=self.ollama_base_url,
                temperature=self.critique_temperature
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, 'content') else str(response)
            
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Missing LLM dependencies for critique: {e}")
            return "Unable to generate critique (missing dependencies)."
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to generate critique: {e}")
            return "Unable to generate critique."
    
    def _refine_output(
        self,
        original_output: str,
        critiques: List[str],
        task_context: Optional[str] = None
    ) -> str:
        """
        Refine an output based on received critiques.
        
        Args:
            original_output: The original output to refine
            critiques: List of critiques received
            task_context: Optional context about the task
            
        Returns:
            Refined output
        """
        critiques_text = "\n\n---\n\n".join(
            f"Critique {i+1}:\n{c}" for i, c in enumerate(critiques)
        )
        
        prompt = f"""You are refining an AI-generated output based on peer critiques.

{f"Task Context: {task_context}" if task_context else ""}

## Original Output:
{original_output}

## Critiques Received:
{critiques_text}

## Instructions:
1. Carefully consider each critique
2. Address valid concerns and suggestions
3. Maintain the strengths of the original output
4. Produce an improved version that incorporates the feedback
5. Do NOT mention the critiques in your output - just provide the refined content

## Refined Output:"""

        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage
            
            llm = ChatOllama(
                model=self.refinement_model,
                base_url=self.ollama_base_url,
                temperature=self.refinement_temperature
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, 'content') else str(response)
            
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Missing LLM dependencies for refinement: {e}")
            return original_output
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to refine output: {e}. Returning original.")
            return original_output
    
    def _select_best_output(
        self,
        outputs: Dict[str, str],
        task_context: Optional[str] = None
    ) -> str:
        """
        Select the best output from multiple refined outputs.
        
        Args:
            outputs: Dictionary of refined outputs
            task_context: Optional context about the task
            
        Returns:
            Selected best output
        """
        if len(outputs) == 1:
            return list(outputs.values())[0]
        
        outputs_text = "\n\n---\n\n".join(
            f"Option {i+1} (from {name}):\n{output}"
            for i, (name, output) in enumerate(outputs.items())
        )
        
        prompt = f"""You are selecting the best output from multiple refined options.

{f"Task Context: {task_context}" if task_context else ""}

## Options:
{outputs_text}

## Instructions:
1. Evaluate each option for correctness, completeness, and quality
2. Select the best option OR synthesize the best elements from multiple options
3. Output ONLY the final selected/synthesized content, no explanations

## Best Output:"""

        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage
            
            llm = ChatOllama(
                model=self.refinement_model,
                base_url=self.ollama_base_url,
                temperature=self.refinement_temperature
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, 'content') else str(response)
            
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Missing LLM dependencies for selection: {e}")
            return list(outputs.values())[0] if outputs else ""
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to select best output: {e}. Using first available.")
            return list(outputs.values())[0] if outputs else ""
    
    def apply(
        self,
        outputs: Dict[str, Any],
        task_context: Optional[str] = None
    ) -> CritiqueResult:
        """
        Apply critique exchange consensus to model outputs.
        
        Args:
            outputs: Dictionary mapping model_name -> output
            task_context: Optional context about the task
            
        Returns:
            CritiqueResult with final refined output
        """
        # Extract text outputs
        text_outputs: Dict[str, str] = {}
        for name, output in outputs.items():
            if hasattr(output, 'output'):
                if output.success and output.output:
                    text_outputs[name] = output.output
            elif isinstance(output, str) and output:
                text_outputs[name] = output
        
        if not text_outputs:
            return CritiqueResult(
                final_output="",
                original_outputs={},
                critiques={},
                refinement_round=0,
                consensus_reached=False
            )
        
        # Single output - no critique needed
        if len(text_outputs) == 1:
            model_name = list(text_outputs.keys())[0]
            return CritiqueResult(
                final_output=text_outputs[model_name],
                original_outputs=text_outputs,
                critiques={},
                refinement_round=0,
                consensus_reached=True
            )
        
        all_critiques: Dict[str, List[Dict[str, str]]] = {
            name: [] for name in text_outputs
        }
        current_outputs = text_outputs.copy()
        
        # Critique and refinement rounds
        for round_num in range(self.max_rounds):
            # Generate critiques: each model critiques others
            for target_name, target_output in current_outputs.items():
                for critic_name in current_outputs:
                    if critic_name != target_name:
                        critique = self._generate_critique(
                            target_output,
                            critic_perspective=f"Reviewer from {critic_name}",
                            task_context=task_context
                        )
                        all_critiques[target_name].append({
                            "from": critic_name,
                            "critique": critique
                        })
            
            # Refine outputs based on critiques
            refined_outputs = {}
            for name, output in current_outputs.items():
                critiques = [c["critique"] for c in all_critiques[name]]
                if critiques:
                    refined = self._refine_output(
                        output,
                        critiques,
                        task_context
                    )
                    refined_outputs[name] = refined
                else:
                    refined_outputs[name] = output
            
            current_outputs = refined_outputs
        
        # Select best final output
        final_output = self._select_best_output(current_outputs, task_context)
        
        return CritiqueResult(
            final_output=final_output,
            original_outputs=text_outputs,
            critiques=all_critiques,
            refinement_round=self.max_rounds,
            consensus_reached=True
        )

