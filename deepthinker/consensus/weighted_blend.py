"""
Weighted Blend Consensus for DeepThinker 2.0.

Merges responses weighted by model confidence/temperature to produce
a synthesized output that combines the best elements from all models.

DeepThinker 2.0 Enhancements:
- Synthesis validation to detect model confusion/refusal
- Retry logic with fallback to best single output
- Stricter synthesis prompts
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Patterns indicating the synthesis model misunderstood the task
SYNTHESIS_FAILURE_PATTERNS = [
    "please provide",
    "i need the text",
    "once you provide",
    "model outputs from",
    "provide the outputs",
    "give me the",
    "share the content",
    "waiting for",
]

# Patterns indicating the model described itself instead of synthesizing
# (happens when outputs are empty and model latches onto model names in prompt)
MODEL_SELF_DESCRIPTION_PATTERNS = [
    "the model is",
    "the models referenced",
    "available in both",
    "parameter sizes",
    "billion parameters",
    "llama3",
    "gemma",
    "cogito",
    "deepseek",
    "mistral",
    "model_0",
    "model_1",
    "represents advancements",
    "large language model",
]


@dataclass
class BlendResult:
    """Result of weighted blend consensus."""
    
    blended_output: str
    contributing_models: List[str]
    weights_used: Dict[str, float]
    synthesis_method: str


class WeightedBlendConsensus:
    """
    Weighted blend consensus that synthesizes multiple model outputs.
    
    Uses an LLM to intelligently merge outputs, weighting contributions
    based on model temperatures (lower temp = higher weight for precision tasks).
    
    DeepThinker 2.0: Added synthesis validation and retry logic.
    """
    
    def __init__(
        self,
        synthesis_model: str = "gemma3:27b",
        synthesis_temperature: float = 0.3,
        ollama_base_url: str = "http://localhost:11434",
        max_synthesis_retries: int = 2
    ):
        """
        Initialize weighted blend consensus.
        
        Args:
            synthesis_model: Model to use for blending/synthesis
            synthesis_temperature: Temperature for synthesis
            ollama_base_url: Ollama server URL
            max_synthesis_retries: Maximum retries if synthesis fails validation
        """
        self.synthesis_model = synthesis_model
        self.synthesis_temperature = synthesis_temperature
        self.ollama_base_url = ollama_base_url
        self.max_synthesis_retries = max_synthesis_retries
    
    def _is_synthesis_failure(self, output: str) -> bool:
        """
        Check if synthesis output indicates model confusion or refusal.
        
        Args:
            output: The synthesis output to validate
            
        Returns:
            True if the output appears to be a synthesis failure
        """
        if not output:
            return True
        
        # Check first 300 chars for failure patterns
        output_lower = output.lower()[:300]
        
        for pattern in SYNTHESIS_FAILURE_PATTERNS:
            if pattern in output_lower:
                logger.warning(f"Synthesis failure detected: pattern '{pattern}' found")
                return True
        
        # Check for model self-description patterns (model describing itself instead of synthesizing)
        for pattern in MODEL_SELF_DESCRIPTION_PATTERNS:
            if pattern in output_lower:
                logger.warning(f"Synthesis failure detected: model self-description '{pattern}' found")
                return True
        
        # Also check for very short outputs (likely incomplete)
        if len(output.strip()) < 50:
            logger.warning(f"Synthesis failure: output too short ({len(output.strip())} chars)")
            return True
        
        return False
    
    def _get_best_single_output(
        self, 
        outputs: Dict[str, str], 
        weights: Dict[str, float]
    ) -> str:
        """
        Get the best single output when synthesis fails.
        
        Selects based on weight, then length as tiebreaker.
        """
        if not outputs:
            return ""
        
        # Sort by weight (descending), then by length (descending)
        scored = [
            (name, weights.get(name, 0.5), len(outputs[name]))
            for name in outputs
        ]
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        best_name = scored[0][0]
        logger.info(f"Using best single output from '{best_name}' as synthesis fallback")
        return outputs[best_name]
    
    def synthesize(self, outputs) -> str:
        """
        Synthesize a list of string outputs into a single output.
        
        This is a compatibility method that wraps the apply() method
        for callers expecting a synthesize interface.
        
        Args:
            outputs: List of output strings to synthesize, or Dict[str, ModelOutput]
            
        Returns:
            Synthesized output string
        """
        if not outputs:
            return ""
        
        # SAFETY: Handle dict input (common mistake - passing model_pool.run_all() result directly)
        if isinstance(outputs, dict):
            logger.warning("synthesize() received dict instead of list - extracting model outputs")
            extracted = []
            for name, output in outputs.items():
                # Check if it's a ModelOutput object
                if hasattr(output, 'output') and hasattr(output, 'success'):
                    if output.success and output.output:
                        extracted.append(output.output)
                elif isinstance(output, str) and output:
                    extracted.append(output)
            outputs = extracted
            if not outputs:
                logger.warning("No valid outputs extracted from dict")
                return ""
        
        if len(outputs) == 1:
            return outputs[0]
        
        # Convert list to dict format expected by apply()
        outputs_dict = {
            f"model_{i}": output 
            for i, output in enumerate(outputs) 
            if output
        }
        
        if not outputs_dict:
            return ""
        
        try:
            result = self.apply(outputs_dict)
            return result.blended_output if result else outputs[0]
        except Exception:
            # Fallback to first output on any error
            return outputs[0]
    
    def _compute_weights(
        self,
        outputs: Dict[str, Any],
        explicit_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute weights for each model output.
        
        Args:
            outputs: Dictionary of model outputs
            explicit_weights: Optional explicit weight overrides
            
        Returns:
            Dictionary mapping model_name -> weight
        """
        weights = {}
        
        for name, output in outputs.items():
            if explicit_weights and name in explicit_weights:
                weights[name] = explicit_weights[name]
            elif hasattr(output, 'temperature'):
                # Lower temperature = higher precision = higher weight
                weights[name] = 1.0 - output.temperature
            else:
                weights[name] = 0.5  # Default weight
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _build_synthesis_prompt(
        self,
        outputs: Dict[str, str],
        weights: Dict[str, float],
        task_context: Optional[str] = None,
        is_retry: bool = False
    ) -> str:
        """
        Build the synthesis prompt with all model outputs included.
        
        IMPORTANT: Uses generic identifiers (Response 1, Response 2) instead of
        actual model names to prevent the synthesis model from describing the models
        instead of synthesizing their content.
        
        Args:
            outputs: Dictionary of model outputs
            weights: Dictionary of weights per model
            task_context: Optional context about the task
            is_retry: If True, use a more directive prompt
            
        Returns:
            The synthesis prompt string
        """
        # Filter out empty/short outputs that could confuse the synthesizer
        valid_outputs = {
            name: output for name, output in outputs.items()
            if output and len(output.strip()) > 50
        }
        
        if not valid_outputs:
            # If all outputs are too short, use originals but warn
            logger.warning("All outputs are short/empty, using original outputs for synthesis")
            valid_outputs = outputs
        
        # Build sections with GENERIC identifiers (not model names!)
        # This prevents the synthesis model from describing the models
        output_sections = []
        for idx, (name, output) in enumerate(valid_outputs.items(), 1):
            weight = weights.get(name, 0.5)
            output_sections.append(
                f"=== RESPONSE {idx} (Weight: {weight:.2f}) ===\n{output}\n=== END RESPONSE {idx} ==="
            )
        
        outputs_block = "\n\n".join(output_sections)
        
        if is_retry:
            # More directive prompt for retry
            prompt = f"""IMPORTANT: The responses are PROVIDED BELOW. Do NOT ask for them - they are already included.

Your task: Synthesize the following {len(valid_outputs)} responses into ONE coherent output.

{f"Context: {task_context}" if task_context else ""}

THE RESPONSES ARE HERE:

{outputs_block}

INSTRUCTIONS:
- The responses above are COMPLETE. Do not request additional information.
- Blend the content, prioritizing higher-weighted responses.
- Output ONLY the synthesized content. No meta-commentary about sources.

BEGIN SYNTHESIS NOW:"""
        else:
            prompt = f"""You are a synthesis expert. Blend the following {len(valid_outputs)} responses into a single, coherent output.

Higher weights = more importance. The responses are provided below - do NOT ask for them.

{f"Task Context: {task_context}" if task_context else ""}

{outputs_block}

Instructions:
1. Analyze all responses for common themes and unique insights
2. Give more weight to responses with higher weights
3. Resolve contradictions by favoring higher-weighted sources
4. Produce a unified, coherent synthesis
5. Do NOT mention response numbers, weights, or ask for inputs - just provide the synthesized content

Synthesized Output:"""
        
        return prompt
    
    def _synthesize_outputs(
        self,
        outputs: Dict[str, str],
        weights: Dict[str, float],
        task_context: Optional[str] = None
    ) -> str:
        """
        Use LLM to synthesize weighted outputs with validation and retry.
        
        Args:
            outputs: Dictionary of model outputs
            weights: Dictionary of weights per model
            task_context: Optional context about the task
            
        Returns:
            Synthesized output string
        """
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        
        llm = ChatOllama(
            model=self.synthesis_model,
            base_url=self.ollama_base_url,
            temperature=self.synthesis_temperature
        )
        
        for attempt in range(self.max_synthesis_retries + 1):
            is_retry = attempt > 0
            prompt = self._build_synthesis_prompt(outputs, weights, task_context, is_retry)
            
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                result = response.content if hasattr(response, 'content') else str(response)
                
                # Validate synthesis output
                if not self._is_synthesis_failure(result):
                    if attempt > 0:
                        logger.info(f"Synthesis succeeded on retry {attempt}")
                    return result
                
                logger.warning(f"Synthesis attempt {attempt + 1} failed validation, retrying...")
                
            except Exception as e:
                logger.warning(f"Synthesis attempt {attempt + 1} failed with error: {e}")
        
        # All retries exhausted - fall back to best single output
        logger.warning("All synthesis attempts failed, using best single output fallback")
        return self._get_best_single_output(outputs, weights)
    
    def apply(
        self,
        outputs: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        task_context: Optional[str] = None
    ) -> BlendResult:
        """
        Apply weighted blend consensus to model outputs.
        
        Args:
            outputs: Dictionary mapping model_name -> output
            weights: Optional explicit weights per model
            task_context: Optional context about the task
            
        Returns:
            BlendResult with synthesized output
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
            return BlendResult(
                blended_output="",
                contributing_models=[],
                weights_used={},
                synthesis_method="none"
            )
        
        # Single output - no blending needed
        if len(text_outputs) == 1:
            model_name = list(text_outputs.keys())[0]
            return BlendResult(
                blended_output=text_outputs[model_name],
                contributing_models=[model_name],
                weights_used={model_name: 1.0},
                synthesis_method="single_output"
            )
        
        # Compute weights
        computed_weights = self._compute_weights(outputs, weights)
        
        # Filter to only include models with text outputs
        filtered_weights = {
            k: v for k, v in computed_weights.items()
            if k in text_outputs
        }
        
        # Normalize filtered weights
        total = sum(filtered_weights.values())
        if total > 0:
            filtered_weights = {k: v / total for k, v in filtered_weights.items()}
        
        # Synthesize
        blended = self._synthesize_outputs(
            text_outputs,
            filtered_weights,
            task_context
        )
        
        return BlendResult(
            blended_output=blended,
            contributing_models=list(text_outputs.keys()),
            weights_used=filtered_weights,
            synthesis_method="llm_synthesis"
        )

