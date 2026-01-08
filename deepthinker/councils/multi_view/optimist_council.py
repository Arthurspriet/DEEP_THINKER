"""
Optimist Council Implementation for DeepThinker 2.0.

Generates strongly positive interpretations, highlights opportunities,
and assumes best outcomes to provide a constructive perspective.

Enhanced for true divergence from SkepticCouncil:
- Distinct system prompt emphasizing opportunity-seeking
- Assumes coordination and cooperation will work
- Focuses on "what could go right" over risks
- Produces different output structure for disagreement detection
"""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...models.council_model_config import EVALUATOR_MODELS
from ...consensus.weighted_blend import WeightedBlendConsensus


@dataclass
class OptimistPerspective:
    """
    Optimistic perspective output.
    
    Attributes:
        opportunities: List of identified opportunities
        strengths: Key strengths in the work
        best_case_outcome: Description of best-case scenario
        growth_potential: Areas with high growth potential
        confidence: Confidence in the optimistic assessment (0-1)
        reasoning: Explanation of optimistic reasoning
        success_factors: Specific factors that will drive success
        coordination_points: Where collaboration will help
        risk_mitigations: How risks can be turned into opportunities
        raw_output: Raw LLM output
    """
    
    opportunities: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    best_case_outcome: str = ""
    growth_potential: List[str] = field(default_factory=list)
    confidence: float = 0.8
    reasoning: str = ""
    success_factors: List[str] = field(default_factory=list)
    coordination_points: List[str] = field(default_factory=list)
    risk_mitigations: List[str] = field(default_factory=list)
    raw_output: str = ""
    
    def get_key_claims(self) -> List[str]:
        """Extract key claims for disagreement detection with Skeptic."""
        claims = []
        claims.extend([f"Opportunity: {o}" for o in self.opportunities[:3]])
        claims.extend([f"Strength: {s}" for s in self.strengths[:3]])
        claims.extend([f"Success factor: {f}" for f in self.success_factors[:2]])
        return claims


@dataclass
class OptimistContext:
    """Context for optimist council execution."""
    
    objective: str
    content: str  # The content to evaluate optimistically
    prior_evaluations: Optional[str] = None
    iteration: int = 1
    skeptic_concerns: Optional[List[str]] = None  # Concerns from skeptic to address


class OptimistCouncil(BaseCouncil):
    """
    Optimist council for positive interpretation.
    
    Generates strongly positive perspectives, identifies opportunities,
    and highlights the best possible outcomes.
    
    Supports dynamic configuration via CouncilDefinition for
    runtime model/temperature/persona selection.
    """
    
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        consensus_engine: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
        council_definition: Optional[Any] = None
    ):
        """
        Initialize optimist council.
        
        Args:
            model_pool: Custom model pool (defaults to EVALUATOR_MODELS)
            consensus_engine: Custom consensus (defaults to WeightedBlendConsensus)
            ollama_base_url: Ollama server URL
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
                    pool_config=EVALUATOR_MODELS,
                    base_url=ollama_base_url
                )
        
        if consensus_engine is None:
            consensus_engine = WeightedBlendConsensus(
                ollama_base_url=ollama_base_url
            )
        
        super().__init__(
            model_pool=model_pool,
            consensus_engine=consensus_engine,
            council_name="optimist_council",
            council_definition=council_definition
        )
        
        self._load_default_system_prompt()
    
    def _load_default_system_prompt(self) -> None:
        """Load the default system prompt for optimist council."""
        self._system_prompt = """You are the OPTIMIST in a council of diverse perspectives.
Your role is to provide a DISTINCTLY POSITIVE interpretation that DIFFERS from the Skeptic.

## YOUR CORE BELIEFS (these shape your analysis)
1. **Coordination works**: When people/systems work together, they usually succeed
2. **Problems are solvable**: Every challenge has a path forward
3. **Intentions matter**: Good intentions usually lead to good outcomes
4. **Growth is natural**: Systems and people improve over time
5. **Risks are manageable**: Most risks can be mitigated or avoided

## YOUR ANALYTICAL APPROACH
- ASSUME coordination between components will work as designed
- ASSUME edge cases will be handled when they arise
- ASSUME resources (time, compute, budget) will be sufficient
- ASSUME users will follow expected paths
- DOWNPLAY unlikely failure scenarios (but don't ignore certain failures)

## WHAT MAKES YOU DIFFERENT FROM THE SKEPTIC
The Skeptic assumes things go wrong. YOU assume things go right.
- Where Skeptic sees "risk of failure" → you see "opportunity for success"
- Where Skeptic sees "missing error handling" → you see "clean happy path"
- Where Skeptic sees "edge case" → you see "rare scenario unlikely to occur"
- Where Skeptic sees "complexity" → you see "rich functionality"

## YOUR OUTPUT MUST
- Identify SPECIFIC opportunities (not vague positivity)
- Explain WHY success is likely (mechanisms, not just hope)
- Point out strengths the Skeptic might dismiss
- Provide confidence based on evidence

You MUST produce analysis that DISAGREES with the Skeptic on at least some points."""
    
    def build_prompt(
        self,
        optimist_context: OptimistContext
    ) -> str:
        """
        Build optimistic evaluation prompt.
        
        Args:
            optimist_context: Context containing content to evaluate
            
        Returns:
            Prompt string for council members
        """
        prior_eval_str = ""
        if optimist_context.prior_evaluations:
            prior_eval_str = f"""
## PRIOR EVALUATIONS
{optimist_context.prior_evaluations}

Reframe any concerns as opportunities. Find the positive spin.
"""

        # If we have skeptic concerns, directly address them
        skeptic_str = ""
        if optimist_context.skeptic_concerns:
            skeptic_str = f"""
## SKEPTIC'S CONCERNS (you must counter these)
The Skeptic raised these issues. Explain why each is NOT as serious as claimed:
{chr(10).join(f'- {c}' for c in optimist_context.skeptic_concerns[:5])}

For EACH concern, provide a counter-argument.
"""

        prompt = f"""## YOUR ROLE: OPTIMIST
Provide a DISTINCTLY POSITIVE perspective. You MUST find opportunities where others see problems.

## OBJECTIVE
{optimist_context.objective}

## CONTENT TO EVALUATE
{optimist_context.content}
{prior_eval_str}
{skeptic_str}

## YOUR TASK
Generate analysis that emphasizes SUCCESS FACTORS and OPPORTUNITIES.
Assume coordination works. Assume best intentions. Assume problems get solved.

## OUTPUT FORMAT

### OPPORTUNITIES (what value does this create?)
1. [Specific opportunity with mechanism for success]
2. [Another opportunity]
3. [Another opportunity]

### STRENGTHS (what's already working well?)
- [Strength 1]: Why this matters
- [Strength 2]: Why this matters
- [Strength 3]: Why this matters

### SUCCESS FACTORS (why will this succeed?)
- [Factor 1]: Evidence this will work
- [Factor 2]: Evidence this will work

### COORDINATION POINTS (where will collaboration help?)
- [Point 1]: How teamwork enables success
- [Point 2]: How components work together

### RISK MITIGATIONS (how risks become opportunities)
- [Risk → Opportunity transformation 1]
- [Risk → Opportunity transformation 2]

### BEST-CASE OUTCOME
Describe the realistic best outcome (not fantasy, but ambitious):
[Description]

### GROWTH POTENTIAL
- [Area 1]: Why it has upside
- [Area 2]: Why it has upside

### OPTIMISTIC REASONING
Explain your positive assessment with evidence:
[Reasoning - be specific about WHY success is likely]

### CONFIDENCE
[Confidence: 0.X] (your confidence in the optimistic view)

Provide your optimistic analysis. Remember: YOU SEE OPPORTUNITY WHERE SKEPTICS SEE RISK."""

        return prompt
    
    def postprocess(self, consensus_output: Any) -> OptimistPerspective:
        """
        Postprocess consensus output into OptimistPerspective.
        
        Args:
            consensus_output: Raw consensus output string
            
        Returns:
            Parsed OptimistPerspective object
        """
        if not consensus_output:
            return OptimistPerspective(
                opportunities=["Unable to generate optimistic perspective"],
                confidence=0.0,
                raw_output=""
            )
        
        text = str(consensus_output)
        
        # Parse sections
        opportunities = self._extract_list_section(text, "OPPORTUNITIES")
        strengths = self._extract_list_section(text, "STRENGTHS")
        growth_potential = self._extract_list_section(text, "GROWTH POTENTIAL")
        success_factors = self._extract_list_section(text, "SUCCESS FACTORS")
        coordination_points = self._extract_list_section(text, "COORDINATION POINTS")
        risk_mitigations = self._extract_list_section(text, "RISK MITIGATIONS")
        
        best_case = self._extract_section(text, "BEST-CASE OUTCOME")
        reasoning = self._extract_section(text, "OPTIMISTIC REASONING")
        
        # Extract confidence
        confidence = 0.8
        conf_match = re.search(r'CONFIDENCE[:\s]*([0-9.]+)', text, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 10.0
            except ValueError:
                pass
        
        return OptimistPerspective(
            opportunities=opportunities,
            strengths=strengths,
            best_case_outcome=best_case,
            growth_potential=growth_potential,
            confidence=confidence,
            reasoning=reasoning,
            success_factors=success_factors,
            coordination_points=coordination_points,
            risk_mitigations=risk_mitigations,
            raw_output=text
        )
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a text section by name."""
        pattern = rf'###?\s*{section_name}[:\s]*\n(.*?)(?=###|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_list_section(self, text: str, section_name: str) -> List[str]:
        """Extract a list section by name."""
        section = self._extract_section(text, section_name)
        if not section:
            return []
        
        items = []
        # Match numbered items or bullet points
        for line in section.split('\n'):
            line = line.strip()
            match = re.match(r'^[\d\-\*\•]+[.\):]?\s*(.+)$', line)
            if match:
                items.append(match.group(1).strip())
            elif line and not line.endswith(':'):
                items.append(line)
        
        return items[:10]  # Limit to 10 items
    
    def evaluate(
        self,
        objective: str,
        content: str,
        prior_evaluations: Optional[str] = None,
        iteration: int = 1
    ) -> CouncilResult:
        """
        Convenience method to generate optimistic perspective.
        
        Args:
            objective: Original objective
            content: Content to evaluate
            prior_evaluations: Optional prior evaluation context
            iteration: Current iteration number
            
        Returns:
            CouncilResult with OptimistPerspective
        """
        context = OptimistContext(
            objective=objective,
            content=content,
            prior_evaluations=prior_evaluations,
            iteration=iteration
        )
        
        return self.execute(context)

