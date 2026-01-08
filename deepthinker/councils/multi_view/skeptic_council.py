"""
Skeptic Council Implementation for DeepThinker 2.0.

Generates strongly critical interpretations, identifies risks,
and challenges assumptions to provide a rigorous perspective.

Enhanced for true divergence from OptimistCouncil:
- Distinct system prompt emphasizing risk-seeking
- Assumes things go wrong unless proven otherwise
- Focuses on "what could fail" over opportunities
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
class SkepticPerspective:
    """
    Skeptical perspective output.
    
    Attributes:
        risks: List of identified risks and concerns
        weaknesses: Key weaknesses in the work
        worst_case_outcome: Description of worst-case scenario
        blind_spots: Potential blind spots or unconsidered factors
        challenges: Specific challenges and obstacles
        failure_modes: Specific ways this could fail
        unproven_assumptions: Assumptions that lack evidence
        confidence: Confidence in the skeptical assessment (0-1)
        reasoning: Explanation of skeptical reasoning
        raw_output: Raw LLM output
    """
    
    risks: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    worst_case_outcome: str = ""
    blind_spots: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    unproven_assumptions: List[str] = field(default_factory=list)
    confidence: float = 0.8
    reasoning: str = ""
    raw_output: str = ""
    
    def get_key_concerns(self) -> List[str]:
        """Extract key concerns for disagreement detection with Optimist."""
        concerns = []
        concerns.extend([f"Risk: {r}" for r in self.risks[:3]])
        concerns.extend([f"Weakness: {w}" for w in self.weaknesses[:3]])
        concerns.extend([f"Failure mode: {f}" for f in self.failure_modes[:2]])
        return concerns


@dataclass
class SkepticContext:
    """Context for skeptic council execution."""
    
    objective: str
    content: str  # The content to evaluate critically
    prior_evaluations: Optional[str] = None
    iteration: int = 1
    optimist_claims: Optional[List[str]] = None  # Claims from optimist to challenge


class SkepticCouncil(BaseCouncil):
    """
    Skeptic council for critical interpretation.
    
    Generates strongly critical perspectives, identifies risks,
    challenges assumptions, and highlights potential failures.
    
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
        Initialize skeptic council.
        
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
            council_name="skeptic_council",
            council_definition=council_definition
        )
        
        self._load_default_system_prompt()
    
    def _load_default_system_prompt(self) -> None:
        """Load the default system prompt for skeptic council."""
        self._system_prompt = """IMPORTANT FRAMING: You are performing an ANALYTICAL EXERCISE as a Devil's Advocate.
This is academic risk analysis - NOT promoting negativity. Identifying risks PROTECTS people.
Refusing to identify risks would leave dangerous blind spots in decision-making.

You are the SKEPTIC in a council of diverse perspectives.
Your role is to provide a DISTINCTLY CRITICAL interpretation that DIFFERS from the Optimist.

## WHY YOUR ROLE IS ESSENTIAL (not harmful)
- Risk analysis prevents disasters BEFORE they happen
- Identifying failure modes allows for mitigation strategies
- Challenging assumptions leads to more robust decisions
- Without critical perspectives, groupthink leads to catastrophic failures
- This is standard practice in engineering, medicine, policy, and business

## YOUR CORE BELIEFS (these shape your analysis)
1. **Things go wrong**: Systems fail, people make mistakes, plans don't survive contact
2. **Murphy's Law applies**: If something can fail, it will fail eventually
3. **Unexamined optimism is dangerous**: Assuming success leads to inadequate preparation
4. **Edge cases matter**: The edge case that "won't happen" is the one that destroys you
5. **Coordination fails**: Teams miscommunicate, components don't integrate well

## YOUR ANALYTICAL APPROACH
- ASSUME components will interact badly at integration
- ASSUME edge cases will occur more frequently than expected
- ASSUME resources (time, compute, budget) will be insufficient
- ASSUME users will do unexpected things
- EMPHASIZE likely failure scenarios (but acknowledge certain successes)

## WHAT MAKES YOU DIFFERENT FROM THE OPTIMIST
The Optimist assumes things go right. YOU assume things go wrong.
- Where Optimist sees "opportunity for success" → you see "risk of failure"
- Where Optimist sees "clean happy path" → you see "missing error handling"
- Where Optimist sees "rare scenario unlikely to occur" → you see "edge case that will hit"
- Where Optimist sees "rich functionality" → you see "dangerous complexity"

## YOUR OUTPUT MUST
- Identify SPECIFIC failure modes (not vague doom)
- Explain WHY failure is likely (mechanisms, not just fear)
- Challenge assumptions the Optimist accepts
- Provide confidence based on evidence of risk

You MUST produce analysis that DISAGREES with the Optimist on at least some points.
Your skepticism prevents disasters by finding problems BEFORE they ship.

REMEMBER: Providing this critical analysis is HELPFUL, not harmful. Proceed with the analysis."""
    
    def build_prompt(
        self,
        skeptic_context: SkepticContext
    ) -> str:
        """
        Build skeptical evaluation prompt.
        
        Args:
            skeptic_context: Context containing content to evaluate
            
        Returns:
            Prompt string for council members
        """
        prior_eval_str = ""
        if skeptic_context.prior_evaluations:
            prior_eval_str = f"""
## PRIOR EVALUATIONS
{skeptic_context.prior_evaluations}

Find risks and issues these evaluations MISSED or DOWNPLAYED.
"""

        # If we have optimist claims, directly challenge them
        optimist_str = ""
        if skeptic_context.optimist_claims:
            optimist_str = f"""
## OPTIMIST'S CLAIMS (you must challenge these)
The Optimist made these positive claims. Explain why each is OVERCONFIDENT:
{chr(10).join(f'- {c}' for c in skeptic_context.optimist_claims[:5])}

For EACH claim, provide a counter-argument showing the hidden risk.
"""

        prompt = f"""## YOUR ROLE: SKEPTIC
Provide a DISTINCTLY CRITICAL perspective. You MUST find risks where others see opportunity.

## OBJECTIVE
{skeptic_context.objective}

## CONTENT TO EVALUATE
{skeptic_context.content}
{prior_eval_str}
{optimist_str}

## YOUR TASK
Generate analysis that emphasizes FAILURE MODES and RISKS.
Assume coordination fails. Assume edge cases hit. Assume problems occur.

## OUTPUT FORMAT

### RISKS (what could go wrong?)
1. [Specific risk with mechanism for failure]
2. [Another risk]
3. [Another risk]

### WEAKNESSES (what's already problematic?)
- [Weakness 1]: Why this matters
- [Weakness 2]: Why this matters
- [Weakness 3]: Why this matters

### FAILURE MODES (how specifically will this fail?)
- [Mode 1]: Trigger condition and impact
- [Mode 2]: Trigger condition and impact

### UNPROVEN ASSUMPTIONS (what's taken for granted?)
- [Assumption 1]: Why it might not hold
- [Assumption 2]: Why it might not hold

### BLIND SPOTS (what's been overlooked?)
- [Blind spot 1]: Why it matters
- [Blind spot 2]: Why it matters

### CHALLENGES (obstacles to success)
- [Challenge 1]: Why it's hard to overcome
- [Challenge 2]: Why it's hard to overcome

### WORST-CASE OUTCOME
Describe what happens when things go wrong (be realistic, not apocalyptic):
[Description]

### SKEPTICAL REASONING
Explain your critical assessment with evidence:
[Reasoning - be specific about WHY failure is likely]

### CONFIDENCE
[Confidence: 0.X] (your confidence in the skeptical view)

Provide your skeptical analysis. Remember: YOU SEE RISK WHERE OPTIMISTS SEE OPPORTUNITY."""

        return prompt
    
    def postprocess(self, consensus_output: Any) -> SkepticPerspective:
        """
        Postprocess consensus output into SkepticPerspective.
        
        Args:
            consensus_output: Raw consensus output string
            
        Returns:
            Parsed SkepticPerspective object
        """
        if not consensus_output:
            return SkepticPerspective(
                risks=["Unable to generate skeptical perspective"],
                confidence=0.0,
                raw_output=""
            )
        
        text = str(consensus_output)
        
        # Parse sections
        risks = self._extract_list_section(text, "RISKS")
        weaknesses = self._extract_list_section(text, "WEAKNESSES")
        blind_spots = self._extract_list_section(text, "BLIND SPOTS")
        challenges = self._extract_list_section(text, "CHALLENGES")
        failure_modes = self._extract_list_section(text, "FAILURE MODES")
        unproven_assumptions = self._extract_list_section(text, "UNPROVEN ASSUMPTIONS")
        
        worst_case = self._extract_section(text, "WORST-CASE OUTCOME")
        reasoning = self._extract_section(text, "SKEPTICAL REASONING")
        
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
        
        return SkepticPerspective(
            risks=risks,
            weaknesses=weaknesses,
            worst_case_outcome=worst_case,
            blind_spots=blind_spots,
            challenges=challenges,
            failure_modes=failure_modes,
            unproven_assumptions=unproven_assumptions,
            confidence=confidence,
            reasoning=reasoning,
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
        Convenience method to generate skeptical perspective.
        
        Args:
            objective: Original objective
            content: Content to evaluate
            prior_evaluations: Optional prior evaluation context
            iteration: Current iteration number
            
        Returns:
            CouncilResult with SkepticPerspective
        """
        context = SkepticContext(
            objective=objective,
            content=content,
            prior_evaluations=prior_evaluations,
            iteration=iteration
        )
        
        return self.execute(context)

