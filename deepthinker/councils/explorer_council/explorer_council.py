"""
Explorer Council for DeepThinker 2.0.

Broad, shallow, fast reconnaissance council for initial landscape mapping.

Purpose:
- Map the problem space quickly
- Identify known facts, unknowns, hypotheses, and questions
- NO recommendations, synthesis, or conclusions

Constraints:
- Max 2000 tokens output
- No web search (fast/shallow)
- Memory write: ephemeral only
- Forbidden: recommendations, synthesis, conclusions, action_items
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...consensus.voting import MajorityVoteConsensus

logger = logging.getLogger(__name__)


@dataclass
class ExplorerContext:
    """
    Input context for Explorer Council.
    
    Minimal context for fast reconnaissance.
    
    Attributes:
        objective: The mission objective to explore
        focus_areas: Optional areas to prioritize
        time_budget_seconds: Max time for exploration
        max_depth: How deep to explore (1=surface, 2=moderate, 3=detailed)
    """
    objective: str
    focus_areas: List[str] = field(default_factory=list)
    time_budget_seconds: int = 60
    max_depth: int = 1
    
    def validate(self) -> bool:
        """Validate the context."""
        return bool(self.objective and len(self.objective) > 0)


@dataclass
class ExplorerOutput:
    """
    Output from Explorer Council.
    
    Focused on landscape mapping, NOT recommendations.
    
    Attributes:
        known_facts: Facts established from the objective/context
        unknowns: Things we don't know and need to find out
        hypotheses: Initial hypotheses to investigate
        questions: Questions that need answers
        landscape_summary: Brief summary of the problem space
        raw_output: The raw LLM output
        confidence_score: Confidence in the exploration (0-1)
        
    FORBIDDEN (these should never be populated):
        - recommendations
        - synthesis
        - conclusions
        - action_items
    """
    known_facts: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    landscape_summary: str = ""
    raw_output: str = ""
    confidence_score: float = 0.5
    
    @classmethod
    def from_text(cls, text: str) -> "ExplorerOutput":
        """
        Parse explorer output from raw text.
        
        Args:
            text: Raw LLM output
            
        Returns:
            Parsed ExplorerOutput
        """
        lines = text.strip().split('\n')
        
        known_facts = []
        unknowns = []
        hypotheses = []
        questions = []
        
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            line_stripped = line.strip()
            
            # Detect section headers
            if any(kw in line_lower for kw in ['known fact', 'established', 'we know']):
                current_section = 'known_facts'
            elif any(kw in line_lower for kw in ['unknown', "don't know", 'unclear', 'need to find']):
                current_section = 'unknowns'
            elif any(kw in line_lower for kw in ['hypothesis', 'hypothes', 'might be', 'possibly']):
                current_section = 'hypotheses'
            elif any(kw in line_lower for kw in ['question', 'need to answer', 'investigate']):
                current_section = 'questions'
            elif line_stripped.startswith(('-', '*', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                content = line_stripped.lstrip('-*•0123456789.) ')
                if content:
                    if current_section == 'known_facts':
                        known_facts.append(content)
                    elif current_section == 'unknowns':
                        unknowns.append(content)
                    elif current_section == 'hypotheses':
                        hypotheses.append(content)
                    elif current_section == 'questions':
                        questions.append(content)
                    # If no section, try to categorize by content
                    elif '?' in content:
                        questions.append(content)
                    elif any(kw in content.lower() for kw in ['might', 'could', 'possibly', 'likely']):
                        hypotheses.append(content)
        
        # Extract summary (first paragraph or first 200 chars)
        summary = text[:300].split('\n\n')[0] if text else ""
        
        # Estimate confidence based on completeness
        total_items = len(known_facts) + len(unknowns) + len(hypotheses) + len(questions)
        confidence = min(0.9, 0.3 + (total_items * 0.05))
        
        return cls(
            known_facts=known_facts[:10],
            unknowns=unknowns[:10],
            hypotheses=hypotheses[:10],
            questions=questions[:15],
            landscape_summary=summary,
            raw_output=text,
            confidence_score=confidence,
        )
    
    def has_content(self) -> bool:
        """Check if exploration produced meaningful content."""
        return bool(self.known_facts or self.unknowns or self.hypotheses or self.questions)
    
    def get_priority_questions(self, limit: int = 5) -> List[str]:
        """Get the most important questions to answer next."""
        return self.questions[:limit]
    
    def get_investigation_targets(self) -> List[str]:
        """Get targets for deeper investigation by EvidenceCouncil."""
        targets = []
        # Unknowns are high priority
        targets.extend(self.unknowns[:3])
        # Questions are next
        targets.extend(self.questions[:3])
        # Hypotheses need validation
        targets.extend([f"Validate: {h}" for h in self.hypotheses[:2]])
        return targets


# Default models for fast exploration
EXPLORER_MODELS = [
    ("llama3.2:3b", 0.5),
    ("gemma3:4b", 0.5),
]


class ExplorerCouncil(BaseCouncil):
    """
    Explorer Council for fast, broad reconnaissance.
    
    Used in the reconnaissance phase to:
    - Quickly map the problem landscape
    - Identify what is known vs unknown
    - Generate initial hypotheses
    - Create questions for deeper investigation
    
    This council does NOT:
    - Make recommendations
    - Generate synthesis or conclusions
    - Produce action items
    - Write to stable memory
    """
    
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        consensus_engine: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
        cognitive_spine: Optional[Any] = None,
        council_definition: Optional[Any] = None,
    ):
        """
        Initialize Explorer Council.
        
        Args:
            model_pool: Custom model pool (defaults to small/fast models)
            consensus_engine: Custom consensus (defaults to MajorityVoteConsensus)
            ollama_base_url: Ollama server URL
            cognitive_spine: Optional CognitiveSpine
            council_definition: Optional dynamic configuration
        """
        if model_pool is None:
            model_pool = ModelPool(
                pool_config=EXPLORER_MODELS,
                base_url=ollama_base_url,
                max_workers=2,
            )
        
        if consensus_engine is None:
            consensus_engine = MajorityVoteConsensus(
                ollama_base_url=ollama_base_url
            )
        
        super().__init__(
            model_pool=model_pool,
            consensus_engine=consensus_engine,
            council_name="explorer_council",
            cognitive_spine=cognitive_spine,
            council_definition=council_definition,
        )
        
        self._load_system_prompt()
    
    def _load_system_prompt(self) -> None:
        """Load the exploration-focused system prompt."""
        self._system_prompt = """You are an Explorer Agent performing RECONNAISSANCE only.

Your job is to quickly MAP THE LANDSCAPE of a problem or objective.

You MUST produce:
1. KNOWN FACTS: What can be established from the objective?
2. UNKNOWNS: What information is missing or unclear?
3. HYPOTHESES: What might be true that needs validation?
4. QUESTIONS: What questions need to be answered?

You MUST NOT produce:
- Recommendations or advice
- Synthesis or conclusions
- Action items or plans
- Final assessments

Keep your exploration FAST and BROAD. Do not go deep. Identify the terrain, then stop.

Format your output with clear sections:
### Known Facts
- [fact 1]
- [fact 2]

### Unknowns
- [unknown 1]
- [unknown 2]

### Hypotheses
- [hypothesis 1]
- [hypothesis 2]

### Questions to Investigate
- [question 1]
- [question 2]

Be concise. Maximum 2000 tokens."""
    
    def build_prompt(self, context: ExplorerContext) -> str:
        """
        Build the exploration prompt.
        
        Args:
            context: ExplorerContext with objective
            
        Returns:
            Prompt string
        """
        focus_str = ""
        if context.focus_areas:
            focus_str = "\n\n### Priority Focus Areas:\n" + "\n".join(
                f"- {area}" for area in context.focus_areas
            )
        
        depth_instruction = ""
        if context.max_depth == 1:
            depth_instruction = "\nStay at SURFACE LEVEL. Quick scan only."
        elif context.max_depth == 2:
            depth_instruction = "\nModerate depth. Cover main aspects."
        else:
            depth_instruction = "\nGo into reasonable detail on key areas."
        
        prompt = f"""## RECONNAISSANCE MISSION

Explore and map the following objective:

### OBJECTIVE
{context.objective}
{focus_str}
{depth_instruction}

Produce a LANDSCAPE MAP with:
1. Known Facts (what we can establish)
2. Unknowns (what we need to find out)
3. Hypotheses (what might be true)
4. Questions (what to investigate next)

DO NOT make recommendations or conclusions. Just map the territory."""
        
        return prompt
    
    def postprocess(self, consensus_output: Any) -> ExplorerOutput:
        """
        Convert consensus output to ExplorerOutput.
        
        Args:
            consensus_output: Raw consensus result
            
        Returns:
            Structured ExplorerOutput
        """
        if not consensus_output:
            return ExplorerOutput()
        
        text = str(consensus_output)
        return ExplorerOutput.from_text(text)
    
    def explore(
        self,
        objective: str,
        focus_areas: Optional[List[str]] = None,
        time_budget_seconds: int = 60,
        max_depth: int = 1,
    ) -> CouncilResult:
        """
        Convenience method to run exploration.
        
        Args:
            objective: What to explore
            focus_areas: Optional priority areas
            time_budget_seconds: Time budget
            max_depth: Exploration depth (1-3)
            
        Returns:
            CouncilResult with ExplorerOutput
        """
        context = ExplorerContext(
            objective=objective,
            focus_areas=focus_areas or [],
            time_budget_seconds=time_budget_seconds,
            max_depth=max_depth,
        )
        
        if not context.validate():
            return CouncilResult(
                output=ExplorerOutput(),
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error="Invalid context: objective is required",
            )
        
        return self.execute(context)
    
    def get_forbidden_output_types(self) -> List[str]:
        """Return list of forbidden output types for this council."""
        return [
            "recommendations",
            "synthesis",
            "conclusions",
            "action_items",
            "final_report",
        ]

