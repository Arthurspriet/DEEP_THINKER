"""
Evidence Council for DeepThinker 2.0.

Narrow, slow, structured evidence gathering council for deep analysis.

Purpose:
- Answer specific questions with evidence
- Gather citations and sources
- Validate or refute hypotheses
- Produce structured evidence items

Constraints:
- Requires specific questions as input
- Web search enabled
- Memory write: stable (validated evidence only)
- Slower, more thorough than ExplorerCouncil
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...consensus.voting import MajorityVoteConsensus

logger = logging.getLogger(__name__)

# Try to import web search
try:
    from ...tools.websearch_tool import WebSearchTool, create_websearch_tool
    WEBSEARCH_AVAILABLE = True
except ImportError:
    WEBSEARCH_AVAILABLE = False
    WebSearchTool = None


@dataclass
class Citation:
    """
    A citation/source for evidence.
    
    Attributes:
        source: Source name or URL
        title: Title of the source
        snippet: Relevant excerpt
        confidence: Confidence in this source (0-1)
    """
    source: str
    title: str = ""
    snippet: str = ""
    confidence: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "confidence": self.confidence,
        }


@dataclass
class EvidenceItem:
    """
    A single piece of evidence.
    
    Attributes:
        claim: The claim or finding
        support: Evidence supporting the claim
        citations: Sources for this evidence
        confidence: Confidence level (0-1)
        answers_question: Which input question this answers
    """
    claim: str
    support: str = ""
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.5
    answers_question: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "support": self.support,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "answers_question": self.answers_question,
        }


@dataclass
class EvidenceContext:
    """
    Input context for Evidence Council.
    
    Focused on answering specific questions with evidence.
    
    Attributes:
        objective: The overall mission objective
        questions: Specific questions to answer (REQUIRED)
        prior_evidence: Previously gathered evidence
        hypotheses_to_validate: Hypotheses that need evidence
        allow_web_search: Whether web search is enabled
        max_sources: Maximum sources to gather per question
        knowledge_context: Optional knowledge from RAG retrieval
    """
    objective: str
    questions: List[str] = field(default_factory=list)
    prior_evidence: List[str] = field(default_factory=list)
    hypotheses_to_validate: List[str] = field(default_factory=list)
    allow_web_search: bool = True
    max_sources: int = 5
    # Knowledge context from RAG retrieval
    knowledge_context: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate the context."""
        return bool(self.objective and self.questions)


@dataclass
class EvidenceOutput:
    """
    Output from Evidence Council.
    
    Structured evidence with citations.
    
    Attributes:
        evidence_items: List of evidence items found
        citations: All citations gathered
        answered_questions: Questions successfully answered
        remaining_questions: Questions still needing answers
        validated_hypotheses: Hypotheses confirmed by evidence
        refuted_hypotheses: Hypotheses contradicted by evidence
        confidence_score: Overall confidence in findings
        raw_output: Raw LLM output
        web_search_count: Number of web searches performed
    """
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    answered_questions: List[str] = field(default_factory=list)
    remaining_questions: List[str] = field(default_factory=list)
    validated_hypotheses: List[str] = field(default_factory=list)
    refuted_hypotheses: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    raw_output: str = ""
    web_search_count: int = 0
    
    @classmethod
    def from_text(
        cls,
        text: str,
        input_questions: Optional[List[str]] = None
    ) -> "EvidenceOutput":
        """
        Parse evidence output from raw text.
        
        Args:
            text: Raw LLM output
            input_questions: Original questions for tracking
            
        Returns:
            Parsed EvidenceOutput
        """
        lines = text.strip().split('\n')
        
        evidence_items = []
        citations = []
        answered = []
        remaining = []
        validated = []
        refuted = []
        
        current_section = None
        current_evidence: Optional[Dict] = None
        
        for line in lines:
            line_lower = line.lower().strip()
            line_stripped = line.strip()
            
            # Detect section headers
            if any(kw in line_lower for kw in ['evidence:', 'finding:', 'found:']):
                current_section = 'evidence'
                if current_evidence:
                    evidence_items.append(EvidenceItem(**current_evidence))
                current_evidence = {"claim": "", "support": "", "citations": [], "confidence": 0.6}
            elif any(kw in line_lower for kw in ['source:', 'citation:', 'reference:']):
                current_section = 'citations'
            elif any(kw in line_lower for kw in ['answered', 'resolved']):
                current_section = 'answered'
            elif any(kw in line_lower for kw in ['remaining', 'unanswered', 'still need']):
                current_section = 'remaining'
            elif any(kw in line_lower for kw in ['validated', 'confirmed', 'supported']):
                current_section = 'validated'
            elif any(kw in line_lower for kw in ['refuted', 'contradicted', 'disproven']):
                current_section = 'refuted'
            elif line_stripped.startswith(('-', '*', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                content = line_stripped.lstrip('-*•0123456789.) ')
                if content:
                    if current_section == 'evidence' and current_evidence:
                        if not current_evidence["claim"]:
                            current_evidence["claim"] = content
                        else:
                            current_evidence["support"] += " " + content
                    elif current_section == 'citations':
                        citations.append(Citation(source=content))
                    elif current_section == 'answered':
                        answered.append(content)
                    elif current_section == 'remaining':
                        remaining.append(content)
                    elif current_section == 'validated':
                        validated.append(content)
                    elif current_section == 'refuted':
                        refuted.append(content)
            elif current_section == 'evidence' and current_evidence and line_stripped:
                # Continue building evidence support
                current_evidence["support"] += " " + line_stripped
        
        # Don't forget the last evidence item
        if current_evidence and current_evidence["claim"]:
            evidence_items.append(EvidenceItem(**current_evidence))
        
        # Track which input questions were answered
        if input_questions:
            for q in input_questions:
                q_lower = q.lower()
                if any(q_lower in a.lower() or a.lower() in q_lower for a in answered):
                    continue
                elif q not in remaining:
                    remaining.append(q)
        
        # Calculate confidence based on evidence quality
        total_confidence = 0.0
        if evidence_items:
            for ev in evidence_items:
                total_confidence += ev.confidence
            avg_confidence = total_confidence / len(evidence_items)
        else:
            avg_confidence = 0.3
        
        return cls(
            evidence_items=evidence_items[:20],
            citations=citations[:15],
            answered_questions=answered[:10],
            remaining_questions=remaining[:10],
            validated_hypotheses=validated[:5],
            refuted_hypotheses=refuted[:5],
            confidence_score=avg_confidence,
            raw_output=text,
        )
    
    def has_strong_evidence(self, min_confidence: float = 0.7) -> bool:
        """Check if we have high-confidence evidence."""
        return any(e.confidence >= min_confidence for e in self.evidence_items)
    
    def get_high_confidence_items(self, min_confidence: float = 0.7) -> List[EvidenceItem]:
        """Get evidence items above confidence threshold."""
        return [e for e in self.evidence_items if e.confidence >= min_confidence]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "citations": [c.to_dict() for c in self.citations],
            "answered_questions": self.answered_questions,
            "remaining_questions": self.remaining_questions,
            "validated_hypotheses": self.validated_hypotheses,
            "refuted_hypotheses": self.refuted_hypotheses,
            "confidence_score": self.confidence_score,
            "web_search_count": self.web_search_count,
        }


# Default models for evidence gathering (medium tier for thoroughness)
EVIDENCE_MODELS = [
    ("gemma3:12b", 0.4),
    ("deepseek-r1:8b", 0.3),
]


class EvidenceCouncil(BaseCouncil):
    """
    Evidence Council for deep, structured evidence gathering.
    
    Used in analysis and deep_analysis phases to:
    - Answer specific questions with evidence
    - Gather and cite sources
    - Validate or refute hypotheses
    - Produce structured, citation-backed findings
    
    This council:
    - Requires specific questions as input
    - Can perform web searches for evidence
    - Writes to stable memory (validated evidence)
    - Is slower and more thorough than ExplorerCouncil
    """
    
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        consensus_engine: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
        enable_websearch: bool = True,
        max_search_results: int = 5,
        cognitive_spine: Optional[Any] = None,
        council_definition: Optional[Any] = None,
    ):
        """
        Initialize Evidence Council.
        
        Args:
            model_pool: Custom model pool
            consensus_engine: Custom consensus
            ollama_base_url: Ollama server URL
            enable_websearch: Whether to enable web search
            max_search_results: Max results per search
            cognitive_spine: Optional CognitiveSpine
            council_definition: Optional dynamic configuration
        """
        if model_pool is None:
            model_pool = ModelPool(
                pool_config=EVIDENCE_MODELS,
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
            council_name="evidence_council",
            cognitive_spine=cognitive_spine,
            council_definition=council_definition,
        )
        
        # Web search setup
        self.enable_websearch = enable_websearch and WEBSEARCH_AVAILABLE
        self.max_search_results = max_search_results
        self._websearch_tool: Optional[Any] = None
        self._last_search_count = 0
        self._input_questions: List[str] = []
        
        self._init_websearch()
        self._load_system_prompt()
    
    def _init_websearch(self) -> None:
        """Initialize web search tool."""
        if self.enable_websearch and WEBSEARCH_AVAILABLE:
            try:
                self._websearch_tool = create_websearch_tool(
                    max_results=self.max_search_results,
                    timeout=15
                )
                logger.debug("WebSearch initialized for EvidenceCouncil")
            except Exception as e:
                logger.warning(f"Failed to init WebSearch: {e}")
                self._websearch_tool = None
                self.enable_websearch = False
    
    def _load_system_prompt(self) -> None:
        """Load the evidence-focused system prompt."""
        self._system_prompt = """You are an Evidence Analyst gathering STRUCTURED EVIDENCE.

Your job is to ANSWER SPECIFIC QUESTIONS with evidence and citations.

For each question, you MUST provide:
1. EVIDENCE: What facts/data answer this question?
2. SUPPORT: What supports this evidence?
3. SOURCES: Where does this evidence come from?
4. CONFIDENCE: How confident are you (0.0-1.0)?

You MUST also:
- Identify which questions you've ANSWERED
- Identify which questions REMAIN UNANSWERED
- Validate or refute any HYPOTHESES provided

Format your output clearly:

### Evidence 1: [Question being answered]
**Claim:** [The key finding]
**Support:** [Supporting details]
**Source:** [Citation]
**Confidence:** [0.0-1.0]

### Answered Questions
- [question 1]
- [question 2]

### Remaining Questions
- [question still needing answers]

### Validated Hypotheses
- [hypothesis confirmed by evidence]

### Refuted Hypotheses  
- [hypothesis contradicted by evidence]

Be thorough and cite sources. Quality over speed."""
    
    def build_prompt(self, context: EvidenceContext) -> str:
        """
        Build the evidence gathering prompt.
        
        Args:
            context: EvidenceContext with questions
            
        Returns:
            Prompt string
        """
        # Store questions for postprocessing
        self._input_questions = context.questions.copy()
        
        questions_str = "\n".join(f"- {q}" for q in context.questions)
        
        prior_str = ""
        if context.prior_evidence:
            prior_str = "\n\n### Prior Evidence (already gathered):\n" + "\n".join(
                f"- {e}" for e in context.prior_evidence[:5]
            )
        
        hypotheses_str = ""
        if context.hypotheses_to_validate:
            hypotheses_str = "\n\n### Hypotheses to Validate:\n" + "\n".join(
                f"- {h}" for h in context.hypotheses_to_validate
            )
        
        # Build knowledge context (from RAG retrieval)
        knowledge_str = ""
        if context.knowledge_context:
            knowledge_str = f"\n\n### Retrieved Knowledge (use as reference):\n{context.knowledge_context}"
        
        prompt = f"""## EVIDENCE GATHERING MISSION

### Objective
{context.objective}

### Questions to Answer (REQUIRED)
{questions_str}
{prior_str}
{hypotheses_str}
{knowledge_str}

For EACH question:
1. Find evidence that answers it
2. Cite your sources
3. Rate your confidence (0.0-1.0)

Indicate which questions you've answered and which remain.
If you can validate or refute any hypotheses, state that clearly."""
        
        return prompt
    
    def postprocess(self, consensus_output: Any) -> EvidenceOutput:
        """
        Convert consensus output to EvidenceOutput.
        
        Args:
            consensus_output: Raw consensus result
            
        Returns:
            Structured EvidenceOutput
        """
        if not consensus_output:
            return EvidenceOutput(remaining_questions=self._input_questions.copy())
        
        text = str(consensus_output)
        output = EvidenceOutput.from_text(text, self._input_questions)
        output.web_search_count = self._last_search_count
        return output
    
    def gather_evidence(
        self,
        objective: str,
        questions: List[str],
        prior_evidence: Optional[List[str]] = None,
        hypotheses_to_validate: Optional[List[str]] = None,
        allow_web_search: bool = True,
    ) -> CouncilResult:
        """
        Convenience method to gather evidence.
        
        Args:
            objective: Mission objective
            questions: Questions to answer (REQUIRED)
            prior_evidence: Previous evidence
            hypotheses_to_validate: Hypotheses to check
            allow_web_search: Enable web search
            
        Returns:
            CouncilResult with EvidenceOutput
        """
        if not questions:
            return CouncilResult(
                output=EvidenceOutput(),
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error="Questions are required for EvidenceCouncil",
            )
        
        context = EvidenceContext(
            objective=objective,
            questions=questions,
            prior_evidence=prior_evidence or [],
            hypotheses_to_validate=hypotheses_to_validate or [],
            allow_web_search=allow_web_search,
        )
        
        # EvidenceCouncil should NOT initiate web searches
        # Only ResearcherCouncil initiates web searches for anti-hallucination enforcement
        web_context = ""
        self._last_search_count = 0
        
        if allow_web_search and self.enable_websearch and self._websearch_tool:
            logger.warning(
                "EvidenceCouncil should not initiate web searches - "
                "delegating to ResearcherCouncil. Web search disabled for this council."
            )
            # Disable web search - ResearcherCouncil is responsible
            allow_web_search = False
            context.allow_web_search = False
        
        # Note: If web research was performed by ResearcherCouncil, it should be
        # included in prior_evidence by the caller, not by EvidenceCouncil
        
        return self.execute(context)
    
    def _perform_searches(self, context: EvidenceContext) -> str:
        """Perform web searches based on questions."""
        results = []
        
        # Search for top 3 questions
        for question in context.questions[:3]:
            try:
                query = question.rstrip('?').strip()
                if len(query) < 10:
                    continue
                
                result = self._websearch_tool._run(query)
                self._last_search_count += 1
                
                if result and "No results found" not in result:
                    results.append(f"### Search: {query}\n{result[:1500]}")
                    
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        return "\n\n".join(results)
    
    def get_required_input_fields(self) -> List[str]:
        """Return required input fields."""
        return ["objective", "questions"]

