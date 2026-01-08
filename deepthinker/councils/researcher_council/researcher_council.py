"""
Researcher Council Implementation for DeepThinker 2.0.

Multi-LLM research synthesis using voting consensus.
Aggregates findings from multiple models to produce comprehensive research.

Enhanced with:
- WebSearch tool integration for real-world evidence
- Auto-search logic based on data_needs and questions
- Tracking of web search activity
- WebSearchGate for mandatory search enforcement in factual domains
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...models.council_model_config import RESEARCHER_MODELS
from ...consensus.voting import MajorityVoteConsensus

# WebSearch tool import
try:
    from ...tools.websearch_tool import WebSearchTool, create_websearch_tool
    WEBSEARCH_AVAILABLE = True
except ImportError:
    WEBSEARCH_AVAILABLE = False
    WebSearchTool = None

# KnowledgeGate import
try:
    from ...tools.knowledge_gate import KnowledgeGate, KnowledgeGateResult
    KNOWLEDGE_GATE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GATE_AVAILABLE = False
    KnowledgeGate = None
    KnowledgeGateResult = None

# ExternalKnowledge import
try:
    from .external_knowledge import ExternalKnowledge
    EXTERNAL_KNOWLEDGE_AVAILABLE = True
except ImportError:
    EXTERNAL_KNOWLEDGE_AVAILABLE = False
    ExternalKnowledge = None

logger = logging.getLogger(__name__)


# =============================================================================
# WebSearchGate - Mandatory Search Enforcement for Factual Domains
# =============================================================================

class FactualDomain(str, Enum):
    """Domains requiring mandatory web search validation."""
    GEOPOLITICS = "geopolitics"
    ECONOMICS = "economics"
    HARDWARE = "hardware"
    REGULATION = "regulation"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    CURRENT_EVENTS = "current_events"


@dataclass
class WebSearchGateResult:
    """
    Result of WebSearchGate validation.
    
    Attributes:
        requires_search: Whether web search is mandatory
        detected_domains: Domains detected in the objective
        min_sources_required: Minimum sources required for this phase
        current_sources: Number of sources currently available
        is_satisfied: Whether search requirements are met
        blocking_reason: Reason if requirements not met
        source_quality_score: Average quality of sources
        low_trust_sources: Sources from low-trust domains
    """
    requires_search: bool = False
    detected_domains: List[str] = field(default_factory=list)
    min_sources_required: int = 0
    current_sources: int = 0
    is_satisfied: bool = True
    blocking_reason: str = ""
    source_quality_score: float = 0.5
    low_trust_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "requires_search": self.requires_search,
            "detected_domains": self.detected_domains,
            "min_sources_required": self.min_sources_required,
            "current_sources": self.current_sources,
            "is_satisfied": self.is_satisfied,
            "blocking_reason": self.blocking_reason,
            "source_quality_score": self.source_quality_score,
            "low_trust_sources_count": len(self.low_trust_sources),
        }


class WebSearchGate:
    """
    Gates phase advancement based on web search requirements.
    
    Enforces mandatory web search for factual domains and validates
    source quality and coverage before allowing phase progression.
    
    Usage:
        gate = WebSearchGate()
        result = gate.check_requirements(objective, phase, sources)
        if not result.is_satisfied:
            # Block phase advancement or trigger additional searches
    """
    
    # Keywords for domain detection
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        FactualDomain.GEOPOLITICS.value: [
            "geopolitics", "geopolitical", "international relations", "foreign policy",
            "diplomacy", "conflict", "war", "sanction", "alliance", "NATO", "UN",
            "sovereignty", "border", "territory", "treaty"
        ],
        FactualDomain.ECONOMICS.value: [
            "economics", "economy", "GDP", "inflation", "recession", "growth",
            "monetary", "fiscal", "trade", "tariff", "unemployment", "labor market",
            "interest rate", "central bank", "federal reserve", "IMF", "World Bank"
        ],
        FactualDomain.HARDWARE.value: [
            "hardware", "chip", "semiconductor", "processor", "CPU", "GPU", "ASIC",
            "fabrication", "foundry", "TSMC", "Intel", "AMD", "NVIDIA", "memory",
            "compute", "datacenter", "server"
        ],
        FactualDomain.REGULATION.value: [
            "regulation", "regulatory", "compliance", "law", "legal", "policy",
            "legislation", "mandate", "requirement", "standard", "certification",
            "GDPR", "antitrust", "privacy", "data protection"
        ],
        FactualDomain.FINANCE.value: [
            "finance", "financial", "stock", "bond", "equity", "market",
            "investment", "portfolio", "asset", "derivative", "cryptocurrency",
            "bitcoin", "blockchain", "banking", "insurance"
        ],
        FactualDomain.TECHNOLOGY.value: [
            "technology", "tech", "AI", "artificial intelligence", "machine learning",
            "software", "cloud", "SaaS", "platform", "algorithm", "data",
            "cybersecurity", "quantum", "5G", "6G"
        ],
        FactualDomain.SCIENCE.value: [
            "scientific", "research study", "experiment", "clinical trial",
            "peer review", "journal", "publication", "hypothesis", "evidence"
        ],
        FactualDomain.CURRENT_EVENTS.value: [
            "recent", "current", "latest", "2024", "2025", "today", "now",
            "this year", "this month", "breaking", "news", "announcement"
        ],
    }
    
    # Minimum sources required by phase
    MIN_SOURCES_BY_PHASE: Dict[str, int] = {
        "reconnaissance": 2,
        "analysis": 3,
        "deep_analysis": 5,
        "synthesis": 3,  # Should use existing sources, not new ones
        "default": 2,
    }
    
    # Low-trust domains to penalize or reject
    LOW_TRUST_DOMAINS: Set[str] = {
        "reddit.com", "quora.com", "yahoo.answers", "answers.com",
        "ehow.com", "wikihow.com", "buzzfeed.com", "medium.com",
        # Social media
        "twitter.com", "x.com", "facebook.com", "tiktok.com",
        # Content farms
        "forbes.com/sites", "entrepreneur.com", "inc.com",
    }
    
    # High-trust domains to prefer
    HIGH_TRUST_DOMAINS: Set[str] = {
        "arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org",
        "gov", "edu", "who.int", "un.org", "worldbank.org", "imf.org",
        "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
        "economist.com", "ft.com", "wsj.com",
    }
    
    def __init__(
        self,
        enable_blocking: bool = True,
        min_quality_score: float = 0.4,
        max_single_source_ratio: float = 0.5
    ):
        """
        Initialize the web search gate.
        
        Args:
            enable_blocking: Whether to block phase advancement on failures
            min_quality_score: Minimum average source quality (0-1)
            max_single_source_ratio: Max fraction of claims from single source
        """
        self.enable_blocking = enable_blocking
        self.min_quality_score = min_quality_score
        self.max_single_source_ratio = max_single_source_ratio
        
        # Compile keyword patterns
        self._domain_patterns: Dict[str, List[re.Pattern]] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            self._domain_patterns[domain] = [
                re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
                for kw in keywords
            ]
    
    def detect_domains(self, objective: str) -> List[str]:
        """
        Detect factual domains in an objective.
        
        Args:
            objective: The mission or phase objective
            
        Returns:
            List of detected domain names
        """
        detected = []
        
        for domain, patterns in self._domain_patterns.items():
            matches = sum(1 for p in patterns if p.search(objective))
            # Require at least 2 keyword matches for detection
            if matches >= 2:
                detected.append(domain)
            # Single match for high-signal keywords
            elif matches >= 1 and domain in [
                FactualDomain.CURRENT_EVENTS.value,
                FactualDomain.SCIENCE.value
            ]:
                detected.append(domain)
        
        return detected
    
    def requires_search(self, objective: str) -> Tuple[bool, List[str]]:
        """
        Check if objective requires web search.
        
        Args:
            objective: The mission or phase objective
            
        Returns:
            Tuple of (requires_search, detected_domains)
        """
        domains = self.detect_domains(objective)
        return len(domains) > 0, domains
    
    def get_min_sources(self, phase_name: str) -> int:
        """Get minimum sources required for a phase."""
        phase_lower = phase_name.lower()
        
        for key, min_sources in self.MIN_SOURCES_BY_PHASE.items():
            if key in phase_lower:
                return min_sources
        
        return self.MIN_SOURCES_BY_PHASE["default"]
    
    def score_source_quality(self, url: str) -> Tuple[float, str]:
        """
        Score source quality based on domain.
        
        Args:
            url: Source URL
            
        Returns:
            Tuple of (quality_score, quality_tier)
        """
        url_lower = url.lower()
        
        # Check high-trust domains
        for domain in self.HIGH_TRUST_DOMAINS:
            if domain in url_lower:
                return 0.9, "HIGH"
        
        # Check low-trust domains
        for domain in self.LOW_TRUST_DOMAINS:
            if domain in url_lower:
                return 0.2, "LOW"
        
        # Check TLD patterns
        if ".gov" in url_lower or ".edu" in url_lower:
            return 0.85, "HIGH"
        elif ".org" in url_lower:
            return 0.7, "MEDIUM"
        
        return 0.5, "MEDIUM"
    
    def check_requirements(
        self,
        objective: str,
        phase_name: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        claims_count: int = 0
    ) -> WebSearchGateResult:
        """
        Check if web search requirements are satisfied.
        
        Args:
            objective: The objective being researched
            phase_name: Current phase name
            sources: List of sources with 'url' and optionally 'quality_score'
            claims_count: Number of claims in output
            
        Returns:
            WebSearchGateResult with validation details
        """
        sources = sources or []
        
        # Detect if search is required
        requires, domains = self.requires_search(objective)
        min_sources = self.get_min_sources(phase_name) if requires else 0
        
        # Score and filter sources
        quality_scores = []
        low_trust = []
        
        for source in sources:
            url = source.get("url", "")
            if not url:
                continue
            
            score, tier = self.score_source_quality(url)
            quality_scores.append(score)
            
            if tier == "LOW":
                low_trust.append(url)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        current_count = len(sources)
        
        # Determine if satisfied
        is_satisfied = True
        blocking_reason = ""
        
        if requires:
            if current_count < min_sources:
                is_satisfied = False
                blocking_reason = (
                    f"Insufficient sources: {current_count}/{min_sources} required "
                    f"for factual domains: {domains}"
                )
            elif avg_quality < self.min_quality_score:
                is_satisfied = False
                blocking_reason = (
                    f"Source quality too low: {avg_quality:.2f} "
                    f"(min: {self.min_quality_score})"
                )
            elif len(low_trust) > len(sources) * 0.5:
                is_satisfied = False
                blocking_reason = (
                    f"Too many low-trust sources: {len(low_trust)}/{len(sources)}"
                )
        
        # Log if blocking
        if not is_satisfied and self.enable_blocking:
            logger.warning(f"WebSearchGate blocking: {blocking_reason}")
        
        return WebSearchGateResult(
            requires_search=requires,
            detected_domains=domains,
            min_sources_required=min_sources,
            current_sources=current_count,
            is_satisfied=is_satisfied,
            blocking_reason=blocking_reason,
            source_quality_score=avg_quality,
            low_trust_sources=low_trust,
        )
    
    def should_block_phase(self, result: WebSearchGateResult) -> Tuple[bool, str]:
        """
        Determine if phase should be blocked based on gate result.
        
        Args:
            result: WebSearchGateResult from check_requirements
            
        Returns:
            Tuple of (should_block, reason)
        """
        if not self.enable_blocking:
            return False, ""
        
        if not result.is_satisfied:
            return True, result.blocking_reason
        
        return False, ""


# Global WebSearchGate instance
_web_search_gate: Optional[WebSearchGate] = None


def get_web_search_gate() -> WebSearchGate:
    """Get the global WebSearchGate instance."""
    global _web_search_gate
    if _web_search_gate is None:
        _web_search_gate = WebSearchGate()
    return _web_search_gate


@dataclass
class ResearchContext:
    """Context for researcher council execution."""
    
    objective: str
    focus_areas: List[str] = field(default_factory=list)
    prior_knowledge: Optional[str] = None
    constraints: Optional[str] = None
    planner_requirements: Optional[str] = None
    # New fields for iterative research
    allow_internet: bool = True
    data_needs: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    requires_evidence: bool = False
    subgoals: List[str] = field(default_factory=list)


@dataclass
class ResearchFindings:
    """Structured research findings with iteration-driving fields."""
    
    summary: str
    key_points: List[str]
    recommendations: List[str]
    sources_suggested: List[str]
    raw_output: str
    # Web research tracking
    web_search_results: List[Dict[str, str]] = field(default_factory=list)
    web_search_count: int = 0
    queries_executed: List[str] = field(default_factory=list)
    # External knowledge artifact (anti-hallucination)
    external_knowledge: Optional[Any] = None
    # Iteration-driving fields (populated by from_text or evaluator)
    gaps: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    evidence_requests: List[str] = field(default_factory=list)
    next_focus_areas: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    iteration: int = 1
    
    @classmethod
    def from_text(cls, text: str, iteration: int = 1) -> "ResearchFindings":
        """Parse research findings from text output including iteration fields.
        
        Enhanced parsing that handles:
        - Markdown bold formatting (**text**:)
        - Numbered items with various formats (1., 1), 1:)
        - Content on same line as headers
        - Fallback extraction when structured parsing fails
        """
        import re
        import logging
        logger = logging.getLogger(__name__)
        
        lines = text.strip().split('\n')
        
        key_points = []
        recommendations = []
        sources = []
        gaps = []
        unresolved_questions = []
        evidence_requests = []
        next_focus_areas = []
        
        current_section = None
        
        def clean_content(content: str) -> str:
            """Clean up content by removing markdown formatting artifacts."""
            # Remove leading markdown bold markers
            content = re.sub(r'^\*\*', '', content)
            # Remove trailing bold markers and colons
            content = re.sub(r'\*\*:?\s*$', '', content)
            # Remove leading/trailing whitespace
            return content.strip()
        
        def extract_list_item(line: str) -> Optional[str]:
            """Extract content from a list item line."""
            line = line.strip()
            # Match: "1. content", "1) content", "- content", "* content", "• content"
            match = re.match(r'^[\d]+[.\):\s]+\s*(.+)$', line)
            if match:
                return clean_content(match.group(1))
            match = re.match(r'^[-*•]\s*(.+)$', line)
            if match:
                return clean_content(match.group(1))
            return None
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Detect section headers (with multiple variations)
            is_header = False
            if any(kw in line_lower for kw in ['key point', 'key finding', '### key', 'key insight']):
                current_section = 'key_points'
                is_header = True
            elif any(kw in line_lower for kw in ['recommend', 'suggestion', 'action']):
                current_section = 'recommendations'
                is_header = True
            elif any(kw in line_lower for kw in ['source', 'reference', 'citation']):
                current_section = 'sources'
                is_header = True
            elif any(kw in line_lower for kw in ['gap', 'missing', 'incomplete']):
                current_section = 'gaps'
                is_header = True
            elif any(kw in line_lower for kw in ['unresolved', 'open question', 'unclear', 'unknown']):
                current_section = 'unresolved_questions'
                is_header = True
            elif any(kw in line_lower for kw in ['evidence', 'verification', 'data need', 'require']):
                current_section = 'evidence_requests'
                is_header = True
            elif any(kw in line_lower for kw in ['focus', 'deeper investigation', 'next area', 'priority']):
                current_section = 'next_focus_areas'
                is_header = True
            
            # Skip if this is a header line
            if is_header:
                continue
            
            # Try to extract list item content
            content = extract_list_item(line_stripped)
            
            if content and len(content) > 5:  # Minimum meaningful content
                if current_section == 'key_points':
                    key_points.append(content)
                elif current_section == 'recommendations':
                    recommendations.append(content)
                elif current_section == 'sources':
                    sources.append(content)
                elif current_section == 'gaps':
                    gaps.append(content)
                elif current_section == 'unresolved_questions':
                    unresolved_questions.append(content)
                elif current_section == 'evidence_requests':
                    evidence_requests.append(content)
                elif current_section == 'next_focus_areas':
                    next_focus_areas.append(content)
        
        # Fallback: If no key_points found, try extracting numbered items from full text
        if not key_points:
            # Look for numbered list items anywhere in the text
            numbered_items = re.findall(r'\d+\.\s+\*?\*?([^*\n]+(?:\*\*)?[^:\n]*)', text)
            for item in numbered_items[:10]:
                cleaned = clean_content(item)
                if cleaned and len(cleaned) > 10:
                    key_points.append(cleaned)
            
            if key_points:
                logger.debug(f"Fallback parsing extracted {len(key_points)} key points")
        
        # Extract confidence score if present
        confidence = 0.5
        conf_match = re.search(r'confidence[:\s]*([0-9.]+)', text, re.IGNORECASE)
        if conf_match:
            try:
                conf_val = float(conf_match.group(1))
                confidence = conf_val if conf_val <= 1.0 else conf_val / 10.0
            except ValueError:
                pass
        
        # Log parsing results for debugging
        if not key_points and len(text) > 100:
            logger.warning(
                f"ResearchFindings parsing found no key_points in {len(text)} char output. "
                f"First 200 chars: {text[:200]}"
            )
        
        return cls(
            summary=text[:500] + "..." if len(text) > 500 else text,
            key_points=key_points[:10],
            recommendations=recommendations[:5],
            sources_suggested=sources[:5],
            raw_output=text,
            gaps=gaps[:5],
            unresolved_questions=unresolved_questions[:5],
            evidence_requests=evidence_requests[:5],
            next_focus_areas=next_focus_areas[:5],
            confidence_score=confidence,
            iteration=iteration
        )
    
    def has_unresolved_work(self) -> bool:
        """Check if there are gaps or questions requiring more iteration."""
        return bool(self.gaps or self.unresolved_questions or self.evidence_requests)
    
    def get_priority_focus_areas(self, limit: int = 3) -> List[str]:
        """Get prioritized focus areas for next iteration."""
        # Prioritize: gaps > unresolved questions > next focus areas
        all_areas = self.gaps[:2] + self.unresolved_questions[:2] + self.next_focus_areas[:2]
        return all_areas[:limit]


class ResearcherCouncil(BaseCouncil):
    """
    Research council for information synthesis.
    
    Multiple research models analyze topics and synthesize findings
    using voting consensus to select the most agreed-upon insights.
    
    Enhanced with:
    - WebSearch tool for real-world evidence gathering
    - Automatic query generation from data_needs and questions
    - Web search tracking for mission iteration control
    
    Supports dynamic configuration via CouncilDefinition for
    runtime model/temperature/persona selection.
    """
    
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        consensus_engine: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
        enable_websearch: bool = True,
        max_search_results: int = 5,
        cognitive_spine: Optional[Any] = None,
        council_definition: Optional[Any] = None
    ):
        """
        Initialize researcher council.
        
        Args:
            model_pool: Custom model pool (defaults to RESEARCHER_MODELS)
            consensus_engine: Custom consensus (defaults to MajorityVoteConsensus)
                            If None and cognitive_spine provided, gets from spine
            ollama_base_url: Ollama server URL
            enable_websearch: Whether to enable web search capability
            max_search_results: Maximum results per search query
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
                    pool_config=RESEARCHER_MODELS,
                    base_url=ollama_base_url
                )
        
        # Get consensus from CognitiveSpine if not provided
        if consensus_engine is None:
            if cognitive_spine is not None:
                consensus_engine = cognitive_spine.get_consensus_engine(
                    "voting", "researcher_council"
                )
            else:
                consensus_engine = MajorityVoteConsensus(
                    ollama_base_url=ollama_base_url
                )
        
        super().__init__(
            model_pool=model_pool,
            consensus_engine=consensus_engine,
            council_name="researcher_council",
            cognitive_spine=cognitive_spine,
            council_definition=council_definition
        )
        
        # WebSearch configuration
        self.enable_websearch = enable_websearch and WEBSEARCH_AVAILABLE
        self.max_search_results = max_search_results
        self._websearch_tool: Optional[WebSearchTool] = None
        
        # Track web searches for iteration control
        self._last_search_count: int = 0
        self._last_queries: List[str] = []
        self._last_search_results: List[Dict[str, str]] = []
        
        # KnowledgeGate for anti-hallucination enforcement
        self._knowledge_gate: Optional[KnowledgeGate] = None
        if KNOWLEDGE_GATE_AVAILABLE:
            try:
                self._knowledge_gate = KnowledgeGate()
                logger.debug("KnowledgeGate initialized for ResearcherCouncil")
            except Exception as e:
                logger.warning(f"Failed to initialize KnowledgeGate: {e}")
                self._knowledge_gate = None
        
        self._load_default_system_prompt()
        self._init_websearch_tool()
    
    def _init_websearch_tool(self) -> None:
        """Initialize the web search tool if available."""
        if self.enable_websearch and WEBSEARCH_AVAILABLE:
            try:
                self._websearch_tool = create_websearch_tool(
                    max_results=self.max_search_results,
                    timeout=15
                )
                logger.debug("WebSearch tool initialized for ResearcherCouncil")
            except Exception as e:
                logger.warning(f"Failed to initialize WebSearch tool: {e}")
                self._websearch_tool = None
                self.enable_websearch = False
    
    def _load_default_system_prompt(self) -> None:
        """Load the default system prompt for researcher council."""
        self._system_prompt = """You are part of a research council of knowledge specialists.
Your role is to thoroughly research and synthesize information on the given topic.
Be comprehensive, accurate, and cite potential sources when possible.

You excel at:
- Finding relevant documentation and best practices
- Identifying implementation patterns and approaches
- Discovering potential pitfalls and edge cases
- Synthesizing complex information into actionable insights
- Providing balanced analysis of different approaches

## CLAIM ASSERTION RULES

Every non-trivial factual claim must be tagged as one of:
- [VERIFIED]: Backed by external source or high-confidence internal knowledge
- [INFERRED]: Logical inference from known facts
- [SPECULATIVE]: Hypothesis or uncertain claim

If a claim is [INFERRED] or [SPECULATIVE] and no external validation exists,
you MUST include explicit uncertainty language in your output.

Examples:
- ❌ "Democracy is declining globally"
- ✅ "[INFERRED] Available evidence suggests democracy may be declining globally, though this requires further verification"

Your research output should include:
- Key findings and insights
- Specific recommendations
- Suggested sources or references
- Potential concerns or caveats"""
    
    def build_prompt(
        self,
        research_context: ResearchContext
    ) -> str:
        """
        Build research prompt from context.
        
        Supports two modes:
        - FULL RESEARCH: When no prior_knowledge exists
        - INCREMENTAL RESEARCH: When prior_knowledge exists (iterative deepening)
        
        Args:
            research_context: Context containing objective and focus areas
            
        Returns:
            Prompt string for council members
        """
        # Determine if this is incremental mode
        is_incremental = bool(
            research_context.prior_knowledge and 
            (research_context.focus_areas or 
             research_context.unresolved_questions or 
             research_context.data_needs)
        )
        
        # Build focus areas
        focus_str = ""
        if research_context.focus_areas:
            focus_str = "\n\n### FOCUS AREAS (prioritize these):\n" + "\n".join(
                f"- {area}" for area in research_context.focus_areas
            )
        
        # Build unresolved questions
        questions_str = ""
        if research_context.unresolved_questions:
            questions_str = "\n\n### UNRESOLVED QUESTIONS (must address):\n" + "\n".join(
                f"- {q}" for q in research_context.unresolved_questions
            )
        
        # Build data needs
        data_needs_str = ""
        if research_context.data_needs:
            data_needs_str = "\n\n### DATA NEEDS (evidence required):\n" + "\n".join(
                f"- {need}" for need in research_context.data_needs
            )
        
        # Build subgoals
        subgoals_str = ""
        if research_context.subgoals:
            subgoals_str = "\n\n### SUBGOALS (from planner):\n" + "\n".join(
                f"- {sg}" for sg in research_context.subgoals
            )
        
        # Build prior knowledge context
        prior_str = ""
        if research_context.prior_knowledge:
            prior_str = f"\n\n## EXISTING KNOWLEDGE (foundation - do not repeat):\n{research_context.prior_knowledge}"
        
        # Build constraints
        constraints_str = ""
        if research_context.constraints:
            constraints_str = f"\n\nConstraints:\n{research_context.constraints}"
        
        # Build planner requirements
        planner_str = ""
        if research_context.planner_requirements:
            planner_str = f"\n\nPlanner Requirements:\n{research_context.planner_requirements}"
        
        if is_incremental:
            # INCREMENTAL RESEARCH MODE
            prompt = f"""## MODE: INCREMENTAL RESEARCH
You are deepening existing research, NOT starting fresh.

## OBJECTIVE
{research_context.objective}
{prior_str}
{focus_str}
{questions_str}
{data_needs_str}
{subgoals_str}
{constraints_str}
{planner_str}

## INSTRUCTIONS
Use the EXISTING KNOWLEDGE as your foundation.
Only deepen topics listed in FOCUS AREAS, UNRESOLVED QUESTIONS, and DATA NEEDS.
{"Perform web search for evidence if available." if research_context.requires_evidence else ""}
Do NOT restate the entire problem or repeat existing knowledge.

Provide ONLY:

### NEW INSIGHTS
What NEW information have you discovered beyond existing knowledge?

### RESOLVED QUESTIONS
Which questions from UNRESOLVED QUESTIONS can you now answer?

### DEEPER ANALYSIS
Provide deeper analysis on the FOCUS AREAS.

### GAPS REMAINING
What topics still need more investigation?
- [Gap 1]
- [Gap 2]

### UNRESOLVED QUESTIONS
What questions remain unclear or need more evidence?
- [Question 1]
- [Question 2]

### EVIDENCE NEEDED
What specific data or verification is still required?
- [Evidence need 1]
- [Evidence need 2]

### NEXT FOCUS AREAS
What should the next iteration investigate?
- [Focus area 1]
- [Focus area 2]

### CONFIDENCE
Rate your confidence in this research (0.0-1.0): [Score]

## CRITICAL: SOURCE CITATION REQUIREMENT (Phase 6.5)
All factual claims MUST cite a source URL. Claims without sources will be marked as uncertain.
When referencing web search results, include the URL: [Claim] (Source: [URL])

Be focused and incremental. Only add value beyond existing knowledge."""
        else:
            # FULL RESEARCH MODE
            prompt = f"""## MODE: COMPREHENSIVE RESEARCH
Conduct thorough research on the following objective:

## OBJECTIVE
{research_context.objective}
{focus_str}
{prior_str}
{constraints_str}
{planner_str}

## INSTRUCTIONS
Provide comprehensive research findings including:

### KEY FINDINGS
List the most important discoveries and insights.

### BEST PRACTICES
Identify established best practices relevant to this objective.

### IMPLEMENTATION APPROACHES
Describe potential approaches and patterns to consider.

### POTENTIAL ISSUES
Highlight potential pitfalls, edge cases, or concerns.

### RECOMMENDATIONS
Provide specific, actionable recommendations.

### SUGGESTED SOURCES
List documentation, libraries, or resources that would be helpful.

### GAPS REMAINING
What topics need deeper investigation?
- [Gap 1]
- [Gap 2]

### UNRESOLVED QUESTIONS
What questions need more evidence or clarification?
- [Question 1]
- [Question 2]

### EVIDENCE NEEDED
What data or facts require verification?
- [Evidence need 1]
- [Evidence need 2]

### NEXT FOCUS AREAS
If more research is needed, what should be prioritized?
- [Focus area 1]
- [Focus area 2]

### CONFIDENCE
Rate your confidence in this research (0.0-1.0): [Score]

## CRITICAL: SOURCE CITATION REQUIREMENT (Phase 6.5)
All factual claims MUST cite a source URL. Claims without sources will be marked as uncertain.
When referencing web search results, include the URL: [Claim] (Source: [URL])

Be thorough but focused. Prioritize quality and accuracy over quantity."""

        return prompt
    
    def postprocess(self, consensus_output: Any) -> ResearchFindings:
        """
        Postprocess consensus output into ResearchFindings.
        
        Args:
            consensus_output: Raw consensus output string
            
        Returns:
            Parsed ResearchFindings object
        """
        if not consensus_output:
            return ResearchFindings(
                summary="",
                key_points=[],
                recommendations=[],
                sources_suggested=[],
                raw_output=""
            )
        
        return ResearchFindings.from_text(str(consensus_output), iteration=self._current_iteration)
    
    # Track current iteration for postprocessing
    _current_iteration: int = 1
    
    def research(
        self,
        objective: str,
        focus_areas: Optional[List[str]] = None,
        prior_knowledge: Optional[str] = None,
        planner_requirements: Optional[str] = None,
        allow_internet: bool = True,
        data_needs: Optional[List[str]] = None,
        unresolved_questions: Optional[List[str]] = None,
        requires_evidence: bool = False,
        subgoals: Optional[List[str]] = None,
        iteration: int = 1
    ) -> CouncilResult:
        """
        Convenience method to conduct research.
        
        Args:
            objective: Research objective
            focus_areas: Specific areas to focus on
            prior_knowledge: Existing knowledge context
            planner_requirements: Requirements from planner
            allow_internet: Whether web search is allowed
            data_needs: Data or evidence needed (from evaluator)
            unresolved_questions: Questions to address (from evaluator)
            requires_evidence: Whether phase requires real-world evidence
            subgoals: Specific subgoals from planner for this iteration
            iteration: Current iteration number
            
        Returns:
            CouncilResult with ResearchFindings
        """
        # Track iteration for postprocessing
        self._current_iteration = iteration
        
        # Auto-enable evidence requirement if there are data needs or questions
        # This ensures web search is triggered when evaluator feedback exists
        effective_requires_evidence = requires_evidence or (
            allow_internet and (
                bool(data_needs) or 
                bool(unresolved_questions) or 
                iteration > 1  # Always require evidence on subsequent iterations
            )
        )
        
        research_context = ResearchContext(
            objective=objective,
            focus_areas=focus_areas or [],
            prior_knowledge=prior_knowledge,
            planner_requirements=planner_requirements,
            allow_internet=allow_internet,
            data_needs=data_needs or [],
            unresolved_questions=unresolved_questions or [],
            requires_evidence=effective_requires_evidence,
            subgoals=subgoals or []
        )
        
        # Log context for debugging
        if iteration > 1:
            logger.info(
                f"Iteration {iteration}: focus_areas={len(research_context.focus_areas)}, "
                f"data_needs={len(research_context.data_needs)}, "
                f"questions={len(research_context.unresolved_questions)}, "
                f"requires_evidence={effective_requires_evidence}"
            )
        
        # Execute the council to get draft output
        result = self.execute(research_context)
        
        # Get draft output for KnowledgeGate assessment
        draft_output = ""
        if result.success and result.output:
            if isinstance(result.output, ResearchFindings):
                draft_output = result.output.raw_output
            else:
                draft_output = str(result.output)
        
        # Assess epistemic sufficiency using KnowledgeGate
        gate_result = None
        if self._knowledge_gate and draft_output:
            gate_result = self._knowledge_gate.assess(
                objective=objective,
                draft_output=draft_output,
                data_needs=research_context.data_needs,
                unresolved_questions=research_context.unresolved_questions,
                allow_internet=allow_internet
            )
            logger.info(
                f"KnowledgeGate: risk_level={gate_result.risk_level}, "
                f"requires_validation={gate_result.requires_external_validation}, "
                f"missing_facts={len(gate_result.missing_facts)}"
            )
        else:
            # Fallback: use existing logic if KnowledgeGate unavailable
            if not self._knowledge_gate:
                logger.warning("KnowledgeGate unavailable - using fallback logic")
            gate_result = None
        
        # Perform web searches if KnowledgeGate requires it (MANDATORY)
        web_context = ""
        external_knowledge = None
        if gate_result and gate_result.requires_external_validation:
            if allow_internet and self.enable_websearch and self._websearch_tool:
                # Note: search_budget_manager would be passed from MissionOrchestrator if available
                # For now, we proceed without budget checking at this level
                web_context = self._perform_auto_searches(research_context, gate_result.missing_facts)
                
                # Create ExternalKnowledge artifact
                if EXTERNAL_KNOWLEDGE_AVAILABLE and ExternalKnowledge:
                    external_knowledge = ExternalKnowledge(
                        queries=self._last_queries.copy(),
                        sources=self._extract_sources_from_results()
                    )
                    # Phase 6.3: Pass structured sources with quality scores
                    structured_sources = self._get_structured_sources_from_search()
                    external_knowledge.calculate_evidence_strength(
                        result_count=self._last_search_count,
                        has_urls=bool(external_knowledge.sources),
                        sources=structured_sources
                    )
                    
                    if not web_context:
                        external_knowledge.search_failed = True
                        external_knowledge.evidence_strength = "weak"
                        external_knowledge.confidence_delta = -0.2
                        logger.warning("MANDATORY search required but failed - applying confidence penalty")
            else:
                # Search required but not available
                logger.warning(
                    f"MANDATORY search required but unavailable "
                    f"(allow_internet={allow_internet}, enable_websearch={self.enable_websearch})"
                )
                if EXTERNAL_KNOWLEDGE_AVAILABLE and ExternalKnowledge:
                    external_knowledge = ExternalKnowledge(
                        queries=[],
                        sources=[],
                        search_failed=True,
                        evidence_strength="weak",
                        confidence_delta=-0.2
                    )
        
        # If search was required but failed, apply confidence penalty
        if gate_result and gate_result.requires_external_validation and not web_context:
            if result.success and result.output and isinstance(result.output, ResearchFindings):
                penalty = gate_result.confidence_penalty if gate_result.confidence_penalty < 0 else -0.2
                result.output.confidence_score = max(0.0, result.output.confidence_score + penalty)
                result.output.raw_output += "\n\n[WARNING: External validation required but unavailable]"
                logger.warning(f"Confidence degraded by {penalty} due to missing external validation")
        
        # Inject web search results into prior knowledge for potential re-execution
        if web_context:
            existing_knowledge = research_context.prior_knowledge or ""
            research_context.prior_knowledge = f"{existing_knowledge}\n\n## WEB RESEARCH FINDINGS\n{web_context}"
        
        # Enhance findings with web search tracking and iteration info
        if result.success and result.output:
            findings = result.output
            if isinstance(findings, ResearchFindings):
                findings.web_search_count = self._last_search_count
                findings.queries_executed = self._last_queries.copy()
                findings.web_search_results = self._last_search_results.copy()
                findings.iteration = iteration
                findings.external_knowledge = external_knowledge
                
                # Apply external knowledge confidence delta if available
                if external_knowledge and external_knowledge.confidence_delta != 0.0:
                    findings.confidence_score = max(0.0, min(1.0, 
                        findings.confidence_score + external_knowledge.confidence_delta))
                
                # Log iteration progress
                logger.info(
                    f"Research iteration {iteration} complete: "
                    f"web_searches={findings.web_search_count}, "
                    f"gaps={len(findings.gaps)}, "
                    f"unresolved={len(findings.unresolved_questions)}, "
                    f"external_knowledge={external_knowledge is not None}"
                )
        
        return result
    
    def _perform_auto_searches(
        self, 
        context: ResearchContext, 
        missing_facts: Optional[List[str]] = None,
        search_budget_manager: Optional[Any] = None
    ) -> str:
        """
        Automatically perform web searches based on context.
        
        MANDATORY execution when called - this is for anti-hallucination enforcement.
        
        Generates queries from:
        - missing_facts (from KnowledgeGate - highest priority)
        - data_needs (from evaluator)
        - unresolved_questions (from evaluator)
        - subgoals (from planner)
        
        Args:
            context: Research context with data needs and questions
            missing_facts: Missing facts identified by KnowledgeGate
            search_budget_manager: Optional SearchTriggerManager for budget tracking
            
        Returns:
            Formatted string of web search results
        """
        queries = self._generate_search_queries(context, missing_facts)
        
        if not queries:
            logger.warning("MANDATORY search required but no queries generated")
            return ""
        
        # Check search budget if manager provided
        if search_budget_manager:
            stats = search_budget_manager.get_search_stats()
            budget_remaining = stats.get("budget_remaining", float('inf'))
            if budget_remaining <= 0:
                logger.warning(
                    f"Search budget exhausted ({stats.get('budget_used', 0)}/{stats.get('budget_max', 0)}) - "
                    "applying confidence penalty instead of searching"
                )
                return ""  # Return empty to trigger confidence degradation
        
        # Reset tracking
        self._last_search_count = 0
        self._last_queries = []
        self._last_search_results = []
        
        all_results = []
        
        # Limit queries based on remaining budget
        max_queries = 3
        if search_budget_manager:
            stats = search_budget_manager.get_search_stats()
            budget_remaining = stats.get("budget_remaining", float('inf'))
            max_queries = min(3, int(budget_remaining))
        
        query_results = []
        for query in queries[:max_queries]:
            try:
                logger.info(f"Performing MANDATORY web search: {query}")
                result = self._websearch_tool._run(query)
                
                self._last_search_count += 1
                self._last_queries.append(query)
                self._last_search_results.append({
                    "query": query,
                    "result": result[:2000] if result else ""
                })
                
                # Track results for logging
                results_count = len(result.split('\n')) if result else 0
                query_results.append({
                    "query": query,
                    "results_count": results_count
                })
                
                # Phase 6.1: Record search in SearchTriggerManager (single source of truth)
                if search_budget_manager:
                    search_budget_manager.record_search("research", query_count=1)
                # Also update _last_search_count for backward compatibility
                # But SearchTriggerManager is the authoritative counter
                
                if result and "No results found" not in result:
                    all_results.append(f"### Search: {query}\n{result}")
                else:
                    logger.warning(f"Web search returned no results for '{query}'")
                    
            except Exception as e:
                logger.error(f"MANDATORY web search failed for '{query}': {e}")
                # Continue with other queries even if one fails
        
        # Log internet usage panel
        try:
            from ...cli import verbose_logger
            if verbose_logger and verbose_logger.enabled:
                search_stats = search_budget_manager.get_search_stats() if search_budget_manager else {}
                verbose_logger.log_internet_usage_panel(
                    enabled=True,
                    searches_executed=self._last_search_count,
                    quota=search_stats.get("budget_max", 10),
                    queries=query_results,
                    search_rationale="MANDATORY search for anti-hallucination enforcement",
                    results_summary={str(i+1): qr["results_count"] for i, qr in enumerate(query_results)}
                )
        except Exception:
            pass  # Don't fail if logging fails
        
        if not all_results:
            logger.warning("MANDATORY search executed but no results obtained")
        
        return "\n\n".join(all_results) if all_results else ""
    
    def _generate_search_queries(
        self, 
        context: ResearchContext, 
        missing_facts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate epistemic search queries from context.
        
        Priority order (epistemic queries, not generic):
        1. missing_facts from KnowledgeGate (highest priority - explicit gaps)
        2. data_needs from evaluator (explicit evidence requests)
        3. unresolved_questions from evaluator (need answers)
        4. focus_areas (targeted investigation)
        5. subgoals from planner
        
        NO FALLBACK to objective truncation - must be fact-oriented.
        
        Args:
            context: Research context
            missing_facts: Missing facts identified by KnowledgeGate
            
        Returns:
            List of atomic, fact-oriented search query strings
        """
        queries = []
        missing_facts = missing_facts or []
        
        # Priority 1: Generate multiple queries per missing fact (2-3 per fact)
        for fact in missing_facts[:3]:  # Top 3 missing facts
            # Generate 2-3 atomic queries per fact
            atomic_queries = self._fact_to_epistemic_queries(fact)
            queries.extend(atomic_queries[:2])  # Max 2 queries per fact
        
        # Priority 2: Add data needs as queries (explicit evidence requests)
        for need in context.data_needs[:2]:
            if len(need) > 10:
                # Convert to epistemic query if needed
                query = self._make_epistemic_query(need)
                if query:
                    queries.append(query)
        
        # Priority 3: Add questions as queries (remove question marks, make fact-oriented)
        for question in context.unresolved_questions[:2]:
            query = question.rstrip('?').strip()
            if len(query) > 10:
                query = self._make_epistemic_query(query)
                if query:
                    queries.append(query)
        
        # Priority 4: Add focus areas as queries (if fact-oriented)
        for area in context.focus_areas[:2]:
            if len(area) > 10:
                query = self._make_epistemic_query(area)
                if query:
                    queries.append(query)
        
        # Priority 5: Add subgoals as queries
        for subgoal in context.subgoals[:1]:
            if len(subgoal) > 10:
                query = self._make_epistemic_query(subgoal)
                if query:
                    queries.append(query)
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen and len(q) > 10:
                seen.add(q_lower)
                unique.append(q)
        
        return unique[:5]  # Max 5 queries total
    
    def _fact_to_epistemic_queries(self, fact: str) -> List[str]:
        """
        Convert a missing fact into 2-3 atomic, fact-oriented search queries.
        
        Args:
            fact: Missing fact statement
            
        Returns:
            List of epistemic queries
        """
        queries = []
        
        # Extract key terms
        words = fact.split()
        key_terms = [w for w in words if len(w) > 4 and w.lower() not in 
                    ['this', 'that', 'these', 'those', 'which', 'where', 'when', 'what', 'how']]
        
        if not key_terms:
            # Fallback: use fact as-is if no key terms
            if len(fact) > 10:
                queries.append(fact[:100])
            return queries
        
        # Strategy 1: "empirical studies [key terms]"
        if len(key_terms) >= 2:
            query1 = f"empirical studies {' '.join(key_terms[:4])}"
            queries.append(query1)
        
        # Strategy 2: "[key terms] research evidence"
        if len(key_terms) >= 2:
            query2 = f"{' '.join(key_terms[:3])} research evidence"
            queries.append(query2)
        
        # Strategy 3: "[key terms] data statistics"
        if len(key_terms) >= 2:
            query3 = f"{' '.join(key_terms[:3])} data statistics"
            queries.append(query3)
        
        return queries[:3]  # Max 3 queries per fact
    
    def _make_epistemic_query(self, text: str) -> str:
        """
        Convert generic text into an epistemic (fact-oriented) query.
        
        Args:
            text: Generic text to convert
            
        Returns:
            Epistemic query string, or empty string if conversion fails
        """
        # Remove question words and make it fact-oriented
        text = text.rstrip('?').strip()
        
        # If already looks like a query, return as-is
        if any(keyword in text.lower() for keyword in 
               ['study', 'research', 'data', 'evidence', 'statistics', 'analysis']):
            return text[:100]
        
        # Add epistemic keywords
        words = text.split()
        if len(words) > 3:
            # Insert "research" or "evidence" early
            query = f"{' '.join(words[:3])} research {' '.join(words[3:5])}"
            return query[:100]
        else:
            # Too short, add epistemic prefix
            return f"research {text}"[:100]
    
    def _get_structured_sources_from_search(self) -> List[Dict[str, Any]]:
        """
        Get structured sources with quality scores from web search tool.
        
        Phase 6.3: Returns sources with quality scores for evidence strength calculation.
        
        Returns:
            List of source dicts with url, quality_score, quality_tier
        """
        if not self._websearch_tool or not hasattr(self._websearch_tool, '_last_search_results'):
            return []
        
        # Get structured results from WebSearchTool
        return getattr(self._websearch_tool, '_last_search_results', [])
    
    def _extract_sources_from_results(self) -> List[str]:
        """
        Extract source URLs from web search results.
        
        Returns:
            List of source URLs (for backward compatibility)
        """
        sources = []
        structured = self._get_structured_sources_from_search()
        for source in structured:
            if isinstance(source, dict) and 'url' in source:
                sources.append(source['url'])
        
        # Fallback to old method if structured sources unavailable
        if not sources:
            for result in self._last_search_results:
                result_text = result.get("result", "")
                # Extract URLs from result text
            import re
            urls = re.findall(r'https?://[^\s\)]+', result_text)
            sources.extend(urls[:3])  # Max 3 URLs per result
        
        # Deduplicate
        return list(dict.fromkeys(sources))[:10]  # Max 10 total URLs
    
    def get_last_search_count(self) -> int:
        """Get the number of web searches performed in last research call."""
        return self._last_search_count
    
    def web_searches_performed(self) -> bool:
        """Check if any web searches were performed in last research call."""
        return self._last_search_count > 0

