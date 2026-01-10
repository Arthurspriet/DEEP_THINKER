"""
Knowledge Router for DeepThinker Memory System.

Routes knowledge items to personas and councils based on domain mappings.
Enables specialized knowledge delivery to different analytical perspectives.

Domain Categories (aligned with CIA Factbook + OWID):
- introduction: Background and overview
- geography: Location, terrain, climate
- people: Demographics, population, society
- environment: Environmental issues, resources
- government: Political structure, leadership
- economy: Economic indicators, trade
- energy: Energy production and consumption
- communications: Telecommunications, internet
- transportation: Infrastructure, ports, roads
- military: Defense, security forces
- terrorism: Terrorist threats
- transnational: Cross-border issues
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# CIA Factbook categories for filtering
KNOWLEDGE_CATEGORIES = {
    "introduction",
    "geography",
    "people",
    "environment",
    "government",
    "economy",
    "energy",
    "communications",
    "transportation",
    "military",
    "terrorism",
    "transnational",
    "other",
}

# Persona-to-domain mappings
# Each persona receives knowledge from specific domains
PERSONA_DOMAIN_MAPPING: Dict[str, List[str]] = {
    # Evidence Hunter gets all knowledge - comprehensive research role
    "evidence_hunter": ["all"],
    
    # Skeptic focuses on risks, security, and threats
    "skeptic": ["military", "terrorism", "transnational", "government"],
    
    # Strategist focuses on planning, economics, and governance
    "strategist": ["economy", "government", "people", "introduction"],
    
    # Systems Thinker focuses on infrastructure and interconnections
    "systems_thinker": ["environment", "energy", "transportation", "communications"],
    
    # Code Architect is primarily code-focused, minimal domain knowledge
    "code_architect": [],
    
    # Security Auditor focuses on security-related domains
    "security_auditor": ["military", "terrorism", "communications", "government"],
    
    # Economist focuses on economic data
    "economist": ["economy", "energy", "transportation", "people"],
    
    # Optimist gets broad overview to find opportunities
    "optimist": ["economy", "introduction", "people", "environment"],
    
    # Stress Tester looks for failure modes
    "stress_tester": ["military", "terrorism", "transnational", "environment"],
    
    # Reductionist focuses on core facts
    "reductionist": ["introduction", "geography", "people"],
}

# Council-to-domain mappings
# Each council type receives knowledge from relevant domains
COUNCIL_DOMAIN_MAPPING: Dict[str, List[str]] = {
    # Researcher council needs broad knowledge
    "researcher": ["all"],
    
    # Planner council needs strategic context
    "planner": ["economy", "government", "introduction", "people"],
    
    # Evaluator council needs verification context
    "evaluator": ["all"],
    
    # Coder council is code-focused
    "coder": [],
    
    # Simulation council needs risk context
    "simulation": ["military", "terrorism", "environment", "transnational"],
    
    # Evidence council needs all for verification
    "evidence": ["all"],
    
    # Explorer council needs broad discovery
    "explorer": ["all"],
}


@dataclass
class KnowledgeItem:
    """
    A single knowledge item with metadata.
    
    Attributes:
        text: The knowledge content
        category: Domain category (economy, military, etc.)
        source: Source identifier (cia_world_factbook, owid, mission, etc.)
        country: Country if applicable
        score: Relevance score from retrieval
        metadata: Additional metadata
    """
    text: str
    category: str = "other"
    source: str = "unknown"
    country: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, doc: Dict[str, Any], score: float = 0.0) -> "KnowledgeItem":
        """Create from RAG document dict."""
        return cls(
            text=doc.get("text", str(doc)),
            category=doc.get("category", "other"),
            source=doc.get("source", "unknown"),
            country=doc.get("country"),
            score=score,
            metadata={k: v for k, v in doc.items() 
                     if k not in ("text", "category", "source", "country")}
        )


@dataclass
class RoutedKnowledge:
    """
    Knowledge routed for a specific persona or council.
    
    Attributes:
        items: Filtered knowledge items
        domains_included: Which domains were included
        total_items: Original count before filtering
        persona_name: Target persona (if any)
        council_type: Target council (if any)
    """
    items: List[KnowledgeItem]
    domains_included: Set[str]
    total_items: int
    persona_name: Optional[str] = None
    council_type: Optional[str] = None
    
    def format_for_prompt(self, max_chars: int = 3000) -> str:
        """Format routed knowledge for prompt injection."""
        if not self.items:
            return ""
        
        parts = []
        char_count = 0
        
        for item in self.items:
            if char_count > max_chars:
                break
            
            # Format source attribution
            source_attr = item.source
            if item.country:
                source_attr = f"{item.source}: {item.country}"
            if item.category and item.category != "other":
                source_attr = f"{source_attr}/{item.category}"
            
            snippet = item.text[:500] if len(item.text) > 500 else item.text
            parts.append(f"[{source_attr}] {snippet}")
            char_count += len(parts[-1])
        
        return "\n\n".join(parts)


class KnowledgeRouter:
    """
    Routes knowledge to personas and councils based on domain relevance.
    
    The router filters knowledge items based on predefined domain mappings,
    ensuring each analytical perspective receives relevant context.
    
    Usage:
        router = KnowledgeRouter()
        
        # Route for a specific persona
        routed = router.route_for_persona("strategist", knowledge_items)
        
        # Route for a council type
        routed = router.route_for_council("planner", knowledge_items)
        
        # Get formatted prompt injection
        prompt_text = routed.format_for_prompt(max_chars=2000)
    """
    
    def __init__(
        self,
        persona_mapping: Optional[Dict[str, List[str]]] = None,
        council_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the knowledge router.
        
        Args:
            persona_mapping: Custom persona-to-domain mapping (optional)
            council_mapping: Custom council-to-domain mapping (optional)
        """
        self._persona_mapping = persona_mapping or PERSONA_DOMAIN_MAPPING
        self._council_mapping = council_mapping or COUNCIL_DOMAIN_MAPPING
    
    def get_domains_for_persona(self, persona_name: str) -> List[str]:
        """
        Get knowledge domains relevant to a persona.
        
        Args:
            persona_name: Name of the persona
            
        Returns:
            List of domain names, or ["all"] for unrestricted
        """
        return self._persona_mapping.get(persona_name, ["all"])
    
    def get_domains_for_council(self, council_type: str) -> List[str]:
        """
        Get knowledge domains relevant to a council type.
        
        Args:
            council_type: Type of council (researcher, planner, etc.)
            
        Returns:
            List of domain names, or ["all"] for unrestricted
        """
        # Normalize council type name
        council_type = council_type.lower().replace("_council", "").replace("council", "")
        return self._council_mapping.get(council_type, ["all"])
    
    def _convert_to_knowledge_items(
        self,
        items: List[Tuple[Any, float]]
    ) -> List[KnowledgeItem]:
        """Convert raw retrieval results to KnowledgeItem list."""
        result = []
        for item, score in items:
            if isinstance(item, dict):
                result.append(KnowledgeItem.from_dict(item, score))
            elif isinstance(item, KnowledgeItem):
                result.append(item)
            else:
                # Handle other types (e.g., MissionSummarySchema)
                text = ""
                if hasattr(item, 'objective'):
                    text = item.objective
                elif hasattr(item, 'key_insights'):
                    text = "; ".join(item.key_insights or [])
                elif hasattr(item, 'text'):
                    text = item.text
                else:
                    text = str(item)
                
                result.append(KnowledgeItem(
                    text=text,
                    category="other",
                    source="mission" if hasattr(item, 'mission_id') else "unknown",
                    score=score,
                ))
        
        return result
    
    def _filter_by_domains(
        self,
        items: List[KnowledgeItem],
        domains: List[str]
    ) -> Tuple[List[KnowledgeItem], Set[str]]:
        """Filter knowledge items by domain."""
        if "all" in domains:
            categories = {item.category for item in items}
            return items, categories
        
        if not domains:
            return [], set()
        
        filtered = []
        included_domains = set()
        
        for item in items:
            if item.category in domains or item.category == "other":
                filtered.append(item)
                included_domains.add(item.category)
        
        return filtered, included_domains
    
    def route_for_persona(
        self,
        persona_name: str,
        knowledge_items: List[Tuple[Any, float]],
        max_items: int = 10,
    ) -> RoutedKnowledge:
        """
        Route knowledge for a specific persona.
        
        Filters knowledge items based on the persona's domain mapping,
        returning only relevant context.
        
        Args:
            persona_name: Name of the target persona
            knowledge_items: List of (item, score) tuples from retrieval
            max_items: Maximum items to include
            
        Returns:
            RoutedKnowledge with filtered items
        """
        # Convert to KnowledgeItem list
        items = self._convert_to_knowledge_items(knowledge_items)
        total = len(items)
        
        # Get domains for this persona
        domains = self.get_domains_for_persona(persona_name)
        
        # Filter by domains
        filtered, included = self._filter_by_domains(items, domains)
        
        # Sort by score and limit
        filtered.sort(key=lambda x: x.score, reverse=True)
        filtered = filtered[:max_items]
        
        logger.debug(
            f"Routed {len(filtered)}/{total} items for persona '{persona_name}' "
            f"(domains: {domains})"
        )
        
        return RoutedKnowledge(
            items=filtered,
            domains_included=included,
            total_items=total,
            persona_name=persona_name,
        )
    
    def route_for_council(
        self,
        council_type: str,
        knowledge_items: List[Tuple[Any, float]],
        max_items: int = 15,
    ) -> RoutedKnowledge:
        """
        Route knowledge for a council type.
        
        Filters knowledge items based on the council's domain mapping,
        returning relevant context for the council's purpose.
        
        Args:
            council_type: Type of council (researcher, planner, etc.)
            knowledge_items: List of (item, score) tuples from retrieval
            max_items: Maximum items to include
            
        Returns:
            RoutedKnowledge with filtered items
        """
        # Convert to KnowledgeItem list
        items = self._convert_to_knowledge_items(knowledge_items)
        total = len(items)
        
        # Get domains for this council
        domains = self.get_domains_for_council(council_type)
        
        # Filter by domains
        filtered, included = self._filter_by_domains(items, domains)
        
        # Sort by score and limit
        filtered.sort(key=lambda x: x.score, reverse=True)
        filtered = filtered[:max_items]
        
        logger.debug(
            f"Routed {len(filtered)}/{total} items for council '{council_type}' "
            f"(domains: {domains})"
        )
        
        return RoutedKnowledge(
            items=filtered,
            domains_included=included,
            total_items=total,
            council_type=council_type,
        )
    
    def route_for_model(
        self,
        model_name: str,
        persona_name: Optional[str],
        knowledge_items: List[Tuple[Any, float]],
        max_items: int = 8,
    ) -> RoutedKnowledge:
        """
        Route knowledge for a specific model with optional persona.
        
        If a persona is assigned, filters by persona domains.
        Otherwise returns all items (up to max).
        
        Args:
            model_name: Name of the model
            persona_name: Optional persona assigned to this model
            knowledge_items: List of (item, score) tuples
            max_items: Maximum items to include
            
        Returns:
            RoutedKnowledge with filtered items
        """
        if persona_name:
            return self.route_for_persona(persona_name, knowledge_items, max_items)
        
        # No persona - return all items
        items = self._convert_to_knowledge_items(knowledge_items)
        items.sort(key=lambda x: x.score, reverse=True)
        items = items[:max_items]
        
        return RoutedKnowledge(
            items=items,
            domains_included={item.category for item in items},
            total_items=len(knowledge_items),
        )
    
    def enrich_context_with_knowledge(
        self,
        context: Any,
        knowledge_items: List[Tuple[Any, float]],
        council_type: Optional[str] = None,
        max_chars: int = 3000,
    ) -> Any:
        """
        Enrich a context object with relevant knowledge.
        
        Looks for 'prior_knowledge' or 'knowledge_context' fields in the
        context and populates them with routed knowledge.
        
        Args:
            context: Context object (ResearchContext, PlannerContext, etc.)
            knowledge_items: List of (item, score) tuples
            council_type: Optional council type for domain filtering
            max_chars: Maximum characters for knowledge text
            
        Returns:
            Context with knowledge populated
        """
        if not knowledge_items:
            return context
        
        # Route knowledge
        if council_type:
            routed = self.route_for_council(council_type, knowledge_items)
        else:
            items = self._convert_to_knowledge_items(knowledge_items)
            routed = RoutedKnowledge(
                items=items[:15],
                domains_included={item.category for item in items},
                total_items=len(items),
            )
        
        # Format for prompt
        knowledge_text = routed.format_for_prompt(max_chars=max_chars)
        
        if not knowledge_text:
            return context
        
        # Try to set knowledge on context
        if hasattr(context, 'knowledge_context'):
            existing = getattr(context, 'knowledge_context') or ""
            if existing:
                context.knowledge_context = f"{existing}\n\n{knowledge_text}"
            else:
                context.knowledge_context = knowledge_text
        elif hasattr(context, 'prior_knowledge'):
            existing = getattr(context, 'prior_knowledge') or ""
            if existing:
                context.prior_knowledge = f"{existing}\n\n## Retrieved Knowledge\n{knowledge_text}"
            else:
                context.prior_knowledge = f"## Retrieved Knowledge\n{knowledge_text}"
        elif hasattr(context, 'context'):
            existing = getattr(context, 'context') or ""
            if existing:
                context.context = f"{existing}\n\n## Retrieved Knowledge\n{knowledge_text}"
            else:
                context.context = f"## Retrieved Knowledge\n{knowledge_text}"
        
        logger.debug(
            f"Enriched context with {len(routed.items)} knowledge items "
            f"({len(knowledge_text)} chars)"
        )
        
        return context


# Module-level singleton
_router: Optional[KnowledgeRouter] = None


def get_knowledge_router() -> KnowledgeRouter:
    """Get the singleton knowledge router instance."""
    global _router
    if _router is None:
        _router = KnowledgeRouter()
    return _router


def route_knowledge_for_persona(
    persona_name: str,
    knowledge_items: List[Tuple[Any, float]],
    max_items: int = 10,
) -> RoutedKnowledge:
    """Route knowledge for a persona (convenience function)."""
    return get_knowledge_router().route_for_persona(
        persona_name, knowledge_items, max_items
    )


def route_knowledge_for_council(
    council_type: str,
    knowledge_items: List[Tuple[Any, float]],
    max_items: int = 15,
) -> RoutedKnowledge:
    """Route knowledge for a council type (convenience function)."""
    return get_knowledge_router().route_for_council(
        council_type, knowledge_items, max_items
    )


def enrich_context(
    context: Any,
    knowledge_items: List[Tuple[Any, float]],
    council_type: Optional[str] = None,
) -> Any:
    """Enrich context with knowledge (convenience function)."""
    return get_knowledge_router().enrich_context_with_knowledge(
        context, knowledge_items, council_type
    )


