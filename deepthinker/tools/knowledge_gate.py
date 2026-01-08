"""
KnowledgeGate - Epistemic Sufficiency Assessment for Anti-Hallucination.

Evaluates whether research output has sufficient grounding to prevent
hallucinated claims. Determines when external validation is mandatory.
"""

import logging
import re
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGateResult:
    """Result from knowledge gate assessment."""
    
    knowledge_sufficient: bool
    missing_facts: List[str]
    risk_level: str  # "low" | "medium" | "high"
    requires_external_validation: bool
    confidence_penalty: float  # 0.0 to -0.5


class KnowledgeGate:
    """
    Gate that evaluates epistemic sufficiency before research output is finalized.
    
    Prevents hallucination by detecting knowledge gaps and requiring external
    validation when claims cannot be grounded internally.
    """
    
    # Keywords that indicate factual claims requiring verification
    FACTUAL_CLAIM_KEYWORDS = [
        "study shows", "research indicates", "data suggests", "evidence shows",
        "according to", "statistics", "percentage", "rate", "trend",
        "increased", "decreased", "declined", "grew", "analysis reveals",
        "findings show", "report states", "survey found"
    ]
    
    def assess(
        self,
        objective: str,
        draft_output: str,
        data_needs: List[str],
        unresolved_questions: List[str],
        allow_internet: bool
    ) -> KnowledgeGateResult:
        """
        Assess epistemic sufficiency of research output.
        
        Args:
            objective: Mission objective
            draft_output: Draft research output to assess
            data_needs: Data needs identified by evaluator
            unresolved_questions: Unresolved questions from evaluator
            allow_internet: Whether internet access is available
            
        Returns:
            KnowledgeGateResult with assessment and requirements
        """
        missing_facts = []
        risk_level = "low"
        
        # Check 1: Data needs or unresolved questions indicate gaps
        if data_needs:
            missing_facts.extend(data_needs[:5])  # Limit to top 5
            risk_level = "medium" if risk_level == "low" else "high"
            logger.debug(f"KnowledgeGate: {len(data_needs)} data needs detected")
        
        if unresolved_questions:
            missing_facts.extend(unresolved_questions[:5])
            risk_level = "medium" if risk_level == "low" else "high"
            logger.debug(f"KnowledgeGate: {len(unresolved_questions)} unresolved questions detected")
        
        # Check 2: Factual claims in output without sources
        factual_claims = self._detect_factual_claims(draft_output)
        if factual_claims and not self._has_sources(draft_output):
            missing_facts.extend(factual_claims[:3])
            risk_level = "high" if risk_level == "medium" else "high"
            logger.debug(f"KnowledgeGate: {len(factual_claims)} unverified factual claims detected")
        
        # Check 3: Objective contains factual requirements
        if self._objective_requires_facts(objective):
            if not self._has_sources(draft_output):
                missing_facts.append("Objective requires factual verification")
                risk_level = "medium" if risk_level == "low" else "high"
        
        # Determine if external validation is required
        requires_external_validation = False
        confidence_penalty = 0.0
        
        if risk_level in ["medium", "high"]:
            if allow_internet:
                requires_external_validation = True
                logger.info(f"KnowledgeGate: External validation required (risk_level={risk_level})")
            else:
                # Internet not available - must degrade confidence
                confidence_penalty = -0.3 if risk_level == "medium" else -0.5
                logger.warning(
                    f"KnowledgeGate: High risk ({risk_level}) but internet disabled. "
                    f"Confidence penalty: {confidence_penalty}"
                )
        
        knowledge_sufficient = risk_level == "low" and not missing_facts
        
        result = KnowledgeGateResult(
            knowledge_sufficient=knowledge_sufficient,
            missing_facts=missing_facts[:10],  # Limit total
            risk_level=risk_level,
            requires_external_validation=requires_external_validation,
            confidence_penalty=confidence_penalty
        )
        
        logger.info(
            f"KnowledgeGate assessment: risk_level={risk_level}, "
            f"requires_validation={requires_external_validation}, "
            f"missing_facts={len(missing_facts)}"
        )
        
        return result
    
    def _detect_factual_claims(self, text: str) -> List[str]:
        """Detect factual claims in text that may need verification."""
        claims = []
        text_lower = text.lower()
        
        for keyword in self.FACTUAL_CLAIM_KEYWORDS:
            if keyword in text_lower:
                # Extract sentence containing the keyword
                sentences = re.split(r'[.!?]\s+', text)
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence) > 20:
                        # Check if sentence doesn't already have uncertainty markers
                        if not any(marker in sentence.lower() for marker in 
                                 ["may", "might", "suggests", "indicates", "possibly", "uncertain"]):
                            claims.append(sentence.strip()[:200])
                            break
        
        return claims[:5]  # Limit to 5 claims
    
    def _has_sources(self, text: str) -> bool:
        """Check if text contains source citations or URLs."""
        # Check for URLs
        url_pattern = r'https?://[^\s]+'
        if re.search(url_pattern, text):
            return True
        
        # Check for citation patterns
        citation_patterns = [
            r'\([A-Z][a-z]+\s+et\s+al\.',  # (Author et al.
            r'\[.*?\]',  # [citation]
            r'\(.*?\d{4}.*?\)',  # (Author 2024)
            r'source:',  # source: ...
            r'reference:',  # reference: ...
            r'according to [A-Z]',  # According to Author
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _objective_requires_facts(self, objective: str) -> bool:
        """Check if objective requires factual verification."""
        factual_keywords = [
            "analyze", "identify", "assess", "evaluate", "compare",
            "statistics", "data", "evidence", "research", "study",
            "trends", "impact", "effects", "causes"
        ]
        
        obj_lower = objective.lower()
        return any(keyword in obj_lower for keyword in factual_keywords)

