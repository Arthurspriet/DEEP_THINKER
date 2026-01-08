"""
Claim Validator for DeepThinker Epistemic Hardening.

Enforces evidence-grounded reasoning by:
- Parsing declarative claims from LLM outputs
- Validating claims against source requirements
- Computing epistemic risk scores
- Blocking phase advancement on validation failures

Claim Types:
- FACT: Must have at least one valid source
- INFERENCE: Must reference at least one upstream fact
- SPECULATION: Must be explicitly tagged as uncertain
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ClaimType(str, Enum):
    """Types of epistemic claims."""
    FACT = "fact"           # Verifiable factual assertion
    INFERENCE = "inference"  # Logical derivation from facts
    SPECULATION = "speculation"  # Uncertain/hypothetical assertion


class ClaimStatus(str, Enum):
    """
    Lifecycle status of a claim (Epistemic Hardening Phase 5).
    
    Claims progress through:
    PROPOSED -> GROUNDED (if evidence found)
             -> CONTESTED (if disputed by council)
             -> REJECTED (if no evidence or contradicted)
    """
    PROPOSED = "proposed"      # Newly extracted, awaiting validation
    GROUNDED = "grounded"      # Validated with evidence
    CONTESTED = "contested"    # Under dispute
    REJECTED = "rejected"      # Failed validation


@dataclass
class Source:
    """
    Represents an evidence source for a claim.
    
    Attributes:
        id: Unique identifier for the source
        url: Optional URL of the source
        title: Title or description
        quality_score: Source quality (0-1)
        quality_tier: Quality tier (HIGH, MEDIUM, LOW, VERY_LOW)
        domain: Domain of the source (e.g., "arxiv.org")
    """
    id: str
    url: Optional[str] = None
    title: str = ""
    quality_score: float = 0.5
    quality_tier: str = "MEDIUM"
    domain: str = ""
    
    def is_high_quality(self) -> bool:
        """Check if source is high quality."""
        return self.quality_tier == "HIGH" or self.quality_score >= 0.7


@dataclass
class Claim:
    """
    A structured epistemic claim extracted from LLM output.
    
    Enhanced for Epistemic Hardening Phase 5 with:
    - Lifecycle status tracking (proposed -> grounded -> contested -> rejected)
    - Focus area association
    - Source URL for direct linking
    - Contest reason tracking
    
    Attributes:
        text: The claim text
        claim_type: Type of claim (fact, inference, speculation)
        source_ids: List of source IDs backing this claim
        source_url: Primary source URL for this claim (Phase 5)
        confidence: Model's confidence in the claim (0-1)
        focus_area: Associated focus area (Phase 5)
        status: Lifecycle status (Phase 5)
        contest_reason: Why the claim is contested (if applicable)
        upstream_claim_ids: For inferences, IDs of supporting claims
        is_tagged: Whether claim was explicitly tagged in output
        line_number: Approximate line number in source output
    """
    text: str
    claim_type: ClaimType
    source_ids: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    confidence: float = 0.5
    focus_area: Optional[str] = None
    status: ClaimStatus = ClaimStatus.PROPOSED
    contest_reason: Optional[str] = None
    upstream_claim_ids: List[str] = field(default_factory=list)
    is_tagged: bool = False
    line_number: int = 0
    
    @property
    def id(self) -> str:
        """Generate a claim ID based on content hash."""
        return f"claim_{hash(self.text) % 100000:05d}"
    
    def is_grounded(self) -> bool:
        """Check if claim meets grounding requirements."""
        # Phase 5: Check status first
        if self.status == ClaimStatus.GROUNDED:
            return True
        if self.status == ClaimStatus.REJECTED:
            return False
        
        # Legacy check based on structure
        if self.claim_type == ClaimType.FACT:
            return len(self.source_ids) >= 1 or self.source_url is not None
        elif self.claim_type == ClaimType.INFERENCE:
            return len(self.upstream_claim_ids) >= 1
        elif self.claim_type == ClaimType.SPECULATION:
            return self.is_tagged  # Must be explicitly tagged
        return False
    
    def promote_to_grounded(self) -> None:
        """Promote claim to grounded status."""
        if self.status != ClaimStatus.REJECTED:
            self.status = ClaimStatus.GROUNDED
            logger.debug(f"[CLAIM] Promoted to grounded: {self.id}")
    
    def contest(self, reason: str) -> None:
        """Mark claim as contested with reason."""
        self.status = ClaimStatus.CONTESTED
        self.contest_reason = reason
        logger.debug(f"[CLAIM] Contested: {self.id} - {reason}")
    
    def reject(self, reason: Optional[str] = None) -> None:
        """Reject the claim."""
        self.status = ClaimStatus.REJECTED
        if reason:
            self.contest_reason = reason
        logger.debug(f"[CLAIM] Rejected: {self.id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "source_ids": self.source_ids,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "focus_area": self.focus_area,
            "status": self.status.value,
            "contest_reason": self.contest_reason,
            "upstream_claim_ids": self.upstream_claim_ids,
            "is_tagged": self.is_tagged,
            "line_number": self.line_number,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create from dictionary."""
        return cls(
            text=data.get("text", ""),
            claim_type=ClaimType(data.get("claim_type", "fact")),
            source_ids=data.get("source_ids", []),
            source_url=data.get("source_url"),
            confidence=data.get("confidence", 0.5),
            focus_area=data.get("focus_area"),
            status=ClaimStatus(data.get("status", "proposed")),
            contest_reason=data.get("contest_reason"),
            upstream_claim_ids=data.get("upstream_claim_ids", []),
            is_tagged=data.get("is_tagged", False),
            line_number=data.get("line_number", 0),
        )


@dataclass
class ClaimValidationResult:
    """
    Result of claim validation.
    
    Attributes:
        is_valid: Whether all claims passed validation
        total_claims: Total number of claims found
        grounded_claims: Number of properly grounded claims
        violations: List of validation violations
        grounded_ratio: Ratio of grounded claims (0-1)
        ungrounded_facts: Facts without sources
        untagged_speculation: Speculations not tagged
        orphan_inferences: Inferences without upstream facts
    """
    is_valid: bool
    total_claims: int
    grounded_claims: int
    violations: List[str] = field(default_factory=list)
    grounded_ratio: float = 0.0
    ungrounded_facts: List[Claim] = field(default_factory=list)
    untagged_speculation: List[Claim] = field(default_factory=list)
    orphan_inferences: List[Claim] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "is_valid": self.is_valid,
            "total_claims": self.total_claims,
            "grounded_claims": self.grounded_claims,
            "grounded_ratio": self.grounded_ratio,
            "violations_count": len(self.violations),
            "ungrounded_facts_count": len(self.ungrounded_facts),
            "untagged_speculation_count": len(self.untagged_speculation),
            "orphan_inferences_count": len(self.orphan_inferences),
        }


@dataclass
class EpistemicRiskScore:
    """
    Quantifies epistemic risk in an output.
    
    Higher scores indicate higher risk of hallucination/ungrounded content.
    
    Attributes:
        claim_to_source_ratio: Claims per source (higher = more risk)
        repetition_penalty: Penalty for repeated assertions (0-1)
        confidence_vs_evidence_delta: Gap between confidence and evidence (0-1)
        speculative_density: Fraction of speculative content (0-1)
        overall_risk: Combined risk score (0-1, higher = worse)
        ungrounded_claim_count: Number of ungrounded claims
        source_quality_avg: Average quality of sources used
    """
    claim_to_source_ratio: float = 0.0
    repetition_penalty: float = 0.0
    confidence_vs_evidence_delta: float = 0.0
    speculative_density: float = 0.0
    overall_risk: float = 0.0
    ungrounded_claim_count: int = 0
    source_quality_avg: float = 0.5
    
    def compute_overall_risk(self) -> float:
        """Compute overall risk score from components."""
        # Weighted combination of risk factors
        weights = {
            "claim_source": 0.30,
            "repetition": 0.15,
            "confidence_gap": 0.25,
            "speculation": 0.30,
        }
        
        # Normalize claim_to_source_ratio (optimal is 1-2 claims per source)
        source_risk = min(1.0, max(0.0, (self.claim_to_source_ratio - 2) / 8))
        
        self.overall_risk = (
            weights["claim_source"] * source_risk +
            weights["repetition"] * self.repetition_penalty +
            weights["confidence_gap"] * self.confidence_vs_evidence_delta +
            weights["speculation"] * self.speculative_density
        )
        
        # Penalty for low source quality
        if self.source_quality_avg < 0.5:
            self.overall_risk = min(1.0, self.overall_risk + 0.1)
        
        return self.overall_risk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "claim_to_source_ratio": self.claim_to_source_ratio,
            "repetition_penalty": self.repetition_penalty,
            "confidence_vs_evidence_delta": self.confidence_vs_evidence_delta,
            "speculative_density": self.speculative_density,
            "overall_risk": self.overall_risk,
            "ungrounded_claim_count": self.ungrounded_claim_count,
            "source_quality_avg": self.source_quality_avg,
        }
    
    def is_high_risk(self) -> bool:
        """Check if epistemic risk is high."""
        return self.overall_risk > 0.7


class ClaimValidator:
    """
    Validates claims against epistemic requirements.
    
    Parses LLM outputs to extract claims, validates them against
    source requirements, and computes epistemic risk scores.
    
    Usage:
        validator = ClaimValidator()
        claims = validator.parse_claims(llm_output, sources)
        result = validator.validate(claims)
        if not result.is_valid:
            # Block phase advancement or downgrade confidence
    """
    
    # Patterns indicating factual claims (declarative assertions)
    FACT_PATTERNS = [
        r'\b(is|are|was|were|has|have|had)\b.{10,}',
        r'\baccording to\b',
        r'\bresearch (shows|indicates|suggests|demonstrates)\b',
        r'\bstudies (show|indicate|suggest|demonstrate)\b',
        r'\b(in \d{4}|since \d{4}|by \d{4})\b',
        r'\b(percent|percentage|\d+%)\b',
        r'\b(million|billion|trillion)\b',
        r'\b(increased|decreased|grew|declined) by\b',
    ]
    
    # Patterns indicating inference
    INFERENCE_PATTERNS = [
        r'\b(therefore|thus|hence|consequently)\b',
        r'\b(implies|suggests that|indicates that)\b',
        r'\b(this means|this shows|this demonstrates)\b',
        r'\b(because|since|as a result)\b',
        r'\b(we can (conclude|infer|deduce))\b',
        r'\b(it follows that)\b',
    ]
    
    # Patterns indicating speculation
    SPECULATION_PATTERNS = [
        r'\b(might|may|could|possibly|potentially|perhaps)\b',
        r'\b(likely|unlikely|probably|probability)\b',
        r'\b(if|assuming|hypothetically)\b',
        r'\b(scenario|speculation|speculative)\b',
        r'\b(uncertain|unclear|unknown)\b',
        r'\b(estimate|projection|forecast)\b',
    ]
    
    # Explicit tags in LLM output
    FACT_TAGS = ["[VERIFIED]", "[FACT]", "[SOURCE:", "[CITED]"]
    INFERENCE_TAGS = ["[INFERRED]", "[INFERENCE]", "[DERIVED]", "[ANALYSIS]"]
    SPECULATION_TAGS = ["[SPECULATIVE]", "[SPECULATION]", "[HYPOTHESIS]", "[UNCERTAIN]"]
    
    def __init__(
        self,
        min_grounded_ratio: float = 0.6,
        strict_mode: bool = False,
        allow_untagged_speculation: bool = True
    ):
        """
        Initialize the validator.
        
        Args:
            min_grounded_ratio: Minimum ratio of grounded claims (0-1)
            strict_mode: If True, treat warnings as errors
            allow_untagged_speculation: If True, allow speculation without tags
        """
        self.min_grounded_ratio = min_grounded_ratio
        self.strict_mode = strict_mode
        self.allow_untagged_speculation = allow_untagged_speculation
        
        # Compile patterns for efficiency
        self._fact_patterns = [re.compile(p, re.IGNORECASE) for p in self.FACT_PATTERNS]
        self._inference_patterns = [re.compile(p, re.IGNORECASE) for p in self.INFERENCE_PATTERNS]
        self._speculation_patterns = [re.compile(p, re.IGNORECASE) for p in self.SPECULATION_PATTERNS]
        
        # Source registry for validation
        self._source_registry: Dict[str, Source] = {}
        self._claim_registry: Dict[str, Claim] = {}
    
    def register_source(self, source: Source) -> None:
        """Register a source for claim validation."""
        self._source_registry[source.id] = source
    
    def register_sources(self, sources: List[Source]) -> None:
        """Register multiple sources."""
        for source in sources:
            self.register_source(source)
    
    def clear_registries(self) -> None:
        """Clear source and claim registries."""
        self._source_registry.clear()
        self._claim_registry.clear()
    
    def parse_claims(
        self,
        output: str,
        available_sources: Optional[List[Source]] = None
    ) -> List[Claim]:
        """
        Parse claims from LLM output.
        
        Args:
            output: Raw LLM output text
            available_sources: List of sources that were available
            
        Returns:
            List of parsed Claim objects
        """
        if available_sources:
            self.register_sources(available_sources)
        
        claims = []
        
        # Split into sentences (simplified sentence boundary detection)
        sentences = self._split_into_sentences(output)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Determine claim type
            claim_type, is_tagged = self._classify_sentence(sentence)
            
            # Extract source references
            source_ids = self._extract_source_references(sentence)
            
            # Determine confidence based on language
            confidence = self._estimate_confidence(sentence)
            
            claim = Claim(
                text=sentence[:500],  # Truncate long sentences
                claim_type=claim_type,
                source_ids=source_ids,
                confidence=confidence,
                is_tagged=is_tagged,
                line_number=i + 1,
            )
            
            claims.append(claim)
            self._claim_registry[claim.id] = claim
        
        # Link inferences to upstream claims
        self._link_inferences(claims)
        
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|vs|etc)\.',
                     r'\1<PERIOD>', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        
        return sentences
    
    def _classify_sentence(self, sentence: str) -> Tuple[ClaimType, bool]:
        """
        Classify a sentence into claim type.
        
        Returns:
            Tuple of (ClaimType, is_explicitly_tagged)
        """
        sentence_lower = sentence.lower()
        
        # Check for explicit tags first
        for tag in self.FACT_TAGS:
            if tag.lower() in sentence_lower or tag in sentence:
                return ClaimType.FACT, True
        
        for tag in self.INFERENCE_TAGS:
            if tag.lower() in sentence_lower or tag in sentence:
                return ClaimType.INFERENCE, True
        
        for tag in self.SPECULATION_TAGS:
            if tag.lower() in sentence_lower or tag in sentence:
                return ClaimType.SPECULATION, True
        
        # Check patterns (speculation first, as it's more specific)
        spec_score = sum(1 for p in self._speculation_patterns if p.search(sentence))
        inf_score = sum(1 for p in self._inference_patterns if p.search(sentence))
        fact_score = sum(1 for p in self._fact_patterns if p.search(sentence))
        
        # Speculation patterns are strong indicators
        if spec_score >= 2:
            return ClaimType.SPECULATION, False
        
        # Inference patterns indicate derived content
        if inf_score >= 1:
            return ClaimType.INFERENCE, False
        
        # Fact patterns indicate factual assertions
        if fact_score >= 1:
            return ClaimType.FACT, False
        
        # Default to inference for analytical content
        return ClaimType.INFERENCE, False
    
    def _extract_source_references(self, sentence: str) -> List[str]:
        """Extract source IDs referenced in a sentence."""
        source_ids = []
        
        # Check for URL references
        urls = re.findall(r'https?://[^\s\)]+', sentence)
        for url in urls:
            # Find matching source by URL
            for sid, source in self._source_registry.items():
                if source.url and url in source.url:
                    source_ids.append(sid)
                    break
            else:
                # Create ad-hoc source ID from URL
                source_ids.append(f"url_{hash(url) % 10000:04d}")
        
        # Check for [SOURCE: X] style references
        source_refs = re.findall(r'\[SOURCE:\s*([^\]]+)\]', sentence, re.IGNORECASE)
        source_ids.extend(source_refs)
        
        # Check for "according to X" patterns
        according_to = re.findall(r'according to\s+([^,.\n]+)', sentence, re.IGNORECASE)
        for ref in according_to:
            source_ids.append(f"ref_{hash(ref.strip()) % 10000:04d}")
        
        return source_ids
    
    def _estimate_confidence(self, sentence: str) -> float:
        """Estimate confidence level from language."""
        sentence_lower = sentence.lower()
        
        # High confidence indicators
        high_conf = ["clearly", "certainly", "definitely", "proven", "established"]
        if any(w in sentence_lower for w in high_conf):
            return 0.9
        
        # Low confidence indicators
        low_conf = ["might", "may", "possibly", "perhaps", "uncertain", "unclear"]
        if any(w in sentence_lower for w in low_conf):
            return 0.3
        
        # Medium confidence indicators
        medium_conf = ["likely", "probably", "suggests", "indicates"]
        if any(w in sentence_lower for w in medium_conf):
            return 0.6
        
        return 0.5  # Default
    
    def _link_inferences(self, claims: List[Claim]) -> None:
        """Link inference claims to their upstream facts."""
        facts = [c for c in claims if c.claim_type == ClaimType.FACT and c.is_grounded()]
        
        for claim in claims:
            if claim.claim_type == ClaimType.INFERENCE:
                # Link to facts that appear before this inference
                for fact in facts:
                    if fact.line_number < claim.line_number:
                        # Simple proximity-based linking
                        claim.upstream_claim_ids.append(fact.id)
    
    def validate(self, claims: List[Claim]) -> ClaimValidationResult:
        """
        Validate a list of claims.
        
        Args:
            claims: List of claims to validate
            
        Returns:
            ClaimValidationResult with validation details
        """
        if not claims:
            return ClaimValidationResult(
                is_valid=True,
                total_claims=0,
                grounded_claims=0,
                grounded_ratio=1.0,
            )
        
        violations = []
        ungrounded_facts = []
        untagged_speculation = []
        orphan_inferences = []
        grounded_count = 0
        
        for claim in claims:
            if claim.claim_type == ClaimType.FACT:
                if not claim.source_ids:
                    violations.append(
                        f"FACT without source: '{claim.text[:100]}...'"
                    )
                    ungrounded_facts.append(claim)
                else:
                    grounded_count += 1
                    
            elif claim.claim_type == ClaimType.INFERENCE:
                if not claim.upstream_claim_ids:
                    violations.append(
                        f"INFERENCE without upstream facts: '{claim.text[:100]}...'"
                    )
                    orphan_inferences.append(claim)
                else:
                    grounded_count += 1
                    
            elif claim.claim_type == ClaimType.SPECULATION:
                if not claim.is_tagged and not self.allow_untagged_speculation:
                    violations.append(
                        f"SPECULATION not tagged: '{claim.text[:100]}...'"
                    )
                    untagged_speculation.append(claim)
                else:
                    grounded_count += 1
        
        grounded_ratio = grounded_count / len(claims) if claims else 1.0
        is_valid = grounded_ratio >= self.min_grounded_ratio
        
        if not is_valid:
            logger.warning(
                f"Claim validation failed: {grounded_ratio:.2%} grounded "
                f"(min: {self.min_grounded_ratio:.2%})"
            )
        
        return ClaimValidationResult(
            is_valid=is_valid,
            total_claims=len(claims),
            grounded_claims=grounded_count,
            violations=violations[:20],  # Limit violations
            grounded_ratio=grounded_ratio,
            ungrounded_facts=ungrounded_facts,
            untagged_speculation=untagged_speculation,
            orphan_inferences=orphan_inferences,
        )
    
    def compute_epistemic_risk(
        self,
        output: str,
        claims: List[Claim],
        sources: List[Source],
        stated_confidence: float = 0.5
    ) -> EpistemicRiskScore:
        """
        Compute epistemic risk score for an output.
        
        Args:
            output: Raw LLM output
            claims: Parsed claims
            sources: Available sources
            stated_confidence: Confidence stated by LLM
            
        Returns:
            EpistemicRiskScore with detailed metrics
        """
        risk = EpistemicRiskScore()
        
        # Claim to source ratio
        num_sources = len(sources) if sources else 1
        risk.claim_to_source_ratio = len(claims) / num_sources
        
        # Repetition penalty (detect repeated phrases)
        risk.repetition_penalty = self._compute_repetition_penalty(output)
        
        # Confidence vs evidence delta
        evidence_strength = self._compute_evidence_strength(claims, sources)
        risk.confidence_vs_evidence_delta = max(0, stated_confidence - evidence_strength)
        
        # Speculative density
        spec_claims = sum(1 for c in claims if c.claim_type == ClaimType.SPECULATION)
        risk.speculative_density = spec_claims / len(claims) if claims else 0
        
        # Ungrounded claims
        risk.ungrounded_claim_count = sum(1 for c in claims if not c.is_grounded())
        
        # Source quality
        if sources:
            risk.source_quality_avg = sum(s.quality_score for s in sources) / len(sources)
        
        # Compute overall
        risk.compute_overall_risk()
        
        return risk
    
    def _compute_repetition_penalty(self, output: str) -> float:
        """Detect repetitive content (a hallucination signal)."""
        words = output.lower().split()
        if len(words) < 50:
            return 0.0
        
        # Check for repeated n-grams
        ngram_counts: Dict[str, int] = {}
        n = 4
        for i in range(len(words) - n):
            ngram = " ".join(words[i:i+n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Penalty for n-grams appearing more than twice
        repeated = sum(1 for c in ngram_counts.values() if c > 2)
        penalty = min(1.0, repeated / 10)
        
        return penalty
    
    def _compute_evidence_strength(
        self,
        claims: List[Claim],
        sources: List[Source]
    ) -> float:
        """Compute overall evidence strength."""
        if not claims:
            return 0.5
        
        # Grounding strength
        grounded = sum(1 for c in claims if c.is_grounded())
        grounding_score = grounded / len(claims)
        
        # Source quality contribution
        if sources:
            quality_score = sum(s.quality_score for s in sources) / len(sources)
        else:
            quality_score = 0.3
        
        # Combine
        return 0.7 * grounding_score + 0.3 * quality_score
    
    def cap_confidence_by_evidence(
        self,
        confidence: float,
        risk: EpistemicRiskScore
    ) -> float:
        """
        Cap confidence based on epistemic risk.
        
        High epistemic risk should reduce stated confidence.
        
        Args:
            confidence: Original confidence score
            risk: Computed epistemic risk
            
        Returns:
            Adjusted confidence score
        """
        if risk.overall_risk > 0.7:
            # High risk: cap at 0.5
            return min(confidence, 0.5)
        elif risk.overall_risk > 0.5:
            # Medium risk: cap at 0.7
            return min(confidence, 0.7)
        elif risk.overall_risk > 0.3:
            # Low-medium risk: small reduction
            return min(confidence, 0.85)
        
        return confidence
    
    def should_block_phase_advancement(
        self,
        validation_result: ClaimValidationResult,
        risk: EpistemicRiskScore
    ) -> Tuple[bool, str]:
        """
        Determine if phase advancement should be blocked.
        
        Args:
            validation_result: Claim validation result
            risk: Epistemic risk score
            
        Returns:
            Tuple of (should_block, reason)
        """
        if not validation_result.is_valid:
            return True, (
                f"Grounded claim ratio too low: {validation_result.grounded_ratio:.2%} "
                f"(min: {self.min_grounded_ratio:.2%})"
            )
        
        if risk.is_high_risk():
            return True, f"High epistemic risk: {risk.overall_risk:.2f}"
        
        if len(validation_result.ungrounded_facts) > 5:
            return True, f"Too many ungrounded facts: {len(validation_result.ungrounded_facts)}"
        
        return False, ""


# Global validator instance
_validator: Optional[ClaimValidator] = None


def get_claim_validator(
    min_grounded_ratio: float = 0.6,
    strict_mode: bool = False
) -> ClaimValidator:
    """Get the global claim validator instance."""
    global _validator
    if _validator is None:
        _validator = ClaimValidator(
            min_grounded_ratio=min_grounded_ratio,
            strict_mode=strict_mode
        )
    return _validator

