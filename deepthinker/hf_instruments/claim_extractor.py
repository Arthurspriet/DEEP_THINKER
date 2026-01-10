"""
Multi-Mode Claim Extraction for HF Instruments.

Provides claim extraction with multiple modes and automatic fallback:
- regex: Baseline regex-based extraction (from existing tooling)
- hf: HuggingFace token classification
- llm-json: LLM-based extraction with JSON output

Constraint 3: Claim extraction must be resilient.
- Existing regex extractor is baseline fallback
- If HF/LLM extraction fails, fall back to regex
- Persist extractor_mode, extraction_time_ms, claim_count per run
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import get_config, HF_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    """
    An extracted atomic claim.
    
    Attributes:
        claim_id: Unique identifier for the claim
        text: The claim text
        claim_type: Type of claim (factual, inference, assumption, uncertainty)
        source_type: Source type (final_answer, phase_output, summary, proof)
        source_ref: Source reference (file path or phase name)
        mission_id: Mission identifier (if known)
        phase: Phase name (if known)
        confidence: Confidence score (0-1)
        spans: Character spans in original text [[start, end], ...]
        entities: Extracted entities
        context: Surrounding context
        timestamp: When the claim was extracted
    """
    claim_id: str
    text: str
    claim_type: str = "factual"
    source_type: str = "unknown"
    source_ref: str = ""
    mission_id: str = ""
    phase: str = ""
    confidence: Optional[float] = None
    spans: List[List[int]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    context: str = ""
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ClaimExtractionResult:
    """
    Result of claim extraction.
    
    Includes metadata for observability:
    - extractor_mode: Actual mode used (may differ from requested if fallback)
    - extraction_time_ms: Time taken for extraction
    - claim_count: Number of claims extracted
    """
    claims: List[ExtractedClaim]
    extractor_mode: str  # regex, hf, llm-json, or regex_fallback
    extraction_time_ms: float
    claim_count: int
    source: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "claims": [c.to_dict() for c in self.claims],
            "extractor_mode": self.extractor_mode,
            "extraction_time_ms": self.extraction_time_ms,
            "claim_count": self.claim_count,
            "source": self.source,
            "error": self.error,
        }


class RegexClaimExtractor:
    """
    Regex-based claim extractor (baseline).
    
    Uses patterns from existing deepthinker/tooling/epistemic/claim_extractor.py
    """
    
    # Patterns for factual claims
    FACTUAL_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:is|are|was|were|has|have|contains|includes|shows|indicates|demonstrates|proves|confirms|reveals|suggests|states|reports|finds|according to)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:\d+%|\d+ percent|\d+ of|\d+ out of)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:study|research|analysis|data|evidence|statistics|survey|report)[^.!?]*[.!?])',
    ]
    
    # Patterns for inferences
    INFERENCE_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:therefore|thus|hence|consequently|it follows|implies|suggests that|indicates that)[^.!?]*[.!?])',
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:likely|probably|possibly|may|might|could|appears|seems)[^.!?]*[.!?])',
    ]
    
    # Patterns for assumptions
    ASSUMPTION_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:assuming|presuming|supposing|if we assume|given that)[^.!?]*[.!?])',
    ]
    
    # Patterns for uncertainty
    UNCERTAINTY_PATTERNS = [
        r'(?:^|\.\s+)([A-Z][^.!?]*(?:uncertain|unclear|unknown|unverified|unconfirmed|speculative|tentative)[^.!?]*[.!?])',
    ]
    
    def __init__(self):
        """Initialize compiled patterns."""
        self._patterns = {
            "factual": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.FACTUAL_PATTERNS],
            "inference": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.INFERENCE_PATTERNS],
            "assumption": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.ASSUMPTION_PATTERNS],
            "uncertainty": [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.UNCERTAINTY_PATTERNS],
        }
    
    def extract(
        self,
        text: str,
        source_type: str = "unknown",
        source_ref: str = "",
        mission_id: str = "",
        phase: str = "",
    ) -> List[ExtractedClaim]:
        """
        Extract claims from text using regex patterns.
        
        Args:
            text: Text to extract claims from
            source_type: Type of source
            source_ref: Reference to source
            mission_id: Mission identifier
            phase: Phase name
            
        Returns:
            List of extracted claims
        """
        claims = []
        seen_texts = set()
        
        for claim_type, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    claim_text = match.group(1).strip()
                    
                    # Skip if too short or too long
                    if len(claim_text) < 20 or len(claim_text) > 500:
                        continue
                    
                    # Deduplicate
                    normalized = claim_text.lower().strip()
                    if normalized in seen_texts:
                        continue
                    seen_texts.add(normalized)
                    
                    # Generate ID
                    claim_id = self._generate_claim_id(claim_text, claim_type)
                    
                    # Extract context
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end].strip()
                    
                    claims.append(ExtractedClaim(
                        claim_id=claim_id,
                        text=claim_text,
                        claim_type=claim_type,
                        source_type=source_type,
                        source_ref=source_ref,
                        mission_id=mission_id,
                        phase=phase,
                        spans=[[match.start(), match.end()]],
                        context=context,
                    ))
        
        # Also extract simple declarative sentences
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30 or len(sentence) > 300:
                continue
            
            if re.match(r'^[A-Z].*$', sentence):
                normalized = sentence.lower().strip()
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    claim_id = self._generate_claim_id(sentence, "factual")
                    claims.append(ExtractedClaim(
                        claim_id=claim_id,
                        text=sentence,
                        claim_type="factual",
                        source_type=source_type,
                        source_ref=source_ref,
                        mission_id=mission_id,
                        phase=phase,
                    ))
        
        return claims
    
    def _generate_claim_id(self, text: str, claim_type: str) -> str:
        """Generate a stable claim ID."""
        normalized = text.lower().strip()
        hash_input = f"{claim_type}:{normalized}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"claim_{claim_type}_{hash_hex}"


class HFClaimExtractor:
    """
    HuggingFace-based claim extractor using token classification.
    
    Note: This is a placeholder implementation. In practice, you would
    need a fine-tuned model for claim detection.
    """
    
    def __init__(self, model_id: str = "dslim/bert-base-NER", device: str = "cpu"):
        """Initialize the HF claim extractor."""
        self.model_id = model_id
        self.device = device
        self._pipeline = None
        self._loaded = False
        
        if HF_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the HF pipeline."""
        try:
            from transformers import pipeline
            
            # Use NER as a proxy for claim detection
            # In practice, use a fine-tuned claim detection model
            self._pipeline = pipeline(
                "ner",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
            self._loaded = True
            logger.info(f"HF claim extractor loaded: {self.model_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load HF claim extractor: {e}")
            self._loaded = False
    
    def extract(
        self,
        text: str,
        source_type: str = "unknown",
        source_ref: str = "",
        mission_id: str = "",
        phase: str = "",
    ) -> List[ExtractedClaim]:
        """Extract claims using HF pipeline."""
        if not self._loaded or not self._pipeline:
            raise RuntimeError("HF claim extractor not loaded")
        
        claims = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(sentence) < 20:
                continue
            
            try:
                # Run NER to find entities
                entities = self._pipeline(sentence[:512])  # Truncate
                
                if entities:
                    # If sentence has entities, treat it as a potential claim
                    entity_texts = [e["word"] for e in entities if e.get("word")]
                    
                    claim_id = self._generate_claim_id(sentence)
                    claims.append(ExtractedClaim(
                        claim_id=claim_id,
                        text=sentence.strip(),
                        claim_type="factual",
                        source_type=source_type,
                        source_ref=source_ref,
                        mission_id=mission_id,
                        phase=phase,
                        entities=entity_texts,
                        confidence=0.7,  # Moderate confidence for HF
                    ))
                    
            except Exception as e:
                logger.warning(f"HF extraction failed for sentence: {e}")
                continue
        
        return claims
    
    def _generate_claim_id(self, text: str) -> str:
        """Generate a stable claim ID."""
        normalized = text.lower().strip()
        hash_hex = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"claim_hf_{hash_hex}"


class LLMClaimExtractor:
    """
    LLM-based claim extractor with JSON output.
    
    Uses the existing model caller to extract claims via prompting.
    """
    
    PROMPT_TEMPLATE = """Extract all factual claims from the following text. 
Return a JSON array of claims, where each claim has:
- "text": the claim text
- "type": one of "factual", "inference", "assumption", "uncertainty"
- "confidence": a number from 0 to 1
- "entities": a list of key entities mentioned

Text to analyze:
{text}

Return ONLY a valid JSON array, no other text. Example:
[{{"text": "The economy grew by 3%", "type": "factual", "confidence": 0.9, "entities": ["economy"]}}]

JSON array:"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """Initialize the LLM extractor."""
        self.model = model
    
    def extract(
        self,
        text: str,
        source_type: str = "unknown",
        source_ref: str = "",
        mission_id: str = "",
        phase: str = "",
        max_retries: int = 2,
    ) -> List[ExtractedClaim]:
        """Extract claims using LLM."""
        from deepthinker.models.model_caller import call_model
        
        prompt = self.PROMPT_TEMPLATE.format(text=text[:4000])  # Truncate
        
        last_error = None
        for attempt in range(max_retries):
            try:
                result = call_model(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": 0.1},
                    timeout=30.0,
                    max_retries=1,
                )
                
                response_text = result.get("response", "")
                
                # Try to parse JSON
                claims_data = self._parse_json_response(response_text)
                
                if claims_data:
                    return self._convert_to_claims(
                        claims_data,
                        source_type,
                        source_ref,
                        mission_id,
                        phase,
                    )
                    
            except Exception as e:
                last_error = e
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
                continue
        
        raise RuntimeError(f"LLM extraction failed after {max_retries} attempts: {last_error}")
    
    def _parse_json_response(self, response: str) -> Optional[List[Dict]]:
        """Parse JSON from LLM response."""
        # Try to find JSON array in response
        response = response.strip()
        
        # Try direct parse
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array from response
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _convert_to_claims(
        self,
        claims_data: List[Dict],
        source_type: str,
        source_ref: str,
        mission_id: str,
        phase: str,
    ) -> List[ExtractedClaim]:
        """Convert parsed JSON to ExtractedClaim objects."""
        claims = []
        
        for item in claims_data:
            if not isinstance(item, dict):
                continue
            
            text = item.get("text", "").strip()
            if not text or len(text) < 10:
                continue
            
            claim_type = item.get("type", "factual")
            if claim_type not in ("factual", "inference", "assumption", "uncertainty"):
                claim_type = "factual"
            
            confidence = item.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                    confidence = max(0, min(1, confidence))
                except (ValueError, TypeError):
                    confidence = None
            
            entities = item.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            entities = [str(e) for e in entities]
            
            claim_id = self._generate_claim_id(text, claim_type)
            
            claims.append(ExtractedClaim(
                claim_id=claim_id,
                text=text,
                claim_type=claim_type,
                source_type=source_type,
                source_ref=source_ref,
                mission_id=mission_id,
                phase=phase,
                confidence=confidence,
                entities=entities,
            ))
        
        return claims
    
    def _generate_claim_id(self, text: str, claim_type: str) -> str:
        """Generate a stable claim ID."""
        normalized = text.lower().strip()
        hash_input = f"{claim_type}:{normalized}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"claim_llm_{hash_hex}"


class ClaimExtractionPipeline:
    """
    Multi-mode claim extraction pipeline with automatic fallback.
    
    Constraint 3: Claim extraction must be resilient.
    - Existing regex extractor is baseline fallback
    - If HF/LLM extraction fails, fall back to regex
    - Persist extractor_mode, extraction_time_ms, claim_count per run
    
    Usage:
        pipeline = ClaimExtractionPipeline()
        result = pipeline.extract(text, source={"mission_id": "abc"})
        
        # Access claims
        for claim in result.claims:
            print(claim.text)
        
        # Check metadata
        print(f"Mode: {result.extractor_mode}, Time: {result.extraction_time_ms}ms")
    """
    
    def __init__(self):
        """Initialize the pipeline with extractors."""
        self._regex_extractor = RegexClaimExtractor()
        self._hf_extractor: Optional[HFClaimExtractor] = None
        self._llm_extractor: Optional[LLMClaimExtractor] = None
    
    def extract(
        self,
        text: str,
        source: Dict[str, Any],
        mode: Optional[str] = None,
    ) -> ClaimExtractionResult:
        """
        Extract claims from text.
        
        Args:
            text: Text to extract claims from
            source: Source information dict with keys:
                - source_type: Type of source (final_answer, phase_output, etc.)
                - source_ref: Reference to source
                - mission_id: Mission identifier
                - phase: Phase name
            mode: Extraction mode (regex, hf, llm-json) or None for config default
            
        Returns:
            ClaimExtractionResult with claims and metadata
        """
        config = get_config()
        
        if mode is None:
            mode = config.claim_extractor_mode
        
        source_type = source.get("source_type", "unknown")
        source_ref = source.get("source_ref", "")
        mission_id = source.get("mission_id", "")
        phase = source.get("phase", "")
        
        start_time = time.time()
        actual_mode = mode
        error = None
        claims = []
        
        try:
            if mode == "hf" and HF_AVAILABLE:
                claims = self._extract_hf(text, source_type, source_ref, mission_id, phase)
            elif mode == "llm-json":
                claims = self._extract_llm(text, source_type, source_ref, mission_id, phase)
            else:
                claims = self._extract_regex(text, source_type, source_ref, mission_id, phase)
                actual_mode = "regex"
                
        except Exception as e:
            logger.warning(f"Claim extraction failed with mode={mode}: {e}, falling back to regex")
            error = str(e)
            
            try:
                claims = self._extract_regex(text, source_type, source_ref, mission_id, phase)
                actual_mode = "regex_fallback"
            except Exception as e2:
                logger.error(f"Regex fallback also failed: {e2}")
                error = f"{error}; regex fallback: {e2}"
                claims = []
                actual_mode = "failed"
        
        extraction_time_ms = (time.time() - start_time) * 1000
        
        return ClaimExtractionResult(
            claims=claims,
            extractor_mode=actual_mode,
            extraction_time_ms=extraction_time_ms,
            claim_count=len(claims),
            source=source,
            error=error,
        )
    
    def _extract_regex(
        self,
        text: str,
        source_type: str,
        source_ref: str,
        mission_id: str,
        phase: str,
    ) -> List[ExtractedClaim]:
        """Extract using regex."""
        return self._regex_extractor.extract(
            text, source_type, source_ref, mission_id, phase
        )
    
    def _extract_hf(
        self,
        text: str,
        source_type: str,
        source_ref: str,
        mission_id: str,
        phase: str,
    ) -> List[ExtractedClaim]:
        """Extract using HF."""
        if self._hf_extractor is None:
            config = get_config()
            self._hf_extractor = HFClaimExtractor(device=config.get_resolved_device())
        
        return self._hf_extractor.extract(
            text, source_type, source_ref, mission_id, phase
        )
    
    def _extract_llm(
        self,
        text: str,
        source_type: str,
        source_ref: str,
        mission_id: str,
        phase: str,
    ) -> List[ExtractedClaim]:
        """Extract using LLM."""
        if self._llm_extractor is None:
            self._llm_extractor = LLMClaimExtractor()
        
        return self._llm_extractor.extract(
            text, source_type, source_ref, mission_id, phase
        )


# Module-level singleton
_pipeline: Optional[ClaimExtractionPipeline] = None


def get_claim_pipeline() -> ClaimExtractionPipeline:
    """Get the global claim extraction pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ClaimExtractionPipeline()
    return _pipeline


def extract_claims(
    text: str,
    source: Dict[str, Any],
    mode: Optional[str] = None,
) -> ClaimExtractionResult:
    """
    Convenience function to extract claims.
    
    Args:
        text: Text to extract claims from
        source: Source information dict
        mode: Extraction mode or None for config default
        
    Returns:
        ClaimExtractionResult
    """
    return get_claim_pipeline().extract(text, source, mode)


__all__ = [
    "ExtractedClaim",
    "ClaimExtractionResult",
    "RegexClaimExtractor",
    "HFClaimExtractor",
    "LLMClaimExtractor",
    "ClaimExtractionPipeline",
    "get_claim_pipeline",
    "extract_claims",
]

