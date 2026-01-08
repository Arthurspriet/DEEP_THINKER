"""
Contradiction Detector for DeepThinker Epistemics.

Detects contradictions between claims using:
- NLI (Natural Language Inference) model if available
- LLM-cheap judge fallback if NLI not available
- Heuristic patterns for quick detection

Runs on top-K load-bearing claims or final synthesis only.

Gated with config flag: CLAIM_GRAPH_ENABLED
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .claim_validator import Claim
from .claim_graph import ClaimGraph, EdgeType

logger = logging.getLogger(__name__)


@dataclass
class ContradictionResult:
    """
    Result of contradiction detection between two claims.
    
    Attributes:
        claim1: First claim
        claim2: Second claim
        is_contradiction: Whether claims contradict
        confidence: Confidence in the result (0-1)
        method: Detection method used
        explanation: Brief explanation
        timestamp: When detection was run
    """
    claim1: Claim
    claim2: Claim
    is_contradiction: bool = False
    confidence: float = 0.0
    method: str = "unknown"
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim1_id": self.claim1.id,
            "claim1_text": self.claim1.text[:200],
            "claim2_id": self.claim2.id,
            "claim2_text": self.claim2.text[:200],
            "is_contradiction": self.is_contradiction,
            "confidence": self.confidence,
            "method": self.method,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }


class ContradictionDetector:
    """
    Detects contradictions between claims.
    
    Supports multiple detection methods:
    1. NLI model (transformers) - most accurate
    2. LLM judge (Ollama) - good fallback
    3. Heuristic patterns - fast but limited
    
    Usage:
        detector = ContradictionDetector()
        
        # Check two claims
        result = detector.check_pair(claim1, claim2)
        if result.is_contradiction:
            handle_contradiction(result)
        
        # Check all pairs in graph
        contradictions = detector.detect_all(graph)
    """
    
    # Heuristic patterns for contradiction
    NEGATION_PATTERNS = [
        (r"\bis\b", r"\bis not\b"),
        (r"\bare\b", r"\bare not\b"),
        (r"\bwas\b", r"\bwas not\b"),
        (r"\bwere\b", r"\bwere not\b"),
        (r"\bcan\b", r"\bcannot\b"),
        (r"\bwill\b", r"\bwill not\b"),
        (r"\bhas\b", r"\bdoes not have\b"),
        (r"\bhave\b", r"\bdo not have\b"),
        (r"\bincreased\b", r"\bdecreased\b"),
        (r"\bhigher\b", r"\blower\b"),
        (r"\bmore\b", r"\bless\b"),
        (r"\btrue\b", r"\bfalse\b"),
        (r"\bsupports\b", r"\bcontradicts\b"),
        (r"\bconfirms\b", r"\bdenies\b"),
    ]
    
    # LLM judge prompt for contradiction detection
    CONTRADICTION_PROMPT = """Analyze these two claims and determine if they contradict each other.

## CLAIM 1
{claim1}

## CLAIM 2
{claim2}

## INSTRUCTIONS
Two claims contradict if they make mutually exclusive assertions about the same topic.
- Direct negation: "X is true" vs "X is false"
- Incompatible values: "X is 10" vs "X is 50"
- Opposite conclusions: "X supports Y" vs "X refutes Y"

## YOUR RESPONSE
Respond with exactly one of:
CONTRADICTION: [YES/NO]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [One sentence explaining why]
"""

    def __init__(
        self,
        threshold: float = 0.7,
        use_nli: bool = True,
        use_llm: bool = True,
        llm_model: str = "llama3.2:1b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the contradiction detector.
        
        Args:
            threshold: Confidence threshold for contradiction
            use_nli: Whether to try NLI model
            use_llm: Whether to use LLM as fallback
            llm_model: LLM model for judge
            ollama_base_url: Ollama server URL
        """
        self.threshold = threshold
        self.use_nli = use_nli
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        
        self._nli_model = None
        self._nli_tokenizer = None
        self._nli_available = False
        
        # Try to load NLI model
        if use_nli:
            self._try_load_nli()
    
    def _try_load_nli(self) -> bool:
        """Try to load NLI model from transformers."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            
            # Use a small NLI model
            model_name = "typeform/distilbert-base-uncased-mnli"
            
            self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            self._nli_available = True
            logger.info(f"[CONTRADICTION] Loaded NLI model: {model_name}")
            return True
            
        except ImportError:
            logger.info(
                "[CONTRADICTION] transformers not available, "
                "falling back to LLM/heuristic"
            )
            return False
        except Exception as e:
            logger.warning(f"[CONTRADICTION] Failed to load NLI model: {e}")
            return False
    
    def check_pair(
        self,
        claim1: Claim,
        claim2: Claim,
    ) -> ContradictionResult:
        """
        Check if two claims contradict each other.
        
        Tries methods in order: NLI -> LLM -> Heuristic
        
        Args:
            claim1: First claim
            claim2: Second claim
            
        Returns:
            ContradictionResult with detection details
        """
        # Skip if same claim
        if claim1.id == claim2.id:
            return ContradictionResult(
                claim1=claim1,
                claim2=claim2,
                is_contradiction=False,
                confidence=1.0,
                method="same_claim",
                explanation="Same claim cannot contradict itself",
            )
        
        # Try NLI first
        if self._nli_available:
            result = self._check_with_nli(claim1, claim2)
            if result.confidence >= self.threshold:
                return result
        
        # Try LLM
        if self.use_llm:
            result = self._check_with_llm(claim1, claim2)
            if result.confidence >= self.threshold:
                return result
        
        # Fall back to heuristic
        return self._check_with_heuristic(claim1, claim2)
    
    def _check_with_nli(
        self,
        claim1: Claim,
        claim2: Claim,
    ) -> ContradictionResult:
        """Check contradiction using NLI model."""
        try:
            import torch
            
            # Tokenize
            inputs = self._nli_tokenizer(
                claim1.text,
                claim2.text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            
            # Run inference
            with torch.no_grad():
                outputs = self._nli_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Get contradiction probability (label 0 for MNLI)
            # Labels: 0=contradiction, 1=neutral, 2=entailment
            contradiction_prob = probs[0][0].item()
            
            is_contradiction = contradiction_prob >= self.threshold
            
            return ContradictionResult(
                claim1=claim1,
                claim2=claim2,
                is_contradiction=is_contradiction,
                confidence=contradiction_prob,
                method="nli",
                explanation=(
                    "NLI model detected contradiction"
                    if is_contradiction else
                    "NLI model found no contradiction"
                ),
            )
            
        except Exception as e:
            logger.warning(f"[CONTRADICTION] NLI failed: {e}")
            return ContradictionResult(
                claim1=claim1,
                claim2=claim2,
                confidence=0.0,
                method="nli_error",
                explanation=f"NLI error: {e}",
            )
    
    def _check_with_llm(
        self,
        claim1: Claim,
        claim2: Claim,
    ) -> ContradictionResult:
        """Check contradiction using LLM judge."""
        try:
            from langchain_ollama import ChatOllama
            
            llm = ChatOllama(
                model=self.llm_model,
                base_url=self.ollama_base_url,
                temperature=0.1,
            )
            
            prompt = self.CONTRADICTION_PROMPT.format(
                claim1=claim1.text[:500],
                claim2=claim2.text[:500],
            )
            
            response = llm.invoke(prompt)
            raw_output = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            is_contradiction = False
            confidence = 0.5
            explanation = "LLM analysis"
            
            if "CONTRADICTION: YES" in raw_output.upper():
                is_contradiction = True
            elif "CONTRADICTION: NO" in raw_output.upper():
                is_contradiction = False
            
            # Extract confidence
            conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", raw_output)
            if conf_match:
                confidence = float(conf_match.group(1))
            
            # Extract explanation
            exp_match = re.search(r"EXPLANATION:\s*(.+)", raw_output)
            if exp_match:
                explanation = exp_match.group(1).strip()
            
            return ContradictionResult(
                claim1=claim1,
                claim2=claim2,
                is_contradiction=is_contradiction,
                confidence=confidence,
                method="llm_judge",
                explanation=explanation,
            )
            
        except ImportError:
            # Try legacy Ollama
            try:
                from langchain_community.llms import Ollama
                
                llm = Ollama(
                    model=self.llm_model,
                    base_url=self.ollama_base_url,
                    temperature=0.1,
                )
                
                prompt = self.CONTRADICTION_PROMPT.format(
                    claim1=claim1.text[:500],
                    claim2=claim2.text[:500],
                )
                
                raw_output = llm(prompt)
                
                is_contradiction = "CONTRADICTION: YES" in raw_output.upper()
                confidence = 0.6 if is_contradiction else 0.4
                
                return ContradictionResult(
                    claim1=claim1,
                    claim2=claim2,
                    is_contradiction=is_contradiction,
                    confidence=confidence,
                    method="llm_judge_legacy",
                    explanation="LLM analysis (legacy)",
                )
                
            except Exception as e:
                logger.warning(f"[CONTRADICTION] LLM fallback failed: {e}")
                return ContradictionResult(
                    claim1=claim1,
                    claim2=claim2,
                    confidence=0.0,
                    method="llm_error",
                    explanation=f"LLM error: {e}",
                )
        except Exception as e:
            logger.warning(f"[CONTRADICTION] LLM failed: {e}")
            return ContradictionResult(
                claim1=claim1,
                claim2=claim2,
                confidence=0.0,
                method="llm_error",
                explanation=f"LLM error: {e}",
            )
    
    def _check_with_heuristic(
        self,
        claim1: Claim,
        claim2: Claim,
    ) -> ContradictionResult:
        """Check contradiction using heuristic patterns."""
        text1 = claim1.text.lower()
        text2 = claim2.text.lower()
        
        # Check for negation pattern pairs
        for pattern1, pattern2 in self.NEGATION_PATTERNS:
            match1_has_p1 = re.search(pattern1, text1, re.IGNORECASE)
            match2_has_p2 = re.search(pattern2, text2, re.IGNORECASE)
            match1_has_p2 = re.search(pattern2, text1, re.IGNORECASE)
            match2_has_p1 = re.search(pattern1, text2, re.IGNORECASE)
            
            # Check if claims have opposite patterns on similar topics
            if (match1_has_p1 and match2_has_p2) or (match1_has_p2 and match2_has_p1):
                # Check for topic overlap (simple word overlap)
                words1 = set(text1.split())
                words2 = set(text2.split())
                overlap = len(words1 & words2) / max(len(words1), len(words2), 1)
                
                if overlap > 0.3:  # Significant topic overlap
                    return ContradictionResult(
                        claim1=claim1,
                        claim2=claim2,
                        is_contradiction=True,
                        confidence=0.6,
                        method="heuristic",
                        explanation=f"Detected negation pattern: {pattern1} vs {pattern2}",
                    )
        
        # No contradiction detected
        return ContradictionResult(
            claim1=claim1,
            claim2=claim2,
            is_contradiction=False,
            confidence=0.4,
            method="heuristic",
            explanation="No contradiction patterns detected",
        )
    
    def detect_all(
        self,
        claims: List[Claim],
        max_pairs: int = 100,
    ) -> List[ContradictionResult]:
        """
        Detect contradictions among all claim pairs.
        
        Args:
            claims: List of claims to check
            max_pairs: Maximum pairs to check (for performance)
            
        Returns:
            List of ContradictionResults (only contradictions)
        """
        contradictions = []
        pairs_checked = 0
        
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if pairs_checked >= max_pairs:
                    break
                
                result = self.check_pair(claim1, claim2)
                pairs_checked += 1
                
                if result.is_contradiction:
                    contradictions.append(result)
            
            if pairs_checked >= max_pairs:
                break
        
        logger.debug(
            f"[CONTRADICTION] Checked {pairs_checked} pairs, "
            f"found {len(contradictions)} contradictions"
        )
        
        return contradictions
    
    def detect_in_graph(
        self,
        graph: ClaimGraph,
        top_k: int = 20,
    ) -> Tuple[List[ContradictionResult], float]:
        """
        Detect contradictions in a claim graph.
        
        Only checks top-K load-bearing claims for efficiency.
        
        Args:
            graph: ClaimGraph to check
            top_k: Number of top claims to check
            
        Returns:
            Tuple of (contradictions, consistency_score)
        """
        # Get top-K load-bearing claims
        top_nodes = graph.get_top_k_load_bearing(top_k)
        top_claims = [node.claim for node in top_nodes]
        
        # Detect contradictions
        contradictions = self.detect_all(top_claims)
        
        # Add edges to graph
        for result in contradictions:
            graph.add_edge(
                source_claim_id=result.claim1.id,
                target_claim_id=result.claim2.id,
                edge_type=EdgeType.CONTRADICTS,
                confidence=result.confidence,
                detected_by=result.method,
            )
        
        # Compute consistency score
        if not top_claims:
            consistency = 1.0
        else:
            consistency = 1.0 - (len(contradictions) / len(top_claims))
            consistency = max(0.0, consistency)
        
        return contradictions, consistency


# Global detector instance
_detector: Optional[ContradictionDetector] = None


def get_contradiction_detector(
    threshold: float = 0.7,
    use_nli: bool = True,
    use_llm: bool = True,
) -> ContradictionDetector:
    """Get global contradiction detector instance."""
    global _detector
    if _detector is None:
        _detector = ContradictionDetector(
            threshold=threshold,
            use_nli=use_nli,
            use_llm=use_llm,
        )
    return _detector

