"""
Cross-Encoder Reranker for HF Instruments.

Provides batched cross-encoder reranking with:
- Provenance tracking (vector_score, rerank_score, rank_before, rank_after)
- Batching + truncation for latency safety
- Graceful error handling (OOM/timeout returns original ordering)

Constraint 2: Reranker output must preserve provenance and be measurable.
Constraint 4: Reranker must be safe for latency.
"""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import get_config, HF_AVAILABLE

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker using HuggingFace transformers.
    
    Implements:
    - Batched scoring for efficiency
    - Truncation for latency safety
    - Provenance tracking for observability
    - Graceful fallback on errors
    
    Usage:
        reranker = CrossEncoderReranker(model_id="cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = reranker.rerank(query, passages, top_k=6)
    """
    
    def __init__(self, model_id: str, device: str = "cpu"):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_id: HuggingFace model ID for cross-encoder
            device: Device to run on (cpu or cuda)
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace not available, reranker will not work")
            return
        
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            logger.info(f"Loading cross-encoder model: {self.model_id}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"
            
            self.model.eval()
            self._loaded = True
            
            logger.info(f"Cross-encoder loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self._loaded = False
    
    def _score_batch(
        self,
        query: str,
        texts: List[str],
        max_length: int
    ) -> List[float]:
        """
        Score a batch of query-text pairs.
        
        Args:
            query: Query string
            texts: List of passage texts
            max_length: Maximum token length per pair
            
        Returns:
            List of scores for each text
        """
        if not self._loaded or not texts:
            return [0.0] * len(texts)
        
        import torch
        
        # Prepare inputs
        pairs = [(query, text) for text in texts]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Score
        with torch.no_grad():
            outputs = self.model(**inputs)
            # For cross-encoders, the logits are the relevance scores
            if hasattr(outputs, 'logits'):
                scores = outputs.logits.squeeze(-1)
            else:
                scores = outputs[0].squeeze(-1)
            
            # Handle single-element case
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            
            scores = scores.cpu().numpy().tolist()
        
        return scores if isinstance(scores, list) else [scores]
    
    def _batch_score_with_timeout(
        self,
        query: str,
        texts: List[str],
        batch_size: int,
        max_length: int,
        timeout: float
    ) -> Optional[List[float]]:
        """
        Score texts in batches with timeout protection.
        
        Args:
            query: Query string
            texts: List of passage texts
            batch_size: Batch size for scoring
            max_length: Maximum token length
            timeout: Timeout in seconds for entire operation
            
        Returns:
            List of scores, or None on timeout/error
        """
        all_scores = []
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            # Check overall timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Reranking timed out after {elapsed:.1f}s")
                return None
            
            batch_texts = texts[i:i + batch_size]
            
            try:
                scores = self._score_batch(query, batch_texts, max_length)
                all_scores.extend(scores)
            except Exception as e:
                logger.warning(f"Batch scoring failed: {e}")
                return None
        
        return all_scores
    
    def rerank(
        self,
        query: str,
        passages: List[Tuple[Dict[str, Any], float]],
        top_k: int = 6,
        mission_id: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank passages with provenance tracking.
        
        Constraint 2: Each passage includes vector_score, rerank_score, 
        rank_before, rank_after, reranked=true.
        
        Constraint 4: On OOM/timeout/error, returns original ordering
        with reranked=False.
        
        Args:
            query: Query string
            passages: List of (document_dict, vector_score) tuples
            top_k: Number of results to return
            mission_id: Optional mission ID for observability
            
        Returns:
            Reranked list of (document_dict, rerank_score) tuples
            with provenance fields added to each document
        """
        if not passages:
            return []
        
        config = get_config()
        
        # If not loaded or HF unavailable, return original ordering
        if not self._loaded:
            logger.warning("Reranker not loaded, returning original ordering")
            return self._add_failed_provenance(passages[:top_k], "not_loaded")
        
        try:
            import torch
            
            start_time = time.time()
            
            # Limit to rerank_topn candidates
            candidates = passages[:config.rerank_topn]
            
            # Truncate query
            query_truncated = query[:config.rerank_max_length]
            
            # Extract and truncate texts
            texts = []
            for doc, _ in candidates:
                text = doc.get("text", "")[:config.rerank_max_length]
                texts.append(text)
            
            # Score with timeout
            scores = self._batch_score_with_timeout(
                query_truncated,
                texts,
                config.rerank_batch_size,
                config.rerank_max_length,
                config.rerank_timeout_seconds
            )
            
            if scores is None:
                logger.warning("Reranking failed, returning original ordering")
                return self._add_failed_provenance(passages[:top_k], "timeout_or_error")
            
            # Build results with provenance
            results = self._attach_provenance_and_sort(
                candidates, 
                scores, 
                top_k,
                mission_id,
                query
            )
            
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Reranking completed in {latency_ms:.1f}ms for {len(candidates)} candidates")
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"Reranker OOM: {e}, returning original ordering")
            return self._add_failed_provenance(passages[:top_k], "cuda_oom")
            
        except Exception as e:
            logger.warning(f"Reranker failed ({type(e).__name__}): {e}, returning original ordering")
            return self._add_failed_provenance(passages[:top_k], str(e))
    
    def _attach_provenance_and_sort(
        self,
        passages: List[Tuple[Dict[str, Any], float]],
        scores: List[float],
        top_k: int,
        mission_id: Optional[str],
        query: str,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Attach provenance fields and sort by rerank score.
        
        Provenance fields:
        - vector_score: Original cosine similarity
        - rerank_score: Cross-encoder score
        - rank_before: Position before reranking (0-indexed)
        - rank_after: Position after reranking (0-indexed)
        - reranked: True
        """
        # Build indexed results
        indexed = []
        for i, ((doc, vector_score), rerank_score) in enumerate(zip(passages, scores)):
            indexed.append({
                "doc": doc,
                "vector_score": vector_score,
                "rerank_score": rerank_score,
                "rank_before": i,
            })
        
        # Sort by rerank score (descending)
        indexed.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Compute observability metrics
        self._emit_observability_events(indexed, mission_id, query, len(passages))
        
        # Build final results with provenance
        results = []
        for rank_after, item in enumerate(indexed[:top_k]):
            doc_with_provenance = {
                **item["doc"],
                "vector_score": item["vector_score"],
                "rerank_score": item["rerank_score"],
                "rank_before": item["rank_before"],
                "rank_after": rank_after,
                "reranked": True,
            }
            results.append((doc_with_provenance, item["rerank_score"]))
        
        return results
    
    def _add_failed_provenance(
        self,
        passages: List[Tuple[Dict[str, Any], float]],
        error: str
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Add provenance for failed reranking (keeps original order)."""
        results = []
        for i, (doc, score) in enumerate(passages):
            doc_with_provenance = {
                **doc,
                "vector_score": score,
                "rerank_score": None,
                "rank_before": i,
                "rank_after": i,
                "reranked": False,
                "rerank_error": error,
            }
            results.append((doc_with_provenance, score))
        return results
    
    def _emit_observability_events(
        self,
        indexed: List[Dict[str, Any]],
        mission_id: Optional[str],
        query: str,
        candidates_count: int
    ) -> None:
        """
        Emit observability events for reranking.
        
        Events:
        - rerank_applied: When reranker runs
        - rerank_swaps: Count of ordering inversions
        - rerank_delta_top1: Top-1 changed yes/no
        """
        try:
            # Import observability tracker
            try:
                from deepthinker.observability.ml_influence import (
                    get_influence_tracker,
                    MLInfluenceEvent,
                )
                tracker = get_influence_tracker()
            except ImportError:
                # Observability not available
                return
            
            query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
            
            # Calculate swap count (inversions)
            swap_count = 0
            for i, item in enumerate(indexed):
                # Count how many items with higher rank_before are now below
                for j in range(i + 1, len(indexed)):
                    if indexed[j]["rank_before"] < item["rank_before"]:
                        swap_count += 1
            
            # Calculate Kendall's tau (simplified)
            n = len(indexed)
            if n > 1:
                max_swaps = n * (n - 1) / 2
                kendall_tau = 1 - (2 * swap_count / max_swaps)
            else:
                kendall_tau = 1.0
            
            # Check if top-1 changed
            top1_changed = indexed[0]["rank_before"] != 0 if indexed else False
            old_top1_id = None
            new_top1_id = None
            score_delta = 0.0
            
            if indexed and top1_changed:
                new_top1_id = indexed[0]["doc"].get("id", "unknown")
                # Find original top-1
                for item in indexed:
                    if item["rank_before"] == 0:
                        old_top1_id = item["doc"].get("id", "unknown")
                        score_delta = indexed[0]["rerank_score"] - item["rerank_score"]
                        break
            
            # Emit events
            events_data = [
                {
                    "event_type": "rerank_applied",
                    "mission_id": mission_id or "unknown",
                    "query_hash": query_hash,
                    "candidates_count": candidates_count,
                    "reranked_count": len(indexed),
                    "model_id": self.model_id,
                },
                {
                    "event_type": "rerank_swaps",
                    "mission_id": mission_id or "unknown",
                    "query_hash": query_hash,
                    "swap_count": swap_count,
                    "kendall_tau": kendall_tau,
                },
                {
                    "event_type": "rerank_delta_top1",
                    "mission_id": mission_id or "unknown",
                    "query_hash": query_hash,
                    "top1_changed": top1_changed,
                    "old_top1_id": old_top1_id,
                    "new_top1_id": new_top1_id,
                    "score_delta": score_delta,
                },
            ]
            
            for event_data in events_data:
                try:
                    event = MLInfluenceEvent(
                        mission_id=event_data.get("mission_id", "unknown"),
                        phase_name="retrieval",
                        predictor_name="hf_reranker",
                        predictor_mode="active",
                        prediction_summary=event_data,
                    )
                    tracker.record_event(event)
                except Exception as e:
                    logger.debug(f"Failed to emit observability event: {e}")
                    
        except Exception as e:
            # Observability failures should never break reranking
            logger.debug(f"Failed to emit observability events: {e}")
    
    def test_rerank(self, num_passages: int = 10) -> Dict[str, Any]:
        """
        Test the reranker with dummy data.
        
        Args:
            num_passages: Number of dummy passages to test with
            
        Returns:
            Test results including latency
        """
        if not self._loaded:
            return {"success": False, "error": "model_not_loaded"}
        
        # Create dummy data
        query = "What is the capital of France?"
        passages = [
            ({"id": f"doc_{i}", "text": f"This is test passage number {i}."}, 0.9 - i * 0.05)
            for i in range(num_passages)
        ]
        
        try:
            start_time = time.time()
            results = self.rerank(query, passages, top_k=5)
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "latency_ms": latency_ms,
                "input_count": num_passages,
                "output_count": len(results),
                "device": self.device,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


__all__ = [
    "CrossEncoderReranker",
]

