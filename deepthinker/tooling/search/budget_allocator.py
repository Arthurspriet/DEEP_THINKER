"""
Search Budget Allocator - Allocates search budget based on mission constraints.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from ..schemas import Claim, ConfidenceScore, SearchBudget, SearchJustification

logger = logging.getLogger(__name__)


class SearchBudgetAllocator:
    """
    Allocates search budget based on:
    - Mission time remaining
    - Claim criticality scores
    - Prior searches (reuse detection)
    """
    
    # Base time estimates (in minutes)
    TIME_PER_SEARCH = 0.5  # 30 seconds per search
    TIME_PER_RESULT = 0.1  # 6 seconds per result processing
    
    def __init__(
        self,
        default_max_searches: int = 10,
        default_depth_per_query: int = 5
    ):
        """
        Initialize budget allocator.
        
        Args:
            default_max_searches: Default max searches if no time budget
            default_depth_per_query: Default results per query
        """
        self.default_max_searches = default_max_searches
        self.default_depth_per_query = default_depth_per_query
    
    def allocate_budget(
        self,
        time_remaining_minutes: Optional[float] = None,
        claims: Optional[List[Claim]] = None,
        confidence_scores: Optional[List[ConfidenceScore]] = None,
        prior_searches: Optional[Dict[str, int]] = None,
        justifications: Optional[List[SearchJustification]] = None
    ) -> SearchBudget:
        """
        Allocate search budget for current phase.
        
        Args:
            time_remaining_minutes: Time remaining in mission (minutes)
            claims: Optional list of claims to prioritize
            confidence_scores: Optional confidence scores for claims
            prior_searches: Optional dict mapping claim_id -> search count
            justifications: Optional list of search justifications
            
        Returns:
            SearchBudget with allocated limits
        """
        # Calculate max searches from time budget
        max_searches = self.default_max_searches
        if time_remaining_minutes is not None:
            # Estimate: each search takes TIME_PER_SEARCH minutes
            # Reserve 20% of time for other operations
            available_time = time_remaining_minutes * 0.8
            max_searches = int(available_time / self.TIME_PER_SEARCH)
            max_searches = max(1, min(max_searches, 20))  # Clamp between 1 and 20
        
        # Calculate depth per query
        depth_per_query = self.default_depth_per_query
        if time_remaining_minutes is not None:
            # Adjust depth based on time remaining
            if time_remaining_minutes < 5:
                depth_per_query = 3  # Shallow when time is short
            elif time_remaining_minutes < 15:
                depth_per_query = 5  # Medium
            else:
                depth_per_query = 7  # Deep when time allows
        
        # Determine priority claims
        priority_claims = []
        if claims and confidence_scores:
            # Prioritize high-risk, high-priority claims
            claim_priority_map = {}
            confidence_map = {score.claim_id: score for score in confidence_scores}
            
            for claim in claims:
                confidence = confidence_map.get(claim.id)
                if not confidence:
                    continue
                
                # Priority score: inverse of confidence (lower confidence = higher priority)
                priority_score = 1.0 - confidence.score
                
                # Boost priority for high-risk claims
                if confidence.risk_level == "HIGH":
                    priority_score *= 1.5
                
                # Boost priority for factual claims
                if claim.claim_type == "factual":
                    priority_score *= 1.2
                
                claim_priority_map[claim.id] = priority_score
            
            # Sort by priority and take top claims
            sorted_claims = sorted(
                claim_priority_map.items(),
                key=lambda x: x[1],
                reverse=True
            )
            priority_claims = [claim_id for claim_id, _ in sorted_claims[:max_searches]]
        
        # If justifications provided, prioritize those
        if justifications:
            # Sort by priority (high > medium > low)
            priority_order = {"high": 3, "medium": 2, "low": 1}
            sorted_justifications = sorted(
                justifications,
                key=lambda j: priority_order.get(j.priority, 0),
                reverse=True
            )
            # Add justification claim IDs to priority list
            justification_claims = [j.claim_id for j in sorted_justifications]
            # Merge with existing priorities (justifications first)
            priority_claims = justification_claims + [
                cid for cid in priority_claims if cid not in justification_claims
            ]
            priority_claims = priority_claims[:max_searches]
        
        # Account for prior searches (reuse detection)
        if prior_searches:
            # Reduce budget for claims that were already searched
            already_searched = set(prior_searches.keys())
            priority_claims = [
                cid for cid in priority_claims 
                if cid not in already_searched
            ]
        
        budget = SearchBudget(
            max_searches=max_searches,
            depth_per_query=depth_per_query,
            priority_claims=priority_claims,
            time_remaining_minutes=time_remaining_minutes or 0.0,
            budget_used=0
        )
        
        logger.info(
            f"Allocated search budget: max_searches={max_searches}, "
            f"depth={depth_per_query}, priority_claims={len(priority_claims)}"
        )
        
        return budget
    
    def check_budget_available(self, budget: SearchBudget) -> bool:
        """
        Check if budget has remaining capacity.
        
        Args:
            budget: Search budget to check
            
        Returns:
            True if budget available
        """
        return budget.budget_used < budget.max_searches
    
    def record_search(
        self,
        budget: SearchBudget,
        claim_id: Optional[str] = None,
        query_count: int = 1
    ) -> None:
        """
        Record that a search was performed.
        
        Args:
            budget: Budget to update
            claim_id: Optional claim ID that was searched
            query_count: Number of queries executed
        """
        budget.budget_used += query_count
        logger.debug(f"Search budget used: {budget.budget_used}/{budget.max_searches}")

