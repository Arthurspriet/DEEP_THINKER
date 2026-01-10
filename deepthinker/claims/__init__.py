"""
Claims Package for DeepThinker.

Provides claim extraction and persistence to kb/claims/.

Usage:
    from deepthinker.claims import ClaimStore, extract_and_store_claims
    
    # Extract and store claims from mission output
    result = extract_and_store_claims(
        text=final_answer,
        mission_id="abc123",
        source_type="final_answer",
        source_ref="synthesis_report"
    )
    
    # Read claims for a mission
    store = ClaimStore()
    claims = store.read_claims("abc123")
"""

from .store import (
    ClaimStore,
    extract_and_store_claims,
    get_claim_store,
)

__all__ = [
    "ClaimStore",
    "extract_and_store_claims",
    "get_claim_store",
]

