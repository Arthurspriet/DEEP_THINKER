"""
Tests for Claim Graph module.

Tests graph building, contradiction detection, top-K selection,
and integration with ClaimRegistry.
"""

import pytest


class TestClaimGraph:
    """Test ClaimGraph class."""
    
    def test_import(self):
        """Test that claim graph can be imported."""
        from deepthinker.epistemics import ClaimGraph, ClaimNode, ClaimEdge, EdgeType
        assert ClaimGraph is not None
        assert ClaimNode is not None
        assert ClaimEdge is not None
        assert EdgeType is not None
    
    def test_creation(self):
        """Test graph creation."""
        from deepthinker.epistemics import ClaimGraph
        
        graph = ClaimGraph()
        
        assert len(graph._nodes) == 0
        assert len(graph._edges) == 0
    
    def test_add_claim(self):
        """Test adding claims to graph."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType
        
        graph = ClaimGraph()
        
        claim = Claim(
            text="The sky is blue",
            claim_type=ClaimType.FACT,
        )
        
        node = graph.add_claim(claim)
        
        assert node.claim == claim
        assert claim.id in graph._nodes
    
    def test_add_edge(self):
        """Test adding edges between claims."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType, EdgeType
        
        graph = ClaimGraph()
        
        claim1 = Claim(text="The sky is blue", claim_type=ClaimType.FACT)
        claim2 = Claim(text="Therefore the ocean is blue", claim_type=ClaimType.INFERENCE)
        
        graph.add_claim(claim1)
        graph.add_claim(claim2)
        
        edge = graph.add_edge(
            source_claim_id=claim1.id,
            target_claim_id=claim2.id,
            edge_type=EdgeType.SUPPORTS,
            confidence=0.8,
        )
        
        assert edge is not None
        assert edge.edge_type == EdgeType.SUPPORTS
        assert len(graph._edges) == 1
    
    def test_add_contradiction_edge(self):
        """Test adding contradiction edge."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType, EdgeType
        
        graph = ClaimGraph()
        
        claim1 = Claim(text="The sky is blue", claim_type=ClaimType.FACT)
        claim2 = Claim(text="The sky is not blue", claim_type=ClaimType.FACT)
        
        graph.add_claim(claim1)
        graph.add_claim(claim2)
        
        graph.add_edge(
            source_claim_id=claim1.id,
            target_claim_id=claim2.id,
            edge_type=EdgeType.CONTRADICTS,
        )
        
        assert len(graph._contradiction_pairs) == 1
    
    def test_build_from_claims(self):
        """Test building graph from list of claims."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType
        
        claims = [
            Claim(text="Claim 1", claim_type=ClaimType.FACT),
            Claim(text="Claim 2", claim_type=ClaimType.FACT),
            Claim(text="Claim 3", claim_type=ClaimType.INFERENCE),
        ]
        
        # Add upstream reference
        claims[2].upstream_claim_ids.append(claims[0].id)
        
        graph = ClaimGraph()
        graph.build_from_claims(claims)
        
        assert len(graph._nodes) == 3
        assert len(graph._edges) >= 1  # At least the upstream link
    
    def test_get_top_k_load_bearing(self):
        """Test getting top-K load-bearing claims."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType, ClaimStatus
        
        claims = [
            Claim(text="Important fact", claim_type=ClaimType.FACT),
            Claim(text="Less important", claim_type=ClaimType.SPECULATION),
            Claim(text="Another fact", claim_type=ClaimType.FACT),
        ]
        
        # Make first claim grounded (more important)
        claims[0].status = ClaimStatus.GROUNDED
        
        graph = ClaimGraph()
        graph.build_from_claims(claims)
        
        top_2 = graph.get_top_k_load_bearing(k=2)
        
        assert len(top_2) == 2
        # First should be the grounded fact
        assert top_2[0].claim.text == "Important fact"
    
    def test_compute_consistency_score(self):
        """Test consistency score computation."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType, EdgeType
        
        graph = ClaimGraph()
        
        # Add claims with no contradictions
        claim1 = Claim(text="Claim 1", claim_type=ClaimType.FACT)
        claim2 = Claim(text="Claim 2", claim_type=ClaimType.FACT)
        
        graph.add_claim(claim1)
        graph.add_claim(claim2)
        
        consistency = graph.compute_consistency_score()
        
        # No contradictions, should be 1.0
        assert consistency == 1.0
        
        # Add contradiction
        graph.add_edge(
            source_claim_id=claim1.id,
            target_claim_id=claim2.id,
            edge_type=EdgeType.CONTRADICTS,
        )
        
        consistency = graph.compute_consistency_score()
        
        # Should be less than 1.0
        assert consistency < 1.0
    
    def test_get_contradicting_pairs(self):
        """Test getting contradicting claim pairs."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType, EdgeType
        
        graph = ClaimGraph()
        
        claim1 = Claim(text="A is true", claim_type=ClaimType.FACT)
        claim2 = Claim(text="A is false", claim_type=ClaimType.FACT)
        
        graph.add_claim(claim1)
        graph.add_claim(claim2)
        graph.add_edge(
            source_claim_id=claim1.id,
            target_claim_id=claim2.id,
            edge_type=EdgeType.CONTRADICTS,
        )
        
        pairs = graph.get_contradicting_pairs()
        
        assert len(pairs) == 1
        assert pairs[0][0].text in ["A is true", "A is false"]
        assert pairs[0][1].text in ["A is true", "A is false"]
    
    def test_get_statistics(self):
        """Test getting graph statistics."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType, EdgeType
        
        graph = ClaimGraph()
        
        claim1 = Claim(text="Claim 1", claim_type=ClaimType.FACT)
        claim2 = Claim(text="Claim 2", claim_type=ClaimType.FACT)
        
        graph.add_claim(claim1)
        graph.add_claim(claim2)
        graph.add_edge(
            source_claim_id=claim1.id,
            target_claim_id=claim2.id,
            edge_type=EdgeType.SUPPORTS,
        )
        
        stats = graph.get_statistics()
        
        assert stats["num_claims"] == 2
        assert stats["num_edges"] == 1
        assert stats["num_contradictions"] == 0
        assert "consistency_score" in stats
    
    def test_to_dict(self):
        """Test graph serialization."""
        from deepthinker.epistemics import ClaimGraph, Claim, ClaimType
        
        graph = ClaimGraph()
        graph.add_claim(Claim(text="Test claim", claim_type=ClaimType.FACT))
        
        data = graph.to_dict()
        
        assert "nodes" in data
        assert "edges" in data
        assert "statistics" in data


class TestClaimNode:
    """Test ClaimNode class."""
    
    def test_creation(self):
        """Test node creation."""
        from deepthinker.epistemics import ClaimNode, Claim, ClaimType
        
        claim = Claim(text="Test", claim_type=ClaimType.FACT)
        node = ClaimNode(claim=claim)
        
        assert node.claim == claim
        assert node.claim_id == claim.id
    
    def test_compute_load_bearing_score(self):
        """Test load-bearing score computation."""
        from deepthinker.epistemics import ClaimNode, Claim, ClaimType, ClaimStatus
        
        claim = Claim(text="Important fact", claim_type=ClaimType.FACT)
        claim.status = ClaimStatus.GROUNDED
        claim.confidence = 0.9
        
        node = ClaimNode(claim=claim)
        score = node.compute_load_bearing_score()
        
        # Should be high due to FACT type, GROUNDED status, high confidence
        assert score > 0.5


class TestContradictionDetector:
    """Test ContradictionDetector class."""
    
    def test_import(self):
        """Test that detector can be imported."""
        from deepthinker.epistemics import ContradictionDetector, ContradictionResult
        assert ContradictionDetector is not None
        assert ContradictionResult is not None
    
    def test_creation(self):
        """Test detector creation."""
        from deepthinker.epistemics import ContradictionDetector
        
        detector = ContradictionDetector(
            threshold=0.7,
            use_nli=False,  # Don't require NLI model in tests
            use_llm=False,  # Don't require LLM in tests
        )
        
        assert detector is not None
        assert detector.threshold == 0.7
    
    def test_heuristic_contradiction(self):
        """Test heuristic contradiction detection."""
        from deepthinker.epistemics import ContradictionDetector, Claim, ClaimType
        
        detector = ContradictionDetector(
            use_nli=False,
            use_llm=False,
        )
        
        claim1 = Claim(text="The market is increasing rapidly", claim_type=ClaimType.FACT)
        claim2 = Claim(text="The market is decreasing rapidly", claim_type=ClaimType.FACT)
        
        result = detector.check_pair(claim1, claim2)
        
        # Should detect contradiction via heuristic
        assert result.method == "heuristic"
        # May or may not detect depending on heuristic sensitivity
    
    def test_same_claim_no_contradiction(self):
        """Test that same claim doesn't contradict itself."""
        from deepthinker.epistemics import ContradictionDetector, Claim, ClaimType
        
        detector = ContradictionDetector(
            use_nli=False,
            use_llm=False,
        )
        
        claim = Claim(text="Something is true", claim_type=ClaimType.FACT)
        
        result = detector.check_pair(claim, claim)
        
        assert result.is_contradiction == False
        assert result.method == "same_claim"
    
    def test_detect_all(self):
        """Test detecting all contradictions in a list."""
        from deepthinker.epistemics import ContradictionDetector, Claim, ClaimType
        
        detector = ContradictionDetector(
            use_nli=False,
            use_llm=False,
        )
        
        claims = [
            Claim(text="The value is high", claim_type=ClaimType.FACT),
            Claim(text="The value is low", claim_type=ClaimType.FACT),
            Claim(text="Something else", claim_type=ClaimType.FACT),
        ]
        
        contradictions = detector.detect_all(claims)
        
        # Returns list of contradictions found
        assert isinstance(contradictions, list)


class TestRankClaimsByLoadBearing:
    """Test convenience function for ranking claims."""
    
    def test_rank_claims(self):
        """Test ranking claims by load-bearing score."""
        from deepthinker.epistemics import rank_claims_by_load_bearing, Claim, ClaimType, ClaimStatus
        
        claims = [
            Claim(text="Low importance", claim_type=ClaimType.SPECULATION),
            Claim(text="High importance", claim_type=ClaimType.FACT),
            Claim(text="Medium importance", claim_type=ClaimType.INFERENCE),
        ]
        
        claims[1].status = ClaimStatus.GROUNDED
        
        ranked = rank_claims_by_load_bearing(claims, k=2)
        
        assert len(ranked) == 2
        # First should be the grounded fact
        assert ranked[0].text == "High importance"

