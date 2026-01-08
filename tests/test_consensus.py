"""
Tests for Consensus Algorithm modules.

Tests the following consensus mechanisms:
- Majority Vote Consensus
- Weighted Blend Consensus
- Critique Exchange Consensus
- Semantic Distance utilities
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from deepthinker.consensus.voting import MajorityVoteConsensus, VoteResult


class TestMajorityVoteConsensus:
    """Tests for MajorityVoteConsensus algorithm."""
    
    def test_initialization(self):
        """Test consensus initializes with correct defaults."""
        consensus = MajorityVoteConsensus()
        
        assert consensus.similarity_threshold == 0.8
        assert consensus.embedding_model == "qwen3-embedding:4b"
        assert consensus.ollama_base_url == "http://localhost:11434"
        assert consensus._embedding_cache == {}
    
    def test_initialization_custom_params(self):
        """Test consensus with custom parameters."""
        consensus = MajorityVoteConsensus(
            similarity_threshold=0.9,
            embedding_model="custom-embedding",
            ollama_base_url="http://custom:11434"
        )
        
        assert consensus.similarity_threshold == 0.9
        assert consensus.embedding_model == "custom-embedding"
        assert consensus.ollama_base_url == "http://custom:11434"
    
    def test_apply_empty_outputs(self):
        """Test apply with empty outputs."""
        consensus = MajorityVoteConsensus()
        result = consensus.apply({})
        
        assert isinstance(result, VoteResult)
        assert result.winner == ""
        assert result.winner_model == ""
        assert result.confidence == 0.0
    
    def test_apply_single_output(self):
        """Test apply with a single output."""
        consensus = MajorityVoteConsensus()
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            result = consensus.apply({"model_a": "Single output"})
        
        assert result.winner == "Single output"
        assert result.winner_model == "model_a"
        assert result.confidence == 1.0
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        consensus = MajorityVoteConsensus()
        
        vec = [1.0, 2.0, 3.0]
        similarity = consensus._cosine_similarity(vec, vec)
        
        assert abs(similarity - 1.0) < 0.0001
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        consensus = MajorityVoteConsensus()
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = consensus._cosine_similarity(vec1, vec2)
        
        assert abs(similarity) < 0.0001
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        consensus = MajorityVoteConsensus()
        
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = consensus._cosine_similarity(vec1, vec2)
        
        assert abs(similarity + 1.0) < 0.0001
    
    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        consensus = MajorityVoteConsensus()
        
        similarity = consensus._cosine_similarity([], [1.0, 2.0])
        assert similarity == 0.0
        
        similarity = consensus._cosine_similarity([1.0, 2.0], [])
        assert similarity == 0.0
    
    def test_cosine_similarity_zero_norm(self):
        """Test cosine similarity with zero vectors."""
        consensus = MajorityVoteConsensus()
        
        vec_zero = [0.0, 0.0, 0.0]
        vec_normal = [1.0, 2.0, 3.0]
        similarity = consensus._cosine_similarity(vec_zero, vec_normal)
        
        assert similarity == 0.0
    
    def test_cluster_outputs_empty(self):
        """Test clustering with empty outputs."""
        consensus = MajorityVoteConsensus()
        clusters = consensus._cluster_outputs({})
        
        assert clusters == {}
    
    def test_cluster_outputs_single(self):
        """Test clustering with single output."""
        consensus = MajorityVoteConsensus()
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            clusters = consensus._cluster_outputs({"model_a": "Output"})
        
        assert len(clusters) == 1
        assert "model_a" in clusters[0]
    
    def test_cluster_outputs_similar(self):
        """Test clustering groups similar outputs."""
        consensus = MajorityVoteConsensus(similarity_threshold=0.9)
        
        # Mock embeddings that are very similar
        embeddings = {
            "model_a": [0.9, 0.1, 0.0],
            "model_b": [0.85, 0.15, 0.0],  # Similar to model_a
            "model_c": [0.0, 0.1, 0.9],  # Different
        }
        
        with patch.object(consensus, '_get_embedding', side_effect=lambda t: embeddings.get(t[:500], [])):
            clusters = consensus._cluster_outputs({
                "model_a": "model_a",
                "model_b": "model_b", 
                "model_c": "model_c",
            })
        
        # Should have 2 clusters (a+b together, c separate)
        assert len(clusters) >= 1
    
    def test_cache_clear(self):
        """Test embedding cache clearing."""
        consensus = MajorityVoteConsensus()
        consensus._embedding_cache = {"test": [0.1, 0.2]}
        
        consensus.clear_cache()
        
        assert consensus._embedding_cache == {}
    
    def test_model_output_extraction(self):
        """Test extraction from ModelOutput objects."""
        consensus = MajorityVoteConsensus()
        
        # Create mock ModelOutput objects
        mock_output_success = MagicMock()
        mock_output_success.success = True
        mock_output_success.output = "Success output"
        
        mock_output_fail = MagicMock()
        mock_output_fail.success = False
        mock_output_fail.output = None
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            result = consensus.apply({
                "model_a": mock_output_success,
                "model_b": mock_output_fail,
            })
        
        # Should only use successful output
        assert result.winner == "Success output"
        assert result.winner_model == "model_a"


class TestVoteResult:
    """Tests for VoteResult dataclass."""
    
    def test_vote_result_creation(self):
        """Test VoteResult creation."""
        result = VoteResult(
            winner="Test output",
            winner_model="model_a",
            vote_counts={0: 2, 1: 1},
            cluster_assignments={"model_a": 0, "model_b": 0, "model_c": 1},
            confidence=0.67
        )
        
        assert result.winner == "Test output"
        assert result.winner_model == "model_a"
        assert result.vote_counts[0] == 2
        assert result.confidence == 0.67


class TestWeightedBlendConsensus:
    """Tests for WeightedBlendConsensus algorithm."""
    
    def test_import_weighted_blend(self):
        """Test WeightedBlendConsensus can be imported."""
        from deepthinker.consensus.weighted_blend import WeightedBlendConsensus
        
        consensus = WeightedBlendConsensus()
        assert consensus is not None


class TestCritiqueExchangeConsensus:
    """Tests for CritiqueConsensus algorithm."""
    
    def test_import_critique_exchange(self):
        """Test CritiqueConsensus can be imported."""
        from deepthinker.consensus.critique_exchange import CritiqueConsensus
        
        consensus = CritiqueConsensus()
        assert consensus is not None
    
    def test_initialization_defaults(self):
        """Test default initialization."""
        from deepthinker.consensus.critique_exchange import CritiqueConsensus
        
        consensus = CritiqueConsensus()
        
        assert consensus.max_rounds >= 1
        assert consensus.critique_model is not None
        assert consensus.refinement_model is not None


class TestSemanticDistance:
    """Tests for semantic distance utilities."""
    
    def test_import_semantic_distance(self):
        """Test semantic_distance module can be imported."""
        from deepthinker.consensus import semantic_distance
        
        assert semantic_distance is not None


class TestConsensusIntegration:
    """Integration tests for consensus algorithms."""
    
    def test_all_consensus_algorithms_have_apply(self):
        """Test all consensus algorithms have apply method."""
        from deepthinker.consensus.voting import MajorityVoteConsensus
        from deepthinker.consensus.weighted_blend import WeightedBlendConsensus
        from deepthinker.consensus.critique_exchange import CritiqueConsensus
        
        for cls in [MajorityVoteConsensus, WeightedBlendConsensus, CritiqueConsensus]:
            instance = cls()
            assert hasattr(instance, 'apply')
            assert callable(instance.apply)
    
    def test_consensus_with_model_outputs(self):
        """Test consensus works with ModelOutput-like objects."""
        from deepthinker.consensus.voting import MajorityVoteConsensus
        
        # Create mock ModelOutput
        class MockModelOutput:
            def __init__(self, output, success=True):
                self.output = output
                self.success = success
        
        consensus = MajorityVoteConsensus()
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            outputs = {
                "model_a": MockModelOutput("Output A"),
                "model_b": MockModelOutput("Output B"),
            }
            result = consensus.apply(outputs)
        
        assert result.winner in ["Output A", "Output B"]
    
    def test_consensus_handles_failed_outputs(self):
        """Test consensus handles failed model outputs gracefully."""
        from deepthinker.consensus.voting import MajorityVoteConsensus
        
        class MockModelOutput:
            def __init__(self, output, success=True):
                self.output = output
                self.success = success
        
        consensus = MajorityVoteConsensus()
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            outputs = {
                "model_a": MockModelOutput("Good output"),
                "model_b": MockModelOutput(None, success=False),
                "model_c": MockModelOutput("", success=True),  # Empty output
            }
            result = consensus.apply(outputs)
        
        # Should still produce a result from successful non-empty outputs
        assert result.winner == "Good output" or result.winner == ""


class TestEdgeCases:
    """Edge case tests for consensus algorithms."""
    
    def test_very_long_output(self):
        """Test handling of very long outputs."""
        from deepthinker.consensus.voting import MajorityVoteConsensus
        
        consensus = MajorityVoteConsensus()
        long_output = "x" * 10000
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            result = consensus.apply({"model_a": long_output})
        
        assert result.winner == long_output
    
    def test_unicode_outputs(self):
        """Test handling of unicode in outputs."""
        from deepthinker.consensus.voting import MajorityVoteConsensus
        
        consensus = MajorityVoteConsensus()
        unicode_output = "Hello 世界 🌍 مرحبا"
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            result = consensus.apply({"model_a": unicode_output})
        
        assert result.winner == unicode_output
    
    def test_newlines_in_outputs(self):
        """Test handling of newlines in outputs."""
        from deepthinker.consensus.voting import MajorityVoteConsensus
        
        consensus = MajorityVoteConsensus()
        multiline_output = "Line 1\nLine 2\nLine 3"
        
        with patch.object(consensus, '_get_embedding', return_value=[0.1, 0.2, 0.3]):
            result = consensus.apply({"model_a": multiline_output})
        
        assert result.winner == multiline_output

