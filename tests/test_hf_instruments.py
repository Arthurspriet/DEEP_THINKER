"""
Tests for HF Instruments Layer.

Smoke tests that skip gracefully if HF dependencies are not installed.
Tests verify that the API works correctly and fails gracefully.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_kb_dir():
    """Create a temporary kb directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "kb"
        kb_dir.mkdir()
        (kb_dir / "rag" / "global").mkdir(parents=True)
        (kb_dir / "claims").mkdir()
        yield kb_dir


@pytest.fixture
def sample_meta_json(temp_kb_dir):
    """Create a sample meta.json file."""
    meta = {
        "embedding_model_id": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "similarity_type": "cosine",
        "normalization": "l2",
        "created_at": "2025-01-10T12:00:00Z",
        "document_count": 100,
        "notes": "Test index"
    }
    meta_path = temp_kb_dir / "rag" / "global" / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta_path


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for HF instruments configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from deepthinker.hf_instruments.config import HFInstrumentsConfig
        
        config = HFInstrumentsConfig()
        
        assert config.enabled is False
        assert config.embeddings_enabled is False
        assert config.reranker_enabled is True
        assert config.claim_extractor_enabled is True
        assert config.device == "auto"
        assert config.cache_max_models == 2
    
    def test_config_from_environment(self):
        """Test configuration from environment variables."""
        from deepthinker.hf_instruments.config import load_config_from_environment
        
        with patch.dict(os.environ, {
            "DEEPTHINKER_HF_INSTRUMENTS_ENABLED": "true",
            "DEEPTHINKER_HF_RERANKER_ENABLED": "true",
            "DEEPTHINKER_HF_DEVICE": "cpu",
        }):
            config = load_config_from_environment()
            
            assert config.enabled is True
            assert config.reranker_enabled is True
            assert config.device == "cpu"
    
    def test_config_validation(self):
        """Test that invalid config values are corrected."""
        from deepthinker.hf_instruments.config import HFInstrumentsConfig
        
        config = HFInstrumentsConfig(
            device="invalid",
            claim_extractor_mode="invalid",
            cache_max_models=-1,
        )
        
        assert config.device == "auto"
        assert config.claim_extractor_mode == "regex"
        assert config.cache_max_models == 1
    
    def test_is_reranker_active(self):
        """Test is_reranker_active method."""
        from deepthinker.hf_instruments.config import HFInstrumentsConfig, HF_AVAILABLE
        
        config = HFInstrumentsConfig(enabled=True, reranker_enabled=True)
        
        # Result depends on whether HF is available
        if HF_AVAILABLE:
            assert config.is_reranker_active() is True
        else:
            assert config.is_reranker_active() is False


# =============================================================================
# Meta Utils Tests
# =============================================================================

class TestMeta:
    """Tests for index metadata utilities."""
    
    def test_read_index_meta(self, sample_meta_json):
        """Test reading index metadata."""
        from deepthinker.hf_instruments.meta import read_index_meta
        
        meta = read_index_meta(sample_meta_json.parent)
        
        assert meta is not None
        assert meta.embedding_model_id == "BAAI/bge-small-en-v1.5"
        assert meta.embedding_dimension == 384
        assert meta.similarity_type == "cosine"
    
    def test_read_nonexistent_meta(self, temp_kb_dir):
        """Test reading meta from directory without meta.json."""
        from deepthinker.hf_instruments.meta import read_index_meta
        
        meta = read_index_meta(temp_kb_dir / "nonexistent")
        
        assert meta is None
    
    def test_write_index_meta(self, temp_kb_dir):
        """Test writing index metadata."""
        from deepthinker.hf_instruments.meta import IndexMeta, write_index_meta, read_index_meta
        
        meta = IndexMeta(
            embedding_model_id="test-model",
            embedding_dimension=768,
        )
        
        test_dir = temp_kb_dir / "test_index"
        success = write_index_meta(test_dir, meta)
        
        assert success is True
        
        # Verify it can be read back
        loaded = read_index_meta(test_dir)
        assert loaded is not None
        assert loaded.embedding_model_id == "test-model"
        assert loaded.embedding_dimension == 768
    
    def test_is_embedding_compatible_no_meta(self):
        """Test compatibility check with no meta.json."""
        from deepthinker.hf_instruments.meta import is_embedding_compatible
        
        result = is_embedding_compatible(None, "any-model")
        
        assert result is False


# =============================================================================
# Manager Tests
# =============================================================================

class TestManager:
    """Tests for HF instrument manager."""
    
    def test_manager_singleton(self):
        """Test that manager is a singleton."""
        from deepthinker.hf_instruments.manager import get_instrument_manager
        
        manager1 = get_instrument_manager()
        manager2 = get_instrument_manager()
        
        assert manager1 is manager2
    
    def test_manager_status(self):
        """Test manager status reporting."""
        from deepthinker.hf_instruments.manager import get_instrument_manager
        
        manager = get_instrument_manager()
        status = manager.get_status()
        
        assert "hf_available" in status
        assert "enabled" in status
        assert "device" in status
        assert "cached_models" in status
    
    def test_get_reranker_disabled(self):
        """Test that reranker returns None when disabled."""
        from deepthinker.hf_instruments.config import set_config, HFInstrumentsConfig
        from deepthinker.hf_instruments.manager import get_reranker
        
        # Ensure disabled
        set_config(HFInstrumentsConfig(enabled=False))
        
        reranker = get_reranker()
        
        assert reranker is None


# =============================================================================
# Claim Extraction Tests
# =============================================================================

class TestClaimExtraction:
    """Tests for claim extraction pipeline."""
    
    def test_regex_extractor(self):
        """Test regex claim extractor."""
        from deepthinker.hf_instruments.claim_extractor import RegexClaimExtractor
        
        extractor = RegexClaimExtractor()
        
        text = """
        The unemployment rate in France decreased by 2% in Q4 2024.
        According to the study, renewable energy adoption has increased.
        This suggests that economic policies are working.
        """
        
        claims = extractor.extract(
            text,
            source_type="test",
            mission_id="test-mission"
        )
        
        assert len(claims) > 0
        assert all(c.mission_id == "test-mission" for c in claims)
        assert all(c.source_type == "test" for c in claims)
    
    def test_extraction_pipeline_regex_mode(self):
        """Test extraction pipeline with regex mode."""
        from deepthinker.hf_instruments.claim_extractor import ClaimExtractionPipeline
        
        pipeline = ClaimExtractionPipeline()
        
        text = "The economy grew by 3% last year. Studies show this is above average."
        source = {"mission_id": "test", "source_type": "test"}
        
        result = pipeline.extract(text, source, mode="regex")
        
        assert result.extractor_mode == "regex"
        assert result.claim_count >= 0
        assert result.extraction_time_ms > 0
    
    def test_extraction_pipeline_fallback(self):
        """Test extraction pipeline falls back to regex on HF failure."""
        from deepthinker.hf_instruments.claim_extractor import ClaimExtractionPipeline
        
        pipeline = ClaimExtractionPipeline()
        
        text = "Some test text with claims."
        source = {"mission_id": "test", "source_type": "test"}
        
        # HF mode should fall back to regex if HF not available
        result = pipeline.extract(text, source, mode="hf")
        
        assert result.extractor_mode in ("hf", "regex_fallback", "regex")


# =============================================================================
# Claims Store Tests
# =============================================================================

class TestClaimsStore:
    """Tests for claims storage."""
    
    def test_store_and_read_claims(self, temp_kb_dir):
        """Test storing and reading claims."""
        from deepthinker.claims.store import ClaimStore
        from deepthinker.hf_instruments.claim_extractor import (
            ExtractedClaim, ClaimExtractionResult
        )
        
        store = ClaimStore(base_dir=temp_kb_dir / "claims")
        
        claims = [
            ExtractedClaim(
                claim_id="claim_1",
                text="Test claim 1",
                claim_type="factual",
            ),
            ExtractedClaim(
                claim_id="claim_2",
                text="Test claim 2",
                claim_type="inference",
            ),
        ]
        
        result = ClaimExtractionResult(
            claims=claims,
            extractor_mode="regex",
            extraction_time_ms=10.5,
            claim_count=2,
        )
        
        # Write
        success = store.write_extraction_result(result, "test-mission")
        assert success is True
        
        # Read back
        loaded_claims = store.read_claims("test-mission")
        assert len(loaded_claims) == 2
        assert loaded_claims[0]["claim_id"] == "claim_1"
        assert loaded_claims[1]["claim_id"] == "claim_2"
    
    def test_read_run_metadata(self, temp_kb_dir):
        """Test reading run metadata."""
        from deepthinker.claims.store import ClaimStore
        from deepthinker.hf_instruments.claim_extractor import (
            ExtractedClaim, ClaimExtractionResult
        )
        
        store = ClaimStore(base_dir=temp_kb_dir / "claims")
        
        result = ClaimExtractionResult(
            claims=[ExtractedClaim(claim_id="c1", text="test")],
            extractor_mode="hf",
            extraction_time_ms=25.0,
            claim_count=1,
        )
        
        store.write_extraction_result(result, "meta-test-mission")
        
        runs = store.read_run_metadata("meta-test-mission")
        
        assert len(runs) == 1
        assert runs[0]["extractor_mode"] == "hf"
        assert runs[0]["extraction_time_ms"] == 25.0
        assert runs[0]["claim_count"] == 1
    
    def test_list_missions(self, temp_kb_dir):
        """Test listing missions with claims."""
        from deepthinker.claims.store import ClaimStore
        from deepthinker.hf_instruments.claim_extractor import (
            ExtractedClaim, ClaimExtractionResult
        )
        
        store = ClaimStore(base_dir=temp_kb_dir / "claims")
        
        for i in range(3):
            result = ClaimExtractionResult(
                claims=[ExtractedClaim(claim_id=f"c{i}", text=f"claim {i}")],
                extractor_mode="regex",
                extraction_time_ms=1.0,
                claim_count=1,
            )
            store.write_extraction_result(result, f"mission-{i}")
        
        missions = store.list_missions()
        
        assert len(missions) == 3
        assert "mission-0" in missions
        assert "mission-1" in missions
        assert "mission-2" in missions


# =============================================================================
# Reranker Tests (conditional on HF availability)
# =============================================================================

class TestReranker:
    """Tests for cross-encoder reranker."""
    
    def test_reranker_provenance_on_failure(self):
        """Test that failed reranking adds appropriate provenance."""
        from deepthinker.hf_instruments.config import HF_AVAILABLE
        
        if not HF_AVAILABLE:
            pytest.skip("HF not available")
        
        from deepthinker.hf_instruments.reranker import CrossEncoderReranker
        
        # Create reranker with invalid model to force failure
        reranker = CrossEncoderReranker(
            model_id="invalid-model-that-does-not-exist",
            device="cpu"
        )
        
        passages = [
            ({"id": "doc1", "text": "Test doc 1"}, 0.9),
            ({"id": "doc2", "text": "Test doc 2"}, 0.8),
        ]
        
        results = reranker.rerank("test query", passages, top_k=2)
        
        # Should return original ordering with failure provenance
        assert len(results) == 2
        assert results[0][0].get("reranked") is False
        assert "rerank_error" in results[0][0]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for HF instruments with RAG store."""
    
    def test_rag_store_rerank_disabled(self, temp_kb_dir):
        """Test RAG store works when reranking is disabled."""
        from deepthinker.hf_instruments.config import set_config, HFInstrumentsConfig
        
        # Ensure disabled
        set_config(HFInstrumentsConfig(enabled=False))
        
        from deepthinker.memory.rag_store import MissionRAGStore
        
        # Create store with mock embedding
        store = MissionRAGStore(
            mission_id="test-mission",
            base_dir=temp_kb_dir,
            embedding_fn=lambda x: [0.1] * 384,
        )
        
        # Add documents
        store.add_text("Test document 1", phase="research")
        store.add_text("Test document 2", phase="research")
        
        # Search should work without reranking
        results = store.search("test query", top_k=2, enable_rerank=False)
        
        assert len(results) == 2
    
    def test_extract_and_store_claims(self, temp_kb_dir):
        """Test the full extract and store flow."""
        from deepthinker.claims.store import get_claim_store, extract_and_store_claims
        
        # Override the store's base_dir
        with patch('deepthinker.claims.store._store', None):
            original_get_store = get_claim_store
            
            def mock_get_store(base_dir=None):
                from deepthinker.claims.store import ClaimStore
                return ClaimStore(base_dir=temp_kb_dir / "claims")
            
            with patch('deepthinker.claims.store.get_claim_store', mock_get_store):
                result = extract_and_store_claims(
                    text="The economy grew by 3% in 2024. This is significant.",
                    mission_id="integration-test",
                    source_type="test",
                    mode="regex"
                )
                
                assert result.claim_count >= 0
                assert result.extractor_mode == "regex"


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for the health check CLI."""
    
    def test_health_check_import(self):
        """Test that health check module can be imported."""
        from deepthinker.hf_instruments import __main__ as health_check
        
        assert hasattr(health_check, 'main')
    
    def test_health_check_functions(self):
        """Test health check helper functions."""
        from deepthinker.hf_instruments.__main__ import print_header, print_status
        
        # These should not raise
        print_header("Test Header")
        print_status("Test", "ok", "details")
        print_status("Test", "error")
        print_status("Test", "warning", "warn details")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

