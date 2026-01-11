"""
Tests for arXiv Connector.

Tests cover:
- Search result parsing
- Get by ID
- Download with caching
- Rate limiting
- Feature flag behavior
- Evidence object creation
"""

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# Sample arXiv Atom XML response for testing
SAMPLE_ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: cat:cs.CL AND ti:alignment</title>
  <id>http://arxiv.org/api/query</id>
  <updated>2025-01-11T00:00:00Z</updated>
  <totalResults>2</totalResults>
  <startIndex>0</startIndex>
  <itemsPerPage>10</itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2501.01234v1</id>
    <updated>2025-01-10T12:00:00Z</updated>
    <published>2025-01-05T12:00:00Z</published>
    <title>Test Paper on AI Alignment</title>
    <summary>This is a test abstract about AI alignment research.</summary>
    <author>
      <name>Jane Doe</name>
    </author>
    <author>
      <name>John Smith</name>
    </author>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2501.01234v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2501.01234v1.pdf" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.05678v2</id>
    <updated>2025-01-08T12:00:00Z</updated>
    <published>2025-01-01T12:00:00Z</published>
    <title>Another Test Paper on Language Models</title>
    <summary>This is another test abstract about language model alignment.</summary>
    <author>
      <name>Alice Johnson</name>
    </author>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2501.05678v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2501.05678v2.pdf" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""

SAMPLE_PDF_CONTENT = b"%PDF-1.4 fake pdf content for testing"


class TestArxivConfig:
    """Tests for ArxivConfig."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        config = ArxivConfig()
        
        assert config.enabled is False
        assert config.ingest_enabled is False
        assert config.api_interval_sec == 3.0
        assert config.dl_interval_sec == 10.0
        assert config.cache_dir == "kb/arxiv/cache"
        assert config.max_results == 50
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        from deepthinker.connectors.arxiv.config import ArxivConfig, reset_arxiv_config
        
        reset_arxiv_config()
        
        with patch.dict(os.environ, {
            "DEEPTHINKER_ARXIV_ENABLED": "true",
            "DEEPTHINKER_ARXIV_INGEST_ENABLED": "true",
            "DEEPTHINKER_ARXIV_API_INTERVAL_SEC": "5",
            "DEEPTHINKER_ARXIV_CACHE_DIR": "/tmp/test_cache",
        }):
            config = ArxivConfig.from_env()
            
            assert config.enabled is True
            assert config.ingest_enabled is True
            assert config.api_interval_sec == 5.0
            assert config.cache_dir == "/tmp/test_cache"
        
        reset_arxiv_config()
    
    def test_is_enabled_property(self):
        """Test is_enabled property."""
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        config = ArxivConfig(enabled=False)
        assert config.is_enabled is False
        
        config = ArxivConfig(enabled=True)
        assert config.is_enabled is True


class TestArxivModels:
    """Tests for ArxivPaper and ArxivEvidence models."""
    
    def test_arxiv_paper_creation(self):
        """Test ArxivPaper dataclass."""
        from deepthinker.connectors.arxiv.models import ArxivPaper
        
        paper = ArxivPaper(
            id="2501.01234",
            version="v1",
            title="Test Paper",
            authors=["Jane Doe", "John Smith"],
            abstract="Test abstract",
            categories=["cs.CL", "cs.AI"],
            primary_category="cs.CL",
        )
        
        assert paper.arxiv_id == "2501.01234v1"
        assert paper.abs_url == "https://arxiv.org/abs/2501.01234v1"
        assert paper.pdf_url == "https://arxiv.org/pdf/2501.01234v1.pdf"
    
    def test_arxiv_paper_to_dict(self):
        """Test ArxivPaper serialization."""
        from deepthinker.connectors.arxiv.models import ArxivPaper
        
        paper = ArxivPaper(
            id="2501.01234",
            title="Test Paper",
            authors=["Jane Doe"],
        )
        
        data = paper.to_dict()
        
        assert data["id"] == "2501.01234"
        assert data["title"] == "Test Paper"
        assert data["authors"] == ["Jane Doe"]
    
    def test_arxiv_evidence_creation(self):
        """Test ArxivEvidence dataclass."""
        from deepthinker.connectors.arxiv.models import ArxivEvidence
        
        evidence = ArxivEvidence(
            arxiv_id="2501.01234",
            version="v1",
            request_url="https://export.arxiv.org/api/query?id_list=2501.01234",
            content_type="metadata",
        )
        
        assert evidence.source == "arxiv"
        assert evidence.arxiv_id == "2501.01234"
        assert evidence.evidence_id.startswith("arxiv_ev_")
    
    def test_arxiv_evidence_factory_methods(self):
        """Test ArxivEvidence factory methods."""
        from deepthinker.connectors.arxiv.models import ArxivEvidence
        
        # Test for_search
        search_ev = ArxivEvidence.for_search(
            request_url="https://export.arxiv.org/api/query?search_query=test",
            result_count=10,
            query="test",
        )
        assert search_ev.content_type == "metadata"
        assert search_ev.metadata["operation"] == "search"
        assert search_ev.metadata["result_count"] == 10
        
        # Test for_paper
        paper_ev = ArxivEvidence.for_paper(
            arxiv_id="2501.01234",
            version="v1",
            request_url="https://export.arxiv.org/api/query?id_list=2501.01234",
            title="Test Paper",
        )
        assert paper_ev.arxiv_id == "2501.01234"
        assert paper_ev.metadata["operation"] == "get"
        
        # Test for_download
        dl_ev = ArxivEvidence.for_download(
            arxiv_id="2501.01234",
            version="v1",
            request_url="https://arxiv.org/pdf/2501.01234v1.pdf",
            local_path="/tmp/test.pdf",
            sha256="abc123",
            content_type="pdf",
        )
        assert dl_ev.content_type == "pdf"
        assert dl_ev.local_path == "/tmp/test.pdf"
        assert dl_ev.sha256 == "abc123"


class TestArxivCache:
    """Tests for ArxivCache."""
    
    def test_cache_put_and_get(self):
        """Test caching content."""
        from deepthinker.connectors.arxiv.cache import ArxivCache
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(cache_dir=tmpdir)
            cache = ArxivCache(config)
            
            content = b"test pdf content"
            metadata = cache.put(
                arxiv_id="2501.01234",
                kind="pdf",
                version="v1",
                content=content,
                request_url="https://arxiv.org/pdf/2501.01234v1.pdf",
            )
            
            assert metadata is not None
            assert metadata["sha256"] == hashlib.sha256(content).hexdigest()
            
            # Test get
            cached = cache.get("2501.01234", "pdf", "v1")
            assert cached is not None
            assert cached["sha256"] == metadata["sha256"]
    
    def test_cache_deduplication(self):
        """Test SHA256 deduplication."""
        from deepthinker.connectors.arxiv.cache import ArxivCache
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(cache_dir=tmpdir)
            cache = ArxivCache(config)
            
            content = b"same content"
            
            # Put same content twice with different IDs
            meta1 = cache.put(
                arxiv_id="2501.01234",
                kind="pdf",
                version="v1",
                content=content,
                request_url="url1",
            )
            
            meta2 = cache.put(
                arxiv_id="2501.05678",
                kind="pdf",
                version="v1",
                content=content,
                request_url="url2",
            )
            
            # Both should have same SHA256
            assert meta1["sha256"] == meta2["sha256"]
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        from deepthinker.connectors.arxiv.cache import ArxivCache
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(cache_dir=tmpdir)
            cache = ArxivCache(config)
            
            cached = cache.get("nonexistent", "pdf", "v1")
            assert cached is None


class TestArxivClient:
    """Tests for ArxivClient."""
    
    def test_search_parsing(self):
        """Test parsing of search results."""
        from deepthinker.connectors.arxiv.client import ArxivClient
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(enabled=True, cache_dir=tmpdir, api_interval_sec=0)
            
            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.text = SAMPLE_ARXIV_RESPONSE
                mock_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_response
                
                client = ArxivClient(config)
                papers, evidence = client.search("cat:cs.CL AND ti:alignment")
                
                assert len(papers) == 2
                assert papers[0].id == "2501.01234"
                assert papers[0].version == "v1"
                assert papers[0].title == "Test Paper on AI Alignment"
                assert papers[0].authors == ["Jane Doe", "John Smith"]
                assert "cs.CL" in papers[0].categories
                
                assert papers[1].id == "2501.05678"
                assert papers[1].version == "v2"
                
                assert evidence.content_type == "metadata"
                assert evidence.metadata["result_count"] == 2
    
    def test_get_by_id(self):
        """Test fetching single paper by ID."""
        from deepthinker.connectors.arxiv.client import ArxivClient
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        single_paper_response = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.01234v1</id>
    <title>Single Test Paper</title>
    <summary>Test abstract</summary>
    <author><name>Test Author</name></author>
    <category term="cs.CL"/>
  </entry>
</feed>
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(enabled=True, cache_dir=tmpdir, api_interval_sec=0)
            
            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.text = single_paper_response
                mock_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_response
                
                client = ArxivClient(config)
                paper, evidence = client.get_by_id("2501.01234v1")
                
                assert paper is not None
                assert paper.title == "Single Test Paper"
                assert evidence.arxiv_id == "2501.01234"
    
    def test_download_pdf_caching(self):
        """Test PDF download with caching."""
        from deepthinker.connectors.arxiv.client import ArxivClient
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(
                enabled=True, 
                cache_dir=tmpdir, 
                api_interval_sec=0,
                dl_interval_sec=0,
            )
            
            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.content = SAMPLE_PDF_CONTENT
                mock_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_response
                
                client = ArxivClient(config)
                
                # First download
                path1, sha1, ev1 = client.download_pdf("2501.01234")
                assert Path(path1).exists()
                assert sha1 == hashlib.sha256(SAMPLE_PDF_CONTENT).hexdigest()
                
                # Second download should hit cache (mock should not be called again)
                mock_get.reset_mock()
                path2, sha2, ev2 = client.download_pdf("2501.01234")
                
                assert path1 == path2
                assert sha1 == sha2
                assert ev2.metadata.get("cache_hit") is True
                # Request should not have been made
                mock_get.assert_not_called()
    
    def test_parse_arxiv_id(self):
        """Test arXiv ID parsing."""
        from deepthinker.connectors.arxiv.client import ArxivClient
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        config = ArxivConfig(enabled=True)
        client = ArxivClient(config)
        
        # Test various ID formats
        assert client._parse_arxiv_id("2501.01234") == ("2501.01234", None)
        assert client._parse_arxiv_id("2501.01234v1") == ("2501.01234", "v1")
        assert client._parse_arxiv_id("2501.01234v12") == ("2501.01234", "v12")
        assert client._parse_arxiv_id("hep-th/9901001") == ("hep-th/9901001", None)
        assert client._parse_arxiv_id("hep-th/9901001v2") == ("hep-th/9901001", "v2")


class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limiter_basic(self):
        """Test basic rate limiting."""
        from deepthinker.connectors.arxiv.client import RateLimiter
        
        limiter = RateLimiter(min_interval_sec=0.1)
        
        # First call should not wait
        wait1 = limiter.wait()
        assert wait1 == 0 or wait1 < 0.01
        
        # Second call immediately after should wait
        start = time.time()
        wait2 = limiter.wait()
        elapsed = time.time() - start
        
        assert elapsed >= 0.09  # Allow small tolerance
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset."""
        from deepthinker.connectors.arxiv.client import RateLimiter
        
        limiter = RateLimiter(min_interval_sec=10.0)
        
        # First call
        limiter.wait()
        
        # Reset
        limiter.reset()
        
        # Should not wait after reset
        wait = limiter.wait()
        assert wait == 0 or wait < 0.01


class TestToolFunctions:
    """Tests for tool functions."""
    
    def test_arxiv_search_disabled(self):
        """Test search when disabled."""
        from deepthinker.connectors.arxiv.tool import arxiv_search
        from deepthinker.connectors.arxiv.config import reset_arxiv_config
        
        reset_arxiv_config()
        
        with patch.dict(os.environ, {"DEEPTHINKER_ARXIV_ENABLED": "false"}):
            result = arxiv_search("test query")
            
            assert result["error"] is not None
            assert "disabled" in result["error"].lower()
            assert result["papers"] == []
        
        reset_arxiv_config()
    
    def test_arxiv_get_disabled(self):
        """Test get when disabled."""
        from deepthinker.connectors.arxiv.tool import arxiv_get
        from deepthinker.connectors.arxiv.config import reset_arxiv_config
        
        reset_arxiv_config()
        
        with patch.dict(os.environ, {"DEEPTHINKER_ARXIV_ENABLED": "false"}):
            result = arxiv_get("2501.01234")
            
            assert result["error"] is not None
            assert result["paper"] is None
            assert result["found"] is False
        
        reset_arxiv_config()
    
    def test_arxiv_download_invalid_kind(self):
        """Test download with invalid kind."""
        from deepthinker.connectors.arxiv.tool import arxiv_download
        
        result = arxiv_download("2501.01234", kind="invalid")
        
        assert result["error"] is not None
        assert "invalid kind" in result["error"].lower()


class TestSearchTriggers:
    """Tests for arXiv search trigger detection."""
    
    def test_should_use_arxiv_keywords(self):
        """Test keyword detection for arXiv."""
        from deepthinker.tools.search_triggers import should_use_arxiv
        from deepthinker.connectors.arxiv.config import reset_arxiv_config
        
        reset_arxiv_config()
        
        with patch.dict(os.environ, {"DEEPTHINKER_ARXIV_ENABLED": "true"}):
            # Reset config to pick up env var
            reset_arxiv_config()
            
            # Should trigger
            assert should_use_arxiv("Find arxiv papers on alignment") is True
            assert should_use_arxiv("Get the latest research papers") is True
            assert should_use_arxiv("Review academic paper on transformers") is True
            assert should_use_arxiv("I need citations for my work") is True
            
            # Should not trigger
            assert should_use_arxiv("What is the weather today?") is False
            assert should_use_arxiv("Write some Python code") is False
        
        reset_arxiv_config()
    
    def test_should_use_arxiv_disabled(self):
        """Test keyword detection when disabled."""
        from deepthinker.tools.search_triggers import should_use_arxiv
        from deepthinker.connectors.arxiv.config import reset_arxiv_config
        
        reset_arxiv_config()
        
        with patch.dict(os.environ, {"DEEPTHINKER_ARXIV_ENABLED": "false"}):
            reset_arxiv_config()
            
            # Should return False even with keywords when disabled
            assert should_use_arxiv("Find arxiv papers") is False
        
        reset_arxiv_config()
    
    def test_get_arxiv_search_queries(self):
        """Test query generation."""
        from deepthinker.tools.search_triggers import get_arxiv_search_queries
        
        queries = get_arxiv_search_queries(
            objective="Find papers on transformer architectures",
            data_needs=["attention mechanism details"],
            focus_areas=["machine learning"],
        )
        
        assert len(queries) > 0
        assert len(queries) <= 3


class TestEvidenceObjectIntegration:
    """Tests for EvidenceObject arXiv integration."""
    
    def test_arxiv_evidence_type_exists(self):
        """Test ARXIV evidence type exists."""
        from deepthinker.metrics.evidence_object import EvidenceType
        
        assert hasattr(EvidenceType, "ARXIV")
        assert EvidenceType.ARXIV.value == "arxiv"
    
    def test_from_arxiv_factory(self):
        """Test from_arxiv factory method."""
        from deepthinker.metrics.evidence_object import EvidenceObject, EvidenceType
        
        evidence = EvidenceObject.from_arxiv(
            arxiv_id="2501.01234",
            title="Test Paper on AI",
            authors=["Jane Doe", "John Smith", "Alice", "Bob"],
            abstract="This is a test abstract.",
            content_type="metadata",
        )
        
        assert evidence.evidence_type == EvidenceType.ARXIV
        assert "2501.01234" in evidence.source
        assert evidence.metadata["arxiv_id"] == "2501.01234"
        assert evidence.confidence == 0.85  # metadata confidence
    
    def test_from_arxiv_pdf_confidence(self):
        """Test PDF download has higher confidence."""
        from deepthinker.metrics.evidence_object import EvidenceObject
        
        evidence = EvidenceObject.from_arxiv(
            arxiv_id="2501.01234",
            title="Test Paper",
            authors=["Jane Doe"],
            abstract="Abstract",
            content_type="pdf",
            local_path="/tmp/test.pdf",
            sha256="abc123",
        )
        
        assert evidence.confidence == 0.9  # PDF confidence
        assert evidence.metadata["local_path"] == "/tmp/test.pdf"
        assert evidence.metadata["sha256"] == "abc123"


class TestRoutingFeatures:
    """Tests for routing feature integration."""
    
    def test_task_type_scholarly_detection(self):
        """Test scholarly task type detection."""
        from deepthinker.routing.features import _classify_task_type
        
        assert _classify_task_type("Find arxiv papers on AI") == "scholarly"
        assert _classify_task_type("Literature review of transformers") == "scholarly"
        assert _classify_task_type("Find academic papers") == "scholarly"
        assert _classify_task_type("Get peer-reviewed citations") == "scholarly"
        
        # Should not be scholarly
        assert _classify_task_type("Write Python code") != "scholarly"
        assert _classify_task_type("Analyze the data") != "scholarly"
    
    def test_scholarly_feature_extraction(self):
        """Test scholarly feature in routing features."""
        from deepthinker.routing.features import extract_routing_features, RoutingContext
        
        context = RoutingContext(
            objective="Find arxiv papers on alignment",
            phase_name="research",
        )
        
        features = extract_routing_features(context)
        
        assert "task_type_scholarly" in features
        assert features["task_type_scholarly"] == 1.0
    
    def test_feature_names_include_scholarly(self):
        """Test feature names include scholarly."""
        from deepthinker.routing.features import get_feature_names
        
        names = get_feature_names()
        assert "task_type_scholarly" in names


class TestAuditLogging:
    """Tests for structured audit logging in arXiv tools."""
    
    def test_tool_logging_emits_arxiv_tool_call(self, caplog):
        """Verify arxiv_tool_call event is logged with correct fields."""
        import logging
        from deepthinker.connectors.arxiv.tool import arxiv_search
        from deepthinker.connectors.arxiv.config import reset_arxiv_config
        
        reset_arxiv_config()
        
        # Enable logging capture at INFO level
        caplog.set_level(logging.INFO)
        
        with patch.dict(os.environ, {"DEEPTHINKER_ARXIV_ENABLED": "false"}):
            reset_arxiv_config()
            
            # Call will fail because disabled, but logging should still occur
            result = arxiv_search("test query", max_results=5)
            
            # Check that arxiv_tool_call was logged
            arxiv_logs = [r for r in caplog.records if r.message == "arxiv_tool_call"]
            assert len(arxiv_logs) >= 2, "Expected at least 2 arxiv_tool_call log entries (start and end)"
            
            # Check the first log has expected fields
            first_log = arxiv_logs[0]
            assert hasattr(first_log, 'action')
            assert first_log.action == "search"
            assert hasattr(first_log, 'query')
            assert first_log.query == "test query"
        
        reset_arxiv_config()
    
    def test_download_logging_includes_cache_hit(self, caplog):
        """Verify download logging includes cache_hit field."""
        import logging
        from deepthinker.connectors.arxiv.tool import arxiv_download
        from deepthinker.connectors.arxiv.config import reset_arxiv_config
        
        reset_arxiv_config()
        caplog.set_level(logging.INFO)
        
        with patch.dict(os.environ, {"DEEPTHINKER_ARXIV_ENABLED": "false"}):
            reset_arxiv_config()
            
            result = arxiv_download("2501.01234", kind="pdf")
            
            # Check that download logging occurred
            arxiv_logs = [r for r in caplog.records if r.message == "arxiv_tool_call"]
            assert len(arxiv_logs) >= 2
            
            # Check the end log has cache_hit field
            end_log = arxiv_logs[-1]
            assert hasattr(end_log, 'cache_hit')
            assert end_log.cache_hit is False  # Should be False since it failed
        
        reset_arxiv_config()


class TestDownloadRouting:
    """Tests for metadata-only default routing."""
    
    def test_scholarly_query_routes_to_search_only(self):
        """Scholarly query triggers search but not download."""
        from deepthinker.tools.search_triggers import should_use_arxiv, should_download_arxiv
        from deepthinker.connectors.arxiv.config import reset_arxiv_config
        
        reset_arxiv_config()
        
        with patch.dict(os.environ, {"DEEPTHINKER_ARXIV_ENABLED": "true"}):
            reset_arxiv_config()
            
            # Test scholarly queries - should trigger search but NOT download
            scholarly_queries = [
                "Find arxiv papers on alignment",
                "Search for academic papers on transformers",
                "Get the latest research papers on LLMs",
                "I need citations for my work on neural networks",
            ]
            
            for query in scholarly_queries:
                assert should_use_arxiv(query) is True, f"Expected should_use_arxiv=True for: {query}"
                assert should_download_arxiv(query) is False, f"Expected should_download_arxiv=False for: {query}"
        
        reset_arxiv_config()
    
    def test_explicit_download_intent_routes_to_download(self):
        """Explicit download intent triggers download function."""
        from deepthinker.tools.search_triggers import should_download_arxiv
        
        # Test explicit download intent queries
        download_queries = [
            "download the arxiv pdf for 2501.01234",
            "get the full text of the paper",
            "retrieve paper pdf",
            "I need to read the paper in full",
            "fetch pdf for this arxiv paper",
        ]
        
        for query in download_queries:
            assert should_download_arxiv(query) is True, f"Expected should_download_arxiv=True for: {query}"
    
    def test_download_respects_plan_metadata(self):
        """Download is triggered when plan metadata has requires_full_text=True."""
        from deepthinker.tools.search_triggers import should_download_arxiv
        
        # Without metadata, scholarly query doesn't trigger download
        assert should_download_arxiv("Find papers on AI") is False
        
        # With requires_full_text=True in plan metadata, it should trigger
        plan_metadata = {"requires_full_text": True}
        assert should_download_arxiv("Find papers on AI", plan_metadata=plan_metadata) is True
        
        # With requires_full_text=False, it should not trigger
        plan_metadata = {"requires_full_text": False}
        assert should_download_arxiv("Find papers on AI", plan_metadata=plan_metadata) is False


class TestEvidenceNeutrality:
    """Tests ensuring ARXIV evidence doesn't inflate confidence."""
    
    def test_arxiv_evidence_does_not_inflate_confidence(self):
        """
        ARXIV evidence type must not affect claim confidence scoring.
        
        The confidence estimator should only use:
        - council_agreement
        - memory_presence
        - linguistic_uncertainty
        - claim_type
        
        NOT evidence_type (including EvidenceType.ARXIV).
        """
        from deepthinker.tooling.epistemic.confidence_estimator import ClaimConfidenceTool
        from deepthinker.tooling.schemas import Claim
        
        tool = ClaimConfidenceTool()
        
        # Create a simple factual claim (context is required)
        claim = Claim(
            id="c1",
            text="AI alignment is important",
            context="Discussion of AI safety topics",
            claim_type="factual"
        )
        
        # Estimate confidence without any evidence or memory
        score = tool.estimate_confidence(claim, check_memory=False)
        
        # Verify no evidence type factor in the signals
        # This is the key assertion: EvidenceType (including ARXIV) should not
        # be a factor in confidence scoring
        assert "evidence_type" not in score.signals, (
            "evidence_type should not be in confidence signals - "
            "confidence should be based on council agreement, memory, and linguistics only"
        )
        
        # Verify only expected signals are present
        expected_signals = {"council_agreement", "memory_presence", "linguistic_uncertainty", "claim_type", "type_penalty"}
        actual_signals = set(score.signals.keys())
        assert actual_signals == expected_signals, (
            f"Unexpected signals in confidence score. "
            f"Expected {expected_signals}, got {actual_signals}"
        )
        
        # Verify score is in valid range (actual value depends on claim content)
        assert 0.0 <= score.score <= 1.0, f"Score out of range: {score.score}"


class TestEmptyParseRobustness:
    """Tests for client robustness on empty parse."""
    
    def test_empty_parse_raises_arxiv_parse_error(self, caplog):
        """HTTP 200 with no entries should raise ArxivParseError and log warning."""
        import logging
        from deepthinker.connectors.arxiv.client import ArxivClient, ArxivParseError
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        # Response with valid XML but no entries
        empty_response = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <id>http://arxiv.org/api/query</id>
  <totalResults>0</totalResults>
</feed>
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(enabled=True, cache_dir=tmpdir, api_interval_sec=0)
            
            caplog.set_level(logging.WARNING)
            
            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.text = empty_response
                mock_response.status_code = 200
                mock_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_response
                
                client = ArxivClient(config)
                
                # Should raise ArxivParseError
                with pytest.raises(ArxivParseError) as exc_info:
                    client.search("nonexistent_query_xyz")
                
                assert "empty results" in str(exc_info.value).lower()
                
                # Check warning was logged
                warning_logs = [
                    r for r in caplog.records 
                    if r.message == "arxiv_parse_empty" and r.levelno == logging.WARNING
                ]
                assert len(warning_logs) >= 1, "Expected arxiv_parse_empty warning to be logged"
    
    def test_get_by_id_empty_parse_returns_none_not_raises(self, caplog):
        """get_by_id with empty parse should return None, not raise."""
        import logging
        from deepthinker.connectors.arxiv.client import ArxivClient
        from deepthinker.connectors.arxiv.config import ArxivConfig
        
        empty_response = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <id>http://arxiv.org/api/query</id>
</feed>
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ArxivConfig(enabled=True, cache_dir=tmpdir, api_interval_sec=0)
            
            caplog.set_level(logging.WARNING)
            
            with patch("requests.Session.get") as mock_get:
                mock_response = MagicMock()
                mock_response.text = empty_response
                mock_response.status_code = 200
                mock_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_response
                
                client = ArxivClient(config)
                
                # Should NOT raise, should return (None, evidence)
                paper, evidence = client.get_by_id("nonexistent.99999")
                
                assert paper is None
                assert evidence is not None
                
                # Check warning was logged
                warning_logs = [
                    r for r in caplog.records 
                    if r.message == "arxiv_parse_empty"
                ]
                assert len(warning_logs) >= 1, "Expected arxiv_parse_empty warning for empty get_by_id"


