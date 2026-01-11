"""
Tests for Constitution Blinding Module.

Tests that sanitize_for_judge properly removes routing/model identifiers
from judge inputs while preserving meaningful content.
"""

import pytest
from deepthinker.constitution.blinding import (
    sanitize_for_judge,
    sanitize_metadata,
    sanitize_evidence,
    create_blinded_judge_input,
    is_identifier_present,
)
from deepthinker.constitution.config import ConstitutionConfig, ConstitutionMode


class TestSanitizeForJudge:
    """Tests for the sanitize_for_judge function."""
    
    def test_removes_model_identifiers(self):
        """Test that model identifiers are removed."""
        text = "The output was generated using llama3.2:7b model."
        result = sanitize_for_judge(text)
        
        assert "llama3" not in result.lower()
        assert "[model]" in result
    
    def test_removes_ollama_models(self):
        """Test removal of various Ollama model names."""
        models = ["llama3", "gemma2", "qwen2.5", "phi3", "mistral", "cogito:14b"]
        
        for model in models:
            text = f"Using {model} for this task."
            result = sanitize_for_judge(text)
            assert model.split(":")[0].lower() not in result.lower(), f"Failed for {model}"
    
    def test_removes_openai_models(self):
        """Test removal of OpenAI model names."""
        text = "Comparing results from gpt-4 and gpt-4o models."
        result = sanitize_for_judge(text)
        
        assert "gpt-4" not in result.lower()
    
    def test_removes_anthropic_models(self):
        """Test removal of Anthropic model names."""
        text = "Using claude-3.5-sonnet for evaluation."
        result = sanitize_for_judge(text)
        
        assert "claude" not in result.lower()
    
    def test_removes_council_identifiers(self):
        """Test that council identifiers are removed."""
        text = "The research_council produced this output."
        result = sanitize_for_judge(text)
        
        assert "research_council" not in result.lower()
        assert "[council]" in result
    
    def test_removes_model_tier_references(self):
        """Test removal of model tier references."""
        text = "Selected MEDIUM tier for this phase."
        result = sanitize_for_judge(text)
        
        # Should be removed or replaced
        assert "[model]" in result
    
    def test_removes_routing_identifiers(self):
        """Test removal of routing/bandit identifiers."""
        text = "Bandit selected arm: LARGE with thompson sampling."
        result = sanitize_for_judge(text)
        
        assert "bandit" not in result.lower()
        assert "thompson" not in result.lower()
    
    def test_preserves_content_meaning(self):
        """Test that meaningful content is preserved."""
        text = "The analysis shows that climate change is accelerating."
        result = sanitize_for_judge(text)
        
        assert "analysis" in result
        assert "climate change" in result
        assert "accelerating" in result
    
    def test_handles_empty_string(self):
        """Test handling of empty string."""
        assert sanitize_for_judge("") == ""
        assert sanitize_for_judge(None) is None
    
    def test_blinding_disabled(self):
        """Test that blinding can be disabled via config."""
        config = ConstitutionConfig(blinding_enabled=False)
        text = "Using llama3.2:7b model."
        result = sanitize_for_judge(text, config)
        
        # Should be unchanged when disabled
        assert result == text
    
    def test_model_patterns(self):
        """Test regex pattern matching for models."""
        # Model with size suffix
        text = "model: cogito:14b selected"
        result = sanitize_for_judge(text)
        assert "cogito" not in result.lower()
        
        # Model reference pattern
        text = "using llama3.2 model for inference"
        result = sanitize_for_judge(text)
        assert "llama" not in result.lower()


class TestSanitizeMetadata:
    """Tests for the sanitize_metadata function."""
    
    def test_removes_model_fields(self):
        """Test that model-related fields are removed."""
        metadata = {
            "models_used": ["llama3", "gemma2"],
            "model_tier": "LARGE",
            "phase_name": "research",
            "objective": "Test objective",
        }
        
        result = sanitize_metadata(metadata)
        
        assert "models_used" not in result
        assert "model_tier" not in result
        assert "phase_name" in result
        assert "objective" in result
    
    def test_removes_council_fields(self):
        """Test that council-related fields are removed."""
        metadata = {
            "councils_used": ["research_council", "evaluator_council"],
            "council_set": "deep",
            "mission_id": "test-123",
        }
        
        result = sanitize_metadata(metadata)
        
        assert "councils_used" not in result
        assert "council_set" not in result
    
    def test_removes_routing_fields(self):
        """Test that routing-related fields are removed."""
        metadata = {
            "routing_decision": {"tier": "LARGE"},
            "bandit_arm": "MEDIUM",
            "arm_selected": "SMALL",
        }
        
        result = sanitize_metadata(metadata)
        
        assert "routing_decision" not in result
        assert "bandit_arm" not in result
        assert "arm_selected" not in result
    
    def test_sanitizes_nested_dicts(self):
        """Test sanitization of nested dictionaries."""
        metadata = {
            "details": {
                "model": "llama3",
                "content": "Important info",
            }
        }
        
        result = sanitize_metadata(metadata)
        
        # Nested model field should be removed
        assert "llama3" not in str(result)


class TestSanitizeEvidence:
    """Tests for the sanitize_evidence function."""
    
    def test_keeps_allowed_fields(self):
        """Test that allowed fields are preserved."""
        evidence = [{
            "evidence_type": "web_search",
            "content_excerpt": "Climate data shows...",
            "confidence": 0.85,
            "source": "https://example.com",
            "internal_id": "ev_123",
            "raw_ref": "raw data...",
        }]
        
        result = sanitize_evidence(evidence)
        
        assert len(result) == 1
        assert result[0]["evidence_type"] == "web_search"
        assert "Climate data" in result[0]["content_excerpt"]
        assert result[0]["confidence"] == 0.85
        assert "internal_id" not in result[0]
        assert "raw_ref" not in result[0]
    
    def test_sanitizes_content_excerpts(self):
        """Test that model refs in content are sanitized."""
        evidence = [{
            "evidence_type": "code_output",
            "content_excerpt": "Using llama3 to process...",
            "confidence": 0.9,
        }]
        
        result = sanitize_evidence(evidence)
        
        assert "llama3" not in result[0]["content_excerpt"].lower()


class TestCreateBlindedJudgeInput:
    """Tests for the create_blinded_judge_input function."""
    
    def test_creates_blinded_input(self):
        """Test creation of complete blinded input."""
        result = create_blinded_judge_input(
            objective="Analyze climate data using llama3",
            output="The research_council found that...",
            phase_name="research",
        )
        
        assert "llama3" not in result["objective"].lower()
        assert "research_council" not in result["output"].lower()
        assert result["phase_name"] == "research"  # Phase name preserved
    
    def test_includes_evidence(self):
        """Test that evidence is included and sanitized."""
        result = create_blinded_judge_input(
            objective="Test",
            output="Test output",
            phase_name="test",
            evidence=[{
                "evidence_type": "web_search",
                "content_excerpt": "Data...",
                "confidence": 0.8,
            }],
        )
        
        assert "evidence" in result
        assert len(result["evidence"]) == 1


class TestIsIdentifierPresent:
    """Tests for the is_identifier_present helper."""
    
    def test_detects_model_identifiers(self):
        """Test detection of model identifiers."""
        assert is_identifier_present("Using llama3 model")
        assert is_identifier_present("gpt-4 response")
        assert is_identifier_present("claude-3 output")
    
    def test_detects_council_identifiers(self):
        """Test detection of council identifiers."""
        assert is_identifier_present("research_council output")
        assert is_identifier_present("evaluator_council analysis")
    
    def test_no_false_positives_on_clean_text(self):
        """Test that clean text is not flagged."""
        clean_text = "The analysis shows climate trends increasing."
        assert not is_identifier_present(clean_text)
    
    def test_sanitized_text_is_clean(self):
        """Test that sanitized text passes the check."""
        original = "Using llama3 model from research_council"
        sanitized = sanitize_for_judge(original)
        assert not is_identifier_present(sanitized)




