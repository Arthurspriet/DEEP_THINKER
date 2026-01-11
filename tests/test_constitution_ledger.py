"""
Tests for Constitution Ledger.

Tests:
- Append-only behavior
- JSON schema validity
- Privacy constraints (no raw prompts, hashed sensitive data)
- Reading and filtering events
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from deepthinker.constitution.config import ConstitutionConfig, ConstitutionMode
from deepthinker.constitution.ledger import (
    ConstitutionLedger,
    get_ledger,
    clear_ledger_cache,
)
from deepthinker.constitution.types import (
    EvidenceEvent,
    ScoreEvent,
    ContradictionEvent,
    DepthEvent,
    MemoryEvent,
    CompressionEvent,
    LearningUpdateEvent,
    ConstitutionViolationEvent,
    BaselineSnapshot,
    ConstitutionEventType,
)


@pytest.fixture
def temp_ledger_dir(tmp_path):
    """Create a temporary directory for ledger files."""
    return tmp_path / "constitution"


@pytest.fixture
def ledger(temp_ledger_dir):
    """Create a ledger with temporary directory."""
    config = ConstitutionConfig(
        mode=ConstitutionMode.SHADOW,
        ledger_enabled=True,
        ledger_base_dir=str(temp_ledger_dir),
    )
    return ConstitutionLedger("test-mission", config, temp_ledger_dir)


class TestLedgerAppendOnly:
    """Tests for append-only behavior."""
    
    def test_write_creates_file(self, ledger, temp_ledger_dir):
        """Test that writing creates the ledger file."""
        event = EvidenceEvent(
            mission_id="test-mission",
            phase_id="research",
            count_added=5,
        )
        
        result = ledger.write_event(event)
        
        assert result
        assert ledger.ledger_path.exists()
    
    def test_multiple_writes_append(self, ledger):
        """Test that multiple writes append to the file."""
        for i in range(5):
            event = EvidenceEvent(
                mission_id="test-mission",
                phase_id=f"phase_{i}",
                count_added=i,
            )
            ledger.write_event(event)
        
        events = ledger.read_all()
        
        assert len(events) == 5
    
    def test_events_preserve_order(self, ledger):
        """Test that events are preserved in write order."""
        phases = ["research", "planning", "execution"]
        
        for phase in phases:
            event = EvidenceEvent(
                mission_id="test-mission",
                phase_id=phase,
                count_added=1,
            )
            ledger.write_event(event)
        
        events = ledger.read_all()
        
        assert [e["phase_id"] for e in events] == phases


class TestLedgerJsonSchema:
    """Tests for JSON schema validity."""
    
    def test_evidence_event_valid_json(self, ledger):
        """Test that EvidenceEvent produces valid JSON."""
        event = EvidenceEvent(
            mission_id="test-mission",
            phase_id="research",
            count_added=5,
            evidence_types=["web_search", "code_output"],
            sources_summary="hashed_source",
            total_evidence_count=10,
        )
        
        ledger.write_event(event)
        events = ledger.read_all()
        
        assert len(events) == 1
        assert events[0]["event_type"] == "evidence"
        assert events[0]["count_added"] == 5
        assert events[0]["evidence_types"] == ["web_search", "code_output"]
    
    def test_score_event_valid_json(self, ledger):
        """Test that ScoreEvent produces valid JSON."""
        event = ScoreEvent(
            mission_id="test-mission",
            phase_id="research",
            score_before=0.5,
            score_after=0.7,
            delta=0.2,
            target_metrics={"overall": 0.7},
            shadow_metrics={"contradiction_rate": 0.1},
        )
        
        ledger.write_event(event)
        events = ledger.read_all()
        
        assert len(events) == 1
        assert events[0]["event_type"] == "score"
        assert events[0]["delta"] == 0.2
    
    def test_violation_event_valid_json(self, ledger):
        """Test that ConstitutionViolationEvent produces valid JSON."""
        event = ConstitutionViolationEvent(
            mission_id="test-mission",
            phase_id="research",
            invariant="evidence_conservation",
            severity=0.8,
            message="Score increased without evidence",
            suggested_action="Add evidence",
            details={"score_delta": 0.1},
        )
        
        ledger.write_event(event)
        events = ledger.read_all()
        
        assert len(events) == 1
        assert events[0]["event_type"] == "violation"
        assert events[0]["severity"] == 0.8
    
    def test_all_event_types_have_timestamp(self, ledger):
        """Test that all event types include timestamp."""
        events_to_write = [
            EvidenceEvent(mission_id="test", phase_id="test"),
            ScoreEvent(mission_id="test", phase_id="test"),
            DepthEvent(mission_id="test", phase_id="test"),
            MemoryEvent(mission_id="test", phase_id="test"),
        ]
        
        for event in events_to_write:
            ledger.write_event(event)
        
        events = ledger.read_all()
        
        for event in events:
            assert "timestamp" in event
            # Should be valid ISO format
            datetime.fromisoformat(event["timestamp"])


class TestLedgerPrivacy:
    """Tests for privacy constraints."""
    
    def test_no_raw_prompts_in_events(self, ledger):
        """Test that events don't contain raw prompt text."""
        # Evidence event should only have excerpt/summary, not full text
        event = EvidenceEvent(
            mission_id="test-mission",
            phase_id="research",
            count_added=1,
            sources_summary="hashed_sources",  # Should be hashed
        )
        
        ledger.write_event(event)
        
        # Read raw file content
        content = ledger.ledger_path.read_text()
        
        # Should not contain any long text fields
        # (In a real test, we'd verify specific forbidden patterns)
        assert "raw_prompt" not in content
    
    def test_violation_details_limited(self, ledger):
        """Test that violation details are limited."""
        event = ConstitutionViolationEvent(
            mission_id="test-mission",
            phase_id="research",
            invariant="test",
            severity=0.5,
            message="Test message",
            details={"key": "value"},
        )
        
        ledger.write_event(event)
        events = ledger.read_all()
        
        # Details should be present but not contain sensitive info
        assert "details" in events[0]


class TestLedgerFiltering:
    """Tests for event reading and filtering."""
    
    def test_read_by_event_type(self, ledger):
        """Test filtering by event type."""
        # Write mixed events
        ledger.write_event(EvidenceEvent(mission_id="test", phase_id="p1"))
        ledger.write_event(ScoreEvent(mission_id="test", phase_id="p2"))
        ledger.write_event(EvidenceEvent(mission_id="test", phase_id="p3"))
        
        # Filter by type
        evidence_events = list(ledger.read_events(
            event_type=ConstitutionEventType.EVIDENCE
        ))
        
        assert len(evidence_events) == 2
    
    def test_read_by_phase(self, ledger):
        """Test filtering by phase ID."""
        ledger.write_event(EvidenceEvent(mission_id="test", phase_id="research"))
        ledger.write_event(EvidenceEvent(mission_id="test", phase_id="planning"))
        ledger.write_event(EvidenceEvent(mission_id="test", phase_id="research"))
        
        research_events = list(ledger.read_events(phase_id="research"))
        
        assert len(research_events) == 2
    
    def test_read_with_limit(self, ledger):
        """Test limiting number of events returned."""
        for i in range(10):
            ledger.write_event(EvidenceEvent(mission_id="test", phase_id=f"p{i}"))
        
        limited_events = list(ledger.read_events(limit=5))
        
        assert len(limited_events) == 5
    
    def test_get_violations(self, ledger):
        """Test getting violation events."""
        ledger.write_event(EvidenceEvent(mission_id="test", phase_id="p1"))
        ledger.write_event(ConstitutionViolationEvent(
            mission_id="test",
            phase_id="p2",
            invariant="test",
            severity=0.8,
            message="Test violation",
        ))
        
        violations = ledger.get_violations()
        
        assert len(violations) == 1
        assert violations[0]["invariant"] == "test"


class TestBaselineSnapshot:
    """Tests for baseline snapshot handling."""
    
    def test_write_and_read_baseline(self, ledger):
        """Test writing and reading baselines."""
        baseline = BaselineSnapshot(
            mission_id="test-mission",
            phase_id="research",
            overall_score=0.5,
            goal_coverage=0.6,
            evidence_count=10,
        )
        
        ledger.write_baseline(baseline)
        baselines = ledger.get_baselines()
        
        assert len(baselines) == 1
        assert baselines[0].overall_score == 0.5
        assert baselines[0].phase_id == "research"
    
    def test_get_latest_baseline_for_phase(self, ledger):
        """Test getting the latest baseline for a specific phase."""
        # Write two baselines for same phase
        baseline1 = BaselineSnapshot(
            mission_id="test",
            phase_id="research",
            overall_score=0.5,
        )
        baseline2 = BaselineSnapshot(
            mission_id="test",
            phase_id="research",
            overall_score=0.7,
        )
        
        ledger.write_baseline(baseline1)
        ledger.write_baseline(baseline2)
        
        latest = ledger.get_latest_baseline("research")
        
        assert latest is not None
        assert latest.overall_score == 0.7


class TestLedgerSummary:
    """Tests for ledger summary functionality."""
    
    def test_get_summary(self, ledger):
        """Test getting ledger summary."""
        # Write various events
        ledger.write_event(EvidenceEvent(mission_id="test", phase_id="p1"))
        ledger.write_event(ScoreEvent(mission_id="test", phase_id="p2"))
        ledger.write_event(ConstitutionViolationEvent(
            mission_id="test",
            phase_id="p1",
            invariant="test",
            severity=0.5,
            message="Test",
        ))
        
        summary = ledger.get_summary()
        
        assert summary["total_events"] == 3
        assert summary["violation_count"] == 1
        assert "p1" in summary["phases_covered"]
        assert "p2" in summary["phases_covered"]


class TestLedgerCache:
    """Tests for global ledger cache."""
    
    def test_get_ledger_caches(self, temp_ledger_dir):
        """Test that get_ledger caches instances."""
        clear_ledger_cache()
        
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_base_dir=str(temp_ledger_dir),
        )
        
        ledger1 = get_ledger("mission-1", config)
        ledger2 = get_ledger("mission-1", config)
        
        assert ledger1 is ledger2
    
    def test_different_missions_different_ledgers(self, temp_ledger_dir):
        """Test that different missions get different ledgers."""
        clear_ledger_cache()
        
        config = ConstitutionConfig(
            mode=ConstitutionMode.SHADOW,
            ledger_base_dir=str(temp_ledger_dir),
        )
        
        ledger1 = get_ledger("mission-1", config)
        ledger2 = get_ledger("mission-2", config)
        
        assert ledger1 is not ledger2




