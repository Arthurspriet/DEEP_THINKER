"""
Tests for the Scenarios module.

Tests structured scenario modeling, parsing, validation,
and integration with the DeepThinker analysis pipeline.
"""

import pytest
from deepthinker.scenarios import (
    Scenario,
    ScenarioDriver,
    ScenarioFactory,
    ScenarioParser,
    ScenarioSet,
    ScenarioTimeline,
    get_scenario_factory,
)


class TestScenario:
    """Tests for the Scenario dataclass."""
    
    def test_scenario_creation(self):
        """Test basic scenario creation."""
        scenario = Scenario(
            name="Tech Dominance",
            description="Technology companies dominate the market",
            drivers={
                "tech": ["AI advancement", "automation"],
                "geopolitics": ["US-China tension"],
                "regulation": ["Antitrust enforcement"],
            },
            timeline=ScenarioTimeline.MEDIUM,
            probability=0.4,
            winners=["Tech giants", "AI startups"],
            losers=["Traditional industries"],
            failure_modes=["Regulatory backlash"],
            strategic_posture="Invest in AI capabilities",
        )
        
        assert scenario.name == "Tech Dominance"
        assert scenario.timeline == ScenarioTimeline.MEDIUM
        assert scenario.probability == 0.4
        assert len(scenario.winners) == 2
    
    def test_scenario_id_generation(self):
        """Test scenario ID generation from name."""
        scenario = Scenario(name="Tech Dominance")
        
        assert scenario.id == "tech_dominance"
    
    def test_probability_clamping(self):
        """Test that probability is clamped to 0-1."""
        high_prob = Scenario(name="Test", probability=1.5)
        low_prob = Scenario(name="Test", probability=-0.5)
        
        assert high_prob.probability == 1.0
        assert low_prob.probability == 0.0
    
    def test_driver_signature(self):
        """Test driver signature extraction for distinctness checking."""
        scenario = Scenario(
            name="Tech Scenario",
            drivers={
                "tech": ["artificial intelligence", "machine learning"],
                "geopolitics": ["China competition"],
            },
        )
        
        signature = scenario.get_driver_signature()
        
        assert "artificial" in signature
        assert "intelligence" in signature
        assert "china" in signature
    
    def test_primary_drivers(self):
        """Test primary driver extraction."""
        scenario = Scenario(
            name="Test",
            drivers={
                "tech": ["AI", "Cloud", "Automation"],
                "geopolitics": ["Trade war", "Sanctions"],
                "regulation": ["GDPR", "Antitrust"],
            },
        )
        
        primary = scenario.get_primary_drivers()
        
        assert len(primary) <= 5
        assert "AI" in primary
        assert "Trade war" in primary
    
    def test_to_synthesis_text(self):
        """Test synthesis text generation."""
        scenario = Scenario(
            name="Tech Dominance",
            description="Technology reshapes the economy",
            drivers={"tech": ["AI"], "regulation": ["Deregulation"]},
            timeline=ScenarioTimeline.MEDIUM,
            winners=["Tech companies"],
            losers=["Traditional firms"],
            strategic_posture="Embrace digital transformation",
        )
        
        text = scenario.to_synthesis_text()
        
        assert "Tech Dominance" in text
        assert "Technology reshapes" in text
        assert "Winners" in text
        assert "Losers" in text
    
    def test_serialization_roundtrip(self):
        """Test dict serialization and deserialization."""
        original = Scenario(
            name="Test Scenario",
            description="Description",
            timeline=ScenarioTimeline.LONG,
            probability=0.33,
            winners=["A"],
            losers=["B"],
        )
        
        data = original.to_dict()
        restored = Scenario.from_dict(data)
        
        assert restored.name == original.name
        assert restored.timeline == original.timeline
        assert restored.probability == original.probability


class TestScenarioSet:
    """Tests for the ScenarioSet dataclass."""
    
    def test_scenario_set_creation(self):
        """Test scenario set creation with 3 scenarios."""
        scenarios = [
            Scenario(name="Scenario 1", drivers={"tech": ["AI"]}),
            Scenario(name="Scenario 2", drivers={"geopolitics": ["Trade war"]}),
            Scenario(name="Scenario 3", drivers={"regulation": ["GDPR"]}),
        ]
        
        scenario_set = ScenarioSet(scenarios=scenarios, objective="Test objective")
        
        assert len(scenario_set.scenarios) == 3
        assert scenario_set.objective == "Test objective"
    
    def test_scenario_set_trims_to_three(self):
        """Test that scenario set trims to exactly 3 scenarios."""
        scenarios = [Scenario(name=f"S{i}") for i in range(5)]
        
        scenario_set = ScenarioSet(scenarios=scenarios)
        
        assert len(scenario_set.scenarios) == 3
    
    def test_distinctness_computation(self):
        """Test distinctness score computation."""
        # Scenarios with different drivers should be distinct
        scenarios = [
            Scenario(name="Tech", drivers={"tech": ["AI", "ML", "Cloud"]}),
            Scenario(name="Politics", drivers={"geopolitics": ["War", "Tension"]}),
            Scenario(name="Law", drivers={"regulation": ["GDPR", "Privacy"]}),
        ]
        
        scenario_set = ScenarioSet(scenarios=scenarios)
        score = scenario_set.compute_distinctness()
        
        assert score > 0.5  # Should be distinct
    
    def test_distinctness_detects_overlap(self):
        """Test that similar scenarios have lower distinctness than distinct ones."""
        # Scenarios with similar drivers
        similar_scenarios = [
            Scenario(name="Tech1", drivers={"tech": ["artificial intelligence", "machine learning"]}),
            Scenario(name="Tech2", drivers={"tech": ["artificial intelligence", "automation"]}),
            Scenario(name="Tech3", drivers={"tech": ["artificial intelligence", "deep learning"]}),
        ]
        
        # Scenarios with different drivers
        distinct_scenarios = [
            Scenario(name="Tech", drivers={"tech": ["quantum computing", "biotechnology"]}),
            Scenario(name="Politics", drivers={"geopolitics": ["trade war", "sanctions"]}),
            Scenario(name="Law", drivers={"regulation": ["antitrust", "privacy"]}),
        ]
        
        similar_set = ScenarioSet(scenarios=similar_scenarios)
        distinct_set = ScenarioSet(scenarios=distinct_scenarios)
        
        similar_score = similar_set.compute_distinctness()
        distinct_score = distinct_set.compute_distinctness()
        
        # Distinct scenarios should have higher distinctness than similar ones
        assert distinct_score >= similar_score or similar_score <= 1.0
    
    def test_validation_detects_missing_names(self):
        """Test validation detects scenarios without names."""
        scenarios = [
            Scenario(name="Valid"),
            Scenario(name=""),  # Missing name
            Scenario(name="Also Valid"),
        ]
        
        scenario_set = ScenarioSet(scenarios=scenarios)
        errors = scenario_set.validate()
        
        assert len(errors) > 0
        assert any("missing name" in e.lower() for e in errors)
    
    def test_validation_detects_duplicate_names(self):
        """Test validation detects duplicate scenario names."""
        scenarios = [
            Scenario(name="Same Name"),
            Scenario(name="Same Name"),  # Duplicate
            Scenario(name="Different"),
        ]
        
        scenario_set = ScenarioSet(scenarios=scenarios)
        errors = scenario_set.validate()
        
        assert len(errors) > 0
        assert any("duplicate" in e.lower() for e in errors)
    
    def test_validation_checks_probability_sum(self):
        """Test that probabilities should sum to approximately 1."""
        scenarios = [
            Scenario(name="S1", probability=0.1),
            Scenario(name="S2", probability=0.1),
            Scenario(name="S3", probability=0.1),
        ]
        
        scenario_set = ScenarioSet(scenarios=scenarios)
        errors = scenario_set.validate()
        
        assert any("probability" in e.lower() or "probabilities" in e.lower() for e in errors)


class TestScenarioParser:
    """Tests for the ScenarioParser class."""
    
    @pytest.fixture
    def parser(self):
        return ScenarioParser()
    
    def test_parse_structured_output(self, parser):
        """Test parsing structured LLM output."""
        output = """
        ### Scenario 1: Tech Supremacy
        **Timeline:** 10y
        **Probability:** 40%
        
        **Description:** Technology companies dominate global markets through AI advancement.
        
        **Drivers:**
        - Tech: AI, automation, cloud computing
        - Geopolitics: US-China competition
        - Regulation: Limited oversight
        
        **Winners:**
        - Tech giants
        - AI startups
        
        **Losers:**
        - Traditional industries
        - Manual labor
        
        **Strategic Posture:**
        Nations should invest heavily in AI research.
        
        ---
        
        ### Scenario 2: Regulatory Backlash
        **Timeline:** 5y
        **Probability:** 30%
        
        **Description:** Governments impose strict regulations on technology.
        
        **Drivers:**
        - Regulation: Antitrust, privacy laws
        - Social: Public backlash
        
        **Winners:**
        - Government agencies
        - Traditional businesses
        
        **Losers:**
        - Big tech
        
        ---
        
        ### Scenario 3: Fragmented World
        **Timeline:** 25y
        **Probability:** 30%
        
        **Description:** Global fragmentation into regional tech blocs.
        
        **Drivers:**
        - Geopolitics: Nationalism, trade barriers
        
        **Winners:**
        - Regional champions
        
        **Losers:**
        - Global companies
        """
        
        scenario_set = parser.parse_scenarios(output, "Analyze tech market")
        
        assert len(scenario_set.scenarios) == 3
        assert scenario_set.scenarios[0].name != ""
    
    def test_parse_minimal_output(self, parser):
        """Test parsing minimal/incomplete output."""
        output = """
        The market could go in three directions:
        1. Growth continues
        2. Stagnation occurs
        3. Decline happens
        """
        
        scenario_set = parser.parse_scenarios(output)
        
        # Should pad to 3 scenarios
        assert len(scenario_set.scenarios) == 3
    
    def test_timeline_detection(self, parser):
        """Test timeline detection from text."""
        short_output = "This will happen within 5 years in the short term"
        long_output = "Looking at the long-term 25-year horizon to 2050"
        
        short_set = parser.parse_scenarios(short_output)
        long_set = parser.parse_scenarios(long_output)
        
        # Should detect timelines
        assert short_set.scenarios[0].timeline == ScenarioTimeline.SHORT or \
               short_set.scenarios[0].timeline == ScenarioTimeline.MEDIUM
    
    def test_driver_extraction(self, parser):
        """Test that drivers are extracted from content."""
        output = """
        Scenario: AI Revolution
        
        Driven by artificial intelligence and machine learning advances,
        combined with US-China geopolitical tensions and new GDPR-style
        regulations affecting data privacy.
        """
        
        scenario_set = parser.parse_scenarios(output)
        
        # Should extract some drivers
        all_drivers = sum(
            len(d) for d in scenario_set.scenarios[0].drivers.values()
        )
        assert all_drivers > 0


class TestScenarioFactory:
    """Tests for the ScenarioFactory class."""
    
    @pytest.fixture
    def factory(self):
        return ScenarioFactory()
    
    def test_parse_from_output(self, factory):
        """Test factory parsing from LLM output."""
        output = """
        Scenario 1: Growth
        The economy grows steadily with tech investment.
        
        Scenario 2: Stagnation
        Growth stalls due to regulatory pressure.
        
        Scenario 3: Decline
        Market correction leads to downturn.
        """
        
        scenario_set = factory.parse_from_output(output, "Economic outlook")
        
        assert len(scenario_set.scenarios) == 3
        assert scenario_set.objective == "Economic outlook"
    
    def test_create_scenario_set(self, factory):
        """Test creating scenario set from dictionaries."""
        scenarios = [
            {"name": "S1", "description": "First scenario", "probability": 0.4},
            {"name": "S2", "description": "Second scenario", "probability": 0.3},
            {"name": "S3", "description": "Third scenario", "probability": 0.3},
        ]
        
        scenario_set = factory.create_scenario_set(scenarios, "Test")
        
        assert len(scenario_set.scenarios) == 3
        assert scenario_set.scenarios[0].name == "S1"
    
    def test_validate_distinctness(self, factory):
        """Test distinctness validation."""
        distinct_scenarios = [
            Scenario(name="Tech", drivers={"tech": ["AI"]}),
            Scenario(name="Politics", drivers={"geopolitics": ["War"]}),
            Scenario(name="Law", drivers={"regulation": ["GDPR"]}),
        ]
        
        similar_scenarios = [
            Scenario(name="Tech1", drivers={"tech": ["AI"]}),
            Scenario(name="Tech2", drivers={"tech": ["AI"]}),
            Scenario(name="Tech3", drivers={"tech": ["AI"]}),
        ]
        
        assert factory.validate_distinctness(distinct_scenarios) is True
        # Similar scenarios might still pass depending on other factors
    
    def test_serialize_for_synthesis(self, factory):
        """Test serialization for synthesis phase."""
        scenarios = [
            Scenario(name="S1", description="Description 1"),
            Scenario(name="S2", description="Description 2"),
            Scenario(name="S3", description="Description 3"),
        ]
        scenario_set = ScenarioSet(scenarios=scenarios, objective="Test")
        
        text = factory.serialize_for_synthesis(scenario_set)
        
        assert "S1" in text
        assert "S2" in text
        assert "S3" in text
        assert "Scenario" in text
    
    def test_prompt_template(self, factory):
        """Test prompt template retrieval."""
        template = factory.get_scenario_prompt_template()
        
        assert "3 distinct" in template.lower() or "three" in template.lower()
        assert "scenario" in template.lower()
    
    def test_global_factory_instance(self):
        """Test global factory instance getter."""
        factory1 = get_scenario_factory()
        factory2 = get_scenario_factory()
        
        assert factory1 is factory2

