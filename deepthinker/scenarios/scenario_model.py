"""
Scenario Model for DeepThinker Formal Scenario Analysis.

Replaces prose-based scenario analysis with structured objects that:
- Define explicit drivers (tech, geopolitics, regulation)
- Specify timelines (5y, 10y, 25y)
- Identify winners and losers
- Enumerate failure modes
- Provide strategic posture recommendations

Enforces exactly 3 distinct, non-overlapping scenarios per analysis.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ScenarioTimeline(str, Enum):
    """Standard scenario timelines."""
    SHORT = "5y"    # 5 years
    MEDIUM = "10y"  # 10 years
    LONG = "25y"    # 25 years


class ScenarioDriver(str, Enum):
    """Categories of scenario drivers."""
    TECH = "tech"
    GEOPOLITICS = "geopolitics"
    REGULATION = "regulation"
    ECONOMICS = "economics"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"


@dataclass
class Scenario:
    """
    A structured future scenario for strategic analysis.
    
    Attributes:
        name: Short, memorable scenario name
        description: Detailed description of the scenario
        drivers: Key drivers organized by category
        timeline: Primary timeline for this scenario
        probability: Estimated probability (0-1)
        winners: Entities that benefit in this scenario
        losers: Entities that are disadvantaged
        failure_modes: Ways this scenario could fail or backfire
        strategic_posture: Recommended posture for mid-sized nations
        key_indicators: Early indicators this scenario is emerging
        assumptions: Key assumptions underlying this scenario
    """
    name: str
    description: str = ""
    drivers: Dict[str, List[str]] = field(default_factory=dict)
    timeline: ScenarioTimeline = ScenarioTimeline.MEDIUM
    probability: float = 0.33
    winners: List[str] = field(default_factory=list)
    losers: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    strategic_posture: str = ""
    key_indicators: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize scenario."""
        # Ensure drivers dict has standard categories
        for driver in ScenarioDriver:
            if driver.value not in self.drivers:
                self.drivers[driver.value] = []
        
        # Clamp probability
        self.probability = max(0.0, min(1.0, self.probability))
    
    @property
    def id(self) -> str:
        """Generate scenario ID from name."""
        return re.sub(r'[^a-z0-9]+', '_', self.name.lower()).strip('_')
    
    def get_primary_drivers(self) -> List[str]:
        """Get the most significant drivers across categories."""
        all_drivers = []
        for category, drivers in self.drivers.items():
            all_drivers.extend(drivers[:2])  # Top 2 from each category
        return all_drivers[:5]
    
    def get_driver_signature(self) -> Set[str]:
        """Get a signature of key driver terms for distinctness checking."""
        signature = set()
        for drivers in self.drivers.values():
            for d in drivers:
                # Extract key terms (words > 4 chars)
                terms = [w.lower() for w in d.split() if len(w) > 4]
                signature.update(terms)
        return signature
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "drivers": self.drivers,
            "timeline": self.timeline.value,
            "probability": self.probability,
            "winners": self.winners,
            "losers": self.losers,
            "failure_modes": self.failure_modes,
            "strategic_posture": self.strategic_posture,
            "key_indicators": self.key_indicators,
            "assumptions": self.assumptions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Create from dictionary."""
        timeline_str = data.get("timeline", "10y")
        try:
            timeline = ScenarioTimeline(timeline_str)
        except ValueError:
            timeline = ScenarioTimeline.MEDIUM
        
        return cls(
            name=data.get("name", "Unnamed Scenario"),
            description=data.get("description", ""),
            drivers=data.get("drivers", {}),
            timeline=timeline,
            probability=data.get("probability", 0.33),
            winners=data.get("winners", []),
            losers=data.get("losers", []),
            failure_modes=data.get("failure_modes", []),
            strategic_posture=data.get("strategic_posture", ""),
            key_indicators=data.get("key_indicators", []),
            assumptions=data.get("assumptions", []),
        )
    
    def to_synthesis_text(self) -> str:
        """Generate synthesis-ready text representation."""
        lines = [
            f"## Scenario: {self.name}",
            f"**Timeline:** {self.timeline.value}",
            f"**Probability:** {self.probability:.0%}",
            "",
            f"### Description",
            self.description,
            "",
            "### Key Drivers",
        ]
        
        for category, drivers in self.drivers.items():
            if drivers:
                lines.append(f"- **{category.title()}:** {', '.join(drivers[:3])}")
        
        if self.winners:
            lines.append(f"\n### Winners\n- " + "\n- ".join(self.winners[:5]))
        
        if self.losers:
            lines.append(f"\n### Losers\n- " + "\n- ".join(self.losers[:5]))
        
        if self.failure_modes:
            lines.append(f"\n### Failure Modes\n- " + "\n- ".join(self.failure_modes[:3]))
        
        if self.strategic_posture:
            lines.append(f"\n### Strategic Posture for Mid-Sized Nations")
            lines.append(self.strategic_posture)
        
        return "\n".join(lines)


@dataclass
class ScenarioSet:
    """
    A set of exactly 3 scenarios for analysis.
    
    Enforces distinctness and completeness requirements.
    """
    scenarios: List[Scenario] = field(default_factory=list)
    objective: str = ""
    distinctness_score: float = 0.0
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate scenario count."""
        if len(self.scenarios) > 3:
            logger.warning(f"ScenarioSet has {len(self.scenarios)} scenarios, trimming to 3")
            self.scenarios = self.scenarios[:3]
    
    def is_valid(self) -> bool:
        """Check if scenario set is valid."""
        return (
            len(self.scenarios) == 3 and
            self.distinctness_score >= 0.5 and
            not self.validation_errors
        )
    
    def compute_distinctness(self) -> float:
        """
        Compute distinctness score between scenarios.
        
        Returns:
            Score from 0-1 where 1 = completely distinct
        """
        if len(self.scenarios) < 2:
            return 1.0
        
        # Compare driver signatures pairwise
        signatures = [s.get_driver_signature() for s in self.scenarios]
        overlaps = []
        
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                if signatures[i] and signatures[j]:
                    intersection = len(signatures[i] & signatures[j])
                    union = len(signatures[i] | signatures[j])
                    overlap = intersection / union if union > 0 else 0
                    overlaps.append(overlap)
        
        # Distinctness = 1 - average overlap
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        self.distinctness_score = 1.0 - avg_overlap
        
        return self.distinctness_score
    
    def validate(self) -> List[str]:
        """
        Validate the scenario set.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check count
        if len(self.scenarios) != 3:
            errors.append(f"Expected 3 scenarios, got {len(self.scenarios)}")
        
        # Check distinctness
        self.compute_distinctness()
        if self.distinctness_score < 0.5:
            errors.append(
                f"Scenarios too similar: distinctness={self.distinctness_score:.2f} (min: 0.5)"
            )
        
        # Check each scenario
        names = set()
        for i, s in enumerate(self.scenarios):
            if not s.name:
                errors.append(f"Scenario {i+1} missing name")
            elif s.name in names:
                errors.append(f"Duplicate scenario name: {s.name}")
            names.add(s.name)
            
            if not s.description:
                errors.append(f"Scenario '{s.name}' missing description")
            
            if not any(s.drivers.values()):
                errors.append(f"Scenario '{s.name}' has no drivers")
            
            if not s.winners and not s.losers:
                errors.append(f"Scenario '{s.name}' missing winners/losers")
        
        # Check probability sum
        total_prob = sum(s.probability for s in self.scenarios)
        if abs(total_prob - 1.0) > 0.1:
            errors.append(
                f"Scenario probabilities sum to {total_prob:.2f}, should be ~1.0"
            )
        
        self.validation_errors = errors
        return errors
    
    def to_synthesis_text(self) -> str:
        """Generate synthesis-ready text for all scenarios."""
        lines = [
            "# Strategic Scenarios Analysis",
            f"**Objective:** {self.objective}",
            f"**Distinctness Score:** {self.distinctness_score:.2f}",
            "",
        ]
        
        for i, scenario in enumerate(self.scenarios, 1):
            lines.append(f"---\n# Scenario {i}")
            lines.append(scenario.to_synthesis_text())
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objective": self.objective,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "distinctness_score": self.distinctness_score,
            "validation_errors": self.validation_errors,
        }


class ScenarioParser:
    """
    Parses scenarios from LLM output into structured Scenario objects.
    """
    
    # Patterns for scenario detection
    SCENARIO_HEADER_PATTERN = re.compile(
        r'(?:##?\s*)?(?:scenario\s*\d*[:\s]*)?([^:\n]+?)(?:\s*[-–]\s*|\s*:\s*|\n)',
        re.IGNORECASE
    )
    
    DRIVER_PATTERNS = {
        "tech": [
            r'\b(AI|artificial intelligence|automation|compute|chips?|semiconductor)',
            r'\b(software|hardware|cloud|quantum|blockchain|crypto)',
            r'\b(5G|6G|network|infrastructure|technology|tech)',
        ],
        "geopolitics": [
            r'\b(US|China|EU|Russia|India|NATO|alliance)',
            r'\b(conflict|war|tension|sanction|trade war)',
            r'\b(sovereignty|power|influence|hegemony)',
        ],
        "regulation": [
            r'\b(regulation|law|policy|compliance|standard)',
            r'\b(ban|restrict|require|mandate|enforce)',
            r'\b(GDPR|antitrust|privacy|security requirement)',
        ],
        "economics": [
            r'\b(GDP|growth|recession|inflation|investment)',
            r'\b(market|trade|tariff|supply chain)',
            r'\b(cost|price|profit|revenue)',
        ],
    }
    
    def __init__(self):
        """Initialize parser with compiled patterns."""
        self._compiled_drivers = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.DRIVER_PATTERNS.items()
        }
    
    def parse_scenarios(self, output: str, objective: str = "") -> ScenarioSet:
        """
        Parse scenarios from LLM output.
        
        Args:
            output: Raw LLM output containing scenario descriptions
            objective: The analysis objective
            
        Returns:
            ScenarioSet with parsed scenarios
        """
        scenarios = []
        
        # Try to split by scenario headers
        sections = self._split_into_sections(output)
        
        for section in sections[:3]:  # Max 3 scenarios
            scenario = self._parse_section(section)
            if scenario:
                scenarios.append(scenario)
        
        # If no scenarios found, try to create from whole output
        if not scenarios:
            scenario = self._parse_section(output)
            if scenario:
                scenarios.append(scenario)
        
        # Pad to 3 if needed
        while len(scenarios) < 3:
            scenarios.append(Scenario(
                name=f"Scenario {len(scenarios) + 1}",
                description="[Scenario not fully specified]",
            ))
        
        scenario_set = ScenarioSet(
            scenarios=scenarios[:3],
            objective=objective,
        )
        scenario_set.validate()
        
        return scenario_set
    
    def _split_into_sections(self, output: str) -> List[str]:
        """Split output into scenario sections."""
        # Try common section markers
        markers = [
            r'(?:^|\n)(?:##?\s*)?scenario\s*\d+[:\s]',
            r'(?:^|\n)###?\s*(?:\d+[.\)]\s*)?scenario',
            r'(?:^|\n)\d+[.\)]\s*scenario',
        ]
        
        for marker in markers:
            parts = re.split(marker, output, flags=re.IGNORECASE)
            if len(parts) >= 3:
                return [p.strip() for p in parts if p.strip()]
        
        # Fallback: split by numbered sections
        numbered = re.split(r'(?:^|\n)\d+[.\)]\s+', output)
        if len(numbered) >= 3:
            return [p.strip() for p in numbered if p.strip()]
        
        # Last resort: return whole output
        return [output]
    
    def _parse_section(self, section: str) -> Optional[Scenario]:
        """Parse a single section into a Scenario."""
        if not section or len(section) < 50:
            return None
        
        # Extract name (first line or header)
        lines = section.strip().split('\n')
        name = self._extract_name(lines[0])
        
        # Extract description (first substantial paragraph)
        description = self._extract_description(section)
        
        # Extract drivers by category
        drivers = self._extract_drivers(section)
        
        # Extract timeline
        timeline = self._extract_timeline(section)
        
        # Extract winners/losers
        winners = self._extract_list(section, ["winner", "benefit", "gain", "advantage"])
        losers = self._extract_list(section, ["loser", "disadvantage", "harm", "lose"])
        
        # Extract failure modes
        failure_modes = self._extract_list(
            section, ["failure", "risk", "downside", "could fail", "backfire"]
        )
        
        # Extract strategic posture
        strategic_posture = self._extract_strategic_posture(section)
        
        # Extract probability
        probability = self._extract_probability(section)
        
        return Scenario(
            name=name,
            description=description,
            drivers=drivers,
            timeline=timeline,
            probability=probability,
            winners=winners,
            losers=losers,
            failure_modes=failure_modes,
            strategic_posture=strategic_posture,
        )
    
    def _extract_name(self, first_line: str) -> str:
        """Extract scenario name from first line."""
        # Remove common prefixes
        name = re.sub(r'^(?:##?\s*)?(?:scenario\s*\d*[:\s]*)?', '', first_line, flags=re.IGNORECASE)
        name = re.sub(r'^[\d.\-)\s]+', '', name)  # Remove numbering
        name = name.strip().strip(':').strip('"').strip("'")
        
        # Truncate if too long
        if len(name) > 100:
            name = name[:100].rsplit(' ', 1)[0] + "..."
        
        return name or "Unnamed Scenario"
    
    def _extract_description(self, section: str) -> str:
        """Extract main description."""
        # Skip first line (name), get first substantial paragraph
        lines = section.split('\n')[1:]
        paragraphs = '\n'.join(lines).split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if len(para) > 100 and not para.startswith(('-', '*', '#', '|')):
                return para[:1000]
        
        return section[:500]
    
    def _extract_drivers(self, section: str) -> Dict[str, List[str]]:
        """Extract drivers by category."""
        drivers: Dict[str, List[str]] = {cat: [] for cat in self._compiled_drivers}
        
        for category, patterns in self._compiled_drivers.items():
            for pattern in patterns:
                matches = pattern.findall(section)
                drivers[category].extend(matches)
            
            # Deduplicate and limit
            drivers[category] = list(dict.fromkeys(drivers[category]))[:5]
        
        return drivers
    
    def _extract_timeline(self, section: str) -> ScenarioTimeline:
        """Extract timeline from section."""
        section_lower = section.lower()
        
        if any(t in section_lower for t in ["25 year", "long term", "2050", "long-term"]):
            return ScenarioTimeline.LONG
        elif any(t in section_lower for t in ["5 year", "short term", "near term", "short-term"]):
            return ScenarioTimeline.SHORT
        
        return ScenarioTimeline.MEDIUM
    
    def _extract_list(self, section: str, keywords: List[str]) -> List[str]:
        """Extract list items near keywords."""
        items = []
        
        for keyword in keywords:
            # Find lines containing keyword
            pattern = rf'(?:^|\n)[^\n]*{keyword}[^\n]*(?:\n[-*]\s*[^\n]+)+'
            matches = re.findall(pattern, section, re.IGNORECASE)
            
            for match in matches:
                # Extract bullet points
                bullets = re.findall(r'[-*]\s*([^\n]+)', match)
                items.extend(bullets)
        
        # Also check for inline lists
        for keyword in keywords:
            pattern = rf'{keyword}s?[:\s]+([^.\n]+(?:,\s*[^.\n]+)*)'
            matches = re.findall(pattern, section, re.IGNORECASE)
            for match in matches:
                items.extend([i.strip() for i in match.split(',')])
        
        # Deduplicate and clean
        items = [i.strip() for i in items if len(i.strip()) > 3]
        return list(dict.fromkeys(items))[:10]
    
    def _extract_strategic_posture(self, section: str) -> str:
        """Extract strategic posture recommendation."""
        keywords = ["strategic posture", "strategy for", "mid-sized nation", "should"]
        
        for keyword in keywords:
            pattern = rf'{keyword}[:\s]*([^.\n]+[.][^.\n]*)'
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:500]
        
        return ""
    
    def _extract_probability(self, section: str) -> float:
        """Extract probability estimate."""
        patterns = [
            r'probability[:\s]*(\d+)%',
            r'(\d+)%\s*(?:probability|likely|chance)',
            r'likelihood[:\s]*(\d+)%',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100
        
        return 0.33  # Default for 3 scenarios


class ScenarioFactory:
    """
    Factory for generating and validating scenario sets.
    """
    
    def __init__(self):
        """Initialize factory."""
        self._parser = ScenarioParser()
    
    def parse_from_output(self, output: str, objective: str = "") -> ScenarioSet:
        """
        Parse scenarios from LLM output.
        
        Args:
            output: Raw LLM output
            objective: Analysis objective
            
        Returns:
            Validated ScenarioSet
        """
        return self._parser.parse_scenarios(output, objective)
    
    def create_scenario_set(
        self,
        scenarios: List[Dict[str, Any]],
        objective: str = ""
    ) -> ScenarioSet:
        """
        Create scenario set from dictionaries.
        
        Args:
            scenarios: List of scenario dictionaries
            objective: Analysis objective
            
        Returns:
            Validated ScenarioSet
        """
        scenario_objs = [Scenario.from_dict(s) for s in scenarios]
        
        scenario_set = ScenarioSet(
            scenarios=scenario_objs[:3],
            objective=objective,
        )
        scenario_set.validate()
        
        return scenario_set
    
    def validate_distinctness(self, scenarios: List[Scenario]) -> bool:
        """
        Validate that scenarios are sufficiently distinct.
        
        Args:
            scenarios: List of scenarios to check
            
        Returns:
            True if scenarios are distinct
        """
        temp_set = ScenarioSet(scenarios=scenarios)
        score = temp_set.compute_distinctness()
        return score >= 0.5
    
    def serialize_for_synthesis(self, scenario_set: ScenarioSet) -> str:
        """
        Serialize scenarios for use in synthesis phase.
        
        Args:
            scenario_set: Validated scenario set
            
        Returns:
            Synthesis-ready text representation
        """
        return scenario_set.to_synthesis_text()
    
    def get_scenario_prompt_template(self) -> str:
        """Get prompt template for scenario generation."""
        return """Generate exactly 3 distinct future scenarios for the following objective.

## Requirements
- Each scenario must have a unique NAME
- Each scenario must specify different DRIVERS across these categories:
  - Tech (technology trends, innovation)
  - Geopolitics (international relations, power dynamics)
  - Regulation (policy, compliance, standards)
- Each scenario must specify a TIMELINE (5 years, 10 years, or 25 years)
- Each scenario must list WINNERS and LOSERS
- Each scenario must identify FAILURE MODES (how the scenario could go wrong)
- Each scenario must provide STRATEGIC POSTURE for mid-sized nations

## Output Format
For each scenario, structure as:

### Scenario 1: [Name]
**Timeline:** [5y/10y/25y]
**Probability:** [X%]

**Description:** [2-3 sentences describing the scenario]

**Drivers:**
- Tech: [key technology drivers]
- Geopolitics: [key geopolitical drivers]
- Regulation: [key regulatory drivers]

**Winners:**
- [Entity 1]
- [Entity 2]

**Losers:**
- [Entity 1]
- [Entity 2]

**Failure Modes:**
- [How this scenario could fail]

**Strategic Posture:**
[Recommended strategy for mid-sized nations]

---

[Repeat for Scenarios 2 and 3]

Ensure scenarios are MUTUALLY EXCLUSIVE - they should represent different futures, not variations of the same theme.
"""


# Global factory instance
_factory: Optional[ScenarioFactory] = None


def get_scenario_factory() -> ScenarioFactory:
    """Get the global scenario factory instance."""
    global _factory
    if _factory is None:
        _factory = ScenarioFactory()
    return _factory

