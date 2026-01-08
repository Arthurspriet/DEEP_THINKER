"""
Workflow plan configuration and parsing.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class WorkflowPlan:
    """
    Structured workflow execution plan.
    
    Attributes:
        objective_analysis: Analysis of what needs to be accomplished
        workflow_strategy: Which agents to use (agent_name -> enabled)
        agent_requirements: Specific requirements for each agent
        success_criteria: List of measurable success criteria
        iteration_strategy: Strategy for iterative refinement
        raw_plan: Original LLM-generated plan text
        new_subgoals: Subgoals identified for subsequent iterations
        data_needs: Data or evidence needed to complete the mission
        priority_areas: Areas to prioritize in next iteration
    """
    
    objective_analysis: str = ""
    workflow_strategy: Dict[str, bool] = field(default_factory=dict)
    agent_requirements: Dict[str, str] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    iteration_strategy: Dict[str, Any] = field(default_factory=dict)
    raw_plan: str = ""
    # New fields for iterative execution
    new_subgoals: List[str] = field(default_factory=list)
    data_needs: List[str] = field(default_factory=list)
    priority_areas: List[str] = field(default_factory=list)
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled in the workflow strategy."""
        return self.workflow_strategy.get(agent_name.lower(), True)
    
    def get_agent_requirements(self, agent_name: str) -> str:
        """Get requirements for a specific agent."""
        return self.agent_requirements.get(agent_name.lower(), "")
    
    def validate(self) -> bool:
        """Validate that the plan has minimum required information."""
        return bool(
            self.objective_analysis and
            self.workflow_strategy and
            self.success_criteria
        )
    
    def summary(self) -> str:
        """Generate a human-readable summary of the plan."""
        lines = ["=" * 60, "WORKFLOW PLAN", "=" * 60, ""]
        
        if self.objective_analysis:
            lines.append("OBJECTIVE ANALYSIS:")
            lines.append(self.objective_analysis[:200] + "..." if len(self.objective_analysis) > 200 else self.objective_analysis)
            lines.append("")
        
        if self.workflow_strategy:
            lines.append("WORKFLOW STRATEGY:")
            for agent, enabled in self.workflow_strategy.items():
                status = "✓" if enabled else "✗"
                lines.append(f"  {status} {agent.capitalize()}")
            lines.append("")
        
        if self.success_criteria:
            lines.append("SUCCESS CRITERIA:")
            for i, criterion in enumerate(self.success_criteria, 1):
                lines.append(f"  {i}. {criterion}")
            lines.append("")
        
        if self.new_subgoals:
            lines.append("SUBGOALS FOR NEXT ITERATION:")
            for i, subgoal in enumerate(self.new_subgoals, 1):
                lines.append(f"  {i}. {subgoal}")
            lines.append("")
        
        if self.data_needs:
            lines.append("DATA NEEDS:")
            for need in self.data_needs:
                lines.append(f"  - {need}")
            lines.append("")
        
        if self.priority_areas:
            lines.append("PRIORITY AREAS:")
            for area in self.priority_areas:
                lines.append(f"  * {area}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def has_pending_work(self) -> bool:
        """Check if there are pending subgoals or data needs."""
        return bool(self.new_subgoals or self.data_needs)
    
    def get_top_subgoals(self, limit: int = 3) -> List[str]:
        """Get top priority subgoals for next iteration."""
        return self.new_subgoals[:limit]


class WorkflowPlanParser:
    """
    Parser for extracting structured plan from LLM output.
    """
    
    @staticmethod
    def parse(plan_text: str) -> WorkflowPlan:
        """
        Parse LLM-generated plan text into structured WorkflowPlan.
        
        Args:
            plan_text: Raw plan text from planner agent
            
        Returns:
            Parsed WorkflowPlan object
        """
        plan = WorkflowPlan(raw_plan=plan_text)
        
        # Extract objective analysis
        plan.objective_analysis = WorkflowPlanParser._extract_section(
            plan_text,
            r"(?:OBJECTIVE ANALYSIS|1\.\s*OBJECTIVE ANALYSIS)",
            r"(?:WORKFLOW STRATEGY|2\.\s*WORKFLOW STRATEGY)"
        )
        
        # Extract workflow strategy
        strategy_text = WorkflowPlanParser._extract_section(
            plan_text,
            r"(?:WORKFLOW STRATEGY|2\.\s*WORKFLOW STRATEGY)",
            r"(?:AGENT REQUIREMENTS|3\.\s*AGENT REQUIREMENTS)"
        )
        plan.workflow_strategy = WorkflowPlanParser._parse_workflow_strategy(strategy_text)
        
        # Extract agent requirements
        requirements_text = WorkflowPlanParser._extract_section(
            plan_text,
            r"(?:AGENT REQUIREMENTS|3\.\s*AGENT REQUIREMENTS)",
            r"(?:SUCCESS CRITERIA|4\.\s*SUCCESS CRITERIA)"
        )
        plan.agent_requirements = WorkflowPlanParser._parse_agent_requirements(requirements_text)
        
        # Extract success criteria
        criteria_text = WorkflowPlanParser._extract_section(
            plan_text,
            r"(?:SUCCESS CRITERIA|4\.\s*SUCCESS CRITERIA)",
            r"(?:ITERATION STRATEGY|5\.\s*ITERATION STRATEGY)"
        )
        plan.success_criteria = WorkflowPlanParser._parse_success_criteria(criteria_text)
        
        # Extract iteration strategy
        iteration_text = WorkflowPlanParser._extract_section(
            plan_text,
            r"(?:ITERATION STRATEGY|5\.\s*ITERATION STRATEGY)",
            None
        )
        plan.iteration_strategy = WorkflowPlanParser._parse_iteration_strategy(iteration_text)
        
        return plan
    
    @staticmethod
    def _extract_section(text: str, start_pattern: str, end_pattern: Optional[str]) -> str:
        """Extract a section of text between two patterns."""
        start_match = re.search(start_pattern, text, re.IGNORECASE | re.MULTILINE)
        if not start_match:
            return ""
        
        start_pos = start_match.end()
        
        if end_pattern:
            end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE)
            if end_match:
                return text[start_pos:start_pos + end_match.start()].strip()
        
        return text[start_pos:].strip()
    
    @staticmethod
    def _parse_workflow_strategy(text: str) -> Dict[str, bool]:
        """Parse workflow strategy section."""
        strategy = {}
        
        # Look for agent mentions with Yes/No
        agents = ["research", "researcher", "code generation", "coder", 
                  "evaluation", "evaluator", "execution", "executor", 
                  "simulation", "simulator"]
        
        for agent in agents:
            # Normalize agent name
            normalized = agent.replace(" generation", "").replace("er", "")
            if normalized.endswith("ion"):
                normalized = normalized[:-3] + "e"
            
            # Look for patterns like "Research: Yes" or "- Research: No"
            pattern = rf"[-•*]?\s*{re.escape(agent)}[:\s]+(\w+)"
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                value = match.group(1).lower()
                enabled = value in ["yes", "true", "enabled", "required"]
                strategy[normalized] = enabled
        
        # Ensure core agents are present
        if "code" not in strategy:
            strategy["code"] = True
        if "evaluate" not in strategy:
            strategy["evaluate"] = True
        
        return strategy
    
    @staticmethod
    def _parse_agent_requirements(text: str) -> Dict[str, str]:
        """Parse agent requirements section."""
        requirements = {}
        
        # Split by agent headers
        agent_patterns = [
            (r"(?:For\s+)?Researcher[:\s]+", "researcher"),
            (r"(?:For\s+)?Coder[:\s]+", "coder"),
            (r"(?:For\s+)?Evaluator[:\s]+", "evaluator"),
            (r"(?:For\s+)?Executor[:\s]+", "executor"),
            (r"(?:For\s+)?Simulator[:\s]+", "simulator")
        ]
        
        for pattern, agent_name in agent_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                start_pos = match.end()
                
                # Find the next agent header or end of text
                next_match = None
                for next_pattern, _ in agent_patterns:
                    temp_match = re.search(next_pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE)
                    if temp_match and (next_match is None or temp_match.start() < next_match.start()):
                        next_match = temp_match
                
                if next_match:
                    agent_text = text[start_pos:start_pos + next_match.start()].strip()
                else:
                    agent_text = text[start_pos:].strip()
                
                if agent_text:
                    requirements[agent_name] = agent_text
        
        return requirements
    
    @staticmethod
    def _parse_success_criteria(text: str) -> List[str]:
        """Parse success criteria section."""
        criteria = []
        
        # Look for numbered or bulleted lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Match patterns like "1. ", "- ", "* ", "• "
            match = re.match(r'^(?:\d+\.|[-•*])\s+(.+)$', line)
            if match:
                criteria.append(match.group(1).strip())
            elif line and not line.endswith(':'):
                # Also capture non-bulleted lines that aren't headers
                if len(line) > 10 and not line.isupper():
                    criteria.append(line)
        
        return criteria[:10]  # Limit to 10 criteria
    
    @staticmethod
    def _parse_iteration_strategy(text: str) -> Dict[str, Any]:
        """Parse iteration strategy section."""
        strategy = {}
        
        # Look for iteration count mentions
        iter_match = re.search(r'(\d+)\s+iteration', text, re.IGNORECASE)
        if iter_match:
            strategy['recommended_iterations'] = int(iter_match.group(1))
        
        # Look for quality threshold mentions
        quality_match = re.search(r'quality.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if quality_match:
            strategy['quality_threshold'] = float(quality_match.group(1))
        
        # Store full text for reference
        strategy['description'] = text[:500] if text else ""
        
        return strategy
    
    @staticmethod
    def _parse_subgoals(text: str) -> List[str]:
        """
        Parse subgoals from plan text.
        
        Looks for sections labeled:
        - "Subgoals", "Sub-goals", "Next Steps", "Follow-up"
        """
        subgoals = []
        
        # Try to find subgoals section
        patterns = [
            r"(?:subgoals?|sub-goals?|next\s+steps?|follow[- ]?up)[:\s]*\n(.+?)(?:\n\n|(?=\n[A-Z][A-Z\s]+:)|\Z)",
            r"(?:should\s+also|additionally)[:\s]*\n(.+?)(?:\n\n|\Z)",
        ]
        
        for pattern in patterns:
            section = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if section:
                section_text = section.group(1)
                # Extract list items
                items = re.findall(r"[-*•]\s*(.+)", section_text)
                subgoals.extend([item.strip() for item in items if len(item.strip()) > 10])
                break
        
        return subgoals[:5]  # Limit to 5
    
    @staticmethod
    def _parse_data_needs(text: str) -> List[str]:
        """
        Parse data needs from plan text.
        
        Looks for sections related to:
        - "Data needed", "Information required", "Evidence needed"
        """
        data_needs = []
        
        patterns = [
            r"(?:data\s+need|information\s+require|evidence\s+need|research\s+require)[sd]?[:\s]*\n(.+?)(?:\n\n|(?=\n[A-Z][A-Z\s]+:)|\Z)",
        ]
        
        for pattern in patterns:
            section = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if section:
                section_text = section.group(1)
                items = re.findall(r"[-*•]\s*(.+)", section_text)
                data_needs.extend([item.strip() for item in items if len(item.strip()) > 10])
                break
        
        return data_needs[:5]
    
    @staticmethod
    def parse_with_extended_fields(plan_text: str) -> "WorkflowPlan":
        """
        Parse plan text including new_subgoals and data_needs fields.
        
        Args:
            plan_text: Raw plan text from planner
            
        Returns:
            WorkflowPlan with all fields populated
        """
        # Get base plan
        plan = WorkflowPlanParser.parse(plan_text)
        
        # Extract additional fields
        plan.new_subgoals = WorkflowPlanParser._parse_subgoals(plan_text)
        plan.data_needs = WorkflowPlanParser._parse_data_needs(plan_text)
        
        # Try to extract priority areas from the plan
        priority_patterns = [
            r"(?:priorit|focus|emphasiz)[ey][:\s]*(.+?)(?:\n\n|\Z)",
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, plan_text, re.IGNORECASE | re.DOTALL)
            if match:
                text = match.group(1)
                items = re.findall(r"[-*•]\s*(.+)", text)
                plan.priority_areas = [item.strip() for item in items[:3]]
                break
        
        return plan

