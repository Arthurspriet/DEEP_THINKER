"""
Plan Reviser for Meta-Cognition.

Produces plan revisions based on reflection, debate, and hypothesis updates.
Updates state fields without rewriting the planner.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.model_pool import ModelPool
    from ..missions.mission_types import MissionState

logger = logging.getLogger(__name__)


class PlanReviser:
    """
    Produces plan revisions based on meta-cognition results.
    
    Updates:
    - state.next_actions: Prioritized list of next actions
    - state.updated_plan: Revised plan elements
    
    Does NOT rewrite the planner or modify existing phases directly.
    Instead, provides guidance for subsequent phases.
    """
    
    def __init__(self, model_pool: "ModelPool"):
        """
        Initialize the plan reviser.
        
        Args:
            model_pool: Model pool for LLM calls
        """
        self.model_pool = model_pool
        self._system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for plan revision."""
        return """You are a strategic planner revising a mission plan based on new insights.

Given the results of reflection and debate, you must:
1. Identify new subgoals that should be added
2. Flag goals that should be removed or invalidated
3. Suggest priority adjustments
4. Note missing data requirements

Be specific and actionable. Focus on high-impact changes.

Output format:
NEW_SUBGOALS:
- [subgoal 1]
- [subgoal 2]

INVALIDATED_GOALS:
- [goal to remove 1]

PRIORITY_ADJUSTMENTS:
- [adjustment 1]

MISSING_DATA:
- [data need 1]

NEXT_ACTIONS:
1. [highest priority action]
2. [second priority action]
3. [third priority action]"""
    
    def _get_smallest_model(self) -> str:
        """Get the smallest available model for efficiency."""
        models = self.model_pool.get_all_models()
        if not models:
            raise ValueError("No models available in model pool")
        
        small_keywords = ["3b", "7b", "8b", "small", "mini", "tiny"]
        for model in models:
            if any(kw in model.lower() for kw in small_keywords):
                return model
        return models[0]
    
    def _ensure_state_initialized(self, state: "MissionState") -> None:
        """Ensure plan revision storage is initialized in state."""
        if not hasattr(state, 'updated_plan') or not state.updated_plan:
            state.updated_plan = {
                "new_subgoals": [],
                "invalidated_goals": [],
                "priority_adjustments": [],
                "missing_data": [],
                "revision_history": [],
            }
        if not hasattr(state, 'next_actions') or not state.next_actions:
            state.next_actions = []
    
    def revise_plan(
        self,
        state: "MissionState",
        phase_name: str,
        reflection: Dict[str, List[str]],
        debate: Dict[str, Any],
        hypotheses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Revise the plan based on meta-cognition results.
        
        Args:
            state: Current mission state
            phase_name: Name of the phase just completed
            reflection: Results from ReflectionEngine
            debate: Results from DebateEngine
            hypotheses: Current hypothesis summary
            
        Returns:
            Dictionary containing:
            - new_subgoals: List of new subgoals
            - invalidated_goals: Goals to remove
            - priority_adjustments: Priority changes
            - missing_data: Data requirements
            - next_actions: Prioritized action list
        """
        self._ensure_state_initialized(state)
        
        result = {
            "new_subgoals": [],
            "invalidated_goals": [],
            "priority_adjustments": [],
            "missing_data": [],
            "next_actions": [],
        }
        
        try:
            # Build revision prompt
            prompt = self._build_revision_prompt(
                state=state,
                phase_name=phase_name,
                reflection=reflection,
                debate=debate,
                hypotheses=hypotheses
            )
            
            model = self._get_smallest_model()
            response = self.model_pool.run_single(
                model_name=model,
                prompt=prompt,
                system_prompt=self._system_prompt,
                temperature=0.4
            )
            
            if response:
                result = self._parse_revision(response)
            
            # Update state with revisions
            self._apply_revisions(state, result, phase_name)
            
        except Exception as e:
            logger.warning(f"Plan revision failed for phase '{phase_name}': {e}")
        
        return result
    
    def _build_revision_prompt(
        self,
        state: "MissionState",
        phase_name: str,
        reflection: Dict[str, List[str]],
        debate: Dict[str, Any],
        hypotheses: Dict[str, Any]
    ) -> str:
        """Build the prompt for plan revision."""
        # Get remaining phases
        remaining_phases = []
        found_current = False
        for phase in state.phases:
            if phase.name == phase_name:
                found_current = True
            elif found_current and phase.status == "pending":
                remaining_phases.append(phase.name)
        
        # Format reflection summary
        reflection_summary = []
        if reflection.get("weaknesses"):
            reflection_summary.append("Weaknesses: " + "; ".join(reflection["weaknesses"][:3]))
        if reflection.get("missing_info"):
            reflection_summary.append("Missing info: " + "; ".join(reflection["missing_info"][:3]))
        if reflection.get("suggestions"):
            reflection_summary.append("Suggestions: " + "; ".join(reflection["suggestions"][:3]))
        
        # Format debate summary
        debate_summary = ""
        if debate.get("contradictions_found"):
            debate_summary = "Contradictions: " + "; ".join(debate["contradictions_found"][:3])
        
        # Format hypothesis summary
        hyp_summary = ""
        if hypotheses:
            active_count = hypotheses.get("total_active", 0)
            avg_conf = hypotheses.get("average_confidence", 0)
            hyp_summary = f"Active hypotheses: {active_count}, Average confidence: {avg_conf:.2f}"
        
        return f"""Revise the mission plan based on these insights from phase "{phase_name}":

MISSION OBJECTIVE:
{state.objective}

REMAINING PHASES:
{', '.join(remaining_phases) if remaining_phases else 'None'}

TIME REMAINING:
{state.remaining_minutes():.1f} minutes

REFLECTION INSIGHTS:
{chr(10).join(reflection_summary) if reflection_summary else 'No major issues identified'}

DEBATE RESULTS:
{debate_summary if debate_summary else 'No contradictions found'}

HYPOTHESIS STATUS:
{hyp_summary if hyp_summary else 'No hypotheses tracked'}

---

Based on these insights, revise the plan. What new subgoals should be added? What should be deprioritized? What data is missing? What are the top 3 next actions?"""
    
    def _parse_revision(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured revision."""
        result = {
            "new_subgoals": [],
            "invalidated_goals": [],
            "priority_adjustments": [],
            "missing_data": [],
            "next_actions": [],
        }
        
        section_map = {
            "new_subgoals": "new_subgoals",
            "new subgoals": "new_subgoals",
            "subgoals": "new_subgoals",
            "invalidated_goals": "invalidated_goals",
            "invalidated goals": "invalidated_goals",
            "removed goals": "invalidated_goals",
            "priority_adjustments": "priority_adjustments",
            "priority adjustments": "priority_adjustments",
            "priorities": "priority_adjustments",
            "missing_data": "missing_data",
            "missing data": "missing_data",
            "data needs": "missing_data",
            "next_actions": "next_actions",
            "next actions": "next_actions",
            "actions": "next_actions",
        }
        
        current_section = None
        lines = response.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section header
            line_lower = line.lower().rstrip(":")
            for header, key in section_map.items():
                if header in line_lower and len(line) < 40:
                    current_section = key
                    break
            else:
                # Not a header - add to current section
                if current_section:
                    # Extract item from line
                    item = line.lstrip("-*0123456789.) ").strip()
                    if item and len(item) > 3:
                        result[current_section].append(item)
        
        return result
    
    def _apply_revisions(
        self,
        state: "MissionState",
        revisions: Dict[str, Any],
        phase_name: str
    ) -> None:
        """Apply revisions to state."""
        # Update next_actions
        if revisions.get("next_actions"):
            # Prepend new actions, keeping existing ones
            existing = state.next_actions or []
            new_actions = revisions["next_actions"]
            # Avoid duplicates
            combined = new_actions + [a for a in existing if a not in new_actions]
            state.next_actions = combined[:10]  # Keep top 10
        
        # Update updated_plan
        if revisions.get("new_subgoals"):
            state.updated_plan["new_subgoals"].extend(revisions["new_subgoals"])
        
        if revisions.get("invalidated_goals"):
            state.updated_plan["invalidated_goals"].extend(revisions["invalidated_goals"])
        
        if revisions.get("priority_adjustments"):
            state.updated_plan["priority_adjustments"].extend(revisions["priority_adjustments"])
        
        if revisions.get("missing_data"):
            state.updated_plan["missing_data"].extend(revisions["missing_data"])
        
        # Record revision history
        state.updated_plan["revision_history"].append({
            "phase": phase_name,
            "new_subgoals_count": len(revisions.get("new_subgoals", [])),
            "invalidated_count": len(revisions.get("invalidated_goals", [])),
        })
    
    def get_next_action(self, state: "MissionState") -> Optional[str]:
        """
        Get the highest priority next action.
        
        Args:
            state: Current mission state
            
        Returns:
            Highest priority action or None
        """
        self._ensure_state_initialized(state)
        
        if state.next_actions:
            return state.next_actions[0]
        return None
    
    def get_revision_summary(self, state: "MissionState") -> Dict[str, Any]:
        """
        Get a summary of all plan revisions.
        
        Args:
            state: Current mission state
            
        Returns:
            Summary dictionary
        """
        self._ensure_state_initialized(state)
        
        return {
            "total_new_subgoals": len(state.updated_plan.get("new_subgoals", [])),
            "total_invalidated": len(state.updated_plan.get("invalidated_goals", [])),
            "pending_actions": len(state.next_actions),
            "missing_data_items": len(state.updated_plan.get("missing_data", [])),
            "revision_count": len(state.updated_plan.get("revision_history", [])),
        }

