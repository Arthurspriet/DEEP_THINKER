"""
Arbiter Agent for DeepThinker 2.0.

The final decision-maker that resolves contradictions between councils
and ensures output consistency and coherence.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from ..outputs.output_types import OutputSpec, OutputFormat

if TYPE_CHECKING:
    from ..missions.mission_types import MissionState

# Sprint 1-2: Metrics Integration for Final Synthesis Scorecard
try:
    from ..metrics import get_metrics_config, get_judge_ensemble, Scorecard
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    get_metrics_config = None
    get_judge_ensemble = None
    Scorecard = None

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    USE_CHAT_OLLAMA = True
except ImportError:
    from langchain_community.llms import Ollama
    USE_CHAT_OLLAMA = False

from ..models.council_model_config import ARBITER_MODEL


@dataclass
class ArbiterDecision:
    """
    Decision from the arbiter.
    
    Enhanced for multi-view synthesis and meta-analysis.
    
    Attributes:
        final_output: The arbitrated final output
        resolution_notes: Notes on how contradictions were resolved
        confidence: Confidence in the decision (0-1)
        council_agreements: Which councils agreed
        council_disagreements: Which councils disagreed
        raw_output: Raw LLM output
        meta_analysis: Analysis of council disagreements
        ranked_insights: Best insights ranked by importance
        optimist_summary: Summary of optimistic perspective
        skeptic_summary: Summary of skeptical perspective
        confidence_breakdown: Per-council confidence scores
    """
    
    final_output: Any
    resolution_notes: str = ""
    confidence: float = 1.0
    council_agreements: List[str] = field(default_factory=list)
    council_disagreements: List[str] = field(default_factory=list)
    raw_output: str = ""
    
    # Enhanced fields for multi-view synthesis
    meta_analysis: str = ""
    ranked_insights: List[str] = field(default_factory=list)
    optimist_summary: str = ""
    skeptic_summary: str = ""
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Cost-aware fields
    raw_quality_score: float = 0.0
    efficiency_penalty: float = 0.0  # cost-normalized
    latency_penalty: float = 0.0
    composite_utility_score: float = 0.0
    
    # Decision justification
    cost_context: Optional[Dict[str, Any]] = None


@dataclass
class CouncilOutput:
    """Output from a single council for arbitration."""
    
    council_name: str
    output: Any
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Arbiter:
    """
    Final decision-maker in DeepThinker 2.0.
    
    Resolves contradictions between councils, ensures output
    consistency, and produces the final coherent result.
    """
    
    def __init__(
        self,
        model_name: str = ARBITER_MODEL,
        temperature: float = 0.3,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize arbiter.
        
        Args:
            model_name: LLM model to use
            temperature: Low temperature for consistent decisions
            ollama_base_url: Ollama server URL
        """
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url
        self._llm = None
        self._system_prompt = self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for arbiter."""
        return """You are the Arbiter, the final decision-maker in DeepThinker 2.0.

Your role is to:
1. Review outputs from all councils including multi-view perspectives
2. Identify contradictions, agreements, and key insights
3. Weigh optimistic vs skeptical viewpoints
4. Synthesize into a balanced, evidence-based conclusion
5. Produce a final, coherent, actionable output
6. Provide meta-analysis of the reasoning process

You prioritize:
- Correctness and accuracy
- Evidence-based synthesis
- Balance between optimism and skepticism
- Clear reasoning for all resolutions
- Identification of key insights and risks

For multi-view synthesis:
- Compare optimist opportunities with skeptic risks
- Identify where perspectives agree (high confidence)
- Reconcile disagreements with reasoned judgment
- Produce both an optimistic path and skeptical assessment
- Synthesize into a balanced recommendation

Your output must include:
- Final report with clear conclusions
- Meta-analysis of council disagreements
- Ranked list of best insights
- Confidence assessment
- Both optimistic and skeptical summaries

Be comprehensive, balanced, and decisive."""
    
    def _get_llm(self) -> Any:
        """Get or create the LLM instance."""
        if self._llm is None:
            if USE_CHAT_OLLAMA:
                # ChatOllama expects just the model name (e.g., "gemma3:27b")
                self._llm = ChatOllama(
                    model=self.model_name,
                    base_url=self.ollama_base_url,
                    temperature=self.temperature
                )
            else:
                # Legacy Ollama class may need prefix
                prefixed_model = f"ollama/{self.model_name}" if not self.model_name.startswith("ollama") else self.model_name
                self._llm = Ollama(
                    model=prefixed_model,
                    base_url=self.ollama_base_url,
                    temperature=self.temperature
                )
        
        return self._llm
    
    def compute_composite_score(
        self,
        raw_quality: float,
        tokens_consumed: int,
        wall_time: float,
        mission_preferences: Any,
    ) -> Tuple[float, float, float]:
        """
        Compute composite utility score with cost and latency penalties.
        
        Args:
            raw_quality: Raw quality score (0-1)
            tokens_consumed: Total tokens consumed
            wall_time: Wall-clock time in seconds
            mission_preferences: MissionPreferences instance
            
        Returns:
            Tuple of (composite_score, efficiency_penalty, latency_penalty)
        """
        # Efficiency penalty: penalize high token usage for given quality
        tokens_per_quality = tokens_consumed / max(raw_quality, 0.1)
        efficiency_baseline = 1000  # Expected tokens for quality=1.0
        efficiency_penalty = max(0, (tokens_per_quality - efficiency_baseline) / efficiency_baseline)
        efficiency_penalty *= mission_preferences.cost_sensitivity
        
        # Latency penalty
        time_per_quality = wall_time / max(raw_quality, 0.1)
        latency_baseline = 30.0  # Expected seconds for quality=1.0
        latency_penalty = max(0, (time_per_quality - latency_baseline) / latency_baseline)
        latency_penalty *= mission_preferences.latency_sensitivity
        
        # Composite score: quality adjusted by penalties
        composite = raw_quality * (1 - efficiency_penalty * 0.3 - latency_penalty * 0.2)
        
        return composite, efficiency_penalty, latency_penalty
    
    def arbitrate(
        self,
        council_outputs: List[CouncilOutput],
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        meta_traces: Optional[Dict[str, Any]] = None
    ) -> ArbiterDecision:
        """
        Arbitrate between council outputs to produce final decision.
        
        Enhanced for multi-view synthesis with:
        - Meta-analysis of disagreements
        - Ranked insights
        - Optimist vs skeptic comparison
        - Per-council confidence breakdown
        - Integration of meta-cognition signals (hypotheses, contradictions)
        
        Args:
            council_outputs: List of outputs from different councils
            objective: Original objective
            context: Additional context
            meta_traces: Optional meta-cognition traces including:
                - hypotheses: Active hypotheses and evidence
                - depth_contracts: Exploration depth used per phase
                - phase_metrics: Difficulty/uncertainty per phase
                - multiview_results: Optimist/Skeptic outputs per phase
            
        Returns:
            ArbiterDecision with final arbitrated output
        """
        if not council_outputs:
            return ArbiterDecision(
                final_output=None,
                resolution_notes="No council outputs to arbitrate",
                confidence=0.0
            )
        
        # Single council - no arbitration needed
        if len(council_outputs) == 1:
            return ArbiterDecision(
                final_output=council_outputs[0].output,
                resolution_notes="Single council output, no arbitration needed",
                confidence=council_outputs[0].confidence,
                council_agreements=[council_outputs[0].council_name],
                confidence_breakdown={council_outputs[0].council_name: council_outputs[0].confidence}
            )
        
        # Separate multi-view outputs from regular council outputs
        optimist_output = None
        skeptic_output = None
        multiview_analysis = None
        regular_outputs = []
        
        for co in council_outputs:
            name_lower = co.council_name.lower()
            if "optimist" in name_lower:
                optimist_output = co
            elif "skeptic" in name_lower:
                skeptic_output = co
            elif "multi_view" in name_lower:
                multiview_analysis = co
            else:
                regular_outputs.append(co)
        
        # Build council outputs summary
        outputs_summary = []
        for co in regular_outputs:
            output_str = str(co.output)
            if len(output_str) > 1000:
                output_str = output_str[:1000] + "..."
            outputs_summary.append(
                f"### {co.council_name.upper()} (Confidence: {co.confidence:.2f})\n{output_str}"
            )
        
        # Add multi-view section
        multiview_section = ""
        if optimist_output or skeptic_output:
            multiview_section = "\n## MULTI-VIEW PERSPECTIVES\n"
            if optimist_output:
                opt_str = str(optimist_output.output)[:800]
                multiview_section += f"\n### OPTIMIST PERSPECTIVE (Confidence: {optimist_output.confidence:.2f})\n{opt_str}\n"
            if skeptic_output:
                skep_str = str(skeptic_output.output)[:800]
                multiview_section += f"\n### SKEPTIC PERSPECTIVE (Confidence: {skeptic_output.confidence:.2f})\n{skep_str}\n"
            if multiview_analysis:
                multiview_section += f"\n### MULTI-VIEW ANALYSIS\n{str(multiview_analysis.output)[:500]}\n"
        
        context_str = ""
        if context:
            # Format context nicely
            ctx_parts = []
            for k, v in context.items():
                if k == "history_context":
                    ctx_parts.append(f"{k}: [truncated for brevity]")
                else:
                    ctx_parts.append(f"{k}: {v}")
            context_str = f"\n\nMission Context:\n" + "\n".join(ctx_parts)
        
        # Format meta-traces if provided
        meta_traces_str = ""
        if meta_traces:
            meta_parts = []
            
            # Hypotheses summary
            hypotheses = meta_traces.get("hypotheses", {})
            active_hyp = hypotheses.get("active", [])
            if active_hyp:
                meta_parts.append("### Active Hypotheses")
                for h in active_hyp[:5]:  # Limit to 5
                    if isinstance(h, dict):
                        hyp_text = h.get("hypothesis", str(h))
                        conf = h.get("confidence", 0.5)
                        meta_parts.append(f"- {hyp_text[:200]} (confidence: {conf:.2f})")
                    else:
                        meta_parts.append(f"- {str(h)[:200]}")
            
            # Depth contracts summary
            depth_contracts = meta_traces.get("depth_contracts", {})
            if depth_contracts:
                meta_parts.append("\n### Exploration Depth")
                for phase_name, contract in list(depth_contracts.items())[:5]:
                    if isinstance(contract, dict):
                        depth = contract.get("exploration_depth", 0.5)
                        tier = contract.get("model_tier", "unknown")
                        meta_parts.append(f"- {phase_name}: depth={depth:.2f}, tier={tier}")
            
            # Phase metrics summary
            phase_metrics = meta_traces.get("phase_metrics", {})
            if phase_metrics:
                meta_parts.append("\n### Phase Analysis")
                for phase_name, metrics in list(phase_metrics.items())[:5]:
                    if isinstance(metrics, dict):
                        diff = metrics.get("difficulty_score", 0.5)
                        unc = metrics.get("uncertainty_score", 0.5)
                        meta_parts.append(f"- {phase_name}: difficulty={diff:.2f}, uncertainty={unc:.2f}")
            
            # Multi-view results summary
            multiview = meta_traces.get("multiview_results", {})
            if multiview:
                meta_parts.append("\n### Multi-View Analysis")
                for phase_name, mv_result in list(multiview.items())[:3]:
                    if isinstance(mv_result, dict):
                        agreement = mv_result.get("agreement_score", 0)
                        meta_parts.append(f"- {phase_name}: agreement={agreement:.1%}")
                        opt = mv_result.get("optimist", {})
                        skep = mv_result.get("skeptic", {})
                        if opt:
                            opps = opt.get("opportunities", [])[:2]
                            if opps:
                                meta_parts.append(f"  Optimist opportunities: {', '.join(str(o)[:50] for o in opps)}")
                        if skep:
                            risks = skep.get("risks", [])[:2]
                            if risks:
                                meta_parts.append(f"  Skeptic risks: {', '.join(str(r)[:50] for r in risks)}")
            
            # Total iterations
            total_reflections = meta_traces.get("total_reflections", 0)
            total_debates = meta_traces.get("total_debates", 0)
            if total_reflections or total_debates:
                meta_parts.append(f"\n### Meta-Cognition Stats")
                meta_parts.append(f"Reflections: {total_reflections}, Debates: {total_debates}")
            
            if meta_parts:
                meta_traces_str = "\n\n## META-COGNITION INSIGHTS\n" + "\n".join(meta_parts)
        
        prompt = f"""Review and synthesize the following council outputs into a final decision:

## OBJECTIVE
{objective}
{context_str}
{meta_traces_str}

## COUNCIL OUTPUTS
{chr(10).join(outputs_summary)}
{multiview_section}

## SYNTHESIS INSTRUCTIONS

1. Identify key agreements and disagreements between councils
2. Compare optimist opportunities with skeptic risks
3. Weigh evidence and reasoning from each perspective
4. Synthesize into a balanced, actionable conclusion
5. Rank the most important insights

## YOUR SYNTHESIS

### META-ANALYSIS
Analyze patterns of agreement/disagreement across councils.
What themes emerge? Where is there consensus vs conflict?

### OPTIMIST PATH
Summarize the optimistic perspective and its key opportunities.

### SKEPTIC PATH
Summarize the skeptical perspective and its key concerns.

### RANKED INSIGHTS
List the top 5 most important insights (numbered 1-5, best first):
1. 
2.
3.
4.
5.

### BALANCED SYNTHESIS
Provide a balanced final synthesis that accounts for both perspectives.

### FINAL OUTPUT
The definitive, actionable conclusion.

### CONFIDENCE
Rate your confidence in this synthesis (0.0-1.0):"""

        try:
            llm = self._get_llm()
            
            if USE_CHAT_OLLAMA:
                messages = [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=prompt)
                ]
                response = llm.invoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                full_prompt = f"{self._system_prompt}\n\n{prompt}"
                response = llm.invoke(full_prompt)
                output = str(response)
            
            # Parse the enhanced arbitration output
            decision = self._parse_enhanced_arbitration(output, council_outputs, optimist_output, skeptic_output)
            
            # === Sprint 1-2: Compute Final Synthesis Scorecard ===
            decision = self._compute_final_scorecard(decision, objective)
            
            return decision
            
        except Exception as e:
            # Fallback: return highest confidence output
            best_output = max(council_outputs, key=lambda x: x.confidence)
            return ArbiterDecision(
                final_output=best_output.output,
                resolution_notes=f"Fallback to highest confidence (error: {str(e)})",
                confidence=best_output.confidence * 0.8,
                council_agreements=[best_output.council_name]
            )
    
    def _parse_arbitration(
        self,
        output: str,
        council_outputs: List[CouncilOutput]
    ) -> ArbiterDecision:
        """Parse arbitration output into structured decision."""
        lines = output.split('\n')
        
        resolution_notes = ""
        final_output = ""
        confidence = 0.8
        
        current_section = None
        final_output_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'resolution' in line_lower:
                current_section = 'resolution'
            elif 'final output' in line_lower:
                current_section = 'final_output'
            elif 'confidence' in line_lower:
                current_section = 'confidence'
                # Try to extract confidence value
                import re
                nums = re.findall(r'0\.\d+|\d+', line)
                if nums:
                    val = float(nums[0])
                    confidence = val if val <= 1 else val / 10
            elif current_section == 'resolution' and line.strip():
                resolution_notes += line.strip() + " "
            elif current_section == 'final_output' and line.strip():
                final_output_lines.append(line)
        
        final_output = '\n'.join(final_output_lines).strip()
        
        # If no final output section found, use the whole output
        if not final_output:
            final_output = output
        
        # Determine agreements/disagreements (simplified)
        agreements = [co.council_name for co in council_outputs if co.confidence > 0.7]
        disagreements = [co.council_name for co in council_outputs if co.confidence <= 0.7]
        
        return ArbiterDecision(
            final_output=final_output,
            resolution_notes=resolution_notes.strip(),
            confidence=confidence,
            council_agreements=agreements,
            council_disagreements=disagreements,
            raw_output=output
        )
    
    def _compute_final_scorecard(
        self,
        decision: ArbiterDecision,
        objective: str,
    ) -> ArbiterDecision:
        """
        Compute final synthesis scorecard for the arbiter decision.
        
        Sprint 1-2: Uses the judge ensemble to score the final output.
        
        Args:
            decision: The arbiter decision to score
            objective: Original mission objective
            
        Returns:
            ArbiterDecision with raw_quality_score updated
        """
        if not METRICS_AVAILABLE or get_metrics_config is None:
            return decision
        
        try:
            config = get_metrics_config()
            if not config.scorecard_enabled:
                return decision
            
            # Get the final output as string
            final_output = str(decision.final_output) if decision.final_output else ""
            if not final_output:
                return decision
            
            # Score the final synthesis
            judge_ensemble = get_judge_ensemble(config)
            result = judge_ensemble.score_final_synthesis(
                objective=objective,
                final_output=final_output,
                models_used=list(decision.confidence_breakdown.keys()),
                councils_used=decision.council_agreements + decision.council_disagreements,
            )
            
            scorecard = result.scorecard
            
            # Update the decision with the scorecard scores
            decision.raw_quality_score = scorecard.overall
            
            # Store full scorecard in cost_context for access
            if decision.cost_context is None:
                decision.cost_context = {}
            
            decision.cost_context["final_scorecard"] = {
                "overall": scorecard.overall,
                "goal_coverage": scorecard.goal_coverage,
                "evidence_grounding": scorecard.evidence_grounding,
                "actionability": scorecard.actionability,
                "consistency": scorecard.consistency,
                "judge_disagreement": scorecard.judge_disagreement,
            }
            
            # Compute composite utility score if not already set
            if decision.composite_utility_score == 0.0:
                decision.composite_utility_score = scorecard.overall
            
        except Exception as e:
            # Don't fail arbitration on scorecard error
            import logging
            logging.getLogger(__name__).debug(f"Final scorecard computation failed: {e}")
        
        return decision
    
    def _parse_enhanced_arbitration(
        self,
        output: str,
        council_outputs: List[CouncilOutput],
        optimist_output: Optional[CouncilOutput],
        skeptic_output: Optional[CouncilOutput]
    ) -> ArbiterDecision:
        """
        Parse enhanced arbitration output with multi-view synthesis.
        
        Args:
            output: Raw LLM output
            council_outputs: All council outputs
            optimist_output: Optimist council output if present
            skeptic_output: Skeptic council output if present
            
        Returns:
            ArbiterDecision with enhanced fields
        """
        import re
        
        # Initialize extracted values
        meta_analysis = ""
        optimist_summary = ""
        skeptic_summary = ""
        ranked_insights = []
        final_output = ""
        balanced_synthesis = ""
        resolution_notes = ""
        confidence = 0.8
        
        # Split into sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in output.split('\n'):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Detect section headers
            if line_stripped.startswith('###'):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                section_name = line_stripped.lstrip('#').strip().lower()
                current_section = section_name
                current_content = []
            else:
                if line_stripped:
                    current_content.append(line_stripped)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Extract specific sections
        for key, value in sections.items():
            if 'meta-analysis' in key or 'meta analysis' in key:
                meta_analysis = value
            elif 'optimist' in key and 'path' in key:
                optimist_summary = value
            elif 'skeptic' in key and 'path' in key:
                skeptic_summary = value
            elif 'ranked' in key or 'insight' in key:
                # Extract numbered insights
                for line in value.split('\n'):
                    match = re.match(r'^\d+[.\)]\s*(.+)$', line)
                    if match:
                        ranked_insights.append(match.group(1).strip())
            elif 'balanced' in key or 'synthesis' in key:
                balanced_synthesis = value
            elif 'final' in key and 'output' in key:
                final_output = value
            elif 'confidence' in key:
                nums = re.findall(r'0\.\d+|\d+\.?\d*', value)
                if nums:
                    val = float(nums[0])
                    confidence = val if val <= 1 else val / 10
        
        # Use balanced synthesis as final output if final output is empty
        if not final_output and balanced_synthesis:
            final_output = balanced_synthesis
        
        # Fallback to whole output
        if not final_output:
            final_output = output
        
        # Build confidence breakdown
        confidence_breakdown = {}
        for co in council_outputs:
            confidence_breakdown[co.council_name] = co.confidence
        
        # Determine agreements/disagreements
        agreements = [co.council_name for co in council_outputs if co.confidence > 0.7]
        disagreements = [co.council_name for co in council_outputs if co.confidence <= 0.7]
        
        # Resolution notes from meta-analysis or balanced synthesis
        resolution_notes = meta_analysis[:500] if meta_analysis else balanced_synthesis[:500]
        
        return ArbiterDecision(
            final_output=final_output,
            resolution_notes=resolution_notes,
            confidence=confidence,
            council_agreements=agreements,
            council_disagreements=disagreements,
            raw_output=output,
            meta_analysis=meta_analysis,
            ranked_insights=ranked_insights[:10],  # Limit to 10
            optimist_summary=optimist_summary,
            skeptic_summary=skeptic_summary,
            confidence_breakdown=confidence_breakdown
        )
    
    def resolve_code_conflict(
        self,
        code_variants: List[str],
        objective: str,
        evaluation_results: Optional[List[Any]] = None
    ) -> ArbiterDecision:
        """
        Specialized method to resolve conflicts between code variants.
        
        Args:
            code_variants: List of code variants to choose from
            objective: Original coding objective
            evaluation_results: Optional evaluation results for each variant
            
        Returns:
            ArbiterDecision with selected/merged code
        """
        # Build code comparison
        code_sections = []
        for i, code in enumerate(code_variants):
            eval_info = ""
            if evaluation_results and i < len(evaluation_results):
                eval_info = f"\nEvaluation: {evaluation_results[i]}"
            code_sections.append(f"### VARIANT {i+1}{eval_info}\n```python\n{code}\n```")
        
        prompt = f"""Select or merge the best code variant for this objective:

## OBJECTIVE
{objective}

## CODE VARIANTS

{chr(10).join(code_sections)}

## INSTRUCTIONS
1. Compare the code variants
2. Evaluate correctness, quality, and completeness
3. Select the best variant OR merge the best elements
4. Output ONLY the final code in a Python code block

## FINAL CODE:"""

        try:
            llm = self._get_llm()
            
            if USE_CHAT_OLLAMA:
                messages = [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=prompt)
                ]
                response = llm.invoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                full_prompt = f"{self._system_prompt}\n\n{prompt}"
                response = llm.invoke(full_prompt)
                output = str(response)
            
            # Extract code from output
            import re
            code_blocks = re.findall(r'```python\n(.*?)```', output, re.DOTALL)
            final_code = code_blocks[0] if code_blocks else output
            
            return ArbiterDecision(
                final_output=final_code.strip(),
                resolution_notes="Code conflict resolved by arbiter",
                confidence=0.85,
                raw_output=output
            )
            
        except Exception as e:
            # Fallback: return first variant
            return ArbiterDecision(
                final_output=code_variants[0] if code_variants else "",
                resolution_notes=f"Fallback to first variant (error: {str(e)})",
                confidence=0.6,
                raw_output=""
            )

    def decide_output_spec(
        self,
        mission_state: "MissionState",
        phase_artifacts: Dict[str, Dict[str, str]]
    ) -> OutputSpec:
        """
        Decide the best primary and secondary output formats for this mission.
        
        Heuristics:
          - If objective mentions 'report', 'analysis', 'brief' → PDF/Markdown report.
          - If objective mentions 'build an app', 'code', 'implementation' → CODE_REPO.
          - If artifacts contain 'graph_*' → add NETWORK_GRAPH_JSON as secondary.
          - If objective mentions 'presentation', 'slides' → SLIDE_DECK_PPTX.
        
        Args:
            mission_state: Current mission state
            phase_artifacts: Artifacts from all phases
            
        Returns:
            OutputSpec with recommended formats
        """
        objective_lower = mission_state.objective.lower()
        primary = OutputFormat.MARKDOWN_REPORT
        secondary: List[OutputFormat] = []

        # Determine primary format based on objective keywords
        if any(k in objective_lower for k in ["report", "analysis", "snapshot", "brief", "review"]):
            primary = OutputFormat.PDF_REPORT
            secondary = [OutputFormat.MARKDOWN_REPORT]
        elif any(k in objective_lower for k in ["build", "implement", "code", "application", "backend", "frontend", "app"]):
            primary = OutputFormat.CODE_REPO
            secondary = [OutputFormat.MARKDOWN_REPORT]
        elif any(k in objective_lower for k in ["network", "graph", "relations", "graph analysis"]):
            primary = OutputFormat.NETWORK_GRAPH_JSON
            secondary = [OutputFormat.MARKDOWN_REPORT]

        # Check for slides/presentation request
        if any(k in objective_lower for k in ["slides", "presentation", "deck", "ppt"]):
            if primary != OutputFormat.SLIDE_DECK_PPTX:
                secondary.append(OutputFormat.SLIDE_DECK_PPTX)
            else:
                primary = OutputFormat.SLIDE_DECK_PPTX

        # Check if any phase produced graph data
        has_graph_data = False
        for phase_name, artifacts_dict in phase_artifacts.items():
            for key in artifacts_dict.keys():
                if "graph" in key.lower():
                    has_graph_data = True
                    break
            if has_graph_data:
                break
        
        if has_graph_data and OutputFormat.NETWORK_GRAPH_JSON not in secondary:
            if primary != OutputFormat.NETWORK_GRAPH_JSON:
                secondary.append(OutputFormat.NETWORK_GRAPH_JSON)

        # Deduplicate secondary formats
        seen = set()
        deduped = []
        for fmt in secondary:
            if fmt not in seen and fmt != primary:
                seen.add(fmt)
                deduped.append(fmt)

        return OutputSpec(
            primary_format=primary,
            secondary_formats=deduped,
            domain_hint=None,
        )

