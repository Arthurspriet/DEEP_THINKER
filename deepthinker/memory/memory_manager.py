"""
Memory Manager for DeepThinker Memory System.

High-level orchestrator providing unified access to:
- Structured mission state
- Per-mission RAG store
- Global RAG store
- Long-term summary memory

This is the single access point for MissionOrchestrator, ReasoningSupervisor,
and MetaController.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .schemas import (
    HypothesisSchema,
    EvidenceSchema,
    PhaseOutputSchema,
    SupervisorSignalsSchema,
    ReflectionSchema,
    DebateSchema,
    PlanRevisionSchema,
    MissionSummarySchema,
    CouncilFeedbackSchema,
)
from .structured_state import StructuredMissionState
from .rag_store import MissionRAGStore, GlobalRAGStore
from .summary_memory import SummaryMemory
from .general_knowledge_store import GeneralKnowledgeStore

if TYPE_CHECKING:
    from ..meta.supervisor import PhaseMetrics

logger = logging.getLogger(__name__)

# Try to import verbose logger for warnings
try:
    from ..cli import verbose_logger
    VERBOSE_AVAILABLE = True
except ImportError:
    VERBOSE_AVAILABLE = False
    verbose_logger = None


class MemoryManager:
    """
    Unified memory orchestrator for DeepThinker missions.
    
    Provides:
    - Runtime writing of phase outputs, reflections, debates, evidence
    - End-of-mission persistence with global RAG update
    - Pre-mission retrieval of relevant past insights
    - Fail-safe operation - errors logged but never crash missions
    
    Usage:
        # At mission creation
        memory = MemoryManager(
            mission_id=state.mission_id,
            objective=state.objective,
            mission_type=state.mission_type,
            time_budget_minutes=state.constraints.time_budget_minutes,
            base_dir=Path(os.getenv("DEEPTHINKER_KB_DIR", "kb")),
            embedding_fn=model_pool.get_embedding_fn(),
        )
        
        # After each phase
        memory.add_phase_output(...)
        memory.save_checkpoint()
        
        # At mission completion
        memory.save_mission()
        memory.update_global_rag()
        memory.save_long_term_summary(summary)
    """
    
    def __init__(
        self,
        mission_id: str,
        objective: str,
        mission_type: Optional[str] = None,
        time_budget_minutes: Optional[float] = None,
        base_dir: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_model: str = "snowflake-arctic-embed:latest",
        ollama_base_url: str = "http://localhost:11434",
        auto_load: bool = True,
    ):
        """
        Initialize memory manager for a mission.
        
        Args:
            mission_id: Unique mission identifier
            objective: Mission objective text
            mission_type: Type of mission (research, coding, strategic, etc.)
            time_budget_minutes: Time budget for the mission
            base_dir: Base directory for storage (default: kb/)
            embedding_fn: Optional custom embedding function
            embedding_model: Ollama embedding model
            ollama_base_url: Ollama server URL
            auto_load: Whether to auto-load existing state if present
        """
        self.mission_id = mission_id
        self.objective = objective
        self.mission_type = mission_type
        self.time_budget_minutes = time_budget_minutes
        self.base_dir = base_dir or Path("kb")
        
        # Store embedding config
        self._embedding_fn = embedding_fn
        self._embedding_model = embedding_model
        self._ollama_base_url = ollama_base_url
        
        # Initialize components
        self._init_state(auto_load)
        self._init_rag_stores()
        self._init_summary_memory()
        
        # Track initialization success
        self._initialized = True
    
    def _init_state(self, auto_load: bool) -> None:
        """Initialize structured mission state."""
        try:
            if auto_load:
                existing = StructuredMissionState.load(
                    mission_id=self.mission_id,
                    base_dir=self.base_dir
                )
                if existing:
                    self.state = existing
                    logger.debug(f"Loaded existing state for mission {self.mission_id}")
                    return
            
            self.state = StructuredMissionState(
                mission_id=self.mission_id,
                objective=self.objective,
                mission_type=self.mission_type,
                time_budget_minutes=self.time_budget_minutes,
                base_dir=self.base_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize state, creating fresh: {e}")
            self.state = StructuredMissionState(
                mission_id=self.mission_id,
                objective=self.objective,
                mission_type=self.mission_type,
                time_budget_minutes=self.time_budget_minutes,
                base_dir=self.base_dir,
            )
    
    def _init_rag_stores(self) -> None:
        """Initialize RAG stores."""
        try:
            self.mission_rag = MissionRAGStore(
                mission_id=self.mission_id,
                base_dir=self.base_dir,
                embedding_fn=self._embedding_fn,
                embedding_model=self._embedding_model,
                ollama_base_url=self._ollama_base_url,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize mission RAG: {e}")
            self.mission_rag = None
        
        try:
            self.global_rag = GlobalRAGStore(
                base_dir=self.base_dir,
                embedding_fn=self._embedding_fn,
                embedding_model=self._embedding_model,
                ollama_base_url=self._ollama_base_url,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize global RAG: {e}")
            self.global_rag = None
        
        try:
            self.general_knowledge = GeneralKnowledgeStore(
                base_dir=self.base_dir,
                embedding_fn=self._embedding_fn,
                embedding_model=self._embedding_model,
                ollama_base_url=self._ollama_base_url,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize general knowledge store: {e}")
            self.general_knowledge = None
    
    def _init_summary_memory(self) -> None:
        """Initialize summary memory."""
        try:
            self.summary_memory = SummaryMemory(
                base_dir=self.base_dir,
                embedding_fn=self._embedding_fn,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize summary memory: {e}")
            self.summary_memory = None
    
    # =========================================================================
    # Runtime Writing Methods
    # =========================================================================
    
    def add_phase_output(
        self,
        phase_name: str,
        summary: Optional[str] = None,
        final_output: Optional[str] = None,
        quality_score: Optional[float] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        council_used: Optional[str] = None,
        models_used: Optional[List[str]] = None,
        iteration_count: int = 0,
    ) -> bool:
        """
        Add phase output to memory.
        
        Args:
            phase_name: Name of the phase
            summary: Optional summary of the phase
            final_output: Final output text
            quality_score: Quality score (0-1)
            artifacts: Phase artifacts dict
            duration_seconds: Execution duration
            council_used: Council that executed the phase
            models_used: Models used in execution
            iteration_count: Number of iterations
            
        Returns:
            True if successful
        """
        try:
            output = PhaseOutputSchema(
                phase_name=phase_name,
                summary=summary,
                final_output=final_output,
                quality_score=quality_score,
                artifacts=artifacts or {},
                duration_seconds=duration_seconds,
                council_used=council_used,
                models_used=models_used or [],
                iteration_count=iteration_count,
            )
            self.state.add_phase_output(output)
            
            # Also add to RAG if we have substantial output
            if self.mission_rag and final_output and len(final_output) > 100:
                self.mission_rag.add_text(
                    text=final_output[:5000],  # Limit size
                    phase=phase_name,
                    artifact_type="phase_output",
                    tags=[phase_name, council_used or "unknown"],
                )
            
            return True
        except Exception as e:
            self._log_warning(f"Failed to add phase output: {e}")
            return False
    
    def add_reflection(
        self,
        phase_name: str,
        assumptions: Optional[List[str]] = None,
        risks: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
        missing_info: Optional[List[str]] = None,
        contradictions: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> bool:
        """
        Add reflection results to memory.
        
        Args:
            phase_name: Phase that was reflected on
            assumptions: Identified assumptions
            risks: Identified risks
            weaknesses: Identified weaknesses
            missing_info: Missing information
            contradictions: Contradictions found
            questions: Open questions
            suggestions: Improvement suggestions
            
        Returns:
            True if successful
        """
        try:
            reflection = ReflectionSchema(
                phase_name=phase_name,
                assumptions=assumptions or [],
                risks=risks or [],
                weaknesses=weaknesses or [],
                missing_info=missing_info or [],
                contradictions=contradictions or [],
                questions=questions or [],
                suggestions=suggestions or [],
                created_at=datetime.utcnow(),
            )
            self.state.add_reflection(reflection)
            return True
        except Exception as e:
            self._log_warning(f"Failed to add reflection: {e}")
            return False
    
    def add_debate(
        self,
        phase_name: str,
        optimist_view: Optional[str] = None,
        skeptic_view: Optional[str] = None,
        key_points: Optional[List[str]] = None,
        contradictions_found: Optional[List[str]] = None,
        confidence_adjustments: Optional[Dict[str, float]] = None,
        agreement_score: Optional[float] = None,
    ) -> bool:
        """
        Add debate results to memory.
        
        Args:
            phase_name: Phase that was debated
            optimist_view: Optimist perspective summary
            skeptic_view: Skeptic perspective summary
            key_points: Key debate points
            contradictions_found: Contradictions identified
            confidence_adjustments: Hypothesis confidence changes
            agreement_score: Agreement between perspectives
            
        Returns:
            True if successful
        """
        try:
            debate = DebateSchema(
                phase_name=phase_name,
                optimist_view=optimist_view,
                skeptic_view=skeptic_view,
                key_points=key_points or [],
                contradictions_found=contradictions_found or [],
                confidence_adjustments=confidence_adjustments or {},
                agreement_score=agreement_score,
                created_at=datetime.utcnow(),
            )
            self.state.add_debate(debate)
            return True
        except Exception as e:
            self._log_warning(f"Failed to add debate: {e}")
            return False
    
    def add_plan_revision(
        self,
        phase_name: str,
        added_subgoals: Optional[List[str]] = None,
        removed_subgoals: Optional[List[str]] = None,
        priority_changes: Optional[List[str]] = None,
        next_actions: Optional[List[str]] = None,
    ) -> bool:
        """
        Add plan revision to memory.
        
        Args:
            phase_name: Phase that triggered revision
            added_subgoals: New subgoals added
            removed_subgoals: Subgoals removed
            priority_changes: Priority adjustments
            next_actions: Recommended next actions
            
        Returns:
            True if successful
        """
        try:
            revision = PlanRevisionSchema(
                phase_name=phase_name,
                added_subgoals=added_subgoals or [],
                removed_subgoals=removed_subgoals or [],
                priority_changes=priority_changes or [],
                next_actions=next_actions or [],
                created_at=datetime.utcnow(),
            )
            self.state.add_plan_revision(revision)
            return True
        except Exception as e:
            self._log_warning(f"Failed to add plan revision: {e}")
            return False
    
    def add_evidence_text(
        self,
        text: str,
        phase: str,
        artifact_type: str = "general",
        hypothesis_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 0.5,
        source: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add evidence text to mission RAG.
        
        Args:
            text: Evidence text
            phase: Source phase
            artifact_type: Type of artifact
            hypothesis_id: Related hypothesis
            tags: Tags for filtering
            confidence: Confidence score
            source: Source of evidence
            
        Returns:
            Evidence ID if successful, None otherwise
        """
        if not self.mission_rag:
            return None
        
        try:
            evidence = EvidenceSchema(
                id=f"ev_{self.mission_id[:8]}_{datetime.utcnow().timestamp():.0f}",
                mission_id=self.mission_id,
                phase=phase,
                text=text,
                artifact_type=artifact_type,
                hypothesis_id=hypothesis_id,
                tags=tags or [],
                confidence=confidence,
                source=source,
                created_at=datetime.utcnow(),
            )
            return self.mission_rag.add_evidence(evidence)
        except Exception as e:
            self._log_warning(f"Failed to add evidence: {e}")
            return None
    
    def add_hypothesis(
        self,
        statement: str,
        confidence: float = 0.5,
        hypothesis_id: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        parent_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Add a hypothesis to memory.
        
        Args:
            statement: Hypothesis statement
            confidence: Initial confidence (0-1)
            hypothesis_id: Optional custom ID
            evidence_ids: Supporting evidence IDs
            parent_ids: Parent hypothesis IDs (for DAG structure)
            
        Returns:
            Hypothesis ID if successful
        """
        try:
            hyp_id = hypothesis_id or f"hyp_{len(self.state.hypotheses) + 1}"
            
            hypothesis = HypothesisSchema(
                id=hyp_id,
                statement=statement,
                confidence=confidence,
                status="active",
                evidence_ids=evidence_ids or [],
                parent_ids=parent_ids or [],
                created_at=datetime.utcnow(),
            )
            self.state.add_hypothesis(hypothesis)
            return hyp_id
        except Exception as e:
            self._log_warning(f"Failed to add hypothesis: {e}")
            return None
    
    def update_hypothesis(
        self,
        hypothesis_id: str,
        confidence: Optional[float] = None,
        evidence_id: Optional[str] = None,
        is_contradiction: bool = False,
    ) -> bool:
        """
        Update an existing hypothesis.
        
        Args:
            hypothesis_id: Hypothesis to update
            confidence: New confidence value
            evidence_id: New evidence supporting/contradicting
            is_contradiction: Whether evidence contradicts
            
        Returns:
            True if successful
        """
        try:
            if confidence is not None:
                return self.state.update_hypothesis_confidence(
                    hypothesis_id=hypothesis_id,
                    confidence=confidence,
                    evidence_id=evidence_id,
                    is_contradiction=is_contradiction,
                )
            return False
        except Exception as e:
            self._log_warning(f"Failed to update hypothesis: {e}")
            return False
    
    def record_supervisor_signals(
        self,
        difficulty: float,
        uncertainty: float,
        progress: float,
        novelty: float,
        confidence: float,
        model_tier: str = "medium",
        convergence: Optional[float] = None,
        loop_detected: bool = False,
        stagnation_count: int = 0,
    ) -> bool:
        """
        Record supervisor signals from phase analysis.
        
        Args:
            difficulty: Difficulty score (0-1)
            uncertainty: Uncertainty score (0-1)
            progress: Progress score (0-1)
            novelty: Novelty score (0-1)
            confidence: Confidence score (0-1)
            model_tier: Model tier used
            convergence: Convergence score
            loop_detected: Whether loop was detected
            stagnation_count: Stagnation iteration count
            
        Returns:
            True if successful
        """
        try:
            self.state.record_supervisor_signals(
                difficulty=difficulty,
                uncertainty=uncertainty,
                progress=progress,
                novelty=novelty,
                confidence=confidence,
                model_tier=model_tier,
                convergence=convergence,
            )
            
            if loop_detected or stagnation_count > 0:
                self.state.record_loop_detection(
                    loop_detected=loop_detected,
                    stagnation_count=stagnation_count,
                )
            
            return True
        except Exception as e:
            self._log_warning(f"Failed to record supervisor signals: {e}")
            return False
    
    def record_from_phase_metrics(self, metrics: "PhaseMetrics", model_tier: str = "medium") -> bool:
        """
        Record from ReasoningSupervisor PhaseMetrics.
        
        Args:
            metrics: PhaseMetrics from ReasoningSupervisor
            model_tier: Model tier used
            
        Returns:
            True if successful
        """
        try:
            return self.record_supervisor_signals(
                difficulty=metrics.difficulty_score,
                uncertainty=metrics.uncertainty_score,
                progress=metrics.progress_score,
                novelty=metrics.novelty_score,
                confidence=metrics.confidence_score,
                model_tier=model_tier,
            )
        except Exception as e:
            self._log_warning(f"Failed to record from phase metrics: {e}")
            return False
    
    def add_council_feedback(
        self,
        council_name: str,
        phase_name: str,
        success: bool,
        output_summary: Optional[str] = None,
        models_used: Optional[List[str]] = None,
        duration_seconds: Optional[float] = None,
        error: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> bool:
        """
        Add council execution feedback.
        
        Args:
            council_name: Name of council
            phase_name: Phase name
            success: Whether execution succeeded
            output_summary: Summary of output
            models_used: Models used
            duration_seconds: Execution duration
            error: Error message if failed
            quality_score: Quality score
            
        Returns:
            True if successful
        """
        try:
            feedback = CouncilFeedbackSchema(
                council_name=council_name,
                phase_name=phase_name,
                success=success,
                output_summary=output_summary,
                models_used=models_used or [],
                duration_seconds=duration_seconds,
                error=error,
                quality_score=quality_score,
                created_at=datetime.utcnow(),
            )
            self.state.add_council_feedback(feedback)
            return True
        except Exception as e:
            self._log_warning(f"Failed to add council feedback: {e}")
            return False
    
    def save_checkpoint(self) -> bool:
        """
        Save current state checkpoint.
        
        Should be called after each phase for resumability.
        
        Returns:
            True if successful
        """
        try:
            state_saved = self.state.save()
            rag_saved = self.mission_rag.persist() if self.mission_rag else True
            return state_saved and rag_saved
        except Exception as e:
            self._log_warning(f"Failed to save checkpoint: {e}")
            return False
    
    # =========================================================================
    # End of Mission Methods
    # =========================================================================
    
    def save_mission(self) -> bool:
        """
        Save complete mission state.
        
        Called at mission completion.
        
        Returns:
            True if successful
        """
        try:
            state_saved = self.state.save()
            rag_saved = self.mission_rag.persist() if self.mission_rag else True
            return state_saved and rag_saved
        except Exception as e:
            self._log_warning(f"Failed to save mission: {e}")
            return False
    
    def update_global_rag(
        self,
        min_confidence: float = 0.5,
        max_documents: int = 50,
    ) -> int:
        """
        Update global RAG with mission evidence.
        
        Args:
            min_confidence: Minimum confidence for inclusion
            max_documents: Maximum documents to add
            
        Returns:
            Number of documents added
        """
        if not self.global_rag or not self.mission_rag:
            return 0
        
        try:
            added = self.global_rag.add_from_mission(
                mission_rag=self.mission_rag,
                min_confidence=min_confidence,
                max_documents=max_documents,
            )
            self.global_rag.persist()
            return added
        except Exception as e:
            self._log_warning(f"Failed to update global RAG: {e}")
            return 0
    
    def save_long_term_summary(self, summary: MissionSummarySchema) -> bool:
        """
        Save mission summary to long-term memory.
        
        Args:
            summary: Mission summary schema
            
        Returns:
            True if successful
        """
        if not self.summary_memory:
            return False
        
        try:
            self.summary_memory.add_summary(summary)
            return self.summary_memory.save()
        except Exception as e:
            self._log_warning(f"Failed to save long-term summary: {e}")
            return False
    
    def build_mission_summary(
        self,
        final_quality_score: Optional[float] = None,
        convergence_score: Optional[float] = None,
        time_taken_minutes: Optional[float] = None,
    ) -> MissionSummarySchema:
        """
        Build mission summary from current state.
        
        Args:
            final_quality_score: Final quality score
            convergence_score: Final convergence score
            time_taken_minutes: Total time taken
            
        Returns:
            MissionSummarySchema ready for storage
        """
        # Extract key insights from phase outputs
        key_insights = []
        for phase_name, output in self.state.phase_outputs.items():
            if isinstance(output, PhaseOutputSchema) and output.summary:
                key_insights.append(f"{phase_name}: {output.summary[:200]}")
        
        # Get hypothesis summaries
        resolved = [
            h.statement[:200] for h in self.state.get_confirmed_hypotheses()
        ]
        unresolved = [
            h.statement[:200] for h in self.state.get_active_hypotheses()
        ]
        
        # Get contradictions
        contradictions = self.state.get_all_contradictions()[:10]
        
        # Get models used
        models_used = list(set(self.state.supervisor_signals.model_tiers_used))
        
        return MissionSummarySchema(
            mission_id=self.mission_id,
            objective=self.objective,
            mission_type=self.mission_type,
            domain=self.state.domain,
            key_insights=key_insights[:10],
            resolved_hypotheses=resolved[:5],
            unresolved_hypotheses=unresolved[:5],
            contradictions=contradictions,
            final_quality_score=final_quality_score,
            convergence_score=convergence_score,
            phases_completed=self.state.total_phases_completed,
            total_iterations=self.state.iteration_count,
            time_taken_minutes=time_taken_minutes,
            models_used=models_used,
            created_at=self.state.created_at,
            completed_at=datetime.utcnow(),
            tags=self.state.tags,
        )
    
    # =========================================================================
    # Pre-Mission Retrieval Methods
    # =========================================================================
    
    def retrieve_relevant_past_insights(
        self,
        objective: str,
        limit: int = 5,
    ) -> List[Tuple[MissionSummarySchema, float]]:
        """
        Retrieve relevant past mission insights for a new objective.
        
        Used at mission start to inform strategy.
        
        Args:
            objective: New mission objective
            limit: Maximum results
            
        Returns:
            List of (summary, relevance_score) tuples
        """
        if not self.summary_memory:
            return []
        
        try:
            return self.summary_memory.query_by_objective_similarity(
                objective=objective,
                limit=limit,
            )
        except Exception as e:
            self._log_warning(f"Failed to retrieve past insights: {e}")
            return []
    
    def retrieve_relevant_evidence(
        self,
        objective: str,
        limit: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant evidence from global RAG.
        
        Args:
            objective: Mission objective for relevance
            limit: Maximum results
            
        Returns:
            List of (evidence_doc, score) tuples
        """
        if not self.global_rag:
            return []
        
        try:
            return self.global_rag.search_global(
                query=objective,
                top_k=limit,
            )
        except Exception as e:
            self._log_warning(f"Failed to retrieve evidence: {e}")
            return []
    
    def seed_initial_hypotheses(
        self,
        objective: str,
        limit: int = 3,
    ) -> List[HypothesisSchema]:
        """
        Generate seed hypotheses from past mission patterns.
        
        Args:
            objective: New mission objective
            limit: Maximum hypotheses to generate
            
        Returns:
            List of seed hypotheses
        """
        seeded = []
        
        try:
            # Get similar past missions
            past_insights = self.retrieve_relevant_past_insights(objective, limit=3)
            
            for summary, score in past_insights:
                if score < 0.4:
                    continue
                
                # Create hypothesis from resolved hypotheses
                for resolved in summary.resolved_hypotheses[:1]:
                    hyp = HypothesisSchema(
                        id=f"seed_{len(seeded) + 1}",
                        statement=f"Based on similar mission: {resolved}",
                        confidence=0.4 + (score * 0.2),  # Base + relevance boost
                        status="active",
                        created_at=datetime.utcnow(),
                    )
                    seeded.append(hyp)
                    
                    if len(seeded) >= limit:
                        break
                
                if len(seeded) >= limit:
                    break
            
            # Add to state
            for hyp in seeded:
                self.state.add_hypothesis(hyp)
            
        except Exception as e:
            self._log_warning(f"Failed to seed hypotheses: {e}")
        
        return seeded
    
    def get_domain_context(self, domain: str) -> Dict[str, Any]:
        """
        Get context from past missions in a domain.
        
        Args:
            domain: Domain to query
            
        Returns:
            Context dictionary with insights and patterns
        """
        context = {
            "insights": [],
            "common_contradictions": [],
            "past_missions": 0,
        }
        
        if not self.summary_memory:
            return context
        
        try:
            # Get insights
            context["insights"] = self.summary_memory.get_insights_for_domain(
                domain=domain,
                limit=10,
            )
            
            # Get common contradictions
            contradictions = self.summary_memory.get_common_contradictions(
                domain=domain,
                limit=5,
            )
            context["common_contradictions"] = [c for c, _ in contradictions]
            
            # Count past missions
            summaries = self.summary_memory.query_by_domain(domain, limit=100)
            context["past_missions"] = len(summaries)
            
        except Exception as e:
            self._log_warning(f"Failed to get domain context: {e}")
        
        return context
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def search_mission_evidence(
        self,
        query: str,
        top_k: int = 6,
        phase_filter: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search mission RAG for relevant evidence.
        
        Args:
            query: Search query
            top_k: Maximum results
            phase_filter: Optional phase filter
            
        Returns:
            List of (document, score) tuples
        """
        if not self.mission_rag:
            return []
        
        try:
            return self.mission_rag.search(
                query=query,
                top_k=top_k,
                phase_filter=phase_filter,
            )
        except Exception as e:
            self._log_warning(f"Failed to search mission evidence: {e}")
            return []
    
    def search_general_knowledge(
        self,
        query: str,
        top_k: int = 5,
        country_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search general knowledge store (e.g., CIA World Factbook).
        
        Args:
            query: Search query
            top_k: Maximum results
            country_filter: Optional country name filter
            category_filter: Optional category filter (geography, economy, etc.)
            
        Returns:
            List of (document, score) tuples
        """
        if not self.general_knowledge or not self.general_knowledge.is_loaded():
            return []
        
        try:
            return self.general_knowledge.search(
                query=query,
                top_k=top_k,
                country_filter=country_filter,
                category_filter=category_filter,
            )
        except Exception as e:
            self._log_warning(f"Failed to search general knowledge: {e}")
            return []
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current memory state.
        
        Returns:
            Summary dictionary
        """
        return {
            "mission_id": self.mission_id,
            "state_stats": self.state.get_summary_stats(),
            "mission_rag_docs": self.mission_rag.get_document_count() if self.mission_rag else 0,
            "global_rag_docs": self.global_rag.get_document_count() if self.global_rag else 0,
            "general_knowledge_docs": self.general_knowledge.get_document_count() if self.general_knowledge else 0,
            "general_knowledge_loaded": self.general_knowledge.is_loaded() if self.general_knowledge else False,
            "summary_memory_stats": self.summary_memory.get_statistics() if self.summary_memory else {},
        }
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _log_warning(self, message: str) -> None:
        """Log warning via appropriate channel."""
        logger.warning(message)
        
        # Also log via verbose logger if available
        if VERBOSE_AVAILABLE and verbose_logger and verbose_logger.enabled:
            try:
                if hasattr(verbose_logger, '_print'):
                    verbose_logger._print(f"⚠️  Memory: {message}")
            except Exception:
                pass
    
    def set_domain(self, domain: str) -> None:
        """Set domain classification for the mission."""
        self.state.domain = domain
    
    def add_tags(self, tags: List[str]) -> None:
        """Add tags to the mission."""
        self.state.tags.extend(tags)
        self.state.tags = list(set(self.state.tags))  # Deduplicate
    
    def increment_iteration(self) -> int:
        """Increment iteration count and return new value."""
        self.state.iteration_count += 1
        return self.state.iteration_count
    
    # =========================================================================
    # Memory Reasoning for Prompt Injection
    # =========================================================================
    
    def reason_over(
        self,
        memory_items: Optional[List[Tuple[Any, float]]] = None,
        objective: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Reason over memory items to extract actionable insights for prompts.
        
        This method processes retrieved memory items and structures them
        for injection into council prompts, making memory usage explicit
        and visible.
        
        Args:
            memory_items: List of (item, relevance_score) tuples from retrieval
                         If None, retrieves based on objective
            objective: Objective for relevance retrieval (if memory_items is None)
            limit: Maximum items to process
            
        Returns:
            Dictionary with:
            - prior_knowledge: List of relevant knowledge snippets
            - known_gaps: List of identified gaps or missing info
            - memory_used_count: Number of memory items used
            - memory_items_titles: Short titles/summaries of items
            - memory_sources: Source artifacts/missions
            - used_in_prompt: Whether memory was included
        """
        result = {
            "prior_knowledge": [],
            "known_gaps": [],
            "memory_used_count": 0,
            "memory_items_titles": [],
            "memory_sources": [],
            "used_in_prompt": False,
        }
        
        try:
            # Retrieve memory if not provided
            if memory_items is None and objective:
                # Get from mission RAG
                mission_results = self.search_mission_evidence(
                    query=objective,
                    top_k=limit,
                )
                # Get from global RAG
                global_results = self.retrieve_relevant_evidence(
                    objective=objective,
                    limit=limit,
                )
                # Get past insights
                past_insights = self.retrieve_relevant_past_insights(
                    objective=objective,
                    limit=3,
                )
                # Get from general knowledge (CIA Factbook, etc.)
                general_results = self.search_general_knowledge(
                    query=objective,
                    top_k=limit,
                )
                
                # Combine results
                memory_items = []
                for doc, score in mission_results:
                    memory_items.append((doc, score))
                for doc, score in global_results:
                    memory_items.append((doc, score))
                for summary, score in past_insights:
                    memory_items.append((summary, score))
                for doc, score in general_results:
                    memory_items.append((doc, score))
            
            if not memory_items:
                return result
            
            # Process memory items
            for item, score in memory_items[:limit]:
                if score < 0.3:  # Skip low relevance
                    continue
                
                # Extract title/summary
                title = self._extract_memory_title(item)
                if title:
                    result["memory_items_titles"].append(title)
                
                # Extract knowledge
                knowledge = self._extract_knowledge(item)
                if knowledge:
                    result["prior_knowledge"].append(knowledge)
                
                # Extract gaps (from past mission summaries)
                gaps = self._extract_gaps(item)
                if gaps:
                    result["known_gaps"].extend(gaps)
                
                # Extract source
                source = self._extract_source(item)
                if source:
                    result["memory_sources"].append(source)
            
            result["memory_used_count"] = len(result["prior_knowledge"])
            result["used_in_prompt"] = result["memory_used_count"] > 0
            
            # Deduplicate known gaps
            result["known_gaps"] = list(set(result["known_gaps"]))[:5]
            
        except Exception as e:
            self._log_warning(f"Failed to reason over memory: {e}")
        
        return result
    
    def _extract_memory_title(self, item: Any) -> Optional[str]:
        """Extract a short title from a memory item."""
        if isinstance(item, dict):
            # Document from RAG
            if "text" in item:
                return item["text"][:100]
            if "title" in item:
                return item["title"]
        
        if hasattr(item, 'objective'):
            return item.objective[:100]
        
        if hasattr(item, 'statement'):
            return item.statement[:100]
        
        return str(item)[:100] if item else None
    
    def _extract_knowledge(self, item: Any) -> Optional[str]:
        """Extract prior knowledge from a memory item."""
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                return text[:500]
        
        # MissionSummarySchema
        if hasattr(item, 'key_insights'):
            insights = item.key_insights or []
            if insights:
                return "; ".join(insights[:3])
        
        # HypothesisSchema
        if hasattr(item, 'statement'):
            return f"Hypothesis (conf={getattr(item, 'confidence', 0.5):.1f}): {item.statement}"
        
        return None
    
    def _extract_gaps(self, item: Any) -> List[str]:
        """Extract known gaps from a memory item."""
        gaps = []
        
        # MissionSummarySchema
        if hasattr(item, 'unresolved_hypotheses'):
            gaps.extend(item.unresolved_hypotheses or [])
        
        if hasattr(item, 'contradictions'):
            gaps.extend(item.contradictions or [])
        
        # ReflectionSchema
        if hasattr(item, 'missing_info'):
            gaps.extend(item.missing_info or [])
        
        return gaps[:5]
    
    def _extract_source(self, item: Any) -> Optional[str]:
        """Extract source identifier from a memory item."""
        if isinstance(item, dict):
            # Check for general knowledge source
            if item.get("source") == "cia_world_factbook":
                country = item.get("country", "unknown")
                category = item.get("category", "")
                return f"CIA Factbook: {country}/{category}"
            return item.get("mission_id") or item.get("id")
        
        if hasattr(item, 'mission_id'):
            return item.mission_id
        
        if hasattr(item, 'id'):
            return item.id
        
        return None
    
    def format_for_prompt(
        self,
        memory_summary: Dict[str, Any],
        include_gaps: bool = True,
    ) -> str:
        """
        Format memory summary for prompt injection.
        
        Args:
            memory_summary: Output from reason_over()
            include_gaps: Whether to include known gaps
            
        Returns:
            Formatted string for prompt injection
        """
        if not memory_summary.get("used_in_prompt"):
            return ""
        
        parts = []
        
        # Prior knowledge
        if memory_summary.get("prior_knowledge"):
            parts.append("## Prior Knowledge from Memory")
            for i, knowledge in enumerate(memory_summary["prior_knowledge"][:5], 1):
                source = memory_summary["memory_sources"][i-1] if i <= len(memory_summary.get("memory_sources", [])) else "unknown"
                parts.append(f"{i}. [from memory: {source}] {knowledge}")
        
        # Known gaps
        if include_gaps and memory_summary.get("known_gaps"):
            parts.append("\n## Known Gaps to Address")
            for gap in memory_summary["known_gaps"][:3]:
                parts.append(f"- {gap}")
        
        return "\n".join(parts) if parts else ""

