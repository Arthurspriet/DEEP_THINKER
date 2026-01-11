"""
Evaluator Council Implementation for DeepThinker 2.0.

Returns EvaluationResult with multiple evaluators scoring independently.
Uses weighted blend consensus for balanced evaluation.

Enhanced with:
- ResearchEvaluation for evaluating research findings
- Actionable deltas (gaps, questions, evidence_requests) for iteration control
"""

import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..base_council import BaseCouncil, CouncilResult
from ...models.model_pool import ModelPool
from ...models.council_model_config import EVALUATOR_MODELS
from ...consensus.weighted_blend import WeightedBlendConsensus
from ...evaluation.evaluation_result import EvaluationResult, IssueItem
from ...evaluation.result_parser import EvaluationResultParser

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorContext:
    """Context for evaluator council execution."""
    
    objective: str
    content_to_evaluate: str  # Renamed from 'code' to match EvaluatorContextSchema
    metric_result: Optional[Any] = None
    previous_evaluation: Optional[Any] = None
    quality_threshold: float = 7.0
    iteration: int = 1
    # New fields for internet/evidence tracking
    allow_internet: bool = False
    web_searches_performed: bool = False
    research_findings: Optional[str] = None
    # Prior analysis context for deep analysis phase
    prior_analysis: Optional[str] = None
    # Knowledge context from RAG retrieval
    knowledge_context: Optional[str] = None
    # Task type: "code", "research", "document", "analysis" - affects evaluation criteria
    task_type: str = "auto"  # "auto" attempts to detect from content


@dataclass
class ResearchEvaluationContext:
    """Context for research evaluation."""
    
    objective: str
    research_summary: str
    key_findings: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    web_searches_performed: int = 0
    iteration: int = 1
    prior_gaps: List[str] = field(default_factory=list)  # Gaps from previous iteration
    # Knowledge context from RAG retrieval
    knowledge_context: Optional[str] = None


@dataclass
class ResearchEvaluation:
    """
    Evaluation result specifically for research findings.
    
    Designed to produce actionable deltas for iteration control.
    """
    
    # Coverage assessment
    completeness_score: float  # 0-1, how complete the research is
    confidence_score: float  # 0-1, confidence in findings
    
    # Actionable deltas for next iteration
    gaps: List[str] = field(default_factory=list)  # Missing topics
    unresolved_questions: List[str] = field(default_factory=list)  # Unclear claims
    evidence_requests: List[str] = field(default_factory=list)  # Facts needing verification
    next_focus_areas: List[str] = field(default_factory=list)  # Topics for deeper investigation
    
    # Quality indicators
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    # Iteration control
    should_continue: bool = True  # Whether another iteration is warranted
    iteration: int = 1
    raw_output: str = ""
    
    @classmethod
    def from_text(cls, text: str, iteration: int = 1) -> "ResearchEvaluation":
        """Parse research evaluation from LLM output."""
        gaps = []
        unresolved_questions = []
        evidence_requests = []
        next_focus_areas = []
        strengths = []
        weaknesses = []
        
        current_section = None
        
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            
            # Detect sections
            if 'gap' in line_lower or 'missing' in line_lower:
                current_section = 'gaps'
            elif 'unresolved' in line_lower or 'question' in line_lower or 'unclear' in line_lower:
                current_section = 'unresolved_questions'
            elif 'evidence' in line_lower or 'verification' in line_lower or 'data need' in line_lower:
                current_section = 'evidence_requests'
            elif 'focus' in line_lower or 'next' in line_lower or 'deeper' in line_lower:
                current_section = 'next_focus_areas'
            elif 'strength' in line_lower or 'positive' in line_lower:
                current_section = 'strengths'
            elif 'weakness' in line_lower or 'concern' in line_lower or 'issue' in line_lower:
                current_section = 'weaknesses'
            elif line.strip().startswith(('-', '*', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                content = line.strip().lstrip('-*•0123456789.) ')
                if content:
                    if current_section == 'gaps':
                        gaps.append(content)
                    elif current_section == 'unresolved_questions':
                        unresolved_questions.append(content)
                    elif current_section == 'evidence_requests':
                        evidence_requests.append(content)
                    elif current_section == 'next_focus_areas':
                        next_focus_areas.append(content)
                    elif current_section == 'strengths':
                        strengths.append(content)
                    elif current_section == 'weaknesses':
                        weaknesses.append(content)
        
        # Extract scores
        completeness = 0.5
        comp_match = re.search(r'completeness[:\s]*([0-9.]+)', text, re.IGNORECASE)
        if comp_match:
            try:
                val = float(comp_match.group(1))
                completeness = val if val <= 1.0 else val / 10.0
            except ValueError:
                pass
        
        confidence = 0.5
        conf_match = re.search(r'confidence[:\s]*([0-9.]+)', text, re.IGNORECASE)
        if conf_match:
            try:
                val = float(conf_match.group(1))
                confidence = val if val <= 1.0 else val / 10.0
            except ValueError:
                pass
        
        # Extract should_continue
        should_continue = True
        if 'should continue: no' in text.lower() or 'continue: false' in text.lower():
            should_continue = False
        elif completeness >= 0.85 and confidence >= 0.8 and not gaps:
            should_continue = False
        
        return cls(
            completeness_score=completeness,
            confidence_score=confidence,
            gaps=gaps[:5],
            unresolved_questions=unresolved_questions[:5],
            evidence_requests=evidence_requests[:5],
            next_focus_areas=next_focus_areas[:5],
            strengths=strengths[:5],
            weaknesses=weaknesses[:5],
            should_continue=should_continue,
            iteration=iteration,
            raw_output=text
        )
    
    def get_priority_issues(self, limit: int = 5) -> List[str]:
        """Get prioritized list of issues to address in next iteration."""
        # Priority: gaps > unresolved questions > evidence requests
        all_issues = self.gaps + self.unresolved_questions + self.evidence_requests
        return all_issues[:limit]
    
    def has_actionable_feedback(self) -> bool:
        """Check if there is actionable feedback for next iteration."""
        return bool(
            self.gaps or 
            self.unresolved_questions or 
            self.evidence_requests or 
            self.next_focus_areas
        )


class EvaluatorCouncil(BaseCouncil):
    """
    Evaluation council for code quality assessment.
    
    Multiple evaluator models independently score code,
    with results blended for a balanced final evaluation.
    
    Supports dynamic configuration via CouncilDefinition for
    runtime model/temperature/persona selection.
    """
    
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        consensus_engine: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
        quality_threshold: float = 7.0,
        cognitive_spine: Optional[Any] = None,
        council_definition: Optional[Any] = None
    ):
        """
        Initialize evaluator council.
        
        Args:
            model_pool: Custom model pool (defaults to EVALUATOR_MODELS)
            consensus_engine: Custom consensus (defaults to WeightedBlendConsensus)
                            If None and cognitive_spine provided, gets from spine
            ollama_base_url: Ollama server URL
            quality_threshold: Threshold for passing evaluation
            cognitive_spine: Optional CognitiveSpine for validation and consensus
            council_definition: Optional CouncilDefinition for dynamic configuration
        """
        # Use council_definition models if provided and no custom pool
        if model_pool is None:
            if council_definition is not None and council_definition.models:
                model_pool = ModelPool(
                    pool_config=council_definition.get_model_configs(),
                    base_url=ollama_base_url
                )
            else:
                model_pool = ModelPool(
                    pool_config=EVALUATOR_MODELS,
                    base_url=ollama_base_url
                )
        
        # Get consensus from CognitiveSpine if not provided
        if consensus_engine is None:
            if cognitive_spine is not None:
                consensus_engine = cognitive_spine.get_consensus_engine(
                    "weighted_blend", "evaluator_council"
                )
            else:
                consensus_engine = WeightedBlendConsensus(
                    ollama_base_url=ollama_base_url
                )
        
        super().__init__(
            model_pool=model_pool,
            consensus_engine=consensus_engine,
            council_name="evaluator_council",
            cognitive_spine=cognitive_spine,
            council_definition=council_definition
        )
        
        self.quality_threshold = quality_threshold
        self._parser = EvaluationResultParser(quality_threshold=quality_threshold)
        self._load_default_system_prompt()
    
    def _load_default_system_prompt(self) -> None:
        """Load the default system prompt for evaluator council."""
        self._system_prompt = """You are part of an evaluation council of code quality experts.
Your role is to thoroughly evaluate code quality and provide constructive feedback.
Be fair, specific, and focused on actionable improvements.

You excel at:
- Assessing code correctness and logic
- Evaluating code style and readability
- Identifying potential bugs and edge cases
- Checking error handling and robustness
- Analyzing performance and efficiency

Your evaluation must include:
- A quality score from 0-10
- Specific issues categorized by severity (critical, major, minor)
- Concrete recommendations for improvement
- Recognition of code strengths

Use a consistent scoring rubric:
- 9-10: Excellent, production-ready
- 7-8: Good, minor improvements needed
- 5-6: Fair, significant improvements needed
- 3-4: Poor, major issues
- 0-2: Very poor, fundamental problems"""
    
    def build_prompt(
        self,
        evaluator_context: EvaluatorContext
    ) -> str:
        """
        Build evaluation prompt from context.
        
        Args:
            evaluator_context: Context containing code and objective
            
        Returns:
            Prompt string for council members
        """
        # Build metric info
        metric_str = ""
        if evaluator_context.metric_result:
            metric_str = f"""
## EXECUTION METRICS
{evaluator_context.metric_result}

Consider these metrics in your evaluation.
"""
        
        # Build previous evaluation context
        prev_eval_str = ""
        if evaluator_context.previous_evaluation:
            prev_eval_str = f"""
## PREVIOUS EVALUATION (Iteration {evaluator_context.iteration - 1})
{evaluator_context.previous_evaluation}

Compare to the previous evaluation and note improvements or regressions.
"""
        
        # Build prior analysis context (for deep analysis phase)
        prior_analysis_str = ""
        if evaluator_context.prior_analysis:
            prior_analysis_str = f"""
## PRIOR ANALYSIS
{evaluator_context.prior_analysis}

Consider this prior analysis context in your evaluation.
"""
        
        # Build knowledge context (from RAG retrieval)
        knowledge_str = ""
        if evaluator_context.knowledge_context:
            knowledge_str = f"""
## RETRIEVED KNOWLEDGE (use as reference)
{evaluator_context.knowledge_context}
"""
        
        # Detect or use specified task type
        task_type = self._detect_task_type(evaluator_context)
        
        # Build task-appropriate evaluation criteria
        if task_type == "research":
            criteria_str = """## EVALUATION CRITERIA
1. **Accuracy**: Are the findings factually correct and well-sourced?
2. **Completeness**: Does the research cover all aspects of the objective?
3. **Source Quality**: Are sources reliable, current, and properly cited?
4. **Analysis Depth**: Is the analysis thorough and insightful?
5. **Clarity**: Is the research clearly organized and communicated?
6. **Objectivity**: Does the research present balanced perspectives?"""
            content_label = "RESEARCH OUTPUT"
        elif task_type == "document":
            criteria_str = """## EVALUATION CRITERIA
1. **Relevance**: Does the document address the objective fully?
2. **Clarity**: Is the content clear and well-organized?
3. **Accuracy**: Are claims accurate and supported?
4. **Completeness**: Does it cover all necessary topics?
5. **Quality**: Is the writing professional and coherent?
6. **Usefulness**: Is the document actionable and useful?"""
            content_label = "DOCUMENT"
        elif task_type == "analysis":
            criteria_str = """## EVALUATION CRITERIA
1. **Rigor**: Is the analysis methodologically sound?
2. **Depth**: Does it go beyond surface-level observations?
3. **Evidence**: Are conclusions supported by evidence?
4. **Insight**: Does it provide valuable new understanding?
5. **Clarity**: Is the analysis clearly presented?
6. **Actionability**: Are recommendations practical?"""
            content_label = "ANALYSIS"
        else:  # code
            criteria_str = """## EVALUATION CRITERIA
1. **Correctness**: Does the code correctly implement the objective?
2. **Code Quality**: Is the code clean, readable, and well-structured?
3. **Error Handling**: Does the code handle errors and edge cases?
4. **Documentation**: Are there adequate docstrings and comments?
5. **Efficiency**: Is the code reasonably efficient?
6. **Best Practices**: Does the code follow best practices?"""
            content_label = "CODE"
        
        prompt = f"""Evaluate the following {content_label.lower()} against the stated objective:

## OBJECTIVE
{evaluator_context.objective}

## {content_label} TO EVALUATE
```
{evaluator_context.content_to_evaluate}
```
{metric_str}
{prev_eval_str}
{prior_analysis_str}
{knowledge_str}
{criteria_str}

## OUTPUT FORMAT
Provide your evaluation in this exact format:

### QUALITY SCORE
[Score: X/10]

### CONFIDENCE
[Confidence: 0.X] (0.0-1.0 scale, how confident you are in this evaluation)

### ISSUES
For each issue, specify severity (critical/major/minor):
- [CRITICAL] Description of critical issue
- [MAJOR] Description of major issue
- [MINOR] Description of minor issue

### RECOMMENDATIONS
1. Specific recommendation 1
2. Specific recommendation 2

### STRENGTHS
- Positive aspect 1
- Positive aspect 2

### MISSING INFORMATION
List any information you would need but don't have to complete a thorough evaluation:
- Missing info 1
- Missing info 2

### OUTSTANDING QUESTIONS
List questions that remain unanswered about the implementation or objective:
- Question 1?
- Question 2?

### DATA NEEDS
List any data, evidence, or external sources that would improve the evaluation:
- Data need 1
- Data need 2

### SUMMARY
Brief overall assessment."""

        return prompt
    
    def _detect_task_type(self, context: EvaluatorContext) -> str:
        """
        Detect the task type from context.
        
        Uses explicit task_type if set to non-auto, otherwise infers from content.
        
        Args:
            context: The evaluator context
            
        Returns:
            Task type string: "code", "research", "document", or "analysis"
        """
        # Use explicit type if provided
        if context.task_type != "auto":
            return context.task_type
        
        content = context.content_to_evaluate.lower()
        objective = context.objective.lower()
        
        # Research indicators
        research_indicators = [
            "research", "findings", "sources", "evidence", "literature",
            "study", "survey", "investigation", "analysis of"
        ]
        if any(ind in objective for ind in research_indicators):
            return "research"
        
        # Code indicators - look for programming patterns
        code_indicators = [
            "def ", "class ", "import ", "function ", "return ",
            "if __name__", "async def", "=> {", "public class",
            ".py", ".js", ".ts", ".java", ".go", ".rs"
        ]
        if any(ind in content for ind in code_indicators):
            return "code"
        
        # Analysis indicators
        analysis_indicators = [
            "analyze", "evaluate", "compare", "assess", "examine",
            "trade-off", "pros and cons", "recommendation"
        ]
        if any(ind in objective for ind in analysis_indicators):
            return "analysis"
        
        # Document indicators
        doc_indicators = [
            "## ", "### ", "**", "1. ", "- ", "summary", "overview",
            "introduction", "conclusion"
        ]
        if sum(1 for ind in doc_indicators if ind in content) >= 3:
            return "document"
        
        # Default to document for non-code content
        if not any(ind in content for ind in code_indicators):
            return "document"
        
        return "code"
    
    def postprocess(self, consensus_output: Any) -> EvaluationResult:
        """
        Postprocess consensus output into EvaluationResult.
        
        Args:
            consensus_output: Raw consensus output string
            
        Returns:
            Parsed EvaluationResult object
        """
        if not consensus_output:
            return EvaluationResult(
                quality_score=0.0,
                passed=False,
                issues=[],
                recommendations=["Evaluation failed to produce output"],
                strengths=[],
                raw_output=""
            )
        
        return self._parser.parse(str(consensus_output))
    
    def postprocess_with_context(
        self,
        consensus_output: Any,
        context: EvaluatorContext
    ) -> EvaluationResult:
        """
        Postprocess with context-aware adjustments.
        
        Applies penalties for:
        - Missing web searches when internet was allowed
        
        Args:
            consensus_output: Raw consensus output string
            context: Evaluator context with flags
            
        Returns:
            Adjusted EvaluationResult object
        """
        result = self.postprocess(consensus_output)
        
        # Apply penalty if internet was allowed but no searches performed
        if context.allow_internet and not context.web_searches_performed:
            penalty = 2.0
            result.quality_score = max(0.0, result.quality_score - penalty)
            result.missing_info.append("Missing real-world evidence from web search")
            result.data_needs.append("Web search for current documentation and best practices")
            result.confidence_score = max(0.2, result.confidence_score - 0.15)
            
            # Re-evaluate pass status
            result.passed = result.quality_score >= self.quality_threshold
        
        return result
    
    def evaluate(
        self,
        objective: str,
        code: str,
        metric_result: Optional[Any] = None,
        iteration: int = 1,
        allow_internet: bool = False,
        web_searches_performed: bool = False,
        research_findings: Optional[str] = None
    ) -> CouncilResult:
        """
        Convenience method to evaluate code.
        
        Args:
            objective: Original objective
            code: Code to evaluate
            metric_result: Optional execution metrics
            iteration: Current iteration number
            allow_internet: Whether internet search was allowed
            web_searches_performed: Whether web searches were actually done
            research_findings: Optional research findings to consider
            
        Returns:
            CouncilResult with EvaluationResult
        """
        evaluator_context = EvaluatorContext(
            objective=objective,
            content_to_evaluate=code,
            metric_result=metric_result,
            quality_threshold=self.quality_threshold,
            iteration=iteration,
            allow_internet=allow_internet,
            web_searches_performed=web_searches_performed,
            research_findings=research_findings
        )
        
        result = self.execute(evaluator_context)
        
        # Apply context-aware post-processing
        if result.success and result.output:
            adjusted_output = self.postprocess_with_context(
                result.output.raw_output if hasattr(result.output, 'raw_output') else str(result.output),
                evaluator_context
            )
            result = CouncilResult(
                output=adjusted_output,
                raw_outputs=result.raw_outputs,
                consensus_details=result.consensus_details,
                council_name=self.council_name,
                success=result.success,
                error=result.error
            )
        
        return result
    
    def evaluate_research(
        self,
        objective: str,
        research_findings: Any,
        iteration: int = 1,
        prior_gaps: Optional[List[str]] = None
    ) -> CouncilResult:
        """
        Evaluate research findings and produce actionable deltas for iteration.
        
        This method is specifically designed to drive iterative research by:
        - Identifying gaps in coverage
        - Surfacing unresolved questions
        - Requesting evidence for unverified claims
        - Suggesting focus areas for next iteration
        
        Args:
            objective: Original research objective
            research_findings: ResearchFindings object or string summary
            iteration: Current iteration number
            prior_gaps: Gaps identified in previous iteration (to check resolution)
            
        Returns:
            CouncilResult with ResearchEvaluation
        """
        # Extract info from findings
        if hasattr(research_findings, 'summary'):
            summary = research_findings.summary
            key_findings = getattr(research_findings, 'key_points', [])
            sources = getattr(research_findings, 'sources_suggested', [])
            web_searches = getattr(research_findings, 'web_search_count', 0)
        else:
            summary = str(research_findings)
            key_findings = []
            sources = []
            web_searches = 0
        
        context = ResearchEvaluationContext(
            objective=objective,
            research_summary=summary,
            key_findings=key_findings,
            sources_used=sources,
            web_searches_performed=web_searches,
            iteration=iteration,
            prior_gaps=prior_gaps or []
        )
        
        # Build specialized prompt for research evaluation
        prompt = self._build_research_evaluation_prompt(context)
        
        # Use a temporary context for the council execution
        eval_context = EvaluatorContext(
            objective=objective,
            content_to_evaluate=summary,  # Research content to evaluate
            iteration=iteration
        )
        
        # Store the specialized prompt for this execution
        self._research_eval_prompt = prompt
        
        # Execute with custom prompt
        result = self._execute_research_evaluation(context)
        
        return result
    
    def _build_research_evaluation_prompt(self, context: ResearchEvaluationContext) -> str:
        """Build prompt for research evaluation."""
        findings_str = "\n".join(f"- {f}" for f in context.key_findings[:10]) if context.key_findings else "No key findings extracted."
        sources_str = "\n".join(f"- {s}" for s in context.sources_used[:5]) if context.sources_used else "No sources cited."
        
        prior_gaps_str = ""
        if context.prior_gaps:
            prior_gaps_str = f"""
## PRIOR GAPS (from iteration {context.iteration - 1})
These gaps were identified previously. Check if they have been addressed:
{chr(10).join(f'- {g}' for g in context.prior_gaps)}
"""
        
        prompt = f"""Evaluate the following research findings for completeness and quality.
Your goal is to identify what is MISSING or UNCLEAR to drive the next research iteration.

## OBJECTIVE
{context.objective}

## RESEARCH SUMMARY
{context.research_summary[:2000]}

## KEY FINDINGS
{findings_str}

## SOURCES USED
{sources_str}

## WEB SEARCHES PERFORMED
{context.web_searches_performed} searches executed

**ANTI-HALLUCINATION CHECK:**
- If web_searches_performed == 0 and this is a factual/analytical objective, flag as CRITICAL issue
- Identify any unverified factual claims (claims without [VERIFIED], [INFERRED], or [SPECULATIVE] tags)
- Flag excessive certainty without external validation
- Note any hallucinated factual assertions

{prior_gaps_str}

## EVALUATION CRITERIA
1. **Coverage**: Does the research adequately cover all aspects of the objective?
2. **Depth**: Is the analysis sufficiently deep, or is it superficial?
3. **Evidence**: Are claims backed by evidence or sources?
4. **Clarity**: Are the findings clear and actionable?
5. **Completeness**: What is still missing?
6. **Hallucination Risk**: Are there unverified factual claims that require external validation?

## OUTPUT FORMAT

### COMPLETENESS SCORE
[Completeness: 0.X] (0.0-1.0, how complete is this research?)

### CONFIDENCE SCORE
[Confidence: 0.X] (0.0-1.0, how confident are you in these findings?)

### GAPS (missing topics that must be researched)
- [Gap 1]: Why this is important
- [Gap 2]: Why this is important

### UNRESOLVED QUESTIONS (unclear or contradictory claims)
- [Question 1]?
- [Question 2]?

### EVIDENCE NEEDED (facts requiring verification)
- [Evidence request 1]: What source would verify this?
- [Evidence request 2]: What source would verify this?

### NEXT FOCUS AREAS (topics for deeper investigation)
- [Focus area 1]: What specifically to investigate
- [Focus area 2]: What specifically to investigate

### STRENGTHS
- [Strength 1]
- [Strength 2]

### WEAKNESSES
- [Weakness 1]
- [Weakness 2]

### SHOULD CONTINUE
[Should Continue: Yes/No] (Is another research iteration warranted?)

Be specific and actionable. Focus on gaps that would significantly improve the research."""

        return prompt
    
    def _execute_research_evaluation(self, context: ResearchEvaluationContext) -> CouncilResult:
        """Execute research evaluation with specialized prompt."""
        from ..base_council import CouncilResult
        
        prompt = self._build_research_evaluation_prompt(context)
        system_prompt = self._get_research_eval_system_prompt()
        
        try:
            # Get responses from model pool using run_all
            responses = self.model_pool.run_all(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            if not responses:
                return CouncilResult(
                    output=None,
                    raw_outputs={},
                    consensus_details=None,
                    council_name=self.council_name,
                    success=False,
                    error="No responses from model pool"
                )
            
            # Use consensus engine - extract text outputs from ModelOutput objects
            # responses is Dict[str, ModelOutput], need to extract output strings
            text_outputs = []
            for name, model_output in responses.items():
                if model_output.success and model_output.output:
                    text_outputs.append(model_output.output)
            
            if not text_outputs:
                return CouncilResult(
                    output=None,
                    raw_outputs=responses,
                    consensus_details=None,
                    council_name=self.council_name,
                    success=False,
                    error="All models produced empty outputs"
                )
            
            consensus_output = self.consensus.synthesize(text_outputs)
            
            # Parse into ResearchEvaluation
            evaluation = ResearchEvaluation.from_text(
                str(consensus_output),
                iteration=context.iteration
            )
            
            logger.info(
                f"Research evaluation iteration {context.iteration}: "
                f"completeness={evaluation.completeness_score:.2f}, "
                f"gaps={len(evaluation.gaps)}, "
                f"questions={len(evaluation.unresolved_questions)}, "
                f"should_continue={evaluation.should_continue}"
            )
            
            return CouncilResult(
                output=evaluation,
                raw_outputs=responses,
                consensus_details=getattr(self.consensus, 'last_details', None),
                council_name=self.council_name,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Research evaluation failed: {e}")
            return CouncilResult(
                output=None,
                raw_outputs={},
                consensus_details=None,
                council_name=self.council_name,
                success=False,
                error=str(e)
            )
    
    def _get_research_eval_system_prompt(self) -> str:
        """Get system prompt for research evaluation."""
        return """You are an expert research evaluator.
Your role is to critically assess research findings and identify gaps, unclear claims, 
and areas needing deeper investigation.

You excel at:
- Identifying what's missing from research
- Spotting unverified or unsupported claims
- Determining when research needs more evidence
- Prioritizing what to investigate next

Be constructive but thorough. Your feedback drives the next iteration of research.
Focus on actionable gaps rather than minor issues."""

