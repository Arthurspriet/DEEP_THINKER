"""
Parser for extracting structured evaluation results from LLM outputs.
"""

import re
from typing import List, Optional

from .evaluation_result import EvaluationResult, IssueItem


class EvaluationResultParser:
    """
    Parses LLM evaluation outputs into structured EvaluationResult objects.
    
    Handles multiple output formats and provides graceful fallback for
    unparseable responses.
    """
    
    def __init__(self, quality_threshold: float = 7.0):
        """
        Initialize parser.
        
        Args:
            quality_threshold: Threshold for pass/fail determination
        """
        self.quality_threshold = quality_threshold
    
    def parse(self, llm_output: str) -> EvaluationResult:
        """
        Parse LLM output into structured EvaluationResult.
        
        Args:
            llm_output: Raw text output from evaluator LLM
            
        Returns:
            Structured EvaluationResult object
        """
        # Extract quality score
        quality_score = self._extract_score(llm_output)
        
        # Determine pass/fail
        passed = self._determine_pass_fail(llm_output, quality_score)
        
        # Extract issues
        issues = self._extract_issues(llm_output)
        
        # Extract recommendations
        recommendations = self._extract_list_items(
            llm_output, 
            ["recommendation", "suggest", "improve"]
        )
        
        # Extract strengths
        strengths = self._extract_list_items(
            llm_output,
            ["strength", "good", "well done", "positive"]
        )
        
        # Extract new fields for iteration control
        missing_info = self._extract_list_items(
            llm_output,
            ["missing information", "missing info", "information needed"]
        )
        
        questions = self._extract_list_items(
            llm_output,
            ["outstanding question", "question", "unanswered"]
        )
        
        data_needs = self._extract_list_items(
            llm_output,
            ["data need", "data needed", "evidence needed", "external source"]
        )
        
        # Extract confidence score
        confidence_score = self._extract_confidence(llm_output)
        
        # Determine if critical information is missing
        critical_missing = self._has_critical_missing(missing_info, questions, issues)
        
        return EvaluationResult(
            quality_score=quality_score,
            passed=passed,
            issues=issues,
            recommendations=recommendations,
            strengths=strengths,
            raw_output=llm_output,
            missing_info=missing_info,
            questions=questions,
            data_needs=data_needs,
            confidence_score=confidence_score,
            critical_missing=critical_missing
        )
    
    def _extract_score(self, text: str) -> float:
        """
        Extract quality score from text using multiple patterns.
        
        Patterns matched:
        - "Quality Score: 7.5/10"
        - "Score: 8"
        - "Rating: 6.5"
        - "7.5 out of 10"
        - Keyword inference (excellent, poor, etc.)
        """
        patterns = [
            r"quality\s*score[:\s]+(\d+\.?\d*)\s*/?\s*10",
            r"score[:\s]+(\d+\.?\d*)\s*/?\s*10",
            r"rating[:\s]+(\d+\.?\d*)\s*/?\s*10",
            r"(\d+\.?\d*)\s*/\s*10",
            r"(\d+\.?\d*)\s+out\s+of\s+10",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Ensure score is in valid range
                return max(0.0, min(10.0, score))
        
        # Fallback: keyword-based inference
        text_lower = text.lower()
        
        # Positive keywords (high scores)
        if any(word in text_lower for word in ["excellent", "outstanding", "exceptional", "perfect"]):
            return 9.0
        if any(word in text_lower for word in ["great", "very good", "strong"]):
            return 8.0
        if any(word in text_lower for word in ["good", "solid", "decent"]):
            return 7.0
        
        # Negative keywords (low scores)
        if any(word in text_lower for word in ["poor", "bad", "terrible", "awful"]):
            return 4.0
        if any(word in text_lower for word in ["weak", "inadequate", "lacking"]):
            return 5.0
        
        # Default: assume medium quality if no score or keywords found
        return 5.0
    
    def _determine_pass_fail(self, text: str, score: float) -> bool:
        """
        Determine if code passes evaluation.
        
        Uses explicit pass/fail indicators in text, or falls back to score threshold.
        """
        text_lower = text.lower()
        
        # Look for explicit pass/fail indicators
        if "pass" in text_lower and "fail" not in text_lower:
            return True
        if "fail" in text_lower and "pass" not in text_lower:
            return False
        
        # Fall back to score threshold
        return score >= self.quality_threshold
    
    def _extract_issues(self, text: str) -> List[IssueItem]:
        """
        Extract issues with severity categorization.
        
        Looks for sections labeled "Issues", "Problems", "Concerns" etc.
        and categorizes by severity keywords.
        """
        issues = []
        
        # Find issue section
        issue_section = re.search(
            r"(?:issues?|problems?|concerns?)[:\s]+(.+?)(?:\n\n|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if not issue_section:
            return issues
        
        issue_text = issue_section.group(1)
        
        # Extract individual issues (lines starting with -, *, or bullet points)
        issue_lines = re.findall(r"[-*•]\s*(.+)", issue_text)
        
        for line in issue_lines:
            severity = self._determine_severity(line)
            # Clean up severity keywords from description
            description = re.sub(
                r"\[(critical|major|minor)\]\s*",
                "",
                line,
                flags=re.IGNORECASE
            ).strip()
            
            issues.append(IssueItem(severity=severity, description=description))
        
        return issues
    
    def _determine_severity(self, issue_text: str) -> str:
        """Determine issue severity from text."""
        text_lower = issue_text.lower()
        
        # Check for explicit severity markers
        if "[critical]" in text_lower or "critical" in text_lower:
            return "critical"
        if "[major]" in text_lower or "major" in text_lower:
            return "major"
        
        # Check for severity keywords
        critical_keywords = ["crash", "error", "broken", "fails", "doesn't work"]
        major_keywords = ["inefficient", "unclear", "confusing", "poor"]
        
        if any(keyword in text_lower for keyword in critical_keywords):
            return "critical"
        if any(keyword in text_lower for keyword in major_keywords):
            return "major"
        
        # Default to minor
        return "minor"
    
    def _extract_list_items(self, text: str, section_keywords: List[str]) -> List[str]:
        """
        Extract list items from sections matching keywords.
        
        Args:
            text: Full text to search
            section_keywords: Keywords that might label the section
            
        Returns:
            List of extracted items
        """
        items = []
        
        # Try each keyword individually to handle partial matches
        for keyword in section_keywords:
            # Build pattern - use word boundary at start to allow plural forms
            pattern = rf"\b{keyword}[s]?[:\s]*\n(.+?)(?:\n\n|(?=\n###|\n[A-Z][A-Z\s]+:)|\Z)"
            
            section = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if section:
                section_text = section.group(1)
                # Extract list items
                list_items = re.findall(r"[-*•]\s*(.+)", section_text)
                items.extend([item.strip() for item in list_items])
                # Found section, no need to check other keywords
                break
        
        return items
    
    def _extract_confidence(self, text: str) -> float:
        """
        Extract confidence score from text.
        
        Looks for patterns like:
        - "Confidence: 0.8"
        - "[Confidence: 0.75]"
        - "confidence score: 85%"
        """
        patterns = [
            r"confidence[:\s]+(\d+\.?\d*)",
            r"\[confidence[:\s]+(\d+\.?\d*)\]",
            r"confidence\s+score[:\s]+(\d+\.?\d*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Normalize if given as percentage
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
        
        # Fallback: infer from language
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["highly confident", "very confident", "certain"]):
            return 0.9
        if any(word in text_lower for word in ["confident", "fairly certain"]):
            return 0.75
        if any(word in text_lower for word in ["uncertain", "unsure", "unclear"]):
            return 0.4
        if any(word in text_lower for word in ["cannot determine", "impossible to assess"]):
            return 0.2
        
        # Default confidence
        return 0.6
    
    def _has_critical_missing(
        self,
        missing_info: List[str],
        questions: List[str],
        issues: List[IssueItem]
    ) -> bool:
        """
        Determine if critical information is missing.
        
        Critical missing is true if:
        - Any missing info mentions "critical", "essential", "required"
        - Many unresolved questions (> 3)
        - Any critical issues related to incomplete analysis
        """
        # Check missing info for critical keywords
        critical_keywords = ["critical", "essential", "required", "fundamental", "must have"]
        for info in missing_info:
            if any(kw in info.lower() for kw in critical_keywords):
                return True
        
        # Too many questions suggests incomplete evaluation
        if len(questions) > 3:
            return True
        
        # Check if any critical issue mentions missing/incomplete
        incomplete_keywords = ["incomplete", "missing", "cannot assess", "unable to evaluate"]
        for issue in issues:
            if issue.severity == "critical":
                if any(kw in issue.description.lower() for kw in incomplete_keywords):
                    return True
        
        return False

