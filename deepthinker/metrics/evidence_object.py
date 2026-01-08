"""
Evidence Object for DeepThinker Metrics.

Provides a standardized schema for tool outputs that can be
used as evidence for claims and grounding.

Converts various tool outputs (web search, code execution, etc.)
to a common format for tracking and attribution.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceType(str, Enum):
    """Types of evidence objects."""
    WEB_SEARCH = "web_search"
    CODE_OUTPUT = "code_output"
    DATABASE_QUERY = "database_query"
    FILE_READ = "file_read"
    API_RESPONSE = "api_response"
    DOCUMENT_EXTRACT = "document_extract"
    USER_INPUT = "user_input"
    MODEL_OUTPUT = "model_output"
    UNKNOWN = "unknown"


@dataclass
class EvidenceObject:
    """
    Standardized evidence object for tool outputs.
    
    Provides a common schema for tracking evidence provenance,
    quality, and attribution across different tool types.
    
    Attributes:
        id: Unique identifier (generated from content hash)
        evidence_type: Type of evidence source
        source: Source identifier (URL, file path, tool name)
        created_at: When the evidence was created
        content_excerpt: Truncated content for display
        raw_ref: Optional reference to full content
        confidence: Confidence in evidence quality (0-1)
        metadata: Additional source-specific metadata
        claim_ids: IDs of claims this evidence supports
    """
    id: str = ""
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    source: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    content_excerpt: str = ""
    raw_ref: Optional[str] = None
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    claim_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content hash."""
        content = f"{self.evidence_type.value}:{self.source}:{self.content_excerpt[:200]}"
        return f"ev_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    @classmethod
    def from_web_search(
        cls,
        url: str,
        title: str,
        snippet: str,
        quality_score: float = 0.5,
        quality_tier: str = "MEDIUM",
    ) -> "EvidenceObject":
        """
        Create an evidence object from web search result.
        
        Args:
            url: Source URL
            title: Page title
            snippet: Content snippet
            quality_score: Quality score (0-1)
            quality_tier: Quality tier string
            
        Returns:
            EvidenceObject for the search result
        """
        return cls(
            evidence_type=EvidenceType.WEB_SEARCH,
            source=url,
            content_excerpt=f"{title}\n{snippet}"[:500],
            raw_ref=url,
            confidence=quality_score,
            metadata={
                "title": title,
                "quality_tier": quality_tier,
            },
        )
    
    @classmethod
    def from_code_output(
        cls,
        code: str,
        output: str,
        success: bool = True,
        execution_time: float = 0.0,
    ) -> "EvidenceObject":
        """
        Create an evidence object from code execution.
        
        Args:
            code: Executed code
            output: Execution output
            success: Whether execution succeeded
            execution_time: Execution time in seconds
            
        Returns:
            EvidenceObject for the code execution
        """
        # Truncate code and output for excerpt
        code_preview = code[:200] + "..." if len(code) > 200 else code
        output_preview = output[:300] if output else "[no output]"
        
        return cls(
            evidence_type=EvidenceType.CODE_OUTPUT,
            source="code_executor",
            content_excerpt=f"Code: {code_preview}\nOutput: {output_preview}",
            confidence=0.9 if success else 0.3,
            metadata={
                "success": success,
                "execution_time_seconds": execution_time,
                "code_length": len(code),
                "output_length": len(output) if output else 0,
            },
        )
    
    @classmethod
    def from_document(
        cls,
        path: str,
        content: str,
        doc_type: str = "text",
    ) -> "EvidenceObject":
        """
        Create an evidence object from document extraction.
        
        Args:
            path: Document path or identifier
            content: Extracted content
            doc_type: Document type (pdf, txt, etc.)
            
        Returns:
            EvidenceObject for the document
        """
        return cls(
            evidence_type=EvidenceType.DOCUMENT_EXTRACT,
            source=path,
            content_excerpt=content[:500],
            raw_ref=path,
            confidence=0.8,
            metadata={
                "doc_type": doc_type,
                "content_length": len(content),
            },
        )
    
    @classmethod
    def from_api_response(
        cls,
        endpoint: str,
        response_data: Any,
        status_code: int = 200,
    ) -> "EvidenceObject":
        """
        Create an evidence object from API response.
        
        Args:
            endpoint: API endpoint
            response_data: Response data
            status_code: HTTP status code
            
        Returns:
            EvidenceObject for the API response
        """
        if isinstance(response_data, dict):
            excerpt = str(response_data)[:500]
        elif isinstance(response_data, str):
            excerpt = response_data[:500]
        else:
            excerpt = str(response_data)[:500]
        
        return cls(
            evidence_type=EvidenceType.API_RESPONSE,
            source=endpoint,
            content_excerpt=excerpt,
            confidence=0.9 if 200 <= status_code < 300 else 0.3,
            metadata={
                "status_code": status_code,
                "response_type": type(response_data).__name__,
            },
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "evidence_type": self.evidence_type.value,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "content_excerpt": self.content_excerpt,
            "raw_ref": self.raw_ref,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "claim_ids": self.claim_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceObject":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        
        evidence_type = data.get("evidence_type", "unknown")
        if isinstance(evidence_type, str):
            evidence_type = EvidenceType(evidence_type)
        
        return cls(
            id=data.get("id", ""),
            evidence_type=evidence_type,
            source=data.get("source", ""),
            created_at=created_at,
            content_excerpt=data.get("content_excerpt", ""),
            raw_ref=data.get("raw_ref"),
            confidence=data.get("confidence", 0.5),
            metadata=data.get("metadata", {}),
            claim_ids=data.get("claim_ids", []),
        )
    
    def link_to_claim(self, claim_id: str) -> None:
        """
        Link this evidence to a claim.
        
        Args:
            claim_id: ID of claim this evidence supports
        """
        if claim_id not in self.claim_ids:
            self.claim_ids.append(claim_id)
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if evidence meets quality threshold."""
        return self.confidence >= threshold
    
    def __str__(self) -> str:
        return (
            f"Evidence({self.evidence_type.value}, "
            f"source={self.source[:30]}..., "
            f"confidence={self.confidence:.2f})"
        )

