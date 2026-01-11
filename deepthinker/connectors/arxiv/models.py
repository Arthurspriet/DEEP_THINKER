"""
Data models for arXiv Connector.

Provides dataclasses for arXiv paper metadata and evidence objects.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ArxivPaper:
    """
    Represents an arXiv paper with metadata.
    
    Attributes:
        id: arXiv ID (e.g., "2501.01234")
        version: Version string (e.g., "v1", "v2") or None for latest
        title: Paper title
        authors: List of author names
        abstract: Paper abstract
        categories: List of arXiv categories (e.g., ["cs.CL", "cs.AI"])
        primary_category: Primary category
        published: Publication date
        updated: Last update date
        links: Dictionary of link types to URLs (pdf, abs, etc.)
        doi: Optional DOI
        journal_ref: Optional journal reference
        comment: Optional author comment
    """
    id: str
    version: Optional[str] = None
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    categories: List[str] = field(default_factory=list)
    primary_category: str = ""
    published: Optional[datetime] = None
    updated: Optional[datetime] = None
    links: Dict[str, str] = field(default_factory=dict)
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comment: Optional[str] = None
    
    @property
    def arxiv_id(self) -> str:
        """Get the full arXiv ID including version if available."""
        if self.version:
            return f"{self.id}{self.version}"
        return self.id
    
    @property
    def abs_url(self) -> str:
        """Get the abstract page URL."""
        return self.links.get("abs", f"https://arxiv.org/abs/{self.arxiv_id}")
    
    @property
    def pdf_url(self) -> str:
        """Get the PDF URL."""
        return self.links.get("pdf", f"https://arxiv.org/pdf/{self.arxiv_id}.pdf")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "version": self.version,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "published": self.published.isoformat() if self.published else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "links": self.links,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comment": self.comment,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArxivPaper":
        """Create from dictionary."""
        published = data.get("published")
        if isinstance(published, str):
            try:
                published = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                published = None
        
        updated = data.get("updated")
        if isinstance(updated, str):
            try:
                updated = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except ValueError:
                updated = None
        
        return cls(
            id=data.get("id", ""),
            version=data.get("version"),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            abstract=data.get("abstract", ""),
            categories=data.get("categories", []),
            primary_category=data.get("primary_category", ""),
            published=published,
            updated=updated,
            links=data.get("links", {}),
            doi=data.get("doi"),
            journal_ref=data.get("journal_ref"),
            comment=data.get("comment"),
        )
    
    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al. ({len(self.authors)} authors)"
        return f"[{self.arxiv_id}] {self.title[:60]}... by {authors_str}"


@dataclass
class ArxivEvidence:
    """
    Evidence object for arXiv operations.
    
    Tracks provenance of arXiv data retrieval for Constitution compliance.
    
    Attributes:
        evidence_id: Unique evidence identifier
        source: Always "arxiv"
        arxiv_id: arXiv paper ID
        version: Paper version
        retrieved_at: When the data was retrieved
        request_url: URL used to retrieve the data
        content_type: Type of content ("metadata", "pdf", "source")
        local_path: Local file path if downloaded
        sha256: SHA256 hash of downloaded content
        metadata: Additional metadata
    """
    evidence_id: str = ""
    source: str = "arxiv"
    arxiv_id: str = ""
    version: Optional[str] = None
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    request_url: str = ""
    content_type: str = "metadata"  # "metadata", "pdf", "source"
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate evidence ID if not provided."""
        if not self.evidence_id:
            self.evidence_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique evidence ID based on content hash."""
        content = f"arxiv:{self.arxiv_id}:{self.content_type}:{self.request_url}"
        if self.sha256:
            content += f":{self.sha256}"
        return f"arxiv_ev_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "arxiv_id": self.arxiv_id,
            "version": self.version,
            "retrieved_at": self.retrieved_at.isoformat(),
            "request_url": self.request_url,
            "content_type": self.content_type,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArxivEvidence":
        """Create from dictionary."""
        retrieved_at = data.get("retrieved_at")
        if isinstance(retrieved_at, str):
            try:
                retrieved_at = datetime.fromisoformat(retrieved_at)
            except ValueError:
                retrieved_at = datetime.utcnow()
        elif retrieved_at is None:
            retrieved_at = datetime.utcnow()
        
        return cls(
            evidence_id=data.get("evidence_id", ""),
            source=data.get("source", "arxiv"),
            arxiv_id=data.get("arxiv_id", ""),
            version=data.get("version"),
            retrieved_at=retrieved_at,
            request_url=data.get("request_url", ""),
            content_type=data.get("content_type", "metadata"),
            local_path=data.get("local_path"),
            sha256=data.get("sha256"),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def for_search(
        cls,
        request_url: str,
        result_count: int,
        query: str,
    ) -> "ArxivEvidence":
        """Create evidence for a search operation."""
        return cls(
            arxiv_id="search",
            request_url=request_url,
            content_type="metadata",
            metadata={
                "operation": "search",
                "query": query,
                "result_count": result_count,
            },
        )
    
    @classmethod
    def for_paper(
        cls,
        arxiv_id: str,
        version: Optional[str],
        request_url: str,
        title: str,
    ) -> "ArxivEvidence":
        """Create evidence for a paper metadata fetch."""
        return cls(
            arxiv_id=arxiv_id,
            version=version,
            request_url=request_url,
            content_type="metadata",
            metadata={
                "operation": "get",
                "title": title[:200],
            },
        )
    
    @classmethod
    def for_download(
        cls,
        arxiv_id: str,
        version: Optional[str],
        request_url: str,
        local_path: str,
        sha256: str,
        content_type: str,  # "pdf" or "source"
    ) -> "ArxivEvidence":
        """Create evidence for a download operation."""
        return cls(
            arxiv_id=arxiv_id,
            version=version,
            request_url=request_url,
            content_type=content_type,
            local_path=local_path,
            sha256=sha256,
            metadata={
                "operation": "download",
            },
        )
    
    def __str__(self) -> str:
        return f"ArxivEvidence({self.content_type}, arxiv_id={self.arxiv_id})"


