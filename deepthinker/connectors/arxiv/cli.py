"""
CLI for arXiv Connector.

Provides command-line interface for arXiv operations.

Usage:
    python -m deepthinker.connectors.arxiv search "cat:cs.CL AND ti:alignment"
    python -m deepthinker.connectors.arxiv get 2501.01234
    python -m deepthinker.connectors.arxiv download 2501.01234 --kind pdf
    python -m deepthinker.connectors.arxiv status
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import get_arxiv_config, is_arxiv_enabled
from .tool import (
    arxiv_download,
    arxiv_get,
    arxiv_search,
    get_arxiv_tool_status,
)


def cmd_search(
    query: str,
    max_results: int = 10,
    start: int = 0,
    sort_by: Optional[str] = None,
    output_json: bool = False,
) -> int:
    """
    Search arXiv for papers.
    
    Args:
        query: Search query
        max_results: Max results to return
        start: Starting index
        sort_by: Sort field
        output_json: Output as JSON
        
    Returns:
        Exit code (0 = success)
    """
    result = arxiv_search(
        query=query,
        max_results=max_results,
        start=start,
        sort_by=sort_by,
    )
    
    if result["error"]:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1
    
    if output_json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    
    print(f"Search: {query}")
    print(f"Results: {result['count']}")
    print("-" * 60)
    
    for paper in result["papers"]:
        arxiv_id = paper.get("arxiv_id", paper.get("id", ""))
        title = paper.get("title", "")[:60]
        authors = ", ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += " et al."
        categories = ", ".join(paper.get("categories", [])[:3])
        
        print(f"\n[{arxiv_id}]")
        print(f"  Title: {title}...")
        print(f"  Authors: {authors}")
        print(f"  Categories: {categories}")
    
    print()
    
    if result["evidence"]:
        print(f"Evidence ID: {result['evidence'].get('evidence_id', 'N/A')}")
    
    return 0


def cmd_get(arxiv_id: str, output_json: bool = False) -> int:
    """
    Get paper metadata by ID.
    
    Args:
        arxiv_id: arXiv paper ID
        output_json: Output as JSON
        
    Returns:
        Exit code (0 = success)
    """
    result = arxiv_get(arxiv_id)
    
    if result["error"]:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1
    
    if not result["found"]:
        print(f"Paper not found: {arxiv_id}")
        return 1
    
    if output_json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    
    paper = result["paper"]
    
    print(f"arXiv ID: {paper.get('arxiv_id', paper.get('id', ''))}")
    print(f"Title: {paper.get('title', '')}")
    print(f"Authors: {', '.join(paper.get('authors', []))}")
    print(f"Categories: {', '.join(paper.get('categories', []))}")
    print(f"Published: {paper.get('published', 'N/A')}")
    print(f"Updated: {paper.get('updated', 'N/A')}")
    
    if paper.get("doi"):
        print(f"DOI: {paper['doi']}")
    if paper.get("journal_ref"):
        print(f"Journal: {paper['journal_ref']}")
    
    print()
    print("Abstract:")
    print("-" * 60)
    abstract = paper.get("abstract", "")
    # Word wrap at ~70 chars
    words = abstract.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 70:
            print(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        print(line)
    
    print()
    print(f"PDF URL: {paper.get('pdf_url', 'N/A')}")
    print(f"Abstract URL: {paper.get('abs_url', 'N/A')}")
    
    if result["evidence"]:
        print()
        print(f"Evidence ID: {result['evidence'].get('evidence_id', 'N/A')}")
    
    return 0


def cmd_download(
    arxiv_id: str,
    kind: str = "pdf",
    out_path: Optional[str] = None,
    output_json: bool = False,
) -> int:
    """
    Download paper PDF or source.
    
    Args:
        arxiv_id: arXiv paper ID
        kind: "pdf" or "source"
        out_path: Optional output path
        output_json: Output as JSON
        
    Returns:
        Exit code (0 = success)
    """
    result = arxiv_download(
        arxiv_id=arxiv_id,
        kind=kind,
        out_path=out_path,
    )
    
    if result["error"]:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1
    
    if output_json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    
    print(f"Downloaded: {arxiv_id} ({kind})")
    print(f"Path: {result['local_path']}")
    print(f"SHA256: {result['sha256']}")
    print(f"Cached: {result['cached']}")
    
    if result["evidence"]:
        print(f"Evidence ID: {result['evidence'].get('evidence_id', 'N/A')}")
    
    return 0


def cmd_status(output_json: bool = False) -> int:
    """
    Show arXiv connector status.
    
    Args:
        output_json: Output as JSON
        
    Returns:
        Exit code (0 = success)
    """
    status = get_arxiv_tool_status()
    
    if output_json:
        print(json.dumps(status, indent=2, default=str))
        return 0
    
    print("arXiv Connector Status")
    print("=" * 40)
    print(f"Enabled: {status['enabled']}")
    print(f"Ingest Enabled: {status['ingest_enabled']}")
    
    print()
    print("Configuration:")
    for key, value in status.get("config", {}).items():
        print(f"  {key}: {value}")
    
    if status.get("cache_stats"):
        print()
        print("Cache Statistics:")
        for key, value in status["cache_stats"].items():
            print(f"  {key}: {value}")
    
    return 0


def print_help() -> None:
    """Print help message."""
    print("arXiv Connector CLI")
    print()
    print("Usage: python -m deepthinker.connectors.arxiv <command> [options]")
    print()
    print("Commands:")
    print("  search <query>              Search arXiv for papers")
    print("  get <arxiv_id>              Get paper metadata by ID")
    print("  download <arxiv_id>         Download paper PDF or source")
    print("  status                      Show connector status")
    print()
    print("Options:")
    print("  --max-results N             Max search results (default: 10)")
    print("  --start N                   Starting index for pagination")
    print("  --sort-by FIELD             Sort by: relevance, lastUpdatedDate, submittedDate")
    print("  --kind TYPE                 Download type: pdf, source (default: pdf)")
    print("  --out PATH                  Output path for download")
    print("  --json                      Output as JSON")
    print()
    print("Examples:")
    print('  python -m deepthinker.connectors.arxiv search "cat:cs.CL AND ti:alignment"')
    print("  python -m deepthinker.connectors.arxiv get 2501.01234v1")
    print("  python -m deepthinker.connectors.arxiv download 2501.01234 --kind pdf")


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]
    
    if not args or args[0] in ("-h", "--help", "help"):
        print_help()
        return 0
    
    command = args[0].lower()
    remaining = args[1:]
    
    # Parse common options
    output_json = "--json" in remaining
    if output_json:
        remaining.remove("--json")
    
    if command == "search":
        if not remaining:
            print("Error: search requires a query", file=sys.stderr)
            return 1
        
        query = remaining[0]
        max_results = 10
        start = 0
        sort_by = None
        
        # Parse options
        i = 1
        while i < len(remaining):
            if remaining[i] == "--max-results" and i + 1 < len(remaining):
                max_results = int(remaining[i + 1])
                i += 2
            elif remaining[i] == "--start" and i + 1 < len(remaining):
                start = int(remaining[i + 1])
                i += 2
            elif remaining[i] == "--sort-by" and i + 1 < len(remaining):
                sort_by = remaining[i + 1]
                i += 2
            else:
                i += 1
        
        return cmd_search(
            query=query,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            output_json=output_json,
        )
    
    elif command == "get":
        if not remaining:
            print("Error: get requires an arxiv_id", file=sys.stderr)
            return 1
        
        arxiv_id = remaining[0]
        return cmd_get(arxiv_id=arxiv_id, output_json=output_json)
    
    elif command == "download":
        if not remaining:
            print("Error: download requires an arxiv_id", file=sys.stderr)
            return 1
        
        arxiv_id = remaining[0]
        kind = "pdf"
        out_path = None
        
        # Parse options
        i = 1
        while i < len(remaining):
            if remaining[i] == "--kind" and i + 1 < len(remaining):
                kind = remaining[i + 1]
                i += 2
            elif remaining[i] == "--out" and i + 1 < len(remaining):
                out_path = remaining[i + 1]
                i += 2
            else:
                i += 1
        
        return cmd_download(
            arxiv_id=arxiv_id,
            kind=kind,
            out_path=out_path,
            output_json=output_json,
        )
    
    elif command == "status":
        return cmd_status(output_json=output_json)
    
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Use --help for usage information", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


