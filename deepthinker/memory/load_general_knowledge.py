#!/usr/bin/env python3
"""
Loader script for General Knowledge data.

This script indexes reference data like the CIA World Factbook into the
GeneralKnowledgeStore for semantic retrieval during missions.

Usage:
    python -m deepthinker.memory.load_general_knowledge
    python -m deepthinker.memory.load_general_knowledge --source cia_facts
    python -m deepthinker.memory.load_general_knowledge --kb-dir /path/to/kb
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepthinker.memory.general_knowledge_store import GeneralKnowledgeStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_cia_facts(
    kb_dir: Path,
    embedding_model: str = "snowflake-arctic-embed:latest",
    ollama_url: str = "http://localhost:11434",
) -> int:
    """
    Load CIA World Factbook data into the general knowledge store.
    
    Args:
        kb_dir: Knowledge base directory (contains General_knowledge/CIA_FACTS/)
        embedding_model: Ollama embedding model to use
        ollama_url: Ollama server URL
        
    Returns:
        Number of documents indexed
    """
    cia_path = kb_dir / "General_knowledge" / "CIA_FACTS" / "countries.json"
    
    if not cia_path.exists():
        logger.error(f"CIA Factbook not found at {cia_path}")
        return 0
    
    logger.info(f"Loading CIA World Factbook from {cia_path}")
    
    # Initialize store
    store = GeneralKnowledgeStore(
        base_dir=kb_dir,
        embedding_model=embedding_model,
        ollama_base_url=ollama_url,
    )
    
    # Progress callback
    def show_progress(processed: int, total: int):
        pct = (processed / total) * 100
        print(f"\r  Progress: {processed}/{total} ({pct:.1f}%)", end="", flush=True)
    
    # Load and index
    print(f"Indexing CIA World Factbook...")
    count = store.load_cia_facts(
        json_path=cia_path,
        batch_size=100,
        progress_callback=show_progress,
    )
    print()  # Newline after progress
    
    # Print statistics
    stats = store.get_statistics()
    print(f"\n✓ Indexed {count} documents")
    print(f"  Countries: {stats['countries_count']}")
    print(f"  Categories: {stats['categories_count']}")
    print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
    print(f"  Storage: {stats['storage_path']}")
    
    return count


def verify_store(kb_dir: Path) -> bool:
    """
    Verify the general knowledge store is properly loaded.
    
    Args:
        kb_dir: Knowledge base directory
        
    Returns:
        True if store is valid and searchable
    """
    print("\nVerifying store...")
    
    store = GeneralKnowledgeStore(base_dir=kb_dir)
    
    if not store.is_loaded():
        print("✗ Store is not loaded or empty")
        return False
    
    # Test a search
    print("  Testing search: 'population of France'")
    results = store.search("population of France", top_k=3)
    
    if not results:
        print("✗ Search returned no results")
        return False
    
    print(f"  ✓ Found {len(results)} results")
    for doc, score in results[:2]:
        print(f"    - {doc['country']}/{doc['category']}: score={score:.3f}")
    
    # Test country lookup
    print("  Testing country lookup: 'Japan'")
    japan_docs = store.get_country_info("Japan", categories=["economy"])
    
    if not japan_docs:
        print("✗ Country lookup failed")
        return False
    
    print(f"  ✓ Found {len(japan_docs)} Japan economy documents")
    
    print("\n✓ Store verification passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Load general knowledge data into DeepThinker knowledge base"
    )
    parser.add_argument(
        "--source",
        choices=["cia_facts", "all"],
        default="all",
        help="Data source to load (default: all)",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("kb"),
        help="Knowledge base directory (default: kb/)",
    )
    parser.add_argument(
        "--embedding-model",
        default="snowflake-arctic-embed:latest",
        help="Ollama embedding model (default: snowflake-arctic-embed:latest)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing store, don't load new data",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve kb directory
    kb_dir = args.kb_dir.resolve()
    if not kb_dir.exists():
        logger.error(f"Knowledge base directory not found: {kb_dir}")
        sys.exit(1)
    
    print(f"Knowledge base: {kb_dir}")
    
    if args.verify_only:
        success = verify_store(kb_dir)
        sys.exit(0 if success else 1)
    
    # Load data
    total_indexed = 0
    
    if args.source in ["cia_facts", "all"]:
        count = load_cia_facts(
            kb_dir=kb_dir,
            embedding_model=args.embedding_model,
            ollama_url=args.ollama_url,
        )
        total_indexed += count
    
    if total_indexed > 0:
        print(f"\n{'='*50}")
        print(f"Total documents indexed: {total_indexed}")
        
        # Verify
        verify_store(kb_dir)
    else:
        print("\nNo documents were indexed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

