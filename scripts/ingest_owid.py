#!/usr/bin/env python3
"""
OWID Dataset Ingestion Script for DeepThinker Knowledge Base.

Ingests Our World in Data (OWID) datasets into the general knowledge store
with AI-powered labeling and metadata extraction.

Usage:
    python scripts/ingest_owid.py --source /path/to/owid-datasets/datasets
    python scripts/ingest_owid.py --source /path/to/owid-datasets/datasets --limit 10  # Test run
    python scripts/ingest_owid.py --source /path/to/owid-datasets/datasets --resume  # Resume from checkpoint
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from deepthinker.models.model_caller import call_model, call_embeddings
from deepthinker.memory.rag_store import EmbeddingProvider, cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
LABELING_MODEL = "gemma3:12b"
EMBEDDING_MODEL = "qwen3-embedding:4b"
OLLAMA_BASE_URL = "http://localhost:11434"
KB_BASE_DIR = Path("kb")
PROGRESS_FILE = "owid_progress.json"
BATCH_SIZE = 50  # Save checkpoint every N datasets


# OWID Category mapping for fallback
OWID_CATEGORY_KEYWORDS = {
    "environment": ["climate", "co2", "emission", "pollution", "air quality", "environment", "carbon", "greenhouse", "deforestation", "forest"],
    "health": ["death", "mortality", "disease", "health", "medical", "cancer", "covid", "vaccine", "life expectancy", "hospital"],
    "economy": ["gdp", "income", "poverty", "economic", "trade", "employment", "labor", "wage", "wealth", "financial"],
    "energy": ["energy", "electricity", "fuel", "oil", "gas", "renewable", "nuclear", "coal", "power"],
    "population": ["population", "birth", "fertility", "migration", "urbanization", "demographic"],
    "education": ["education", "literacy", "school", "university", "learning", "student"],
    "food": ["food", "agriculture", "crop", "farming", "calorie", "nutrition", "hunger", "famine"],
    "technology": ["internet", "technology", "digital", "computer", "mobile", "communication"],
    "governance": ["democracy", "government", "corruption", "freedom", "rights", "vote", "election"],
    "conflict": ["war", "conflict", "military", "violence", "terrorism", "peace", "weapon"],
    "transport": ["transport", "aviation", "vehicle", "car", "travel", "road"],
    "water": ["water", "sanitation", "drinking", "access to water"],
}


class OWIDDatasetParser:
    """Parses OWID dataset folders and extracts metadata."""
    
    def __init__(self, dataset_path: Path):
        self.path = dataset_path
        self.name = dataset_path.name
        
    def parse(self) -> Dict[str, Any]:
        """Parse all available metadata from the dataset folder."""
        result = {
            "name": self.name,
            "path": str(self.path),
            "datapackage": None,
            "readme": None,
            "csv_info": None,
            "owid_tags": [],
            "description": "",
            "sources": [],
            "fields": [],
        }
        
        # Parse datapackage.json
        datapackage_path = self.path / "datapackage.json"
        if datapackage_path.exists():
            try:
                with open(datapackage_path, "r", encoding="utf-8") as f:
                    dp = json.load(f)
                result["datapackage"] = dp
                result["description"] = dp.get("description", "")
                result["owid_tags"] = dp.get("owidTags", [])
                result["sources"] = dp.get("sources", [])
                
                # Extract field info
                resources = dp.get("resources", [])
                if resources:
                    schema = resources[0].get("schema", {})
                    fields = schema.get("fields", [])
                    result["fields"] = [
                        {
                            "name": f.get("name"),
                            "type": f.get("type"),
                            "description": f.get("description"),
                        }
                        for f in fields
                    ]
            except Exception as e:
                logger.warning(f"Failed to parse datapackage.json for {self.name}: {e}")
        
        # Parse README.md
        readme_path = self.path / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    result["readme"] = f.read()
            except Exception as e:
                logger.warning(f"Failed to read README.md for {self.name}: {e}")
        
        # Get CSV info (row count, columns)
        csv_files = list(self.path.glob("*.csv"))
        if csv_files:
            try:
                csv_path = csv_files[0]
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    headers = next(reader, [])
                    row_count = sum(1 for _ in reader)
                result["csv_info"] = {
                    "file": csv_path.name,
                    "columns": headers,
                    "row_count": row_count,
                }
            except Exception as e:
                logger.warning(f"Failed to parse CSV for {self.name}: {e}")
        
        return result


class AILabeler:
    """Uses LLM to generate labels and metadata for datasets."""
    
    def __init__(
        self,
        model: str = LABELING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
    
    def label_dataset(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI labels for a dataset.
        
        Returns:
            Dictionary with:
            - primary_category: Main category (e.g., "environment", "health")
            - secondary_tags: List of specific topic tags
            - geographic_scope: "global", "regional", or specific countries
            - temporal_range: Time period covered
            - data_quality: Assessment of data quality
            - summary: Brief description of what the dataset contains
        """
        # Build context for the LLM
        context_parts = [f"Dataset: {parsed_data['name']}"]
        
        if parsed_data.get("description"):
            context_parts.append(f"Description: {parsed_data['description'][:500]}")
        
        if parsed_data.get("owid_tags"):
            context_parts.append(f"Original Tags: {', '.join(parsed_data['owid_tags'])}")
        
        if parsed_data.get("sources"):
            sources_text = ", ".join(s.get("name", "") for s in parsed_data["sources"][:3])
            context_parts.append(f"Sources: {sources_text}")
        
        if parsed_data.get("csv_info"):
            csv_info = parsed_data["csv_info"]
            context_parts.append(f"Columns: {', '.join(csv_info['columns'][:10])}")
            context_parts.append(f"Rows: {csv_info['row_count']}")
        
        if parsed_data.get("fields"):
            field_names = [f["name"] for f in parsed_data["fields"][:8]]
            context_parts.append(f"Fields: {', '.join(field_names)}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Analyze this dataset and provide structured metadata. Respond ONLY with valid JSON, no other text.

{context}

Respond with this exact JSON structure:
{{
    "primary_category": "<one of: environment, health, economy, energy, population, education, food, technology, governance, conflict, transport, water, other>",
    "secondary_tags": ["<tag1>", "<tag2>", "<up to 5 specific topic tags>"],
    "geographic_scope": "<global, regional, or comma-separated country names>",
    "temporal_range": "<e.g., '1990-2020' or 'historical' or 'unknown'>",
    "data_quality": "<high, medium, or low based on source reliability and completeness>",
    "summary": "<One sentence describing what this dataset measures or tracks>"
}}

JSON response:"""

        try:
            response = call_model(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1},
                timeout=self.timeout,
                max_retries=2,
                base_url=self.base_url,
            )
            
            response_text = response.get("response", "")
            
            # Extract JSON from response
            labels = self._parse_json_response(response_text)
            
            if labels:
                return labels
            else:
                logger.warning(f"Failed to parse LLM response for {parsed_data['name']}, using fallback")
                return self._fallback_labels(parsed_data)
                
        except Exception as e:
            logger.warning(f"LLM labeling failed for {parsed_data['name']}: {e}")
            return self._fallback_labels(parsed_data)
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        text = text.strip()
        
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON block
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON with nested arrays
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _fallback_labels(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labels using keyword matching when LLM fails."""
        name_lower = parsed_data["name"].lower()
        desc_lower = (parsed_data.get("description") or "").lower()
        combined = name_lower + " " + desc_lower
        
        # Find primary category
        primary_category = "other"
        max_matches = 0
        
        for category, keywords in OWID_CATEGORY_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in combined)
            if matches > max_matches:
                max_matches = matches
                primary_category = category
        
        # Use OWID tags if available
        secondary_tags = parsed_data.get("owid_tags", [])[:5]
        if not secondary_tags:
            secondary_tags = [primary_category]
        
        return {
            "primary_category": primary_category,
            "secondary_tags": secondary_tags,
            "geographic_scope": "global",
            "temporal_range": "unknown",
            "data_quality": "medium",
            "summary": parsed_data.get("description", "")[:200] or f"Dataset about {primary_category}",
        }


class DocumentChunker:
    """Creates searchable document chunks from parsed dataset metadata."""
    
    def create_chunks(
        self,
        parsed_data: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Create document chunks for RAG indexing.
        
        Returns:
            List of chunk dictionaries ready for embedding
        """
        chunks = []
        dataset_id = self._generate_id(parsed_data["name"])
        
        # Main metadata chunk (always created)
        main_text = self._build_main_chunk_text(parsed_data, labels)
        chunks.append({
            "id": f"owid_{dataset_id}_main",
            "text": main_text,
            "dataset_name": parsed_data["name"],
            "source": "owid",
            "category": labels.get("primary_category", "other"),
            "tags": labels.get("secondary_tags", []),
            "geographic_scope": labels.get("geographic_scope", "global"),
            "temporal_range": labels.get("temporal_range", "unknown"),
            "data_quality": labels.get("data_quality", "medium"),
            "chunk_type": "main",
            "created_at": datetime.utcnow().isoformat(),
        })
        
        # Fields chunk (if many fields)
        fields = parsed_data.get("fields", [])
        if len(fields) > 5:
            fields_text = self._build_fields_chunk_text(parsed_data, fields)
            chunks.append({
                "id": f"owid_{dataset_id}_fields",
                "text": fields_text,
                "dataset_name": parsed_data["name"],
                "source": "owid",
                "category": labels.get("primary_category", "other"),
                "tags": labels.get("secondary_tags", []),
                "chunk_type": "fields",
                "created_at": datetime.utcnow().isoformat(),
            })
        
        return chunks
    
    def _build_main_chunk_text(
        self,
        parsed_data: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> str:
        """Build the main searchable text chunk."""
        parts = [
            f"Dataset: {parsed_data['name']}",
            f"Category: {labels.get('primary_category', 'unknown').title()}",
            f"Topics: {', '.join(labels.get('secondary_tags', []))}",
        ]
        
        if labels.get("summary"):
            parts.append(f"Summary: {labels['summary']}")
        elif parsed_data.get("description"):
            parts.append(f"Description: {parsed_data['description'][:500]}")
        
        if labels.get("geographic_scope"):
            parts.append(f"Geographic Coverage: {labels['geographic_scope']}")
        
        if labels.get("temporal_range") and labels["temporal_range"] != "unknown":
            parts.append(f"Time Period: {labels['temporal_range']}")
        
        if parsed_data.get("sources"):
            source_names = [s.get("name", "") for s in parsed_data["sources"][:3] if s.get("name")]
            if source_names:
                parts.append(f"Data Sources: {', '.join(source_names)}")
        
        csv_info = parsed_data.get("csv_info")
        if csv_info:
            parts.append(f"Data Points: {csv_info.get('row_count', 'unknown')} rows")
            columns = csv_info.get("columns", [])[:8]
            if columns:
                parts.append(f"Key Columns: {', '.join(columns)}")
        
        return "\n".join(parts)
    
    def _build_fields_chunk_text(
        self,
        parsed_data: Dict[str, Any],
        fields: List[Dict[str, Any]],
    ) -> str:
        """Build a chunk describing the data fields."""
        parts = [
            f"Dataset Fields: {parsed_data['name']}",
            "",
            "Available data columns:",
        ]
        
        for field in fields[:20]:  # Limit to 20 fields
            field_name = field.get("name", "Unknown")
            field_type = field.get("type", "")
            field_desc = field.get("description", "")
            
            if field_desc:
                parts.append(f"- {field_name} ({field_type}): {field_desc[:100]}")
            else:
                parts.append(f"- {field_name} ({field_type})")
        
        if len(fields) > 20:
            parts.append(f"... and {len(fields) - 20} more fields")
        
        return "\n".join(parts)
    
    def _generate_id(self, name: str) -> str:
        """Generate a clean ID from dataset name."""
        # Remove special characters, lowercase, replace spaces with underscores
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        clean = clean.lower().replace(' ', '_')[:50]
        return clean


class OWIDIngestionPipeline:
    """Main pipeline for ingesting OWID datasets into the knowledge base."""
    
    def __init__(
        self,
        source_dir: Path,
        kb_dir: Path = KB_BASE_DIR,
        labeling_model: str = LABELING_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.source_dir = source_dir
        self.kb_dir = kb_dir
        self.store_dir = kb_dir / "general_knowledge"
        self.progress_file = self.store_dir / PROGRESS_FILE
        
        self.labeler = AILabeler(model=labeling_model)
        self.chunker = DocumentChunker()
        self.embedding_provider = EmbeddingProvider(embedding_model=embedding_model)
        
        # Storage
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._source_index: Dict[str, List[int]] = {}  # owid, cia -> doc indices
        self._category_index: Dict[str, List[int]] = {}  # category -> doc indices
        self._dataset_index: Dict[str, List[int]] = {}  # dataset_name -> doc indices
        
        # Progress tracking
        self._processed_datasets: set = set()
        
    def run(
        self,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> int:
        """
        Run the ingestion pipeline.
        
        Args:
            limit: Maximum number of datasets to process (for testing)
            resume: Whether to resume from last checkpoint
            
        Returns:
            Number of documents indexed
        """
        # Load existing data and progress
        self._load_existing()
        if resume:
            self._load_progress()
        
        # Find all dataset folders
        dataset_folders = sorted([
            d for d in self.source_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if limit:
            dataset_folders = dataset_folders[:limit]
        
        logger.info(f"Found {len(dataset_folders)} dataset folders to process")
        
        # Filter out already processed
        to_process = [
            d for d in dataset_folders
            if d.name not in self._processed_datasets
        ]
        
        if resume and len(to_process) < len(dataset_folders):
            logger.info(f"Resuming: {len(dataset_folders) - len(to_process)} already processed, {len(to_process)} remaining")
        
        # Process datasets
        processed_count = 0
        start_time = time.time()
        
        for i, dataset_path in enumerate(to_process):
            try:
                logger.info(f"[{i+1}/{len(to_process)}] Processing: {dataset_path.name}")
                
                # Parse dataset
                parser = OWIDDatasetParser(dataset_path)
                parsed_data = parser.parse()
                
                # Generate AI labels
                labels = self.labeler.label_dataset(parsed_data)
                logger.debug(f"  Labels: {labels.get('primary_category')} - {labels.get('secondary_tags')}")
                
                # Create chunks
                chunks = self.chunker.create_chunks(parsed_data, labels)
                logger.debug(f"  Created {len(chunks)} chunks")
                
                # Generate embeddings and store
                for chunk in chunks:
                    self._add_chunk(chunk)
                
                # Track progress
                self._processed_datasets.add(dataset_path.name)
                processed_count += 1
                
                # Checkpoint save
                if processed_count % BATCH_SIZE == 0:
                    logger.info(f"Checkpoint: {processed_count} datasets processed, saving...")
                    self._persist()
                    self._save_progress()
                
                # Rate limiting to avoid overwhelming Ollama
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process {dataset_path.name}: {e}")
                continue
        
        # Final save
        self._persist()
        self._save_progress()
        
        elapsed = time.time() - start_time
        logger.info(f"Completed: {processed_count} datasets processed in {elapsed:.1f}s")
        logger.info(f"Total documents in store: {len(self._documents)}")
        
        return processed_count
    
    def _add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a chunk with embedding to the store."""
        # Generate embedding
        embedding = self.embedding_provider.get_embedding(chunk["text"])
        
        if not embedding:
            logger.warning(f"Failed to get embedding for chunk {chunk['id']}")
            embedding = []
        
        # Add to documents
        doc_idx = len(self._documents)
        self._documents.append(chunk)
        
        # Update indices
        source = chunk.get("source", "unknown")
        if source not in self._source_index:
            self._source_index[source] = []
        self._source_index[source].append(doc_idx)
        
        category = chunk.get("category", "other")
        if category not in self._category_index:
            self._category_index[category] = []
        self._category_index[category].append(doc_idx)
        
        dataset_name = chunk.get("dataset_name", "")
        if dataset_name:
            if dataset_name not in self._dataset_index:
                self._dataset_index[dataset_name] = []
            self._dataset_index[dataset_name].append(doc_idx)
        
        # Add embedding
        if embedding:
            emb_array = np.array(embedding).reshape(1, -1)
            if self._embeddings is None or self._embeddings.size == 0:
                self._embeddings = emb_array
            else:
                if self._embeddings.shape[1] == emb_array.shape[1]:
                    self._embeddings = np.vstack([self._embeddings, emb_array])
                else:
                    logger.warning(f"Embedding dimension mismatch")
        else:
            # Add zero embedding as placeholder
            if self._embeddings is not None and self._embeddings.size > 0:
                zero_emb = np.zeros((1, self._embeddings.shape[1]))
                self._embeddings = np.vstack([self._embeddings, zero_emb])
    
    def _load_existing(self) -> None:
        """Load existing data from the general knowledge store."""
        try:
            docs_path = self.store_dir / "documents.json"
            index_path = self.store_dir / "index.npy"
            
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self._documents = data.get("documents", [])
                self._source_index = data.get("source_index", {})
                self._category_index = data.get("category_index", {})
                self._dataset_index = data.get("dataset_index", {})
                
                # Backwards compat: may have country_index instead of source_index
                if "country_index" in data and not self._source_index:
                    # Keep country_index for CIA data
                    pass
                
                logger.info(f"Loaded {len(self._documents)} existing documents")
            
            if index_path.exists():
                self._embeddings = np.load(index_path)
            else:
                self._embeddings = np.array([])
                
        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")
            self._documents = []
            self._embeddings = np.array([])
    
    def _load_progress(self) -> None:
        """Load ingestion progress for resume."""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, "r") as f:
                    progress = json.load(f)
                self._processed_datasets = set(progress.get("processed_datasets", []))
                logger.info(f"Loaded progress: {len(self._processed_datasets)} datasets already processed")
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
    
    def _save_progress(self) -> None:
        """Save ingestion progress for resume."""
        try:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            progress = {
                "processed_datasets": list(self._processed_datasets),
                "last_updated": datetime.utcnow().isoformat(),
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def _persist(self) -> None:
        """Persist documents and embeddings to disk."""
        try:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents with all indices
            docs_data = {
                "documents": self._documents,
                "source_index": self._source_index,
                "category_index": self._category_index,
                "dataset_index": self._dataset_index,
            }
            
            docs_path = self.store_dir / "documents.json"
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if self._embeddings is not None and self._embeddings.size > 0:
                index_path = self.store_dir / "index.npy"
                np.save(index_path, self._embeddings)
            
            logger.debug(f"Persisted {len(self._documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to persist: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest OWID datasets into DeepThinker knowledge base"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to OWID datasets directory",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=KB_BASE_DIR,
        help="Knowledge base directory (default: kb/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of datasets to process (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--labeling-model",
        type=str,
        default=LABELING_MODEL,
        help=f"Ollama model for labeling (default: {LABELING_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Ollama model for embeddings (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        sys.exit(1)
    
    logger.info(f"Starting OWID ingestion from {args.source}")
    logger.info(f"Labeling model: {args.labeling_model}")
    logger.info(f"Embedding model: {args.embedding_model}")
    
    pipeline = OWIDIngestionPipeline(
        source_dir=args.source,
        kb_dir=args.kb_dir,
        labeling_model=args.labeling_model,
        embedding_model=args.embedding_model,
    )
    
    count = pipeline.run(limit=args.limit, resume=args.resume)
    
    logger.info(f"Done! Indexed {count} datasets")


if __name__ == "__main__":
    main()




