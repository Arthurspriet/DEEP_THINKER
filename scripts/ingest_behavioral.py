#!/usr/bin/env python3
"""
Behavioral Science Ingestion Script for DeepThinker Knowledge Base.

Ingests cognitive biases, behavioral economics concepts, and decision-making
heuristics into the general knowledge store.

Sources:
- Buster Benson's Cognitive Bias Cheat Sheet (GitHub JSON)
- The Decision Lab (web scraping for detailed explanations)
- Wikipedia API (for additional context and definitions)

Usage:
    python scripts/ingest_behavioral.py
    python scripts/ingest_behavioral.py --skip-scrape  # Only use JSON source
    python scripts/ingest_behavioral.py --enrich  # Use LLM to add examples
    python scripts/ingest_behavioral.py --limit 10  # Test with limited items
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
import urllib.request
import ssl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from deepthinker.models.model_caller import call_model
from deepthinker.memory.rag_store import EmbeddingProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
ENRICHMENT_MODEL = "gemma3:12b"
EMBEDDING_MODEL = "qwen3-embedding:4b"
OLLAMA_BASE_URL = "http://localhost:11434"
KB_BASE_DIR = Path("kb")
PROGRESS_FILE = "behavioral_progress.json"
BATCH_SIZE = 25  # Save checkpoint every N biases

# Source URLs
BENSON_JSON_URL = "https://raw.githubusercontent.com/busterbenson/public/master/cognitive-bias-cheat-sheet.json"
DECISION_LAB_BASE_URL = "https://thedecisionlab.com/biases"

# Meta-category mappings (from Buster Benson's taxonomy)
META_CATEGORIES = {
    "1. Too Much Information": {
        "id": "too_much_information",
        "description": "What we notice - biases that affect what information we pay attention to",
        "problem": "There is too much information to process",
        "solution": "We filter aggressively"
    },
    "2. Not Enough Meaning": {
        "id": "not_enough_meaning",
        "description": "How we construct stories - biases that affect how we interpret information",
        "problem": "The world is confusing and we need to make sense of it",
        "solution": "We fill in gaps and find patterns"
    },
    "3. Need To Act Fast": {
        "id": "need_to_act_fast",
        "description": "What drives action - biases that affect our decision-making speed",
        "problem": "We need to act quickly with incomplete information",
        "solution": "We favor confident, immediate options"
    },
    "4. What Should We Remember?": {
        "id": "what_to_remember",
        "description": "How memory works - biases that affect what we remember",
        "problem": "We can't remember everything",
        "solution": "We simplify and reinforce memories"
    },
}

# Categories for behavioral content
BEHAVIORAL_CATEGORIES = {
    "cognitive_bias": "Systematic patterns of deviation from rationality in judgment",
    "heuristic": "Mental shortcuts that ease cognitive load in decision-making",
    "nudge": "Interventions that alter behavior without forbidding options",
    "influence": "Principles of persuasion and social influence",
    "decision_making": "Frameworks and patterns in how decisions are made",
    "memory": "Biases and effects related to memory encoding and retrieval",
    "social": "Biases related to social perception and group dynamics",
}


class BensonJSONParser:
    """Parses the Buster Benson cognitive bias cheat sheet JSON."""
    
    def __init__(self):
        self.biases: List[Dict[str, Any]] = []
        
    def download_and_parse(self) -> List[Dict[str, Any]]:
        """Download and parse the JSON from GitHub."""
        logger.info(f"Downloading cognitive bias data from GitHub...")
        
        try:
            # Create SSL context that doesn't verify certificates (for simplicity)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(BENSON_JSON_URL, context=ctx, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            logger.info("Successfully downloaded cognitive bias JSON")
            return self._parse_hierarchical_json(data)
            
        except Exception as e:
            logger.error(f"Failed to download JSON: {e}")
            # Try to load from local cache if available
            cache_path = KB_BASE_DIR / "behavioral_cache" / "benson_biases.json"
            if cache_path.exists():
                logger.info("Loading from local cache...")
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self._parse_hierarchical_json(data)
            raise
    
    def _parse_hierarchical_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse the hierarchical JSON structure into flat bias list."""
        biases = []
        
        # The JSON has a hierarchical structure:
        # biases -> meta_categories -> sub_categories -> individual biases
        for meta_cat in data.get("children", []):
            meta_name = meta_cat.get("name", "")
            meta_info = META_CATEGORIES.get(meta_name, {
                "id": self._generate_id(meta_name),
                "description": "",
                "problem": "",
                "solution": ""
            })
            
            for sub_cat in meta_cat.get("children", []):
                sub_name = sub_cat.get("name", "")
                related_biases = []
                
                # Collect all biases in this sub-category
                for bias in sub_cat.get("children", []):
                    bias_name = bias.get("name", "")
                    if bias_name:
                        related_biases.append(bias_name)
                
                # Now add each bias with its context
                for bias in sub_cat.get("children", []):
                    bias_name = bias.get("name", "")
                    if not bias_name:
                        continue
                    
                    # Get related biases (same sub-category, excluding self)
                    related = [b for b in related_biases if b != bias_name]
                    
                    biases.append({
                        "name": bias_name,
                        "meta_category": meta_info["id"],
                        "meta_category_name": meta_name,
                        "meta_category_description": meta_info["description"],
                        "meta_category_problem": meta_info.get("problem", ""),
                        "meta_category_solution": meta_info.get("solution", ""),
                        "sub_category": sub_name,
                        "related_biases": related[:5],  # Limit to 5 related
                        "source": "benson_taxonomy",
                    })
        
        logger.info(f"Parsed {len(biases)} cognitive biases from taxonomy")
        return biases
    
    def _generate_id(self, name: str) -> str:
        """Generate a clean ID from a name."""
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        return clean.lower().replace(' ', '_')[:50]


class DecisionLabScraper:
    """Scrapes The Decision Lab for detailed bias explanations."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay  # Delay between requests to be polite
        self.scraped_count = 0
        
    def scrape_bias_page(self, bias_name: str) -> Optional[Dict[str, Any]]:
        """Scrape a single bias page from The Decision Lab."""
        # Convert bias name to URL slug
        slug = self._name_to_slug(bias_name)
        url = f"{DECISION_LAB_BASE_URL}/{slug}"
        
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'DeepThinker/1.0 (Research; Educational)',
                    'Accept': 'text/html',
                }
            )
            
            with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
                html = response.read().decode('utf-8')
            
            # Extract content from HTML (simple extraction)
            description = self._extract_description(html)
            example = self._extract_example(html)
            
            if description:
                self.scraped_count += 1
                return {
                    "description": description,
                    "example": example,
                    "url": url,
                }
            
        except Exception as e:
            logger.debug(f"Could not scrape {bias_name}: {e}")
        
        return None
    
    def _name_to_slug(self, name: str) -> str:
        """Convert bias name to URL slug."""
        # Handle common name variations
        slug = name.lower()
        slug = re.sub(r'[''`]', '', slug)  # Remove apostrophes
        slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars
        slug = re.sub(r'\s+', '-', slug)  # Spaces to hyphens
        slug = re.sub(r'-+', '-', slug)  # Multiple hyphens to single
        slug = slug.strip('-')
        return slug
    
    def _extract_description(self, html: str) -> Optional[str]:
        """Extract main description from HTML."""
        # Look for the main content description
        # This is a simplified extraction - would need proper HTML parsing for production
        patterns = [
            r'<meta name="description" content="([^"]+)"',
            r'<p class="[^"]*intro[^"]*">([^<]+)</p>',
            r'<div class="[^"]*entry-content[^"]*">.*?<p>([^<]+)</p>',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                text = match.group(1)
                # Clean up HTML entities
                text = text.replace('&amp;', '&')
                text = text.replace('&lt;', '<')
                text = text.replace('&gt;', '>')
                text = text.replace('&#8217;', "'")
                text = text.replace('&#8220;', '"')
                text = text.replace('&#8221;', '"')
                text = re.sub(r'<[^>]+>', '', text)  # Remove any remaining HTML tags
                return text.strip()
        
        return None
    
    def _extract_example(self, html: str) -> Optional[str]:
        """Extract an example from the HTML."""
        # Look for example sections
        patterns = [
            r'<h[23][^>]*>.*?[Ee]xample.*?</h[23]>.*?<p>([^<]+)</p>',
            r'[Ff]or example[,:]?\s+([^.]+\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
        
        return None


class WikipediaEnricher:
    """Enriches bias data with Wikipedia definitions."""
    
    WIKI_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    
    def get_summary(self, term: str) -> Optional[Dict[str, Any]]:
        """Get Wikipedia summary for a term."""
        try:
            # URL encode the term
            encoded_term = quote(term.replace(' ', '_'))
            url = f"{self.WIKI_API_URL}{encoded_term}"
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'DeepThinker/1.0 (Research; Educational)',
                    'Accept': 'application/json',
                }
            )
            
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if data.get("type") == "standard":
                return {
                    "extract": data.get("extract", ""),
                    "description": data.get("description", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
                
        except Exception as e:
            logger.debug(f"Wikipedia lookup failed for {term}: {e}")
        
        return None


class AIEnricher:
    """Uses LLM to enrich bias data with examples and practical implications."""
    
    def __init__(
        self,
        model: str = ENRICHMENT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
    
    def enrich_bias(self, bias_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add AI-generated examples and implications."""
        name = bias_data.get("name", "")
        description = bias_data.get("description", "")
        sub_category = bias_data.get("sub_category", "")
        
        prompt = f"""Analyze this cognitive bias and provide practical information. Respond ONLY with valid JSON.

Bias: {name}
Category context: {sub_category}
{f'Description: {description[:300]}' if description else ''}

Provide:
{{
    "definition": "<Clear 1-2 sentence definition of this bias>",
    "real_world_example": "<Specific everyday example where this bias occurs>",
    "business_implication": "<How this bias affects business/marketing decisions>",
    "debiasing_strategy": "<One technique to counteract this bias>",
    "related_concepts": ["<2-3 related psychological concepts>"]
}}

JSON response:"""

        try:
            response = call_model(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.3},
                timeout=self.timeout,
                max_retries=2,
                base_url=self.base_url,
            )
            
            response_text = response.get("response", "")
            enrichment = self._parse_json_response(response_text)
            
            if enrichment:
                return enrichment
                
        except Exception as e:
            logger.warning(f"AI enrichment failed for {name}: {e}")
        
        return {}
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        text = text.strip()
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None


class BiasDocumentChunker:
    """Creates searchable document chunks from bias data."""
    
    def create_chunk(self, bias_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a document chunk for a single bias."""
        name = bias_data.get("name", "Unknown")
        chunk_id = self._generate_id(name)
        
        # Build searchable text
        text_parts = [
            f"Cognitive Bias: {name}",
            f"Category: {bias_data.get('meta_category', 'cognitive_bias').replace('_', ' ').title()}",
        ]
        
        if bias_data.get("sub_category"):
            text_parts.append(f"Why this happens: {bias_data['sub_category']}")
        
        # Add description from various sources
        if bias_data.get("definition"):
            text_parts.append(f"Definition: {bias_data['definition']}")
        elif bias_data.get("description"):
            text_parts.append(f"Description: {bias_data['description']}")
        elif bias_data.get("wiki_extract"):
            text_parts.append(f"Description: {bias_data['wiki_extract'][:500]}")
        
        # Add example if available
        if bias_data.get("real_world_example"):
            text_parts.append(f"Example: {bias_data['real_world_example']}")
        elif bias_data.get("example"):
            text_parts.append(f"Example: {bias_data['example']}")
        
        # Add business context
        if bias_data.get("business_implication"):
            text_parts.append(f"Business Impact: {bias_data['business_implication']}")
        
        # Add debiasing strategy
        if bias_data.get("debiasing_strategy"):
            text_parts.append(f"How to avoid: {bias_data['debiasing_strategy']}")
        
        # Add related biases
        related = bias_data.get("related_biases", [])
        if related:
            text_parts.append(f"Related biases: {', '.join(related[:5])}")
        
        # Determine primary category
        category = self._determine_category(bias_data)
        
        # Build tags
        tags = self._build_tags(bias_data)
        
        return {
            "id": f"behavioral_{chunk_id}",
            "text": "\n".join(text_parts),
            "name": name,
            "source": "behavioral_science",
            "category": category,
            "meta_category": bias_data.get("meta_category", ""),
            "sub_category": bias_data.get("sub_category", ""),
            "related_biases": related[:5],
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
            "data_source": bias_data.get("source", "benson_taxonomy"),
            "has_example": bool(bias_data.get("real_world_example") or bias_data.get("example")),
            "has_definition": bool(bias_data.get("definition") or bias_data.get("description") or bias_data.get("wiki_extract")),
        }
    
    def _generate_id(self, name: str) -> str:
        """Generate a clean ID from bias name."""
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        return clean.lower().replace(' ', '_')[:50]
    
    def _determine_category(self, bias_data: Dict[str, Any]) -> str:
        """Determine the primary category for the bias."""
        meta_cat = bias_data.get("meta_category", "")
        
        if "memory" in meta_cat or "remember" in meta_cat.lower():
            return "memory"
        elif "social" in bias_data.get("name", "").lower():
            return "social"
        else:
            return "cognitive_bias"
    
    def _build_tags(self, bias_data: Dict[str, Any]) -> List[str]:
        """Build relevant tags for the bias."""
        tags = ["behavioral-science", "psychology"]
        
        name_lower = bias_data.get("name", "").lower()
        sub_cat = bias_data.get("sub_category", "").lower()
        
        # Add category-based tags
        if "memory" in name_lower or "memory" in sub_cat:
            tags.append("memory")
        if "social" in name_lower or "group" in sub_cat:
            tags.append("social")
        if "decision" in sub_cat or "choice" in sub_cat:
            tags.append("decision-making")
        if "confirm" in name_lower or "belief" in sub_cat:
            tags.append("confirmation")
        if "probability" in sub_cat or "risk" in name_lower:
            tags.append("risk-assessment")
        if "attention" in name_lower or "notice" in sub_cat:
            tags.append("attention")
        
        # Add meta-category tag
        meta = bias_data.get("meta_category", "")
        if meta:
            tags.append(meta.replace("_", "-"))
        
        return list(set(tags))[:8]  # Limit to 8 unique tags


class BehavioralIngestionPipeline:
    """Main pipeline for ingesting behavioral science content."""
    
    def __init__(
        self,
        kb_dir: Path = KB_BASE_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        enrichment_model: str = ENRICHMENT_MODEL,
        skip_scrape: bool = False,
        skip_wiki: bool = False,
        use_ai_enrichment: bool = False,
    ):
        self.kb_dir = kb_dir
        self.store_dir = kb_dir / "general_knowledge"
        self.cache_dir = kb_dir / "behavioral_cache"
        self.progress_file = self.store_dir / PROGRESS_FILE
        
        self.skip_scrape = skip_scrape
        self.skip_wiki = skip_wiki
        self.use_ai_enrichment = use_ai_enrichment
        
        # Components
        self.parser = BensonJSONParser()
        self.scraper = DecisionLabScraper() if not skip_scrape else None
        self.wiki_enricher = WikipediaEnricher() if not skip_wiki else None
        self.ai_enricher = AIEnricher(model=enrichment_model) if use_ai_enrichment else None
        self.chunker = BiasDocumentChunker()
        self.embedding_provider = EmbeddingProvider(embedding_model=embedding_model)
        
        # Storage
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._source_index: Dict[str, List[int]] = {}
        self._category_index: Dict[str, List[int]] = {}
        self._dataset_index: Dict[str, List[int]] = {}
        
        # Progress
        self._processed_biases: set = set()
    
    def run(
        self,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> int:
        """Run the ingestion pipeline."""
        # Ensure directories exist
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_existing()
        if resume:
            self._load_progress()
        
        # Parse source data
        try:
            biases = self.parser.download_and_parse()
        except Exception as e:
            logger.error(f"Failed to get bias data: {e}")
            return 0
        
        # Cache the raw data
        self._cache_raw_data(biases)
        
        if limit:
            biases = biases[:limit]
        
        # Filter already processed
        to_process = [
            b for b in biases
            if b.get("name") not in self._processed_biases
        ]
        
        if resume and len(to_process) < len(biases):
            logger.info(f"Resuming: {len(biases) - len(to_process)} already processed, {len(to_process)} remaining")
        
        logger.info(f"Processing {len(to_process)} cognitive biases...")
        
        processed_count = 0
        start_time = time.time()
        
        for i, bias_data in enumerate(to_process):
            try:
                name = bias_data.get("name", "Unknown")
                logger.info(f"[{i+1}/{len(to_process)}] Processing: {name}")
                
                # Enrich with Decision Lab content
                if self.scraper:
                    scraped = self.scraper.scrape_bias_page(name)
                    if scraped:
                        bias_data["description"] = scraped.get("description", "")
                        bias_data["example"] = scraped.get("example", "")
                        bias_data["source_url"] = scraped.get("url", "")
                        logger.debug(f"  + Decision Lab content")
                    time.sleep(self.scraper.delay)  # Be polite
                
                # Enrich with Wikipedia
                if self.wiki_enricher:
                    wiki_data = self.wiki_enricher.get_summary(name)
                    if wiki_data:
                        bias_data["wiki_extract"] = wiki_data.get("extract", "")
                        bias_data["wiki_description"] = wiki_data.get("description", "")
                        bias_data["wiki_url"] = wiki_data.get("url", "")
                        logger.debug(f"  + Wikipedia content")
                    time.sleep(0.2)  # Rate limit
                
                # AI enrichment (optional)
                if self.ai_enricher:
                    enrichment = self.ai_enricher.enrich_bias(bias_data)
                    if enrichment:
                        bias_data.update(enrichment)
                        logger.debug(f"  + AI enrichment")
                
                # Create chunk
                chunk = self.chunker.create_chunk(bias_data)
                
                # Add to store
                self._add_chunk(chunk)
                
                # Track progress
                self._processed_biases.add(name)
                processed_count += 1
                
                # Checkpoint
                if processed_count % BATCH_SIZE == 0:
                    logger.info(f"Checkpoint: {processed_count} biases processed, saving...")
                    self._persist()
                    self._save_progress()
                
            except Exception as e:
                logger.error(f"Failed to process {bias_data.get('name', 'Unknown')}: {e}")
                continue
        
        # Final save
        self._persist()
        self._save_progress()
        
        elapsed = time.time() - start_time
        logger.info(f"Completed: {processed_count} biases processed in {elapsed:.1f}s")
        logger.info(f"Total behavioral documents in store: {self._count_behavioral_docs()}")
        
        if self.scraper:
            logger.info(f"Successfully scraped {self.scraper.scraped_count} bias pages from Decision Lab")
        
        return processed_count
    
    def _add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a chunk with embedding to the store."""
        # Generate embedding
        embedding = self.embedding_provider.get_embedding(chunk["text"])
        
        if not embedding:
            logger.warning(f"Failed to get embedding for {chunk['id']}")
            embedding = []
        
        # Add to documents
        doc_idx = len(self._documents)
        self._documents.append(chunk)
        
        # Update indices
        source = chunk.get("source", "behavioral_science")
        if source not in self._source_index:
            self._source_index[source] = []
        self._source_index[source].append(doc_idx)
        
        category = chunk.get("category", "cognitive_bias")
        if category not in self._category_index:
            self._category_index[category] = []
        self._category_index[category].append(doc_idx)
        
        # Index by name for lookup
        name = chunk.get("name", "")
        if name:
            if name not in self._dataset_index:
                self._dataset_index[name] = []
            self._dataset_index[name].append(doc_idx)
        
        # Add embedding
        if embedding:
            emb_array = np.array(embedding).reshape(1, -1)
            if self._embeddings is None or self._embeddings.size == 0:
                self._embeddings = emb_array
            else:
                if self._embeddings.shape[1] == emb_array.shape[1]:
                    self._embeddings = np.vstack([self._embeddings, emb_array])
                else:
                    logger.warning("Embedding dimension mismatch")
        else:
            if self._embeddings is not None and self._embeddings.size > 0:
                zero_emb = np.zeros((1, self._embeddings.shape[1]))
                self._embeddings = np.vstack([self._embeddings, zero_emb])
    
    def _cache_raw_data(self, biases: List[Dict[str, Any]]) -> None:
        """Cache the raw bias data."""
        try:
            cache_file = self.cache_dir / "benson_biases.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"biases": biases, "cached_at": datetime.utcnow().isoformat()}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache raw data: {e}")
    
    def _count_behavioral_docs(self) -> int:
        """Count behavioral science documents in store."""
        return len(self._source_index.get("behavioral_science", []))
    
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
        """Load ingestion progress."""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, "r") as f:
                    progress = json.load(f)
                self._processed_biases = set(progress.get("processed_biases", []))
                logger.info(f"Loaded progress: {len(self._processed_biases)} biases already processed")
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
    
    def _save_progress(self) -> None:
        """Save ingestion progress."""
        try:
            progress = {
                "processed_biases": list(self._processed_biases),
                "last_updated": datetime.utcnow().isoformat(),
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def _persist(self) -> None:
        """Persist documents and embeddings to disk."""
        try:
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
        description="Ingest behavioral science content into DeepThinker knowledge base"
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
        help="Limit number of biases to process (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping Decision Lab (faster, less detailed)",
    )
    parser.add_argument(
        "--skip-wiki",
        action="store_true",
        help="Skip Wikipedia enrichment",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Use LLM to add examples and explanations (slower)",
    )
    parser.add_argument(
        "--enrichment-model",
        type=str,
        default=ENRICHMENT_MODEL,
        help=f"Ollama model for enrichment (default: {ENRICHMENT_MODEL})",
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
    
    logger.info("Starting behavioral science content ingestion")
    logger.info(f"Embedding model: {args.embedding_model}")
    if args.enrich:
        logger.info(f"Enrichment model: {args.enrichment_model}")
    if args.skip_scrape:
        logger.info("Skipping Decision Lab scraping")
    if args.skip_wiki:
        logger.info("Skipping Wikipedia enrichment")
    
    pipeline = BehavioralIngestionPipeline(
        kb_dir=args.kb_dir,
        embedding_model=args.embedding_model,
        enrichment_model=args.enrichment_model,
        skip_scrape=args.skip_scrape,
        skip_wiki=args.skip_wiki,
        use_ai_enrichment=args.enrich,
    )
    
    count = pipeline.run(limit=args.limit, resume=args.resume)
    
    logger.info(f"Done! Indexed {count} cognitive biases")


if __name__ == "__main__":
    main()


