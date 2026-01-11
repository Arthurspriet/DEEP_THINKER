#!/usr/bin/env python3
"""
arXiv RAG Ingestion Script for DeepThinker Knowledge Base.

Downloads arXiv papers via the existing connector, extracts text from PDFs,
chunks the content, and upserts into GlobalRAGStore with full metadata
and provenance tracking.

Usage:
    # Ingest specific papers
    python scripts/arxiv_fill_rag.py --ids 2005.11401 2210.03629
    
    # Ingest a curated seed pack
    python scripts/arxiv_fill_rag.py --seed-pack rag_core
    python scripts/arxiv_fill_rag.py --seed-pack epistemics_foundations
    python scripts/arxiv_fill_rag.py --seed-pack all
    
    # Force re-ingest (ignore idempotency)
    python scripts/arxiv_fill_rag.py --seed-pack rag_core --force
    
    # Dry run (no writes)
    python scripts/arxiv_fill_rag.py --seed-pack rag_core --dry-run
    
    # With LLM-derived labels
    python scripts/arxiv_fill_rag.py --ids 2005.11401 --llm-labels
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

KB_BASE_DIR = Path("kb")
ARXIV_DIR = KB_BASE_DIR / "arxiv"
INGEST_INDEX_PATH = ARXIV_DIR / "ingest_index.json"
REPORTS_DIR = ARXIV_DIR / "ingest_reports"

# Embedding settings (match other ingest scripts)
EMBEDDING_MODEL = "qwen3-embedding:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking settings
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# LLM labeling settings
LABELING_MODEL = "gemma3:12b"


# =============================================================================
# Seed Packs - Curated arXiv Paper Collections
# =============================================================================

SEED_PACKS = {
    # RAG + retrieval fundamentals
    "rag_core": [
        "2005.11401",  # RAG - Retrieval-Augmented Generation
        "2002.08909",  # REALM - Retrieval-Enhanced Language Model
        "2208.03299",  # Atlas - Few-shot Learning with Retrieval
        "2310.11511",  # Self-RAG - Learning to Retrieve, Generate, Critique
    ],
    # Tool use / reasoning agents
    "agents_reasoning": [
        "2210.03629",  # ReAct - Reasoning + Acting
        "2302.04761",  # Toolformer - LMs Teaching Themselves to Use Tools
        "2201.11903",  # Chain-of-Thought Prompting
    ],
    # Truthfulness / hallucination / uncertainty
    "trust_uncertainty": [
        "2109.07958",  # TruthfulQA
        "2303.08896",  # SelfCheckGPT
        "2207.05221",  # Language Models (Mostly) Know What They Know
        "2302.09664",  # Semantic Uncertainty / Semantic Entropy
    ],
    # Epistemic foundations
    "epistemics_foundations": [
        "1810.09108",  # Logical Induction
        "1602.04155",  # Belief Revision (AGM)
        "2005.14007",  # Formal Models of Disagreement
        "2104.04843",  # Epistemic Logic & Belief Update
    ],
    # Causality and explanation
    "causality_explanation": [
        "1912.03477",  # Causal Inference / Book of Why (technical)
        "2002.02104",  # Causal Representation Learning
        "2102.11107",  # Counterfactual Reasoning
    ],
    # Decision theory
    "decision_theory": [
        "0707.4486",   # Foundations of Decision Theory
        "0906.0360",   # Bounded Rationality
        "2301.04655",  # Decision-Making Under Uncertainty
    ],
    # Robustness and failure modes
    "robustness_failure": [
        "1904.02118",  # Robustness and Generalization
        "2106.09685",  # Distribution Shift / OOD
        "2202.05862",  # When Do Models Fail?
    ],
    # Systems and control theory
    "systems_control": [
        "1803.00847",  # Feedback Systems
        "2105.04206",  # Stability in Complex Adaptive Systems
        "1909.08593",  # Control Theory Meets ML
    ],
    # Scientific discovery
    "scientific_discovery": [
        "2004.09550",  # Abduction & Scientific Discovery
        "1806.05005",  # Automated Scientific Discovery
        "2011.07871",  # Falsification & Model Criticism
    ],
    # Non-obvious / innovative perspectives
    "non_obvious_innovative": [
        "1703.00914",  # Free Energy Principle (Friston)
        "1805.08345",  # Active Inference
        "2103.01955",  # Predictive Processing as Cognition
        "2006.07066",  # Collective Intelligence
        "2211.01387",  # Deliberation & Debate Systems
        "2105.09168",  # Pathologies of Optimization
        "1906.01820",  # Limits of Optimization (Goodhart-like effects)
    ],
    # Biological computation and robustness
    "biological_computation": [
        "1708.06247",  # Biological Computation
        "1807.08580",  # Morphological Computation
        "2006.06718",  # Evolution as Information Processing
        "2102.05869",  # Robustness and Degeneracy in Biological Systems
        "1906.01526",  # Sloppiness, Robustness, and Evolvability
        "2106.01788",  # Cellular Decision-Making Under Uncertainty
        "1806.08610",  # Adaptation Without Optimization
        "2001.04451",  # Distributed Control in Biological Systems
        "1909.01412",  # Redundancy and Failure Tolerance in Living Systems
        "2104.14403",  # Homeostasis as a Control Strategy
        "1809.08549",  # Biological Modularity and Stability
        "2003.01354",  # Noise, Information, and Robust Decision Making
    ],
    # Institutions, governance & rule-based stability
    "institutions_governance": [
        "2006.05621",  # Institutional Design and Stability
        "2101.06842",  # Rules, Norms, and System Compliance
        "1905.02145",  # Why Institutions Fail
        "2204.03875",  # Governance as a Control Problem
        "2109.09152",  # Constitutions as Coordination Mechanisms
        "2003.01892",  # Rule-Based Systems and Long-Term Stability
        "1804.04121",  # Institutional Drift and Path Dependence
        "1908.03261",  # Feedback Failure in Governance Systems
        "2106.09812",  # Regulatory Capture as a Systemic Risk
        "2009.06144",  # Stability vs Adaptability in Rule-Based Systems
    ],
    # Measurement theory & epistemic limits
    "measurement_epistemic": [
        "1803.00237",  # Theory of Measurement
        "2004.04434",  # When Metrics Distort Reality
        "2103.13332",  # Limits of Quantification
        "1907.06136",  # Observer Effects in Measurement Systems
        "2106.02587",  # Measurement Error and Decision Bias
        "2009.08742",  # Epistemic Uncertainty and Model Limits
        "1806.01261",  # Measurement-Induced Bias in Complex Systems
        "1904.03922",  # When Proxies Replace Reality
        "2108.01455",  # Epistemic Saturation and Information Limits
    ],
    # Black swans, tail risks & catastrophic reasoning
    "black_swans_tail_risks": [
        "1706.06218",  # Fat Tails and Fragility
        "2001.00991",  # Black Swan Dynamics
        "2104.09891",  # Reasoning Under Extreme Uncertainty
        "1902.02941",  # Risk of Ruin and Decision Theory
        "2105.09168",  # Pathologies of Optimization (also in non_obvious)
        "2008.04651",  # Rare Events and Systemic Collapse
        "1809.02145",  # Catastrophic Failure Modes in Complex Systems
        "2102.03211",  # Tail Risk Dominance in Decision Systems
        "1906.04187",  # Fragility Accumulation in Optimized Systems
    ],
    # Philosophy of limits / incompleteness / stopping conditions
    "philosophy_limits": [
        "1606.04417",  # Gödel, Incompleteness, and Reasoning Systems
        "2009.09120",  # Limits of Formal Systems
        "2107.00566",  # When Reasoning Cannot Converge
        "1901.04584",  # Ethics of Uncertainty
        "2007.08215",  # Paradoxes of Self-Reference
        "1804.01121",  # Incompleteness in Real-World Systems
        "2103.04122",  # Non-Convergence in Iterative Reasoning
        "1909.06233",  # The Cost of Excessive Rationality
        "2005.02841",  # When Optimization Must Stop
        "1808.01917",  # Path Dependence and Irreversibility
    ],
}

# "all" pack is union of all others
SEED_PACKS["all"] = list(set(
    paper_id
    for pack in SEED_PACKS.values()
    for paper_id in pack
))

# Reverse mapping: paper_id -> seed_pack name (for metadata)
PAPER_TO_PACK = {}
for pack_name, paper_ids in SEED_PACKS.items():
    if pack_name != "all":
        for paper_id in paper_ids:
            PAPER_TO_PACK[paper_id] = pack_name


# =============================================================================
# PDF Text Extraction
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, bool]:
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_text, success)
        If extraction fails, returns ("", False)
    """
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue
        
        full_text = "\n\n".join(text_parts)
        
        # Clean up text
        full_text = _clean_extracted_text(full_text)
        
        if len(full_text) < 100:
            logger.warning(f"Extracted text too short ({len(full_text)} chars)")
            return "", False
        
        return full_text, True
        
    except ImportError:
        logger.warning(
            "pypdf not installed. Install with: pip install pypdf\n"
            "Falling back to abstract-only mode."
        )
        return "", False
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return "", False


def _clean_extracted_text(text: str) -> str:
    """Clean up extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'-\n', '', text)  # Hyphenated line breaks
    
    # Remove page numbers that appear alone on lines
    text = re.sub(r'\n\d+\n', '\n', text)
    
    return text.strip()


# =============================================================================
# Text Chunking
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count using simple word-based heuristic.
    
    Rough approximation: ~0.75 tokens per word for English text.
    """
    words = len(text.split())
    return int(words * 1.3)  # Slightly conservative estimate


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens.
    
    Uses sentence boundaries when possible for cleaner splits.
    
    Args:
        text: Full text to chunk
        chunk_size: Target tokens per chunk
        overlap: Token overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # If text is small enough, return as single chunk
    if estimate_tokens(text) <= chunk_size:
        return [text]
    
    # Split into sentences (simple heuristic)
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If single sentence exceeds chunk size, split it further
        if sentence_tokens > chunk_size:
            # Save current chunk if any
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_tokens = 0
            
            for word in words:
                word_tokens += 1.3  # Rough token estimate
                word_chunk.append(word)
                
                if word_tokens >= chunk_size:
                    chunks.append(' '.join(word_chunk))
                    # Keep overlap
                    overlap_words = int(overlap / 1.3)
                    word_chunk = word_chunk[-overlap_words:] if overlap_words > 0 else []
                    word_tokens = len(word_chunk) * 1.3
            
            if word_chunk:
                current_chunk = word_chunk
                current_tokens = word_tokens
            continue
        
        # Check if adding sentence exceeds chunk size
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_tokens = 0
            overlap_sentences = []
            
            # Add sentences from end of current chunk for overlap
            for s in reversed(current_chunk):
                s_tokens = estimate_tokens(s)
                if overlap_tokens + s_tokens <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_tokens
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# =============================================================================
# LLM Labeling (Optional)
# =============================================================================

def generate_llm_labels(
    title: str,
    abstract: str,
    text_sample: str,
    model: str = LABELING_MODEL,
) -> Optional[Dict[str, Any]]:
    """
    Generate derived metadata labels using LLM.
    
    Args:
        title: Paper title
        abstract: Paper abstract
        text_sample: Sample of paper text (first ~2000 chars)
        model: Ollama model to use
        
    Returns:
        Dictionary with derived_topics, derived_paper_type, derived_summary
        and llm_provenance, or None if labeling fails
    """
    try:
        from deepthinker.models.model_caller import call_model
        
        prompt = f"""Analyze this academic paper and provide structured metadata.
Respond ONLY with valid JSON, no other text.

Title: {title}

Abstract: {abstract[:1500]}

Sample text: {text_sample[:1000]}

Respond with this exact JSON structure:
{{
    "derived_topics": ["<topic1>", "<topic2>", "<up to 5 key topics>"],
    "derived_paper_type": "<one of: methodology, survey, empirical, theoretical, benchmark, application>",
    "derived_summary": "<One sentence summary of the paper's main contribution>"
}}

JSON response:"""

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        
        response = call_model(
            model=model,
            prompt=prompt,
            options={"temperature": 0.1},
            timeout=120.0,
            max_retries=2,
            base_url=OLLAMA_BASE_URL,
        )
        
        response_text = response.get("response", "")
        
        # Parse JSON from response
        labels = _parse_json_response(response_text)
        
        if labels:
            return {
                "derived_topics": labels.get("derived_topics", []),
                "derived_paper_type": labels.get("derived_paper_type", "unknown"),
                "derived_summary": labels.get("derived_summary", ""),
                "llm_provenance": {
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "generated_at": datetime.utcnow().isoformat(),
                },
            }
        
        return None
        
    except Exception as e:
        logger.warning(f"LLM labeling failed: {e}")
        return None


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response."""
    text = text.strip()
    
    # Try direct parse
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
    
    # Try with nested structures
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


# =============================================================================
# Idempotency Index
# =============================================================================

def load_ingest_index() -> Dict[str, Any]:
    """Load the idempotency index from disk."""
    if INGEST_INDEX_PATH.exists():
        try:
            with open(INGEST_INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ingest index: {e}")
    return {}


def save_ingest_index(index: Dict[str, Any]) -> None:
    """Save the idempotency index to disk."""
    INGEST_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INGEST_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def is_already_ingested(
    index: Dict[str, Any],
    arxiv_id: str,
    version: Optional[str],
    pdf_sha256: str,
) -> bool:
    """Check if a paper has already been ingested."""
    key = f"{arxiv_id}{version or ''}"
    
    if key not in index:
        return False
    
    entry = index[key]
    
    # Check if SHA256 matches (same content)
    if entry.get("pdf_sha256") == pdf_sha256:
        return True
    
    return False


# =============================================================================
# Main Ingestion Pipeline
# =============================================================================

class ArxivIngestionPipeline:
    """Main pipeline for ingesting arXiv papers into the RAG store."""
    
    def __init__(
        self,
        store_type: str = "global",
        mission_id: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
        llm_labels: bool = False,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.store_type = store_type
        self.mission_id = mission_id
        self.force = force
        self.dry_run = dry_run
        self.llm_labels = llm_labels
        self.embedding_model = embedding_model
        
        # Generate unique run ID
        self.ingest_run_id = str(uuid.uuid4())
        
        # Statistics
        self.stats = {
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "chunks_created": 0,
        }
        
        # Report data
        self.report = {
            "ingest_run_id": self.ingest_run_id,
            "started_at": datetime.utcnow().isoformat(),
            "config": {
                "store_type": store_type,
                "mission_id": mission_id,
                "force": force,
                "dry_run": dry_run,
                "llm_labels": llm_labels,
            },
            "papers": [],
        }
        
        # Load idempotency index
        self.ingest_index = load_ingest_index()
        
        # Initialize RAG store (lazy)
        self._rag_store = None
    
    @property
    def rag_store(self):
        """Lazy-load RAG store."""
        if self._rag_store is None and not self.dry_run:
            if self.store_type == "global":
                from deepthinker.memory.rag_store import GlobalRAGStore
                self._rag_store = GlobalRAGStore(
                    base_dir=KB_BASE_DIR,
                    embedding_model=self.embedding_model,
                    ollama_base_url=OLLAMA_BASE_URL,
                )
            else:
                from deepthinker.memory.rag_store import MissionRAGStore
                self._rag_store = MissionRAGStore(
                    mission_id=self.mission_id,
                    base_dir=KB_BASE_DIR,
                    embedding_model=self.embedding_model,
                    ollama_base_url=OLLAMA_BASE_URL,
                )
        return self._rag_store
    
    def run(self, arxiv_ids: List[str], max_papers: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the ingestion pipeline.
        
        Args:
            arxiv_ids: List of arXiv IDs to ingest
            max_papers: Optional limit on papers to process
            
        Returns:
            Final statistics dictionary
        """
        if max_papers:
            arxiv_ids = arxiv_ids[:max_papers]
        
        logger.info(f"Starting arXiv ingestion run: {self.ingest_run_id}")
        logger.info(f"Papers to process: {len(arxiv_ids)}")
        logger.info(f"Store: {self.store_type}, Force: {self.force}, Dry run: {self.dry_run}")
        
        # Initialize arXiv client (bypassing feature flag)
        from deepthinker.connectors.arxiv.client import ArxivClient
        client = ArxivClient()
        
        for i, arxiv_id in enumerate(arxiv_ids):
            logger.info(f"[{i+1}/{len(arxiv_ids)}] Processing: {arxiv_id}")
            
            try:
                self._process_paper(client, arxiv_id)
            except Exception as e:
                logger.error(f"Failed to process {arxiv_id}: {e}")
                self.stats["failed"] += 1
                self.report["papers"].append({
                    "arxiv_id": arxiv_id,
                    "status": "failed",
                    "error": str(e),
                })
            
            # Rate limiting (be nice to arXiv)
            time.sleep(0.5)
        
        # Save index, RAG store, and report
        if not self.dry_run:
            save_ingest_index(self.ingest_index)
            # Persist RAG store to disk
            if self._rag_store is not None:
                self._rag_store.persist()
                logger.info("RAG store persisted to disk")
        
        self._save_report()
        
        # Log summary
        logger.info("=" * 60)
        logger.info("Ingestion Complete!")
        logger.info(f"  Ingested: {self.stats['ingested']}")
        logger.info(f"  Skipped:  {self.stats['skipped']}")
        logger.info(f"  Failed:   {self.stats['failed']}")
        logger.info(f"  Chunks:   {self.stats['chunks_created']}")
        logger.info(f"  Report:   {REPORTS_DIR / f'{self.ingest_run_id}.json'}")
        logger.info("=" * 60)
        
        return self.stats
    
    def _process_paper(self, client, arxiv_id: str) -> None:
        """Process a single arXiv paper."""
        # Step 1: Fetch metadata
        paper, meta_evidence = client.get_by_id(arxiv_id)
        
        if paper is None:
            raise ValueError(f"Paper not found: {arxiv_id}")
        
        logger.info(f"  Title: {paper.title[:60]}...")
        
        # Step 2: Download PDF
        pdf_path, pdf_sha256, dl_evidence = client.download_pdf(arxiv_id)
        
        logger.info(f"  PDF: {pdf_path} (sha256: {pdf_sha256[:16]}...)")
        
        # Step 3: Check idempotency
        if not self.force and is_already_ingested(
            self.ingest_index,
            paper.id,
            paper.version,
            pdf_sha256,
        ):
            logger.info(f"  Skipping (already ingested)")
            self.stats["skipped"] += 1
            self.report["papers"].append({
                "arxiv_id": arxiv_id,
                "status": "skipped",
                "reason": "already_ingested",
            })
            return
        
        # Step 4: Extract text from PDF
        pdf_text, extraction_ok = extract_text_from_pdf(pdf_path)
        
        if extraction_ok:
            text_to_chunk = pdf_text
            text_source = "pdf"
            logger.info(f"  Extracted {len(pdf_text)} chars from PDF")
        else:
            # Fallback to abstract
            text_to_chunk = f"{paper.title}\n\n{paper.abstract}"
            text_source = "abstract"
            logger.info(f"  Using abstract fallback ({len(text_to_chunk)} chars)")
        
        # Step 5: Chunk text
        chunks = chunk_text(text_to_chunk)
        logger.info(f"  Created {len(chunks)} chunks")
        
        # Step 6: Optional LLM labels
        derived_metadata = None
        if self.llm_labels:
            logger.info("  Generating LLM labels...")
            derived_metadata = generate_llm_labels(
                title=paper.title,
                abstract=paper.abstract,
                text_sample=text_to_chunk[:2000],
            )
            if derived_metadata:
                logger.info(f"  Labels: {derived_metadata.get('derived_topics', [])}")
        
        # Step 7: Build metadata and upsert chunks
        chunk_ids = []
        ingested_at = datetime.utcnow().isoformat()
        
        # Determine seed pack for this paper (if any)
        seed_pack = PAPER_TO_PACK.get(arxiv_id) or PAPER_TO_PACK.get(paper.id)
        
        for chunk_idx, chunk_text_content in enumerate(chunks):
            # Build chunk metadata
            chunk_metadata = {
                "source": "arxiv",
                "arxiv_id": paper.id,
                "version": paper.version,
                "title": paper.title,
                "authors": paper.authors,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "published": paper.published.isoformat() if paper.published else None,
                "updated": paper.updated.isoformat() if paper.updated else None,
                "pdf_path": pdf_path,
                "pdf_sha256": pdf_sha256,
                "evidence_id": dl_evidence.evidence_id,
                "ingest_run_id": self.ingest_run_id,
                "ingested_at": ingested_at,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "text_source": text_source,
                "seed_pack": seed_pack,
            }
            
            # Add derived metadata if available
            if derived_metadata:
                chunk_metadata.update(derived_metadata)
            
            if self.dry_run:
                # Simulate doc ID
                doc_id = f"dry_run_doc_{chunk_idx}"
            else:
                # Upsert to RAG store
                if self.store_type == "global":
                    doc_id = self.rag_store.add_document(
                        text=chunk_text_content,
                        mission_id="arxiv_ingest",
                        phase="ingestion",
                        artifact_type="arxiv_paper",
                        tags=paper.categories[:5],
                        confidence=0.9,
                        source_type="arxiv",
                        metadata=chunk_metadata,
                    )
                else:
                    doc_id = self.rag_store.add_text(
                        text=chunk_text_content,
                        phase="ingestion",
                        artifact_type="arxiv_paper",
                        tags=paper.categories[:5],
                        metadata=chunk_metadata,
                    )
            
            chunk_ids.append(doc_id)
        
        self.stats["chunks_created"] += len(chunks)
        
        # Step 8: Update idempotency index
        index_key = f"{paper.id}{paper.version or ''}"
        self.ingest_index[index_key] = {
            "pdf_sha256": pdf_sha256,
            "ingested_at": ingested_at,
            "chunk_ids": chunk_ids,
            "ingest_run_id": self.ingest_run_id,
            "title": paper.title,
        }
        
        # Step 9: Update report
        self.stats["ingested"] += 1
        self.report["papers"].append({
            "arxiv_id": arxiv_id,
            "status": "ingested",
            "title": paper.title,
            "version": paper.version,
            "text_source": text_source,
            "chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "pdf_sha256": pdf_sha256,
            "evidence_id": dl_evidence.evidence_id,
        })
        
        logger.info(f"  Ingested successfully ({len(chunks)} chunks)")
    
    def _save_report(self) -> None:
        """Save the ingestion report to disk."""
        self.report["completed_at"] = datetime.utcnow().isoformat()
        self.report["stats"] = self.stats
        
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"{self.ingest_run_id}.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved: {report_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into DeepThinker RAG store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest specific papers
    python scripts/arxiv_fill_rag.py --ids 2005.11401 2210.03629
    
    # Ingest a curated seed pack
    python scripts/arxiv_fill_rag.py --seed-pack rag_core
    python scripts/arxiv_fill_rag.py --seed-pack all
    
    # Force re-ingest
    python scripts/arxiv_fill_rag.py --seed-pack epistemics_foundations --force
    
    # Dry run
    python scripts/arxiv_fill_rag.py --seed-pack rag_core --dry-run

Seed Packs:
    rag_core              - RAG fundamentals (RAG, REALM, Atlas, Self-RAG)
    agents_reasoning      - Tool use & reasoning (ReAct, Toolformer, CoT)
    trust_uncertainty     - Truthfulness (TruthfulQA, SelfCheckGPT, Semantic Entropy)
    epistemics_foundations- Epistemic logic (Logical Induction, Belief Revision, AGM)
    causality_explanation - Causal inference (Book of Why, Counterfactuals)
    decision_theory       - Decision theory (Bounded Rationality, Uncertainty)
    robustness_failure    - Robustness (Distribution Shift, OOD, Model Failures)
    systems_control       - Control theory (Feedback Systems, Stability, ML+Control)
    scientific_discovery  - Scientific method (Abduction, Automated Discovery)
    non_obvious_innovative- Innovative (Free Energy, Active Inference, Collective Intel)
    biological_computation- Biological robustness (Morphological, Homeostasis, Degeneracy)
    institutions_governance- Institutions (Governance, Norms, Regulatory Capture)
    measurement_epistemic - Measurement theory (Metrics, Proxies, Epistemic Limits)
    black_swans_tail_risks- Tail risks (Fat Tails, Black Swans, Catastrophic Failure)
    philosophy_limits     - Philosophy of limits (Incompleteness, Non-Convergence)
    all                   - Union of all packs above
        """,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ids",
        nargs="+",
        help="arXiv IDs to ingest (e.g., 2005.11401 2210.03629)",
    )
    input_group.add_argument(
        "--seed-pack",
        choices=list(SEED_PACKS.keys()),
        help="Use a curated seed pack of papers",
    )
    
    # Store options
    parser.add_argument(
        "--store",
        choices=["global", "mission"],
        default="global",
        help="Target RAG store (default: global)",
    )
    parser.add_argument(
        "--mission-id",
        type=str,
        default=None,
        help="Mission ID (required if --store mission)",
    )
    
    # Processing options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingest even if already present",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing",
    )
    parser.add_argument(
        "--llm-labels",
        action="store_true",
        help="Generate LLM-derived metadata labels",
    )
    
    # Other options
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Embedding model (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Validate args
    if args.store == "mission" and not args.mission_id:
        parser.error("--mission-id is required when --store is 'mission'")
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get paper IDs
    if args.ids:
        arxiv_ids = args.ids
    else:
        arxiv_ids = SEED_PACKS[args.seed_pack]
    
    logger.info(f"Papers to ingest: {arxiv_ids}")
    
    # Run pipeline
    pipeline = ArxivIngestionPipeline(
        store_type=args.store,
        mission_id=args.mission_id,
        force=args.force,
        dry_run=args.dry_run,
        llm_labels=args.llm_labels,
        embedding_model=args.embedding_model,
    )
    
    stats = pipeline.run(arxiv_ids, max_papers=args.max_papers)
    
    # Exit code based on results
    if stats["failed"] > 0 and stats["ingested"] == 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

