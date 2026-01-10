"""
HF Instruments Health Check CLI.

Run with: python -m deepthinker.hf_instruments

Provides a health check for:
- torch/transformers availability
- Device detection
- Model loading status
- Index compatibility checks
"""

import sys
from pathlib import Path


def print_header(text: str) -> None:
    """Print a header."""
    print(f"\n{text}")
    print("=" * len(text))


def print_status(name: str, status: str, details: str = "") -> None:
    """Print a status line."""
    status_icon = "✓" if status == "ok" else "✗" if status == "error" else "○"
    if details:
        print(f"  {status_icon} {name}: {details}")
    else:
        print(f"  {status_icon} {name}")


def main() -> int:
    """Run health check."""
    print_header("HF Instruments Health Check")
    
    # Check dependencies
    print_header("Dependencies")
    
    try:
        import torch
        torch_version = torch.__version__
        print_status("torch", "ok", f"available ({torch_version})")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
            print_status("CUDA", "ok", f"available ({cuda_device})")
        else:
            print_status("CUDA", "warning", "not available")
            
    except ImportError:
        print_status("torch", "error", "NOT INSTALLED")
        torch = None
        cuda_available = False
    
    try:
        import transformers
        transformers_version = transformers.__version__
        print_status("transformers", "ok", f"available ({transformers_version})")
    except ImportError:
        print_status("transformers", "error", "NOT INSTALLED")
        transformers = None
    
    # Check config
    print_header("Configuration")
    
    from .config import get_config, HF_AVAILABLE
    
    config = get_config()
    
    print_status("HF_AVAILABLE", "ok" if HF_AVAILABLE else "error", str(HF_AVAILABLE))
    print_status("enabled", "ok" if config.enabled else "warning", str(config.enabled))
    print_status("device", "ok", config.get_resolved_device())
    print_status("reranker_enabled", "ok" if config.reranker_enabled else "warning", str(config.reranker_enabled))
    print_status("embeddings_enabled", "ok" if config.embeddings_enabled else "warning", str(config.embeddings_enabled))
    print_status("claim_extractor_enabled", "ok" if config.claim_extractor_enabled else "warning", str(config.claim_extractor_enabled))
    
    # Check models
    print_header("Models")
    
    print_status("rerank_model_id", "ok", config.rerank_model_id)
    print_status("embed_model_id", "ok", config.embed_model_id)
    print_status("rerank_topn", "ok", str(config.rerank_topn))
    print_status("rerank_batch_size", "ok", str(config.rerank_batch_size))
    print_status("rerank_max_length", "ok", str(config.rerank_max_length))
    print_status("claim_extractor_mode", "ok", config.claim_extractor_mode)
    
    # Check index metadata
    print_header("Index Metadata")
    
    from .meta import get_global_rag_meta, get_general_knowledge_meta
    
    global_meta = get_global_rag_meta()
    if global_meta:
        print_status("kb/rag/global/meta.json", "ok", f"dim={global_meta.embedding_dimension}, model={global_meta.embedding_model_id}")
    else:
        print_status("kb/rag/global/meta.json", "warning", "not found - HF embeddings will be disabled for this index")
    
    gk_meta = get_general_knowledge_meta()
    if gk_meta:
        print_status("kb/general_knowledge/meta.json", "ok", f"dim={gk_meta.embedding_dimension}, model={gk_meta.embedding_model_id}")
    else:
        print_status("kb/general_knowledge/meta.json", "warning", "not found")
    
    # Test reranker if available
    if HF_AVAILABLE and config.enabled and config.reranker_enabled:
        print_header("Reranker Test")
        
        try:
            from .manager import get_reranker
            
            reranker = get_reranker()
            if reranker:
                result = reranker.test_rerank(num_passages=5)
                if result.get("success"):
                    print_status("reranker_load", "ok", "model loaded")
                    print_status("reranker_test", "ok", f"completed in {result['latency_ms']:.1f}ms")
                else:
                    print_status("reranker_test", "error", result.get("error", "unknown error"))
            else:
                print_status("reranker_load", "warning", "reranker not available")
                
        except Exception as e:
            print_status("reranker_test", "error", str(e))
    else:
        print_header("Reranker Test")
        print_status("reranker_test", "warning", "skipped (not enabled or HF not available)")
    
    # Test embedder compatibility
    if HF_AVAILABLE and config.enabled and config.embeddings_enabled:
        print_header("Embedder Compatibility")
        
        try:
            from .meta import get_hf_model_dimension
            
            hf_dim = get_hf_model_dimension(config.embed_model_id)
            if hf_dim:
                print_status("hf_embed_dimension", "ok", str(hf_dim))
                
                if global_meta:
                    if global_meta.embedding_dimension == hf_dim:
                        print_status("global_index_compat", "ok", "COMPATIBLE")
                    else:
                        print_status("global_index_compat", "error", f"INCOMPATIBLE (index={global_meta.embedding_dimension}, HF={hf_dim})")
            else:
                print_status("hf_embed_dimension", "warning", "could not determine")
                
        except Exception as e:
            print_status("embedder_compat", "error", str(e))
    else:
        print_header("Embedder Compatibility")
        print_status("embedder_compat", "warning", "skipped (not enabled or HF not available)")
    
    # Summary
    print_header("Summary")
    
    if not HF_AVAILABLE:
        print("  HF instruments are NOT available (missing torch/transformers)")
        print("  Install with: pip install torch transformers")
        return 1
    
    if not config.enabled:
        print("  HF instruments are available but DISABLED")
        print("  Enable with: export DEEPTHINKER_HF_INSTRUMENTS_ENABLED=true")
        return 0
    
    print("  HF instruments are available and enabled")
    active = []
    if config.is_reranker_active():
        active.append("reranker")
    if config.is_embeddings_active():
        active.append("embeddings")
    if config.is_claim_extractor_active():
        active.append("claim_extractor")
    
    if active:
        print(f"  Active: {', '.join(active)}")
    else:
        print("  No instruments currently active")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

