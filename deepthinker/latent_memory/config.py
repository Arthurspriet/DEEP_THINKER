"""
Configuration for Latent Mission Memory Module.

All constants are explicitly defined here. No environment variables.
No defaults. No magic.
"""

from pathlib import Path

# Feature flag - disabled by default
LATENT_MEMORY_ENABLED = False

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda"  # Do not auto-detect
DTYPE = "float16"

# Token limits
MAX_TOKENS_DOC = 1024
MAX_TOKENS_QUERY = 256

# Memory configuration
MEMORY_TOKENS_PER_DOC = 8
TOP_K_RETRIEVAL = 5

# Storage paths (using kb/ to match existing structure)
INDEX_PATH = "kb/latent_memory/faiss.index"
STORE_PATH = "kb/latent_memory/store.pkl"



