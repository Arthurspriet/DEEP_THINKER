"""
CLI entry point for arXiv connector package.

Usage:
    python -m deepthinker.connectors.arxiv search "query"
    python -m deepthinker.connectors.arxiv get 2501.01234
    python -m deepthinker.connectors.arxiv download 2501.01234 --kind pdf
    python -m deepthinker.connectors.arxiv status
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())


