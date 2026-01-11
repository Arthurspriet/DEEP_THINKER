"""
CLI entry point for bio_priors package.

Usage:
    python -m deepthinker.bio_priors validate
    python -m deepthinker.bio_priors list
    python -m deepthinker.bio_priors run-demo
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())



