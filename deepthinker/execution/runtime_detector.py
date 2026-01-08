"""
Runtime detector for multi-language code execution.

Detects programming language from code and maps to appropriate Docker images.
"""

import re
from typing import Optional


class RuntimeDetector:
    """
    Detects programming language from code.
    
    Supports: Python, Node.js, Bash
    """
    
    # Language detection patterns
    PYTHON_PATTERNS = [
        r'^import\s+\w+',
        r'^from\s+\w+\s+import',
        r'^def\s+\w+\s*\(',
        r'^class\s+\w+',
        r'^\s*@\w+',
        r'^\s*if\s+__name__\s*==\s*["\']__main__["\']',
    ]
    
    NODE_PATTERNS = [
        r'^const\s+\w+',
        r'^let\s+\w+',
        r'^var\s+\w+',
        r'^function\s+\w+',
        r'^require\s*\(',
        r'^import\s+.*\s+from\s+',
        r'^export\s+',
        r'^module\.exports',
    ]
    
    BASH_PATTERNS = [
        r'^#!/bin/(ba)?sh',
        r'^\$\{',
        r'^\$\(',
        r'^if\s+\[',
        r'^for\s+\w+\s+in',
        r'^while\s+\[',
    ]
    
    def detect_language(self, code: str) -> str:
        """
        Detect programming language from code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Language name: "python", "node", "bash", or "unknown"
        """
        if not code or not code.strip():
            return "unknown"
        
        lines = code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return "unknown"
        
        # Check first few lines for shebang
        for line in non_empty_lines[:5]:
            if re.match(r'^#!/usr/bin/env\s+(\w+)', line):
                match = re.match(r'^#!/usr/bin/env\s+(\w+)', line)
                if match:
                    lang = match.group(1).lower()
                    if lang in ["python", "python3", "node", "nodejs"]:
                        return "python" if lang.startswith("python") else "node"
            if re.match(r'^#!/bin/(ba)?sh', line):
                return "bash"
        
        # Pattern matching
        python_score = 0
        node_score = 0
        bash_score = 0
        
        for line in non_empty_lines[:50]:  # Check first 50 lines
            for pattern in self.PYTHON_PATTERNS:
                if re.match(pattern, line):
                    python_score += 1
                    break
            
            for pattern in self.NODE_PATTERNS:
                if re.match(pattern, line):
                    node_score += 1
                    break
            
            for pattern in self.BASH_PATTERNS:
                if re.match(pattern, line):
                    bash_score += 1
                    break
        
        # Return language with highest score
        scores = {
            "python": python_score,
            "node": node_score,
            "bash": bash_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            # Default to Python if no patterns match
            return "python"
        
        return max(scores, key=scores.get)
    
    def get_docker_image(self, language: str) -> Optional[str]:
        """
        Get recommended Docker image for language.
        
        Args:
            language: Language name (python, node, bash)
            
        Returns:
            Docker image name, or None if not supported
        """
        image_map = {
            "python": "deepthinker-sandbox:latest",
            "node": "deepthinker-sandbox-node:latest",
            "bash": "deepthinker-sandbox:latest",  # Bash can run in Python image
        }
        
        return image_map.get(language.lower())


# Global detector instance
_default_detector = RuntimeDetector()


def get_default_detector() -> RuntimeDetector:
    """Get the default runtime detector."""
    return _default_detector

