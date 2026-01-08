"""
Security scanner for static code analysis before execution.

Detects dangerous patterns and potential security risks in generated code.
"""

import ast
import re
from typing import List, Set, Dict, Any
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk level classification for detected patterns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Represents a detected security issue in code."""
    
    risk_level: RiskLevel
    category: str
    description: str
    line_number: int = 0
    code_snippet: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "risk_level": self.risk_level.value,
            "category": self.category,
            "description": self.description,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet
        }


class SecurityScanner:
    """
    Static code analyzer for detecting security risks.
    
    Scans generated code for dangerous patterns, suspicious imports,
    and potential security vulnerabilities before execution.
    """
    
    # Allowed safe imports for ML tasks
    SAFE_IMPORTS = {
        'numpy', 'np',
        'pandas', 'pd',
        'sklearn', 'scipy',
        'math', 'statistics',
        'itertools', 'collections',
        'typing', 'dataclasses',
        'abc', 'enum'
    }
    
    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile',
        '__import__', 'open',
        'input', 'raw_input'
    }
    
    # Dangerous modules and their risk levels
    DANGEROUS_MODULES = {
        'os': RiskLevel.CRITICAL,
        'sys': RiskLevel.HIGH,
        'subprocess': RiskLevel.CRITICAL,
        'socket': RiskLevel.CRITICAL,
        'urllib': RiskLevel.HIGH,
        'requests': RiskLevel.HIGH,
        'http': RiskLevel.HIGH,
        'ftplib': RiskLevel.CRITICAL,
        'telnetlib': RiskLevel.CRITICAL,
        'pickle': RiskLevel.HIGH,
        'shelve': RiskLevel.HIGH,
        'marshal': RiskLevel.HIGH,
        'tempfile': RiskLevel.MEDIUM,
        'shutil': RiskLevel.HIGH,
        'glob': RiskLevel.MEDIUM,
        'pathlib': RiskLevel.MEDIUM,
        'importlib': RiskLevel.CRITICAL,
        'ctypes': RiskLevel.CRITICAL,
        'multiprocessing': RiskLevel.HIGH,
        'threading': RiskLevel.MEDIUM,
    }
    
    def __init__(
        self,
        strict_mode: bool = True,
        scan_level: str = "strict"
    ):
        """
        Initialize security scanner.
        
        Args:
            strict_mode: If True, block any non-whitelisted imports (deprecated, use scan_level)
            scan_level: Security scan level - "strict", "warn", "log", or "disabled"
        """
        self.strict_mode = strict_mode
        self.scan_level = scan_level
        self.issues: List[SecurityIssue] = []
        
        # Validate scan_level
        valid_levels = ["strict", "warn", "log", "disabled"]
        if scan_level not in valid_levels:
            raise ValueError(
                f"scan_level must be one of {valid_levels}, got '{scan_level}'"
            )
    
    def scan_code(self, code: str) -> List[SecurityIssue]:
        """
        Scan code for security issues.
        
        Args:
            code: Python code to analyze
            
        Returns:
            List of detected security issues
        """
        # Skip scanning if disabled
        if self.scan_level == "disabled":
            return []
        
        self.issues = []
        
        try:
            tree = ast.parse(code)
            self._analyze_ast(tree, code)
        except SyntaxError as e:
            self.issues.append(SecurityIssue(
                risk_level=RiskLevel.HIGH,
                category="syntax_error",
                description=f"Syntax error in code: {str(e)}",
                line_number=getattr(e, 'lineno', 0)
            ))
            return self.issues
        
        # Additional regex-based checks for obfuscated code
        self._check_obfuscation_patterns(code)
        
        return self.issues
    
    def _analyze_ast(self, tree: ast.AST, code: str):
        """Analyze AST for security issues."""
        code_lines = code.split('\n')
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                self._check_imports(node, code_lines)
            elif isinstance(node, ast.ImportFrom):
                self._check_import_from(node, code_lines)
            
            # Check function calls
            elif isinstance(node, ast.Call):
                self._check_function_call(node, code_lines)
            
            # Check attribute access
            elif isinstance(node, ast.Attribute):
                self._check_attribute_access(node, code_lines)
            
            # Check for global/nonlocal (potential namespace pollution)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                self._add_issue(
                    RiskLevel.MEDIUM,
                    "namespace_manipulation",
                    f"Use of {node.__class__.__name__} statement detected",
                    node,
                    code_lines
                )
    
    def _check_imports(self, node: ast.Import, code_lines: List[str]):
        """Check import statements."""
        for alias in node.names:
            module = alias.name.split('.')[0]
            
            if module in self.DANGEROUS_MODULES:
                self._add_issue(
                    self.DANGEROUS_MODULES[module],
                    "dangerous_import",
                    f"Import of dangerous module '{module}' detected",
                    node,
                    code_lines
                )
            elif self.strict_mode and module not in self.SAFE_IMPORTS:
                # Check if it's a sklearn submodule
                if not (alias.name.startswith('sklearn.') or 
                       alias.name.startswith('numpy.') or
                       alias.name.startswith('pandas.')):
                    self._add_issue(
                        RiskLevel.MEDIUM,
                        "unknown_import",
                        f"Non-whitelisted import '{module}' detected",
                        node,
                        code_lines
                    )
    
    def _check_import_from(self, node: ast.ImportFrom, code_lines: List[str]):
        """Check from...import statements."""
        if node.module:
            module = node.module.split('.')[0]
            
            if module in self.DANGEROUS_MODULES:
                self._add_issue(
                    self.DANGEROUS_MODULES[module],
                    "dangerous_import",
                    f"Import from dangerous module '{module}' detected",
                    node,
                    code_lines
                )
            elif self.strict_mode and module not in self.SAFE_IMPORTS:
                if not (node.module.startswith('sklearn.') or 
                       node.module.startswith('numpy.') or
                       node.module.startswith('pandas.')):
                    self._add_issue(
                        RiskLevel.MEDIUM,
                        "unknown_import",
                        f"Non-whitelisted import from '{module}' detected",
                        node,
                        code_lines
                    )
    
    def _check_function_call(self, node: ast.Call, code_lines: List[str]):
        """Check function calls for dangerous built-ins."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.DANGEROUS_BUILTINS:
                self._add_issue(
                    RiskLevel.CRITICAL,
                    "dangerous_builtin",
                    f"Call to dangerous built-in function '{func_name}'",
                    node,
                    code_lines
                )
    
    def _check_attribute_access(self, node: ast.Attribute, code_lines: List[str]):
        """Check attribute access for suspicious patterns."""
        # Check for __dict__, __class__, __bases__, etc.
        if node.attr.startswith('__') and node.attr.endswith('__'):
            if node.attr in ['__dict__', '__class__', '__bases__', '__subclasses__']:
                self._add_issue(
                    RiskLevel.HIGH,
                    "introspection",
                    f"Suspicious introspection via '{node.attr}' detected",
                    node,
                    code_lines
                )
    
    def _check_obfuscation_patterns(self, code: str):
        """Check for code obfuscation patterns using regex."""
        # Check for base64 encoding (common in obfuscated code)
        if re.search(r'base64\.(b64decode|b64encode)', code):
            self.issues.append(SecurityIssue(
                risk_level=RiskLevel.HIGH,
                category="obfuscation",
                description="Base64 encoding/decoding detected (possible obfuscation)"
            ))
        
        # Check for chr() chains (common obfuscation)
        if re.search(r'chr\(\d+\).*chr\(\d+\).*chr\(\d+\)', code):
            self.issues.append(SecurityIssue(
                risk_level=RiskLevel.HIGH,
                category="obfuscation",
                description="Character code obfuscation pattern detected"
            ))
        
        # Check for eval/exec with string manipulation
        if re.search(r'(eval|exec)\s*\(.*(join|decode|chr)', code):
            self.issues.append(SecurityIssue(
                risk_level=RiskLevel.CRITICAL,
                category="obfuscation",
                description="Dynamic code execution with obfuscation detected"
            ))
    
    def _add_issue(
        self,
        risk_level: RiskLevel,
        category: str,
        description: str,
        node: ast.AST,
        code_lines: List[str]
    ):
        """Add a security issue to the list."""
        line_number = getattr(node, 'lineno', 0)
        code_snippet = ""
        
        if 0 < line_number <= len(code_lines):
            code_snippet = code_lines[line_number - 1].strip()
        
        self.issues.append(SecurityIssue(
            risk_level=risk_level,
            category=category,
            description=description,
            line_number=line_number,
            code_snippet=code_snippet
        ))
    
    def get_max_risk_level(self) -> RiskLevel:
        """Get the highest risk level from all detected issues."""
        if not self.issues:
            return RiskLevel.LOW
        
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_risk = RiskLevel.LOW
        
        for issue in self.issues:
            if risk_order.index(issue.risk_level) > risk_order.index(max_risk):
                max_risk = issue.risk_level
        
        return max_risk
    
    def is_safe(self, max_allowed_risk: RiskLevel = RiskLevel.MEDIUM) -> bool:
        """
        Check if code is safe based on maximum allowed risk level and scan level.
        
        Args:
            max_allowed_risk: Maximum acceptable risk level (used for strict mode)
            
        Returns:
            True if code is safe, False otherwise
        """
        # Disabled scan level always returns safe
        if self.scan_level == "disabled":
            return True
        
        # Log level never blocks
        if self.scan_level == "log":
            return True
        
        # Warn level only blocks CRITICAL
        if self.scan_level == "warn":
            max_risk = self.get_max_risk_level()
            return max_risk != RiskLevel.CRITICAL
        
        # Strict level (default) - block HIGH and CRITICAL
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_risk_index = risk_order.index(self.get_max_risk_level())
        allowed_index = risk_order.index(max_allowed_risk)
        
        return max_risk_index <= allowed_index
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate security scan report.
        
        Returns:
            Dictionary with scan results and statistics
        """
        issue_counts = {level: 0 for level in RiskLevel}
        for issue in self.issues:
            issue_counts[issue.risk_level] += 1
        
        return {
            "total_issues": len(self.issues),
            "max_risk_level": self.get_max_risk_level().value,
            "is_safe": self.is_safe(),
            "issue_counts": {level.value: count for level, count in issue_counts.items()},
            "issues": [issue.to_dict() for issue in self.issues]
        }

