"""
Tests for the Security Scanner module.

Tests various aspects of static code analysis including:
- Safe code detection
- Dangerous import detection
- Dangerous builtin detection
- Obfuscation pattern detection
- Introspection detection
- Syntax error handling
"""

import pytest
from deepthinker.execution.security_scanner import SecurityScanner, SecurityIssue, RiskLevel


class TestSecurityScannerBasics:
    """Basic functionality tests for SecurityScanner."""
    
    def test_scanner_initialization(self):
        """Test scanner initializes correctly."""
        scanner = SecurityScanner()
        assert scanner.strict_mode is True
        assert scanner.issues == []
        
        scanner_lenient = SecurityScanner(strict_mode=False)
        assert scanner_lenient.strict_mode is False
    
    def test_empty_code(self):
        """Test scanning empty code."""
        scanner = SecurityScanner()
        issues = scanner.scan_code("")
        assert issues == []
        assert scanner.is_safe()
    
    def test_safe_code(self, sample_python_code):
        """Test that safe ML code passes without issues."""
        scanner = SecurityScanner()
        issues = scanner.scan_code(sample_python_code["safe"])
        
        # Should have no critical or high issues
        critical_issues = [i for i in issues if i.risk_level == RiskLevel.CRITICAL]
        high_issues = [i for i in issues if i.risk_level == RiskLevel.HIGH]
        
        assert len(critical_issues) == 0
        assert len(high_issues) == 0
        assert scanner.is_safe()


class TestDangerousImports:
    """Tests for dangerous import detection."""
    
    def test_os_import(self):
        """Test detection of os module import."""
        scanner = SecurityScanner()
        code = "import os"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        os_issues = [i for i in issues if "os" in i.description.lower()]
        assert len(os_issues) > 0
        assert os_issues[0].risk_level == RiskLevel.CRITICAL
    
    def test_subprocess_import(self):
        """Test detection of subprocess module import."""
        scanner = SecurityScanner()
        code = "import subprocess"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        assert any(i.risk_level == RiskLevel.CRITICAL for i in issues)
    
    def test_socket_import(self):
        """Test detection of socket module import."""
        scanner = SecurityScanner()
        code = "import socket"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        assert any(i.risk_level == RiskLevel.CRITICAL for i in issues)
    
    def test_pickle_import(self):
        """Test detection of pickle module import (deserialization risk)."""
        scanner = SecurityScanner()
        code = "import pickle"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        pickle_issues = [i for i in issues if "pickle" in i.description.lower()]
        assert len(pickle_issues) > 0
        assert pickle_issues[0].risk_level == RiskLevel.HIGH
    
    def test_from_import_dangerous(self):
        """Test detection of dangerous from...import statements."""
        scanner = SecurityScanner()
        code = "from os import system"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        assert any("os" in i.description.lower() for i in issues)
    
    def test_dangerous_code_sample(self, sample_python_code):
        """Test full dangerous code sample."""
        scanner = SecurityScanner()
        issues = scanner.scan_code(sample_python_code["dangerous_import"])
        
        assert len(issues) >= 2  # os and subprocess
        assert not scanner.is_safe()


class TestDangerousBuiltins:
    """Tests for dangerous builtin function detection."""
    
    def test_eval_detection(self):
        """Test detection of eval() usage."""
        scanner = SecurityScanner()
        code = "result = eval('1 + 1')"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        eval_issues = [i for i in issues if "eval" in i.description.lower()]
        assert len(eval_issues) > 0
        assert eval_issues[0].risk_level == RiskLevel.CRITICAL
    
    def test_exec_detection(self):
        """Test detection of exec() usage."""
        scanner = SecurityScanner()
        code = "exec('print(1)')"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        exec_issues = [i for i in issues if "exec" in i.description.lower()]
        assert len(exec_issues) > 0
        assert exec_issues[0].risk_level == RiskLevel.CRITICAL
    
    def test_compile_detection(self):
        """Test detection of compile() usage."""
        scanner = SecurityScanner()
        code = "code = compile('x = 1', '<string>', 'exec')"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        compile_issues = [i for i in issues if "compile" in i.description.lower()]
        assert len(compile_issues) > 0
    
    def test_open_detection(self):
        """Test detection of open() usage."""
        scanner = SecurityScanner()
        code = "f = open('/etc/passwd', 'r')"
        issues = scanner.scan_code(code)
        
        assert len(issues) > 0
        open_issues = [i for i in issues if "open" in i.description.lower()]
        assert len(open_issues) > 0
    
    def test_dangerous_builtin_sample(self, sample_python_code):
        """Test full dangerous builtin sample."""
        scanner = SecurityScanner()
        issues = scanner.scan_code(sample_python_code["dangerous_builtin"])
        
        assert len(issues) > 0
        assert not scanner.is_safe()


class TestObfuscationPatterns:
    """Tests for code obfuscation pattern detection."""
    
    def test_base64_detection(self):
        """Test detection of base64 encoding."""
        scanner = SecurityScanner()
        code = """
import base64
decoded = base64.b64decode("SGVsbG8=")
"""
        issues = scanner.scan_code(code)
        
        obfuscation_issues = [i for i in issues if i.category == "obfuscation"]
        assert len(obfuscation_issues) > 0
    
    def test_chr_chain_detection(self):
        """Test detection of chr() obfuscation chains."""
        scanner = SecurityScanner()
        code = "s = chr(72) + chr(101) + chr(108) + chr(108) + chr(111)"
        issues = scanner.scan_code(code)
        
        obfuscation_issues = [i for i in issues if i.category == "obfuscation"]
        assert len(obfuscation_issues) > 0
    
    def test_dynamic_exec_obfuscation(self):
        """Test detection of dynamic code execution with obfuscation."""
        scanner = SecurityScanner()
        code = "eval(''.join(['p','r','i','n','t']))"
        issues = scanner.scan_code(code)
        
        critical_issues = [i for i in issues if i.risk_level == RiskLevel.CRITICAL]
        assert len(critical_issues) > 0
    
    def test_obfuscated_sample(self, sample_python_code):
        """Test full obfuscated code sample."""
        scanner = SecurityScanner()
        issues = scanner.scan_code(sample_python_code["obfuscated"])
        
        assert len(issues) >= 2  # base64 + exec
        assert not scanner.is_safe()


class TestIntrospection:
    """Tests for suspicious introspection detection."""
    
    def test_dict_access(self):
        """Test detection of __dict__ access."""
        scanner = SecurityScanner()
        code = "members = obj.__dict__"
        issues = scanner.scan_code(code)
        
        introspection_issues = [i for i in issues if i.category == "introspection"]
        assert len(introspection_issues) > 0
    
    def test_class_access(self):
        """Test detection of __class__ access."""
        scanner = SecurityScanner()
        code = "cls = obj.__class__"
        issues = scanner.scan_code(code)
        
        introspection_issues = [i for i in issues if i.category == "introspection"]
        assert len(introspection_issues) > 0
    
    def test_bases_access(self):
        """Test detection of __bases__ access."""
        scanner = SecurityScanner()
        code = "bases = MyClass.__bases__"
        issues = scanner.scan_code(code)
        
        introspection_issues = [i for i in issues if i.category == "introspection"]
        assert len(introspection_issues) > 0
    
    def test_subclasses_access(self):
        """Test detection of __subclasses__ access."""
        scanner = SecurityScanner()
        code = "subs = str.__subclasses__()"
        issues = scanner.scan_code(code)
        
        introspection_issues = [i for i in issues if i.category == "introspection"]
        assert len(introspection_issues) > 0


class TestSyntaxErrors:
    """Tests for syntax error handling."""
    
    def test_syntax_error_handling(self, sample_python_code):
        """Test that syntax errors are caught and reported."""
        scanner = SecurityScanner()
        issues = scanner.scan_code(sample_python_code["syntax_error"])
        
        assert len(issues) > 0
        syntax_issues = [i for i in issues if i.category == "syntax_error"]
        assert len(syntax_issues) > 0
        assert syntax_issues[0].risk_level == RiskLevel.HIGH
    
    def test_indentation_error(self):
        """Test handling of indentation errors."""
        scanner = SecurityScanner()
        code = """
def foo():
print("hello")
"""
        issues = scanner.scan_code(code)
        
        syntax_issues = [i for i in issues if i.category == "syntax_error"]
        assert len(syntax_issues) > 0


class TestRiskLevelAssessment:
    """Tests for risk level assessment functionality."""
    
    def test_get_max_risk_level_empty(self):
        """Test max risk level with no issues."""
        scanner = SecurityScanner()
        scanner.scan_code("x = 1")
        
        assert scanner.get_max_risk_level() == RiskLevel.LOW
    
    def test_get_max_risk_level_critical(self):
        """Test max risk level with critical issues."""
        scanner = SecurityScanner()
        scanner.scan_code("import os")
        
        assert scanner.get_max_risk_level() == RiskLevel.CRITICAL
    
    def test_is_safe_default_threshold(self):
        """Test is_safe with default threshold."""
        scanner = SecurityScanner()
        
        # Low risk code should be safe
        scanner.scan_code("x = 1 + 1")
        assert scanner.is_safe()
        
        # High risk code should not be safe
        scanner.scan_code("import os")
        assert not scanner.is_safe()
    
    def test_is_safe_custom_threshold(self):
        """Test is_safe with custom threshold."""
        scanner = SecurityScanner()
        scanner.scan_code("import tempfile")  # Medium risk
        
        # Should be safe with high threshold
        assert scanner.is_safe(max_allowed_risk=RiskLevel.HIGH)
        
        # Should not be safe with low threshold
        assert not scanner.is_safe(max_allowed_risk=RiskLevel.LOW)


class TestReportGeneration:
    """Tests for security report generation."""
    
    def test_report_structure(self):
        """Test report has correct structure."""
        scanner = SecurityScanner()
        scanner.scan_code("import os\neval('1')")
        
        report = scanner.get_report()
        
        assert "total_issues" in report
        assert "max_risk_level" in report
        assert "is_safe" in report
        assert "issue_counts" in report
        assert "issues" in report
    
    def test_report_counts(self):
        """Test report issue counts."""
        scanner = SecurityScanner()
        scanner.scan_code("import os")
        
        report = scanner.get_report()
        
        assert report["total_issues"] >= 1
        assert report["issue_counts"]["critical"] >= 1
    
    def test_report_issue_details(self):
        """Test report issue details."""
        scanner = SecurityScanner()
        scanner.scan_code("import os")
        
        report = scanner.get_report()
        
        assert len(report["issues"]) > 0
        issue = report["issues"][0]
        assert "risk_level" in issue
        assert "category" in issue
        assert "description" in issue


class TestStrictMode:
    """Tests for strict mode functionality."""
    
    def test_strict_mode_unknown_import(self):
        """Test strict mode flags unknown imports."""
        scanner = SecurityScanner(strict_mode=True)
        code = "import unknown_module"
        issues = scanner.scan_code(code)
        
        unknown_issues = [i for i in issues if i.category == "unknown_import"]
        assert len(unknown_issues) > 0
    
    def test_lenient_mode_unknown_import(self):
        """Test lenient mode allows unknown imports."""
        scanner = SecurityScanner(strict_mode=False)
        code = "import unknown_module"
        issues = scanner.scan_code(code)
        
        unknown_issues = [i for i in issues if i.category == "unknown_import"]
        assert len(unknown_issues) == 0
    
    def test_sklearn_submodules_allowed(self):
        """Test that sklearn submodules are allowed even in strict mode."""
        scanner = SecurityScanner(strict_mode=True)
        code = "from sklearn.ensemble import RandomForestClassifier"
        issues = scanner.scan_code(code)
        
        unknown_issues = [i for i in issues if i.category == "unknown_import"]
        assert len(unknown_issues) == 0

