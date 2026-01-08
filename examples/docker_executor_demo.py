"""
Demo: Docker-Based Secure Code Execution

This script demonstrates how to use the Docker executor for secure code execution.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepthinker.execution.docker_executor import DockerExecutor, DOCKER_AVAILABLE
from deepthinker.execution.security_scanner import SecurityScanner, RiskLevel
from deepthinker.execution.data_config import DataConfig, DockerConfig


def demo_security_scanner():
    """Demonstrate security scanner functionality."""
    print("=" * 60)
    print("SECURITY SCANNER DEMO")
    print("=" * 60)
    
    # Safe code
    safe_code = """
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SafeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=3)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
"""
    
    print("\n1. Scanning SAFE code:")
    print("-" * 60)
    scanner = SecurityScanner(strict_mode=True)
    issues = scanner.scan_code(safe_code)
    
    print(f"Total Issues: {len(issues)}")
    print(f"Max Risk Level: {scanner.get_max_risk_level().value}")
    print(f"Is Safe: {scanner.is_safe()}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  [{issue.risk_level.value.upper()}] {issue.description}")
    else:
        print("✅ No security issues detected!")
    
    # Dangerous code
    dangerous_code = """
import os
import subprocess

class DangerousModel:
    def fit(self, X, y):
        os.system("echo 'malicious command'")
        subprocess.run(["ls", "-la"])
    
    def predict(self, X):
        return eval("[0] * len(X)")
"""
    
    print("\n2. Scanning DANGEROUS code:")
    print("-" * 60)
    scanner = SecurityScanner(strict_mode=True)
    issues = scanner.scan_code(dangerous_code)
    
    print(f"Total Issues: {len(issues)}")
    print(f"Max Risk Level: {scanner.get_max_risk_level().value}")
    print(f"Is Safe: {scanner.is_safe()}")
    
    if issues:
        print("\n⚠️  Security Issues Detected:")
        for issue in issues:
            print(f"  [{issue.risk_level.value.upper()}] {issue.description}")
            if issue.line_number > 0:
                print(f"    Line {issue.line_number}: {issue.code_snippet}")
    
    # Generate report
    print("\n3. Security Report:")
    print("-" * 60)
    report = scanner.get_report()
    print(f"Total Issues: {report['total_issues']}")
    print(f"Max Risk: {report['max_risk_level']}")
    print(f"Is Safe: {report['is_safe']}")
    print(f"Issue Breakdown:")
    for level, count in report['issue_counts'].items():
        if count > 0:
            print(f"  {level.upper()}: {count}")


def demo_docker_executor():
    """Demonstrate Docker executor functionality."""
    print("\n" + "=" * 60)
    print("DOCKER EXECUTOR DEMO")
    print("=" * 60)
    
    if not DOCKER_AVAILABLE:
        print("❌ Docker SDK not installed!")
        print("Install with: pip install docker>=7.0.0")
        return
    
    if not DockerExecutor.is_available():
        print("❌ Docker daemon not running or not accessible!")
        print("Ensure Docker is running and your user has access.")
        return
    
    print("✅ Docker is available!")
    
    # Create test dataset
    import pandas as pd
    import numpy as np
    import tempfile
    
    print("\n1. Creating test dataset...")
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 3, 100)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["target"] = y
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        data_path = f.name
    
    print(f"✅ Created dataset: {data_path}")
    
    # Configure Docker execution
    print("\n2. Configuring Docker executor...")
    config = DataConfig(
        data_path=data_path,
        task_type="classification",
        target_column="target",
        execution_backend="docker",
        execution_timeout=30,
        docker_config=DockerConfig(
            memory_limit="512m",
            cpu_limit=1.0,
            enable_security_scanning=True,
            auto_build_image=True
        )
    )
    print("✅ Configuration created")
    
    # Test safe code execution
    safe_code = """
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class TestModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
"""
    
    print("\n3. Executing SAFE code in Docker container...")
    print("-" * 60)
    
    executor = DockerExecutor(
        memory_limit=config.docker_config.memory_limit,
        cpu_limit=config.docker_config.cpu_limit,
        timeout=config.execution_timeout,
        enable_security_scanning=config.docker_config.enable_security_scanning,
        auto_build_image=config.docker_config.auto_build_image
    )
    
    exec_result, y_test, y_pred = executor.execute_model_on_data(safe_code, config)
    
    if exec_result.success:
        print(f"✅ Execution successful!")
        print(f"   Execution time: {exec_result.execution_time:.2f}s")
        print(f"   Test samples: {len(y_test)}")
        print(f"   Predictions: {len(y_pred)}")
        print(f"   Sample predictions: {y_pred[:5].tolist()}")
    else:
        print(f"❌ Execution failed!")
        print(f"   Error type: {exec_result.error_type}")
        print(f"   Error message: {exec_result.error_message}")
    
    executor.cleanup()
    
    # Test dangerous code (should be blocked)
    dangerous_code = """
import os
import numpy as np

class MaliciousModel:
    def fit(self, X, y):
        os.system("echo 'attempting malicious operation'")
    
    def predict(self, X):
        return [0] * len(X)
"""
    
    print("\n4. Attempting to execute DANGEROUS code...")
    print("-" * 60)
    
    executor = DockerExecutor(
        memory_limit="512m",
        cpu_limit=1.0,
        timeout=30,
        enable_security_scanning=True,
        auto_build_image=True
    )
    
    exec_result, y_test, y_pred = executor.execute_model_on_data(dangerous_code, config)
    
    if exec_result.success:
        print("❌ WARNING: Dangerous code was executed!")
    else:
        print("✅ Dangerous code was blocked by security scanner!")
        print(f"   Error type: {exec_result.error_type}")
        print(f"   Error message: {exec_result.error_message[:200]}...")
    
    executor.cleanup()
    
    # Cleanup
    import os
    os.unlink(data_path)
    print("\n✅ Demo complete!")


def main():
    """Run all demos."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "DeepThinker Docker Executor Demo" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Run security scanner demo
    demo_security_scanner()
    
    # Run Docker executor demo
    demo_docker_executor()
    
    print("\n" + "=" * 60)
    print("For more information, see DOCKER_EXECUTOR.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

