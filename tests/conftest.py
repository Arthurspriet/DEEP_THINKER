"""
Pytest fixtures and configuration for DeepThinker test suite.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure deepthinker package is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="deepthinker_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_ollama_response():
    """Mock response from Ollama API."""
    def _make_response(content: str = "Test response"):
        mock = MagicMock()
        mock.content = content
        return mock
    return _make_response


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mock LLM response")
    return mock


@pytest.fixture
def sample_python_code():
    """Sample Python code for security scanner tests."""
    return {
        "safe": '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test
''',
        "dangerous_import": '''
import os
import subprocess

def run_command(cmd):
    return subprocess.run(cmd, shell=True)
''',
        "dangerous_builtin": '''
code = "print('hello')"
eval(code)
''',
        "obfuscated": '''
import base64
decoded = base64.b64decode("SGVsbG8=")
exec(decoded)
''',
        "introspection": '''
class MyClass:
    pass

obj = MyClass()
members = obj.__class__.__bases__
''',
        "syntax_error": '''
def broken(
    return None
''',
    }


@pytest.fixture
def sample_model_outputs():
    """Sample model outputs for consensus algorithm tests."""
    return {
        "similar_outputs": {
            "model_a": "The answer is 42 because it represents the meaning of life.",
            "model_b": "42 is the answer as it symbolizes the meaning of life.",
            "model_c": "The meaning of life is represented by 42.",
        },
        "diverse_outputs": {
            "model_a": "The capital of France is Paris.",
            "model_b": "Python is a programming language.",
            "model_c": "The weather today is sunny.",
        },
        "partial_agreement": {
            "model_a": "Use a decision tree classifier for this dataset.",
            "model_b": "Random forest would be the best choice here.",
            "model_c": "Decision tree or random forest both work well.",
        },
    }


@pytest.fixture
def mission_constraints():
    """Sample mission constraints for testing."""
    from deepthinker.missions.mission_types import MissionConstraints
    return MissionConstraints(
        time_budget_minutes=30,
        max_iterations=10,
        allow_internet=True,
        allow_code_execution=True,
        notes="Test mission"
    )


@pytest.fixture
def mock_mission_store(temp_dir):
    """Create a mock mission store with a temp directory."""
    from deepthinker.missions.mission_store import MissionStore
    
    store = MissionStore(store_dir=str(temp_dir))
    return store


# FastAPI test client fixtures
@pytest.fixture
def test_app():
    """Create a test FastAPI application instance."""
    from api.server import app
    return app


@pytest.fixture
async def async_client(test_app):
    """Create an async HTTP client for testing FastAPI endpoints."""
    from httpx import AsyncClient, ASGITransport
    
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

