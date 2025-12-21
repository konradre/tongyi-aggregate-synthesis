"""Pytest configuration and fixtures."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load test environment
test_env = Path(__file__).parent / ".env"
if test_env.exists():
    load_dotenv(test_env)


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (require external services)")
    config.addinivalue_line("markers", "slow: Slow tests (LLM synthesis)")


@pytest.fixture
def sample_query():
    """Sample search query for tests."""
    return "python async programming tutorial"


@pytest.fixture
def sample_sources():
    """Sample source data for testing."""
    from src.connectors.base import Source
    return [
        Source(
            id="sx_test001",
            title="Python Async IO Guide",
            url="https://example.com/async-guide",
            content="Learn how to use async/await in Python for concurrent programming.",
            score=0.95,
            connector="searxng",
        ),
        Source(
            id="tv_test002",
            title="Async Programming Best Practices",
            url="https://example.com/async-best-practices",
            content="Best practices for writing async code in Python applications.",
            score=0.88,
            connector="tavily",
        ),
        Source(
            id="lu_test003",
            title="Python Concurrency Deep Dive",
            url="https://example.com/concurrency",
            content="Deep dive into Python's concurrency model and asyncio library.",
            score=0.82,
            connector="linkup",
        ),
    ]


@pytest.fixture
def searxng_configured():
    """Check if SearXNG is configured."""
    host = os.getenv("RESEARCH_SEARXNG_HOST", "")
    return bool(host)


@pytest.fixture
def tavily_configured():
    """Check if Tavily is configured."""
    key = os.getenv("RESEARCH_TAVILY_API_KEY", "")
    return bool(key)


@pytest.fixture
def linkup_configured():
    """Check if LinkUp is configured."""
    key = os.getenv("RESEARCH_LINKUP_API_KEY", "")
    return bool(key)


@pytest.fixture
def llm_configured():
    """Check if LLM is configured."""
    base = os.getenv("RESEARCH_LLM_API_BASE", "")
    return bool(base)
