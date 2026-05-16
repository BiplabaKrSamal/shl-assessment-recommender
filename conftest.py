"""
Shared pytest configuration and fixtures.
Session-scoped fixtures ensure expensive operations (model load, FAISS build)
happen once per test run, not once per test function.
"""
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Suppress noisy logging during tests
# ---------------------------------------------------------------------------
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Session-scoped retriever (FAISS index built once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def catalog_path():
    return str(Path(__file__).parent / "data" / "catalog.json")


@pytest.fixture(scope="session")
def shared_retriever(catalog_path):
    """
    Single retriever instance for the entire test session.
    FAISS index and sentence-transformer model loaded once.
    """
    from app.retriever import CatalogRetriever
    r = CatalogRetriever(catalog_path=catalog_path)
    # Warm up FAISS
    r.search("test", k=1)
    return r


# ---------------------------------------------------------------------------
# Option to skip live LLM tests
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that call the live LLM API (requires GEMINI_API_KEY)"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires live LLM API")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="Add --live flag to run live LLM tests")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
