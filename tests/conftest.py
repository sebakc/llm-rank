"""Pytest fixtures: load sample HF/OpenRouter data and isolate cache dir."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("LLM_RANK_CACHE_DIR", str(cache_dir))
    # cache module reads env at import; reassign module attribute too
    from llm_rank import cache
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    yield cache_dir


@pytest.fixture
def hf_rows():
    return json.loads((FIXTURES / "hf_sample.json").read_text())


@pytest.fixture
def or_rows():
    return json.loads((FIXTURES / "openrouter_sample.json").read_text())
