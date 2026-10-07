import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    # Prevent real API calls and keep tests isolated from personal .env settings.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_MODEL", "test-chat")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("AUTO_INGEST", "false")
    monkeypatch.setenv("CHAT_MODE", "llm")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    monkeypatch.setenv("TOP_K", "5")
    def deny_network(*args, **kwargs):
        raise AssertionError("Unit tests must not call a live model API")
    monkeypatch.setattr("httpx.post", deny_network)
