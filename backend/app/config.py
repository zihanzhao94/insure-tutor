"""Environment settings shared by ingestion, retrieval, and the API."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_api_key: str
    model_base_url: str
    model_provider: str
    chat_model: str
    embedding_backend: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    auto_ingest: bool
    chat_mode: str


def get_settings() -> Settings:
    """Load and validate settings; existing environment variables take precedence over .env."""
    load_dotenv(ROOT / ".env", override=False)
    provider = os.getenv("MODEL_PROVIDER", "openai")
    if provider not in {"anthropic", "openai"}:
        raise ValueError("MODEL_PROVIDER must be anthropic or openai.")
    backend = os.getenv("EMBEDDING_BACKEND") or ("openai" if provider == "openai" else "local")
    if backend not in {"local", "openai"}:
        raise ValueError("EMBEDDING_BACKEND must be local or openai.")
    size = int(os.getenv("CHUNK_SIZE", "1000"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k = int(os.getenv("TOP_K", "5"))
    if size < 100 or not 0 <= overlap < size or not 1 <= top_k <= 12:
        raise ValueError("Invalid CHUNK_SIZE, CHUNK_OVERLAP, or TOP_K settings.")
    mode = os.getenv("CHAT_MODE", "llm")
    if mode not in {"llm", "extractive"}:
        raise ValueError("CHAT_MODE must be llm or extractive.")
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR") or ROOT / "data"),
        model_api_key=os.getenv("MODEL_API_KEY") or os.getenv(
            "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY", ""),
        model_base_url=(os.getenv("MODEL_BASE_URL") or (
            "https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com/v1")).rstrip("/"),
        model_provider=provider,
        chat_model=os.getenv("CHAT_MODEL") or (
            "claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4.1-mini"),
        embedding_backend=backend,
        embedding_model=os.getenv("EMBEDDING_MODEL") or (
            "text-embedding-3-small" if backend == "openai" else
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        chunk_size=size, chunk_overlap=overlap, top_k=top_k,
        auto_ingest=os.getenv("AUTO_INGEST", "true").lower() == "true", chat_mode=mode,
    )
