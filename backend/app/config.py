"""Server configuration. Model credentials must never be sent to the browser."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_api_key: str
    model_base_url: str
    chat_model: str
    embedding_model: str


def get_settings() -> Settings:
    default_data_dir = Path(__file__).resolve().parents[2] / "data"
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", str(default_data_dir))),
        model_api_key=os.getenv("MODEL_API_KEY", ""),
        model_base_url=os.getenv("MODEL_BASE_URL", ""),
        chat_model=os.getenv("CHAT_MODEL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
    )
