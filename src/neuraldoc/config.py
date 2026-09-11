"""Central application configuration (Pydantic Settings, v2 style)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALDOC_", env_file=".env", extra="ignore")

    app_name: str = "NeuralDoc"
    env: str = "development"
    database_url: str = "sqlite:///./data/neuraldoc.db"
    index_dir: Path = Path("./data/indices")
    upload_dir: Path = Path("./data/uploads")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 native dim; kept explicit for index init

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    anomaly_contamination: float = 0.08

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
