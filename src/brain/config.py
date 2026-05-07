"""Configuration for brain.

Settings are loaded from environment variables (BRAIN_*) or a .env file.
Defaults assume a fresh agent-only vault under ~/brain-vault and a private
DuckDB database under ~/.brain/db.duckdb.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Where the agent's markdown vault lives. v1 is a fresh vault, separate
    # from any personal Obsidian vault.
    vault_dir: Path = Field(default=Path.home() / "brain-vault")

    # DuckDB file. Disposable — fully rebuildable from the vault.
    db_dir: Path = Field(default=Path.home() / ".brain")
    db_filename: str = Field(default="db.duckdb")

    # Embedding model. bge-small-en-v1.5 → 384-dim, ~30MB ONNX, fast on CPU.
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384)

    # Chunking targets, measured roughly in characters since we don't tokenize
    # ahead of embedding (bge-small handles up to 512 tokens ≈ ~2000 chars).
    chunk_target_chars: int = Field(default=1500)
    chunk_overlap_chars: int = Field(default=150)

    # Hybrid search: weight on vector vs FTS results in RRF fusion.
    rrf_k: int = Field(default=60)

    @property
    def db_path(self) -> Path:
        return self.db_dir / self.db_filename

    def ensure_dirs(self) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
