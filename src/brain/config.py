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

    # Asymmetric embedding prefixes. Some models (e5 family) need
    # "query: " / "passage: " prepended at embed time. Applied only at
    # embedding, never stored in chunk content or hashes. Changing these
    # invalidates stored vectors → `brain index --rebuild`.
    query_prefix: str = Field(default="")
    passage_prefix: str = Field(default="")

    # FTS stemmer for the DuckDB full-text index. 'porter' (English) is
    # the DuckDB default; 'spanish' or 'none' suit mixed-language vaults.
    fts_stemmer: str = Field(default="porter")

    # Chunking targets, measured roughly in characters since we don't tokenize
    # ahead of embedding (bge-small handles up to 512 tokens ≈ ~2000 chars).
    chunk_target_chars: int = Field(default=1500)
    chunk_overlap_chars: int = Field(default=150)

    # Hybrid search: weight on vector vs FTS results in RRF fusion.
    rrf_k: int = Field(default=60)

    # Cross-encoder reranking. Off by default — the model is ~120MB and adds
    # ~100ms per query. Worth turning on once the vault has enough overlap
    # that hybrid_search alone is noisy. Set BRAIN_RERANK_ENABLED=true to opt in.
    rerank_enabled: bool = Field(default=False)
    rerank_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")
    # When reranking, fetch this many candidates from hybrid_search before
    # rescoring down to the requested k. Bigger pool → better recall, slower.
    rerank_candidate_pool: int = Field(default=25)

    @property
    def db_path(self) -> Path:
        return self.db_dir / self.db_filename

    def ensure_dirs(self) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
