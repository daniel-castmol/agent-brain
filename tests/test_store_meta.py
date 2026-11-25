"""Embedding-config guard tests — meta table, mismatch errors, rebuild."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from brain.store import EmbeddingConfigMismatchError, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(db_path=tmp_path / "test.duckdb", embedding_dim=384)
    s.connect()
    yield s
    s.close()


def _insert_note(store: Store, path: str) -> None:
    store.connect().execute(
        "INSERT INTO notes (path, title, frontmatter, tags, content_hash, modified_at, indexed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [path, path, json.dumps({}), [], "x", datetime.now(), datetime.now()],
    )


def test_verify_stamps_config_when_absent(store: Store):
    # Pre-v3 databases have no embedding_config row: first verify stamps it.
    store.verify_embedding_config("m1", 384, "", "")
    assert store.get_embedding_config() == {
        "model": "m1",
        "dim": 384,
        "query_prefix": "",
        "passage_prefix": "",
    }


def test_verify_noop_when_config_matches(store: Store):
    store.verify_embedding_config("m1", 384, "q: ", "p: ")
    store.verify_embedding_config("m1", 384, "q: ", "p: ")  # must not raise


def test_verify_raises_on_model_change_same_dim(store: Store):
    store.verify_embedding_config("BAAI/bge-small-en-v1.5", 384, "", "")
    with pytest.raises(EmbeddingConfigMismatchError, match="--rebuild"):
        store.verify_embedding_config(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384, "", ""
        )


def test_verify_raises_on_prefix_change(store: Store):
    store.verify_embedding_config("m1", 384, "", "")
    with pytest.raises(EmbeddingConfigMismatchError, match="query_prefix"):
        store.verify_embedding_config("m1", 384, "query: ", "")


def test_rebuild_schema_drops_everything_and_recreates(store: Store):
    _insert_note(store, "a.md")
    store.verify_embedding_config("m1", 384, "", "")
    store.rebuild_schema()
    conn = store.connect()
    assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
    assert store.get_embedding_config() is None
    store.verify_embedding_config("m2", 384, "", "")  # fresh stamp, no raise
