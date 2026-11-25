"""FTS dirty-flag tests — real DuckDB, no embedder.

The FTS index must be rebuilt only when chunk data changed (dirty flag),
not on every query.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from brain.chunker import Chunk
from brain.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(db_path=tmp_path / "test.duckdb", embedding_dim=384)
    s.connect()
    yield s
    s.close()


def _insert_note(store: Store, path: str, body: str = "quantum espresso") -> None:
    conn = store.connect()
    conn.execute(
        "INSERT INTO notes (path, title, frontmatter, tags, content_hash, modified_at, indexed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [path, path, json.dumps({}), [], "x", datetime.now(), datetime.now()],
    )
    conn.execute(
        "INSERT INTO chunks (id, note_path, chunk_index, heading_path, content, body,"
        " content_hash, char_count, embedding)"
        " VALUES (nextval('chunk_id_seq'), ?, 0, 'H', ?, ?, ?, ?, ?::FLOAT[])",
        [path, body, body, f"h-{path}", len(body), [0.0] * 384],
    )


def _chunk(path: str, body: str) -> Chunk:
    return Chunk(
        note_path=path,
        chunk_index=0,
        heading_path="H",
        content=body,
        body=body,
        content_hash=f"h-{body}",
        char_count=len(body),
    )


def test_fts_search_clears_dirty_flag_and_finds_content(store: Store):
    _insert_note(store, "a.md", body="quantum espresso")
    hits = store.fts_search("quantum", k=5)
    assert hits and hits[0].note_path == "a.md"
    assert store._fts_dirty is False
    # Second call runs against the existing index (flag still clean).
    assert store.fts_search("quantum", k=5)
    assert store._fts_dirty is False


def test_write_marks_dirty_and_new_content_searchable(store: Store):
    _insert_note(store, "a.md", body="quantum espresso")
    store.fts_search("quantum", k=5)
    assert store._fts_dirty is False

    store.replace_chunks("a.md", [_chunk("a.md", "zanzibar memory")], [[0.0] * 384])
    assert store._fts_dirty is True
    hits = store.fts_search("zanzibar", k=5)
    assert hits and hits[0].note_path == "a.md"


def test_delete_marks_dirty(store: Store):
    _insert_note(store, "a.md")
    store.fts_search("quantum", k=5)
    store.delete_note("a.md")
    assert store._fts_dirty is True


def test_fts_search_on_empty_store_returns_empty(store: Store):
    assert store.fts_search("anything", k=5) == []
