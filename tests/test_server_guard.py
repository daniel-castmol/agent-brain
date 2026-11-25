"""The MCP server's embedding-config guard must hold on every call,
not just the first — a retry after the first error must not hand back
an unverified store."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.store import EmbeddingConfigMismatchError, Store


def test_server_guard_blocks_every_call_on_mismatch(tmp_path: Path, monkeypatch):
    from brain import server
    from brain.config import settings

    # A db stamped with a different (same-dim) model.
    s = Store(db_path=tmp_path / "db.duckdb", embedding_dim=384)
    s.connect()
    s.verify_embedding_config("other/model", 384, "", "")
    s.close()

    monkeypatch.setattr(settings, "db_dir", tmp_path)
    monkeypatch.setattr(settings, "db_filename", "db.duckdb")
    monkeypatch.setattr(server, "_store", None)

    with pytest.raises(EmbeddingConfigMismatchError):
        server._get_store()
    # The retry must raise again — not silently return a mixed-space store.
    with pytest.raises(EmbeddingConfigMismatchError):
        server._get_store()
    assert server._store is None
