"""Report tests — pure-function SQL against an in-memory DuckDB.

No embedder, no parser, no MCP. Each test seeds a tiny Store with
direct SQL inserts so we can isolate the report's query logic from
the rest of the indexing pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from brain.report import build_report
from brain.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(db_path=tmp_path / "test.duckdb", embedding_dim=384)
    s.connect()
    yield s
    s.close()


def _insert_note(
    store: Store,
    *,
    path: str,
    title: str = "test",
    tags: list[str] | None = None,
    modified_at: datetime | None = None,
    chunks_with_headings: list[str] | None = None,
) -> None:
    """Direct SQL insert so we don't depend on the embedder."""
    conn = store.connect()
    modified_at = modified_at or datetime.now()
    tags = tags or []
    conn.execute(
        """
        INSERT INTO notes (path, title, frontmatter, tags, content_hash, modified_at, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [path, title, json.dumps({}), tags, "deadbeef", modified_at, datetime.now()],
    )
    for i, heading in enumerate(chunks_with_headings or []):
        conn.execute(
            """
            INSERT INTO chunks (id, note_path, chunk_index, heading_path,
                                content, body, content_hash, char_count, embedding)
            VALUES (nextval('chunk_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[])
            """,
            [path, i, heading, "content", "body", f"hash{i}", 100, [0.0] * 384],
        )


# ----------------------------------------------------------------------
# Vault summary
# ----------------------------------------------------------------------


def test_empty_vault_returns_zeros(store: Store):
    r = build_report(store)
    assert r["vault"]["notes"] == 0
    assert r["vault"]["chunks"] == 0
    assert r["vault"]["links"] == 0
    assert r["vault"]["total_chars"] == 0
    assert r["vault"]["oldest_note"] is None
    assert r["vault"]["newest_note"] is None


def test_vault_counts_match_inserts(store: Store):
    _insert_note(store, path="a.md", chunks_with_headings=["H1", "H2"])
    _insert_note(store, path="b.md", chunks_with_headings=["H1"])
    r = build_report(store)
    assert r["vault"]["notes"] == 2
    assert r["vault"]["chunks"] == 3
    assert r["vault"]["total_chars"] == 300  # 3 chunks * 100 char_count


def test_vault_oldest_and_newest_resolve(store: Store):
    older = datetime(2026, 1, 1, 12, 0, 0)
    newer = datetime(2026, 5, 1, 12, 0, 0)
    _insert_note(store, path="old.md", title="Old one", modified_at=older)
    _insert_note(store, path="new.md", title="New one", modified_at=newer)
    r = build_report(store)
    assert r["vault"]["oldest_note"]["path"] == "old.md"
    assert r["vault"]["newest_note"]["path"] == "new.md"
    assert r["vault"]["age_days"] >= 119  # Jan 1 → May 1 ≈ 120


# ----------------------------------------------------------------------
# Authorship
# ----------------------------------------------------------------------


def test_authorship_splits_agent_from_user(store: Store):
    _insert_note(store, path="agent1.md", tags=["agent", "research"])
    _insert_note(store, path="agent2.md", tags=["agent"])
    _insert_note(store, path="user1.md", tags=["personal"])
    r = build_report(store)
    assert r["authorship"]["agent_authored"] == 2
    assert r["authorship"]["user_authored"] == 1
    assert r["authorship"]["mixed_authored"] == 0


def test_mixed_authorship_detected_via_agent_notes_heading(store: Store):
    """A user-authored note that has a chunk under '## Agent notes' is
    a mixed-authorship note — the agent appended to it."""
    _insert_note(
        store,
        path="user-with-agent-section.md",
        tags=["personal"],
        chunks_with_headings=["My intro", "Agent notes > 2026-05-06 21:30"],
    )
    _insert_note(store, path="pure-user.md", tags=["personal"], chunks_with_headings=["Just my stuff"])
    r = build_report(store)
    assert r["authorship"]["mixed_authored"] == 1


# ----------------------------------------------------------------------
# Activity
# ----------------------------------------------------------------------


def test_activity_bins_count_correctly(store: Store):
    now = datetime.now()
    _insert_note(store, path="today.md", modified_at=now - timedelta(hours=2))
    _insert_note(store, path="yesterday.md", modified_at=now - timedelta(days=1, hours=1))
    _insert_note(store, path="last-week.md", modified_at=now - timedelta(days=5))
    _insert_note(store, path="last-month.md", modified_at=now - timedelta(days=20))
    _insert_note(store, path="ancient.md", modified_at=now - timedelta(days=100))
    r = build_report(store)
    a = r["activity"]
    assert a["modified_24h"] == 1  # only today
    assert a["modified_7d"] == 3  # today + yesterday + last-week
    assert a["modified_30d"] == 4  # all except ancient


def test_recent_notes_returns_top_5_in_order(store: Store):
    now = datetime.now()
    for i in range(7):
        _insert_note(
            store,
            path=f"note-{i}.md",
            title=f"Note {i}",
            modified_at=now - timedelta(hours=i),
        )
    r = build_report(store)
    recent = r["activity"]["recent_notes"]
    assert len(recent) == 5
    # Most-recent first → note-0, note-1, note-2, note-3, note-4
    assert [n["path"] for n in recent] == [f"note-{i}.md" for i in range(5)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
