"""Vault health report — pure SQL against the existing DuckDB.

Three sections:

- **vault**: counts (notes, chunks, links), total chars, oldest/newest
  note, vault age in days
- **authorship**: agent-authored vs user-authored vs mixed (user notes
  with an agent-appended `## Agent notes` section)
- **activity**: how many notes were modified in the last 24h / 7d / 30d,
  plus the 5 most-recently-modified

Cheap (~10-50ms total). No embedder, no parser. Used by the
`vault_report()` MCP tool.
"""

from __future__ import annotations

from typing import Any

from brain.store import Store


def _vault_section(store: Store) -> dict[str, Any]:
    conn = store.connect()
    notes = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
    total_chars = conn.execute(
        "SELECT coalesce(sum(char_count), 0) FROM chunks"
    ).fetchone()[0]

    oldest_row = conn.execute(
        "SELECT path, title, modified_at FROM notes ORDER BY modified_at ASC LIMIT 1"
    ).fetchone()
    newest_row = conn.execute(
        "SELECT path, title, modified_at FROM notes ORDER BY modified_at DESC LIMIT 1"
    ).fetchone()

    age_days = 0
    if oldest_row and newest_row:
        delta = newest_row[2] - oldest_row[2]
        age_days = delta.days

    return {
        "notes": int(notes),
        "chunks": int(chunks),
        "links": int(links),
        "total_chars": int(total_chars),
        "oldest_note": _note_dict(oldest_row),
        "newest_note": _note_dict(newest_row),
        "age_days": age_days,
    }


def _authorship_section(store: Store) -> dict[str, int]:
    conn = store.connect()
    agent = conn.execute(
        "SELECT count(*) FROM notes WHERE list_contains(tags, 'agent')"
    ).fetchone()[0]
    user = conn.execute(
        "SELECT count(*) FROM notes WHERE NOT list_contains(tags, 'agent')"
    ).fetchone()[0]
    # A user-authored note is "mixed" if any of its chunks live under
    # the `## Agent notes` heading (the marker the writer adds when the
    # agent appends to a user-authored note).
    mixed = conn.execute(
        """
        SELECT count(DISTINCT n.path)
        FROM notes n
        JOIN chunks c ON c.note_path = n.path
        WHERE NOT list_contains(n.tags, 'agent')
          AND c.heading_path LIKE '%Agent notes%'
        """
    ).fetchone()[0]
    return {
        "agent_authored": int(agent),
        "user_authored": int(user),
        "mixed_authored": int(mixed),
    }


def _activity_section(store: Store) -> dict[str, Any]:
    conn = store.connect()
    bins = conn.execute(
        """
        SELECT
            count(*) FILTER (WHERE modified_at > now() - INTERVAL '1 day')   AS d1,
            count(*) FILTER (WHERE modified_at > now() - INTERVAL '7 days')  AS d7,
            count(*) FILTER (WHERE modified_at > now() - INTERVAL '30 days') AS d30
        FROM notes
        """
    ).fetchone()
    recent_rows = conn.execute(
        """
        SELECT path, title, modified_at
        FROM notes
        ORDER BY modified_at DESC
        LIMIT 5
        """
    ).fetchall()
    return {
        "modified_24h": int(bins[0]),
        "modified_7d": int(bins[1]),
        "modified_30d": int(bins[2]),
        "recent_notes": [_note_dict(r) for r in recent_rows],
    }


def _note_dict(row: tuple | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "path": row[0],
        "title": row[1],
        "modified_at": row[2].isoformat() if row[2] else None,
    }


def build_report(store: Store) -> dict[str, Any]:
    """Build the full vault report dict."""
    return {
        "vault": _vault_section(store),
        "authorship": _authorship_section(store),
        "activity": _activity_section(store),
    }
