"""Writer tests — pure functions, no DB or model needed.

Covers the v2 write-back primitives:
- Path validation (rejects traversal, absolute, non-.md)
- Frontmatter stamping with agent provenance
- Heading insertion for inline appends
- The compose orchestrator used by the MCP tools
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from brain.writer import (
    InvalidPathError,
    build_note,
    compose_appended_body,
    format_agent_append,
    insert_under_heading,
    validate_vault_path,
)


# ----------------------------------------------------------------------
# Path validation — system boundary, must reject untrusted input.
# ----------------------------------------------------------------------


def test_validate_path_rejects_traversal(tmp_path: Path):
    with pytest.raises(InvalidPathError):
        validate_vault_path("../escape.md", tmp_path)


def test_validate_path_rejects_absolute(tmp_path: Path):
    with pytest.raises(InvalidPathError):
        validate_vault_path("/etc/passwd.md", tmp_path)


def test_validate_path_rejects_non_md(tmp_path: Path):
    with pytest.raises(InvalidPathError):
        validate_vault_path("research/notes.txt", tmp_path)


def test_validate_path_accepts_nested(tmp_path: Path):
    note = validate_vault_path("research/llm/ranking.md", tmp_path)
    assert note.rel == "research/llm/ranking.md"
    assert note.abs == tmp_path / "research" / "llm" / "ranking.md"


def test_validate_path_normalizes_dot_segments(tmp_path: Path):
    # `./foo/./bar.md` should normalize, but never escape the vault.
    note = validate_vault_path("./foo/./bar.md", tmp_path)
    assert note.rel == "foo/bar.md"


# ----------------------------------------------------------------------
# build_note — stamps agent provenance into frontmatter.
# ----------------------------------------------------------------------


def test_build_note_stamps_author_and_timestamp():
    now = datetime(2026, 5, 6, 21, 30, 0)
    text = build_note("# Hello\n\nBody.", frontmatter=None, tags=None, now=now)
    assert "author: agent" in text
    assert "created_at: '2026-05-06T21:30:00'" in text or "created_at: 2026-05-06T21:30:00" in text
    assert "- agent" in text  # tag list contains agent
    # Body preserved verbatim under the frontmatter block.
    assert text.endswith("# Hello\n\nBody.")


def test_build_note_preserves_user_frontmatter():
    now = datetime(2026, 5, 6, 21, 30, 0)
    text = build_note(
        "Body.",
        frontmatter={"title": "LLM ranking", "source": "deep-research"},
        tags=["research/llm"],
        now=now,
    )
    assert "title: LLM ranking" in text
    assert "source: deep-research" in text
    assert "author: agent" in text


def test_build_note_dedupes_agent_tag():
    now = datetime(2026, 5, 6, 21, 30, 0)
    # User explicitly passes 'agent' — should not appear twice.
    text = build_note("Body.", tags=["agent", "research/llm"], now=now)
    # Count occurrences in the YAML tag list, not in the body.
    fm_section = text.split("---", 2)[1]
    assert fm_section.count("- agent\n") == 1
    assert "- research/llm" in fm_section


# ----------------------------------------------------------------------
# insert_under_heading — append under a heading, creating it if missing.
# ----------------------------------------------------------------------


def test_insert_under_heading_creates_when_missing():
    existing = "# Title\n\nMy content.\n"
    result = insert_under_heading(existing, "## Agent notes", "### 2026-05-06 10:00\n\nFirst entry.\n")
    assert "## Agent notes" in result
    # Original content preserved.
    assert "My content." in result
    # New section comes after original content.
    assert result.index("My content.") < result.index("## Agent notes")
    assert "First entry." in result


def test_insert_under_heading_appends_when_present():
    existing = (
        "# Title\n\n"
        "My content.\n\n"
        "## Agent notes\n\n"
        "### 2026-05-06 10:00\n\n"
        "Old entry.\n"
    )
    result = insert_under_heading(existing, "## Agent notes", "### 2026-05-06 11:00\n\nNew entry.\n")
    # Both entries present, new one after old one.
    assert "Old entry." in result
    assert "New entry." in result
    assert result.index("Old entry.") < result.index("New entry.")
    # Heading wasn't duplicated.
    assert result.count("## Agent notes") == 1


def test_insert_under_heading_respects_section_boundary():
    """New content goes inside the target section, not into a later section."""
    existing = (
        "# Title\n\n"
        "## Agent notes\n\n"
        "Old agent entry.\n\n"
        "## Other section\n\n"
        "Unrelated content.\n"
    )
    result = insert_under_heading(existing, "## Agent notes", "New agent entry.\n")
    # New entry must appear before "Other section" heading.
    assert result.index("New agent entry.") < result.index("## Other section")
    # Unrelated content is untouched and still at the end.
    assert "Unrelated content." in result


# ----------------------------------------------------------------------
# format_agent_append — wraps content in a timestamped subheading.
# ----------------------------------------------------------------------


def test_format_agent_append_adds_timestamp_subheading():
    now = datetime(2026, 5, 6, 21, 30, 0)
    out = format_agent_append("Some agent content.", now)
    assert out.startswith("### 2026-05-06 21:30")
    assert "Some agent content." in out
    assert out.endswith("\n")


# ----------------------------------------------------------------------
# compose_appended_body — orchestrator used by append_to_note MCP tool.
# ----------------------------------------------------------------------


def test_compose_append_uses_default_heading_for_user_notes():
    existing = "# My spec\n\nMy hand-written content.\n"
    now = datetime(2026, 5, 6, 21, 30, 0)
    result = compose_appended_body(existing, "Agent thought.", heading=None, now=now)
    assert "## Agent notes" in result
    assert "### 2026-05-06 21:30" in result
    assert "Agent thought." in result


def test_compose_append_with_explicit_heading():
    existing = "# Doc\n\n## Decisions\n\nOriginal decision.\n"
    now = datetime(2026, 5, 6, 21, 30, 0)
    result = compose_appended_body(existing, "Agent decision.", heading="## Decisions", now=now)
    assert "Original decision." in result
    assert "Agent decision." in result
    # Both should sit under the same Decisions heading (not a new one).
    assert result.count("## Decisions") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
