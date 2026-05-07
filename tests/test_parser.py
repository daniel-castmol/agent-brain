"""Parser tests — pure functions, no DB or model needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.parser import _extract_inline_tags, _extract_wikilinks, _split_sections, parse_file


def test_extract_inline_tags():
    text = "This is #foo and #bar/baz, but not #1 or # heading."
    tags = _extract_inline_tags(text)
    assert "foo" in tags
    assert "bar/baz" in tags
    assert "1" not in tags  # numeric-only rejected


def test_extract_wikilinks():
    text = "See [[Note A]] and [[Note B|alias]] and a duplicate [[Note A]]."
    links = _extract_wikilinks(text)
    assert len(links) == 2
    targets = {l.target for l in links}
    assert targets == {"Note A", "Note B"}
    aliases = {l.target: l.alias for l in links}
    assert aliases["Note B"] == "alias"
    assert aliases["Note A"] is None


def test_split_sections_simple():
    body = (
        "Intro paragraph.\n\n"
        "## First\n"
        "First body.\n\n"
        "### Sub\n"
        "Sub body.\n\n"
        "## Second\n"
        "Second body."
    )
    sections = _split_sections(body)
    paths = [s.heading_path for s in sections]
    assert paths == ["(intro)", "First", "First > Sub", "Second"]
    assert sections[0].text.strip() == "Intro paragraph."
    assert sections[1].text.strip() == "First body."
    assert sections[2].text.strip() == "Sub body."
    assert sections[3].text.strip() == "Second body."


def test_split_sections_no_intro():
    body = "## Only\nbody."
    sections = _split_sections(body)
    assert [s.heading_path for s in sections] == ["Only"]


def test_parse_file(tmp_path: Path):
    note_path = tmp_path / "ideas" / "agent-memory.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(
        "---\n"
        "tags: [project/brain, idea]\n"
        "---\n\n"
        "# Agent Memory\n\n"
        "Linked to [[Other Note]] and #experiment.\n\n"
        "## Storage\n"
        "DuckDB.\n",
        encoding="utf-8",
    )
    note = parse_file(note_path, tmp_path)

    assert note.path == "ideas/agent-memory.md"
    assert note.title == "Agent Memory"
    assert "project/brain" in note.tags
    assert "experiment" in note.tags
    assert any(wl.target == "Other Note" for wl in note.wikilinks)
    paths = [s.heading_path for s in note.sections]
    assert "Agent Memory" in paths
    assert "Agent Memory > Storage" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
