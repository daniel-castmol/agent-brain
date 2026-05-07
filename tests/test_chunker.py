"""Chunker tests."""

from __future__ import annotations

from brain.chunker import _split_paragraphs, chunk_section
from brain.parser import Section


def test_split_paragraphs_under_target():
    text = "Para one.\n\nPara two."
    chunks = _split_paragraphs(text, target=1000, overlap=0)
    assert chunks == ["Para one.\n\nPara two."]


def test_split_paragraphs_over_target():
    text = "\n\n".join(["A" * 400, "B" * 400, "C" * 400])
    chunks = _split_paragraphs(text, target=500, overlap=0)
    assert len(chunks) == 3


def test_chunk_section_prepends_heading_path():
    section = Section(heading_path="Architecture > Storage", level=3, text="DuckDB it is.")
    chunks = chunk_section(
        section, note_path="x.md", start_index=0, target_chars=1500, overlap_chars=0
    )
    assert len(chunks) == 1
    assert chunks[0].content.startswith("# Architecture > Storage")
    assert "DuckDB it is." in chunks[0].content


def test_chunk_section_intro_no_heading_prefix():
    section = Section(heading_path="(intro)", level=0, text="just an intro.")
    chunks = chunk_section(
        section, note_path="x.md", start_index=0, target_chars=1500, overlap_chars=0
    )
    assert chunks[0].content == "just an intro."
