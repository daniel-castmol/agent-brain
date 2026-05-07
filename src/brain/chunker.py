"""Chunking.

Sections from parser.py become one or more chunks. If a section is small,
it's a single chunk. If it's larger than `target_chars`, it's split on
paragraph boundaries (blank lines), with optional overlap.

Each chunk carries its heading path so the embedded text reads like:

    ## Architecture > Storage
    DuckDB stores notes in two tables...

This gives the embedding model context that improves retrieval of
deeply-nested ideas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from brain.parser import Note, Section


@dataclass
class Chunk:
    note_path: str
    chunk_index: int
    heading_path: str
    content: str  # what gets embedded — heading_path + body
    body: str  # the raw section body without the heading prefix
    content_hash: str
    char_count: int


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_paragraphs(text: str, target: int, overlap: int) -> list[str]:
    """Greedy paragraph-boundary split with size cap and optional overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if not current:
            current = p
            continue
        if len(current) + len(p) + 2 <= target:
            current = f"{current}\n\n{p}"
        else:
            chunks.append(current)
            # Overlap: tail of the previous chunk becomes head of the next.
            if overlap > 0 and len(current) > overlap:
                tail = current[-overlap:]
                current = f"{tail}\n\n{p}"
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


def chunk_section(
    section: Section,
    note_path: str,
    start_index: int,
    target_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    body = section.text
    if len(body) <= target_chars:
        pieces = [body] if body.strip() else []
    else:
        pieces = _split_paragraphs(body, target_chars, overlap_chars)

    out: list[Chunk] = []
    for i, body_text in enumerate(pieces):
        # Prepend the heading path so the embedding sees its breadcrumb.
        if section.heading_path and section.heading_path != "(intro)":
            content = f"# {section.heading_path}\n\n{body_text}"
        else:
            content = body_text
        out.append(
            Chunk(
                note_path=note_path,
                chunk_index=start_index + i,
                heading_path=section.heading_path,
                content=content,
                body=body_text,
                content_hash=_hash(content),
                char_count=len(content),
            )
        )
    return out


def chunk_note(note: Note, target_chars: int, overlap_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in note.sections:
        section_chunks = chunk_section(
            section,
            note_path=note.path,
            start_index=len(chunks),
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        chunks.extend(section_chunks)
    return chunks
