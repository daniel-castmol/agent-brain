"""Markdown parsing.

Each .md file becomes a `Note` with frontmatter, tags, wikilinks, and
section-aware structure. Chunking happens downstream in chunker.py.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter

# [[target]] or [[target|alias]]
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
# Inline #tag — must be word-boundary, not # used as heading marker.
# We require the # to be preceded by whitespace or start-of-line, and the tag
# itself must contain at least one letter so we don't match e.g. "#1234".
INLINE_TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]*)")
# ATX heading: 1–6 hashes, then space, then text.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Section:
    """A logical section of a note, defined by its heading path."""

    heading_path: str  # e.g. "Architecture > Storage"
    level: int  # 0 for content before any heading; otherwise heading depth
    text: str  # raw markdown content of the section, headings excluded


@dataclass
class WikiLink:
    target: str  # raw target text (may or may not resolve to a real file)
    alias: str | None = None


@dataclass
class Note:
    """A parsed markdown file."""

    path: str  # vault-relative path, e.g. "ideas/agent-memory.md"
    abs_path: Path
    title: str
    frontmatter: dict[str, Any]
    tags: list[str]  # union of frontmatter tags + inline #tags
    wikilinks: list[WikiLink]
    sections: list[Section]
    content_hash: str  # sha256 of raw file bytes
    raw_content: str  # markdown body (frontmatter stripped)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_inline_tags(text: str) -> list[str]:
    return sorted({m.group(1) for m in INLINE_TAG_RE.finditer(text)})


def _extract_wikilinks(text: str) -> list[WikiLink]:
    seen: dict[str, WikiLink] = {}
    for m in WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        alias = m.group(2).strip() if m.group(2) else None
        # First occurrence wins for alias resolution.
        if target not in seen:
            seen[target] = WikiLink(target=target, alias=alias)
    return list(seen.values())


def _split_sections(body: str) -> list[Section]:
    """Walk the body line by line, tracking the heading stack."""
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current_lines: list[str] = []

    def flush(level: int) -> None:
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines.clear()
            return
        path = " > ".join(t for _, t in heading_stack) or "(intro)"
        sections.append(Section(heading_path=path, level=level, text=text))
        current_lines.clear()

    current_level = 0
    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m:
            # Flush whatever we accumulated under the previous heading path.
            flush(current_level)
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            # Pop any deeper-or-equal headings off the stack.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_level = level
        else:
            current_lines.append(line)
    flush(current_level)
    return sections


def parse_file(abs_path: Path, vault_root: Path) -> Note:
    """Parse a single markdown file into a Note."""
    raw_bytes = abs_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    fm = frontmatter.loads(raw_text)
    body = fm.content
    fm_meta = dict(fm.metadata)

    # Title: H1 if present, else frontmatter `title`, else filename stem.
    title = ""
    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            title = m.group(2).strip()
            break
    if not title:
        title = str(fm_meta.get("title") or abs_path.stem)

    # Tags: frontmatter list + inline #tags.
    fm_tags = fm_meta.get("tags") or []
    if isinstance(fm_tags, str):
        fm_tags = [fm_tags]
    inline_tags = _extract_inline_tags(body)
    tags = sorted({*fm_tags, *inline_tags})

    sections = _split_sections(body)
    wikilinks = _extract_wikilinks(body)
    rel_path = str(abs_path.relative_to(vault_root)).replace("\\", "/")

    return Note(
        path=rel_path,
        abs_path=abs_path,
        title=title,
        frontmatter=fm_meta,
        tags=tags,
        wikilinks=wikilinks,
        sections=sections,
        content_hash=content_hash,
        raw_content=body,
    )


def walk_vault(vault_root: Path) -> list[Path]:
    """Find every .md file in the vault, skipping dotfiles and node_modules."""
    out: list[Path] = []
    for p in vault_root.rglob("*.md"):
        if any(part.startswith(".") for part in p.relative_to(vault_root).parts):
            continue
        if "node_modules" in p.parts:
            continue
        out.append(p)
    return sorted(out)
