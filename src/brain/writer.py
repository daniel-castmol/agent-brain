"""Write-back primitives for v2.

Pure-function string-in/string-out logic for:
- Validating that an MCP-supplied path is safely inside the vault
- Stamping agent provenance into new notes' frontmatter
- Inserting timestamped agent contributions under a heading inside an
  existing user-authored note (so user content is never silently mutated)

The MCP tools in `server.py` are thin glue: validate the path, read the
existing file (if any), call into here to produce the final markdown text,
write it to disk, and trigger a single-file re-index.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Aliased to avoid clashing with our `frontmatter=` keyword argument.
import frontmatter as _fm

DEFAULT_AGENT_HEADING = "## Agent notes"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class InvalidPathError(ValueError):
    """Raised when an MCP-supplied path is unsafe or malformed."""


@dataclass
class NotePath:
    """A validated, vault-relative markdown path."""

    rel: str  # vault-relative, normalized, forward slashes (e.g. "research/llm.md")
    abs: Path  # absolute path on disk (may not yet exist)


# ----------------------------------------------------------------------
# Path validation — system boundary, must reject untrusted input.
# ----------------------------------------------------------------------


def validate_vault_path(raw_path: str, vault_dir: Path) -> NotePath:
    """Validate a user/agent-supplied path resolves inside the vault.

    Rules:
    - non-empty, ends in `.md`
    - no absolute paths (POSIX or Windows-style)
    - no `..` segments after normalization
    - resolves to a path under `vault_dir`
    """
    if not raw_path:
        raise InvalidPathError("path is empty")
    if not raw_path.endswith(".md"):
        raise InvalidPathError(f"path must end in .md: {raw_path!r}")
    # Block POSIX abs (`/foo`) and Windows abs (`C:\foo`).
    if raw_path.startswith("/") or (len(raw_path) >= 2 and raw_path[1] == ":"):
        raise InvalidPathError(f"absolute paths not allowed: {raw_path!r}")

    normalized = os.path.normpath(raw_path).replace("\\", "/")
    if normalized.startswith("..") or "/../" in normalized or normalized == "..":
        raise InvalidPathError(f"path traversal not allowed: {raw_path!r}")
    # `normpath` strips leading `./` already.
    if normalized.startswith("/"):
        raise InvalidPathError(f"absolute paths not allowed: {raw_path!r}")

    abs_path = (vault_dir / normalized).resolve(strict=False)
    vault_resolved = vault_dir.resolve(strict=False)
    try:
        abs_path.relative_to(vault_resolved)
    except ValueError as e:
        raise InvalidPathError(f"path escapes vault: {raw_path!r}") from e

    return NotePath(rel=normalized, abs=abs_path)


# ----------------------------------------------------------------------
# Frontmatter stamping — agent provenance for new notes.
# ----------------------------------------------------------------------


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_note(
    body: str,
    *,
    frontmatter: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    now: datetime | None = None,
) -> str:
    """Produce the full markdown text for a new agent-authored note.

    Always stamps `author: agent`, `created_at: <ISO timestamp>`, and adds
    `agent` to the tag list (deduped). User-supplied frontmatter is
    preserved; user tags merge with the agent tag.
    """
    now = now or datetime.now()
    meta: dict[str, Any] = dict(frontmatter or {})
    meta["author"] = "agent"
    meta["created_at"] = now.isoformat()

    fm_tags = meta.get("tags") or []
    if isinstance(fm_tags, str):
        fm_tags = [fm_tags]
    user_tags = list(tags or [])
    meta["tags"] = _dedupe_preserve_order(["agent", *fm_tags, *user_tags])

    post = _fm.Post(body, **meta)
    return _fm.dumps(post)


# ----------------------------------------------------------------------
# Heading insertion — insert content under an existing or new heading.
# ----------------------------------------------------------------------


def _heading_level(line: str) -> int | None:
    m = HEADING_RE.match(line)
    return len(m.group(1)) if m else None


def insert_under_heading(existing: str, heading: str, new_block: str) -> str:
    """Insert `new_block` at the end of the section under `heading`.

    If `heading` doesn't exist in `existing`, it's created at the end of
    the file. The new block lands before any subsequent section of the
    same or higher level.
    """
    target_level = _heading_level(heading)
    if target_level is None:
        raise ValueError(f"heading must be a markdown ATX heading: {heading!r}")
    target_stripped = heading.strip()

    lines = existing.splitlines()
    # Find the heading line.
    target_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == target_stripped:
            target_idx = i
            break

    new_block_lines = new_block.rstrip("\n").split("\n")

    if target_idx is None:
        # Heading missing — append heading + content at end.
        prefix = existing.rstrip("\n")
        joined_new = "\n".join(new_block_lines)
        sep = "\n\n" if prefix else ""
        return f"{prefix}{sep}{heading}\n\n{joined_new}\n"

    # Find the boundary: next heading at level <= target_level, or EOF.
    end_idx = len(lines)
    for j in range(target_idx + 1, len(lines)):
        lvl = _heading_level(lines[j])
        if lvl is not None and lvl <= target_level:
            end_idx = j
            break

    # Trim trailing blank lines inside the section so we control spacing.
    last_content_idx = end_idx - 1
    while last_content_idx > target_idx and lines[last_content_idx].strip() == "":
        last_content_idx -= 1

    pre = lines[: last_content_idx + 1]
    post = lines[end_idx:]
    middle = ["", *new_block_lines]
    if post:
        middle += [""]
    rebuilt = pre + middle + post
    return "\n".join(rebuilt) + "\n"


# ----------------------------------------------------------------------
# Append composition — used by the append_to_note MCP tool.
# ----------------------------------------------------------------------


def format_agent_append(content: str, now: datetime) -> str:
    """Wrap content in a `### YYYY-MM-DD HH:MM` subheading."""
    ts = now.strftime("%Y-%m-%d %H:%M")
    return f"### {ts}\n\n{content.strip()}\n"


def compose_appended_body(
    existing: str,
    content: str,
    *,
    heading: str | None = None,
    now: datetime | None = None,
    is_agent_authored: bool = False,
) -> str:
    """Decide where appended content goes, then return the new full body.

    - explicit `heading` → always insert under it (creating if missing)
    - no heading + agent-authored file → append at end of file (no marker)
    - no heading + user-authored file → insert under `## Agent notes`
    """
    now = now or datetime.now()
    formatted = format_agent_append(content, now)

    if heading is None and is_agent_authored:
        prefix = existing.rstrip("\n")
        sep = "\n\n" if prefix else ""
        return f"{prefix}{sep}{formatted}"

    target = heading if heading is not None else DEFAULT_AGENT_HEADING
    return insert_under_heading(existing, target, formatted)
