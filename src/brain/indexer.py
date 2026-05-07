"""Vault → DuckDB sync.

Drives the full pipeline:
    walk vault → parse → diff against store → re-chunk + embed changes → upsert.

Incremental: skips files whose SHA256 hasn't changed. Even within a changed
file, individual chunks whose hashes match existing rows could in principle
be reused, but for v1 we just re-embed the whole file when it changes —
simpler and still cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain.chunker import chunk_note
from brain.config import Settings
from brain.embedder import Embedder
from brain.parser import parse_file, walk_vault
from brain.store import Store


@dataclass
class IndexStats:
    scanned: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_embedded: int = 0


def _resolve_wikilinks(
    targets: list[tuple[str, str | None]], known_paths: set[str], known_titles: dict[str, str]
) -> list[tuple[str, str | None, str | None]]:
    """Resolve wikilink targets to vault paths.

    A target like `[[Agent Memory]]` resolves to:
      - the path with the matching title, OR
      - the path whose stem matches (case-insensitive), OR
      - None (dangling link, kept for later resolution).
    """
    out: list[tuple[str, str | None, str | None]] = []
    for target_name, alias in targets:
        target_lower = target_name.lower()
        # Title match.
        path = known_titles.get(target_lower)
        if not path:
            # Stem match against known paths.
            for p in known_paths:
                stem = Path(p).stem.lower()
                if stem == target_lower:
                    path = p
                    break
        out.append((target_name, path, alias))
    return out


def index_vault(settings: Settings, embedder: Embedder, store: Store) -> IndexStats:
    """Full incremental sync of the vault into DuckDB."""
    stats = IndexStats()
    settings.ensure_dirs()

    files = walk_vault(settings.vault_dir)
    on_disk: set[str] = set()
    notes_to_resolve: list[tuple[str, list[tuple[str, str | None]]]] = []

    for abs_path in files:
        stats.scanned += 1
        note = parse_file(abs_path, settings.vault_dir)
        on_disk.add(note.path)

        existing_hash = store.get_note_hash(note.path)
        if existing_hash == note.content_hash:
            stats.unchanged += 1
            continue

        chunks = chunk_note(
            note,
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        embeddings = embedder.embed_batch([c.content for c in chunks]) if chunks else []

        modified_at = datetime.fromtimestamp(abs_path.stat().st_mtime)
        with store.transaction():
            # FK from chunks → notes(path) means we must clear old chunks
            # before upserting the parent row. See store.delete_chunks_for.
            store.delete_chunks_for(note.path)
            store.upsert_note(note, modified_at=modified_at)
            store.replace_chunks(note.path, chunks, embeddings)

        notes_to_resolve.append(
            (note.path, [(wl.target, wl.alias) for wl in note.wikilinks])
        )
        stats.chunks_embedded += len(chunks)
        if existing_hash is None:
            stats.new += 1
        else:
            stats.updated += 1

    # Garbage-collect notes that disappeared from the vault.
    in_db = store.all_note_paths()
    for stale_path in in_db - on_disk:
        store.delete_note(stale_path)
        stats.deleted += 1

    # Resolve wikilinks once we know all known paths/titles.
    known_paths = on_disk
    # Build a title → path lookup from the DB.
    conn = store.connect()
    title_rows = conn.execute("SELECT path, title FROM notes").fetchall()
    known_titles = {t.lower(): p for p, t in title_rows if t}
    for source_path, raw_targets in notes_to_resolve:
        resolved = _resolve_wikilinks(raw_targets, known_paths, known_titles)
        store.replace_links(source_path, resolved)

    return stats


def index_single_file(
    settings: Settings,
    embedder: Embedder,
    store: Store,
    rel_path: str,
) -> None:
    """Re-index a single note after a write or append.

    Used by the v2 write tools so we don't pay a full vault walk on every
    `write_note` / `append_to_note`. Idempotent: skips re-embedding when the
    file's content hash hasn't changed. If the file no longer exists on disk,
    removes the note + its chunks + its outgoing links from the store.
    """
    abs_path = settings.vault_dir / rel_path
    if not abs_path.exists():
        store.delete_note(rel_path)
        return

    note = parse_file(abs_path, settings.vault_dir)
    existing_hash = store.get_note_hash(note.path)
    if existing_hash == note.content_hash:
        return

    chunks = chunk_note(
        note,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    embeddings = embedder.embed_batch([c.content for c in chunks]) if chunks else []

    modified_at = datetime.fromtimestamp(abs_path.stat().st_mtime)
    with store.transaction():
        # FK from chunks → notes(path) means we must clear old chunks
        # before upserting the parent row. See store.delete_chunks_for.
        store.delete_chunks_for(note.path)
        store.upsert_note(note, modified_at=modified_at)
        store.replace_chunks(note.path, chunks, embeddings)

    # Resolve this note's wikilinks using current store state. Include this
    # note's own path so self-links resolve.
    known_paths = store.all_note_paths() | {note.path}
    conn = store.connect()
    title_rows = conn.execute("SELECT path, title FROM notes").fetchall()
    known_titles = {t.lower(): p for p, t in title_rows if t}
    raw_targets = [(wl.target, wl.alias) for wl in note.wikilinks]
    resolved = _resolve_wikilinks(raw_targets, known_paths, known_titles)
    store.replace_links(note.path, resolved)
