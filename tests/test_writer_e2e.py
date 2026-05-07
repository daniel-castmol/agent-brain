"""End-to-end smoke test for v2 write-back.

Loads the real fastembed model and a temp DuckDB to prove the full
write → re-index → search → read round-trip. Marked `integration` so
the fast suite stays under 100ms.

Run with:  uv run pytest -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.config import Settings
from brain.embedder import make_embedder
from brain.indexer import index_single_file
from brain.store import Store
from brain.writer import build_note, compose_appended_body, validate_vault_path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(vault_dir=tmp_path / "vault", db_dir=tmp_path / "db")


@pytest.fixture
def store(settings: Settings) -> Store:
    settings.ensure_dirs()
    s = Store(db_path=settings.db_path, embedding_dim=settings.embedding_dim)
    s.connect()
    yield s
    s.close()


@pytest.fixture(scope="module")
def embedder():
    # Module-scoped so we pay the model-load cost once for all integration tests.
    return make_embedder(model_name="BAAI/bge-small-en-v1.5", expected_dim=384)


@pytest.mark.integration
def test_write_then_search_roundtrip(settings: Settings, store: Store, embedder):
    np = validate_vault_path("research/llm-ranking.md", settings.vault_dir)
    text = build_note(
        "# LLM ranking research\n\nWe compared bge-small to nomic-embed.",
        frontmatter={"title": "LLM ranking"},
        tags=["research/llm"],
    )
    np.abs.parent.mkdir(parents=True, exist_ok=True)
    np.abs.write_text(text, encoding="utf-8")
    index_single_file(settings, embedder, store, np.rel)

    # Note row + chunks present.
    assert store.get_note_hash(np.rel) is not None
    note = store.read_note(np.rel)
    assert note["frontmatter"]["author"] == "agent"
    assert "agent" in note["tags"]
    assert "research/llm" in note["tags"]

    # Searchable.
    qvec = embedder.embed_query("which embedding model performs better")
    hits = store.hybrid_search(
        query_embedding=qvec, query_text="embedding model comparison", k=5, rrf_k=60
    )
    assert any(h.note_path == np.rel for h in hits)


@pytest.mark.integration
def test_append_to_user_note_uses_agent_notes_section(
    settings: Settings, store: Store, embedder
):
    # A user-authored note (no `author: agent` in frontmatter).
    np = validate_vault_path("projects/cortex-spec.md", settings.vault_dir)
    np.abs.parent.mkdir(parents=True, exist_ok=True)
    np.abs.write_text(
        "---\ntitle: Cortex spec\n---\n\n# Cortex spec\n\nMy hand-written intro.\n",
        encoding="utf-8",
    )
    index_single_file(settings, embedder, store, np.rel)

    # Append something — heading omitted, file is user-authored, so it should
    # land under `## Agent notes`.
    existing = np.abs.read_text(encoding="utf-8")
    new_text = compose_appended_body(
        existing,
        "Agent's contribution: consider HNSW after 5k chunks.",
        heading=None,
        is_agent_authored=False,
    )
    np.abs.write_text(new_text, encoding="utf-8")
    index_single_file(settings, embedder, store, np.rel)

    on_disk = np.abs.read_text(encoding="utf-8")
    assert "My hand-written intro." in on_disk  # original preserved
    assert "## Agent notes" in on_disk
    assert "Agent's contribution" in on_disk

    # Searchable by the new content.
    qvec = embedder.embed_query("when to enable HNSW indexing")
    hits = store.hybrid_search(
        query_embedding=qvec, query_text="HNSW index threshold", k=5, rrf_k=60
    )
    assert any(h.note_path == np.rel for h in hits)
