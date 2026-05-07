"""End-to-end reranker test using the real fastembed cross-encoder.

Loads the actual ONNX model (~120MB, cached after first run). Marked
`integration` so the fast suite stays under 200ms.

Run with:  uv run pytest -m integration tests/test_reranker_e2e.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.config import Settings
from brain.embedder import make_embedder
from brain.indexer import index_single_file
from brain.reranker import make_reranker, rerank_candidates
from brain.store import Store
from brain.writer import build_note, validate_vault_path


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
    return make_embedder(model_name="BAAI/bge-small-en-v1.5", expected_dim=384)


@pytest.fixture(scope="module")
def reranker():
    # Use the smallest cross-encoder to keep first-run download fast.
    return make_reranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


@pytest.mark.integration
def test_reranker_promotes_relevant_chunk_above_distractor(
    settings: Settings, store: Store, embedder, reranker
):
    """Seed three notes with distinct topics; verify the reranker scores
    the topically-relevant note higher than the distractors after RRF."""
    notes = {
        "research/provenance.md": (
            "# Provenance in agent memory\n\n"
            "Every note written by the agent is stamped with author=agent in "
            "frontmatter and tagged #agent so the user can distinguish "
            "agent-written content from their own hand-written notes."
        ),
        "infra/duckdb-setup.md": (
            "# DuckDB setup\n\n"
            "Chunks are stored in a FLOAT[384] array column for cosine "
            "similarity search. The vss extension can add an HNSW index "
            "once the vault grows past 5k chunks."
        ),
        "ops/wsl-bridge.md": (
            "# WSL to Windows bridge\n\n"
            "Claude Desktop on Windows calls the brain MCP server in WSL "
            "via wsl.exe. The vault lives on the Windows filesystem so "
            "Obsidian can watch it without EISDIR errors."
        ),
    }
    for path, body in notes.items():
        np = validate_vault_path(path, settings.vault_dir)
        np.abs.parent.mkdir(parents=True, exist_ok=True)
        np.abs.write_text(build_note(body), encoding="utf-8")
        index_single_file(settings, embedder, store, np.rel)

    query = "how does the agent mark notes as written by itself"
    qvec = embedder.embed_query(query)
    hits = store.hybrid_search(
        query_embedding=qvec, query_text=query, k=10, rrf_k=60
    )

    # Sanity: hybrid search returned all three.
    assert {h.note_path for h in hits} == set(notes.keys())

    reranked = rerank_candidates(
        reranker=reranker,
        query=query,
        candidates=list(hits),  # copy so we don't mutate the fixture-scoped list
        text_of=lambda h: f"{h.heading_path}\n\n{h.body}" if h.heading_path else h.body,
        k=3,
    )
    # The provenance note is the only one that actually answers the query.
    # After reranking it must be #1 (RRF could put it anywhere in top-3).
    assert reranked[0].note_path == "research/provenance.md", (
        f"Expected provenance.md as top hit after rerank, got order: "
        f"{[(h.note_path, round(h.score, 3)) for h in reranked]}"
    )
    # And it should beat the next-best by a clear margin (>0.5 logit).
    assert reranked[0].score > reranked[1].score + 0.5
