"""Per-chunk embedding reuse — real parser/chunker/store, fake embedder."""

from __future__ import annotations

from pathlib import Path

from brain.config import Settings
from brain.indexer import index_vault
from brain.store import Store

NOTE_V1 = """# Title

## Alpha

Alpha body text here.

## Beta

Beta body text here.

## Gamma

Gamma body text here.
"""

# Only the Beta section changes.
NOTE_V2 = NOTE_V1.replace("Beta body text here.", "Beta body was rewritten.")


class CountingFakeEmbedder:
    """Returns a different vector per call so reuse is distinguishable
    from deterministic re-embedding."""

    def __init__(self):
        self.embedded: list[str] = []

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out = []
        for t in texts:
            self.embedded.append(t)
            out.append([float(len(self.embedded))] * 384)
        return out


def test_changed_file_reembeds_only_changed_chunks(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text(NOTE_V1, encoding="utf-8")
    settings = Settings(vault_dir=vault, db_dir=tmp_path / "db")
    settings.ensure_dirs()
    store = Store(db_path=settings.db_path, embedding_dim=384)
    store.connect()
    fake = CountingFakeEmbedder()

    stats1 = index_vault(settings, fake, store)
    assert stats1.chunks_embedded == 3
    map1 = store.get_chunk_embeddings("note.md")
    assert len(map1) == 3

    (vault / "note.md").write_text(NOTE_V2, encoding="utf-8")
    stats2 = index_vault(settings, fake, store)

    assert len(fake.embedded) == 4  # 3 initial + only the changed Beta chunk
    assert stats2.chunks_embedded == 1
    assert stats2.chunks_reused == 2

    # Reused vectors survive byte-identically.
    map2 = store.get_chunk_embeddings("note.md")
    shared = set(map1) & set(map2)
    assert len(shared) == 2  # Alpha and Gamma hashes unchanged
    for h in shared:
        assert map2[h] == map1[h]

    store.close()
