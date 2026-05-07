"""DuckDB-backed storage layer.

Schema:
    notes(path PK, title, frontmatter JSON, tags VARCHAR[], content_hash, modified_at, indexed_at)
    chunks(id PK, note_path FK, chunk_index, heading_path, content, body, content_hash, char_count, embedding FLOAT[N])
    links(source_path, target_name, target_path, alias, PK(source_path, target_name))

The embedding column is FLOAT[dim] (fixed-size ARRAY) so we can later add an
HNSW index via the `vss` extension. At Obsidian-vault scale a brute-force
ORDER BY array_cosine_distance is fine, so v1 ships without HNSW and we
turn it on once the vault grows.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from brain.chunker import Chunk
from brain.parser import Note


@dataclass
class SearchHit:
    note_path: str
    title: str
    chunk_index: int
    heading_path: str
    body: str
    score: float


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notes (
    path        VARCHAR PRIMARY KEY,
    title       VARCHAR,
    frontmatter JSON,
    tags        VARCHAR[],
    content_hash VARCHAR,
    modified_at TIMESTAMP,
    indexed_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT PRIMARY KEY,
    note_path    VARCHAR REFERENCES notes(path),
    chunk_index  INTEGER,
    heading_path VARCHAR,
    content      VARCHAR,
    body         VARCHAR,
    content_hash VARCHAR,
    char_count   INTEGER,
    embedding    FLOAT[{dim}]
);

CREATE INDEX IF NOT EXISTS idx_chunks_note_path ON chunks(note_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

CREATE TABLE IF NOT EXISTS links (
    source_path VARCHAR,
    target_name VARCHAR,
    target_path VARCHAR,
    alias       VARCHAR,
    PRIMARY KEY (source_path, target_name)
);

CREATE SEQUENCE IF NOT EXISTS chunk_id_seq START 1;
"""


class Store:
    def __init__(self, db_path: Path, embedding_dim: int):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute(SCHEMA_SQL.format(dim=self.embedding_dim))
            # FTS extension for hybrid search.
            self._conn.execute("INSTALL fts; LOAD fts;")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self):
        conn = self.connect()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Upserts
    # ------------------------------------------------------------------
    def upsert_note(self, note: Note, modified_at: datetime) -> None:
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO notes (path, title, frontmatter, tags, content_hash, modified_at, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (path) DO UPDATE SET
                title = excluded.title,
                frontmatter = excluded.frontmatter,
                tags = excluded.tags,
                content_hash = excluded.content_hash,
                modified_at = excluded.modified_at,
                indexed_at = excluded.indexed_at
            """,
            [
                note.path,
                note.title,
                json.dumps(note.frontmatter, default=str),
                note.tags,
                note.content_hash,
                modified_at,
                datetime.utcnow(),
            ],
        )

    def replace_chunks(
        self,
        note_path: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")
        conn = self.connect()
        conn.execute("DELETE FROM chunks WHERE note_path = ?", [note_path])
        for chunk, vec in zip(chunks, embeddings, strict=True):
            conn.execute(
                """
                INSERT INTO chunks (id, note_path, chunk_index, heading_path,
                                    content, body, content_hash, char_count, embedding)
                VALUES (nextval('chunk_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[])
                """,
                [
                    chunk.note_path,
                    chunk.chunk_index,
                    chunk.heading_path,
                    chunk.content,
                    chunk.body,
                    chunk.content_hash,
                    chunk.char_count,
                    vec,
                ],
            )

    def replace_links(self, note_path: str, links: list[tuple[str, str | None, str | None]]) -> None:
        """links: list of (target_name, target_path_or_None, alias_or_None)."""
        conn = self.connect()
        conn.execute("DELETE FROM links WHERE source_path = ?", [note_path])
        for target_name, target_path, alias in links:
            conn.execute(
                """
                INSERT INTO links (source_path, target_name, target_path, alias)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (source_path, target_name) DO NOTHING
                """,
                [note_path, target_name, target_path, alias],
            )

    def delete_note(self, note_path: str) -> None:
        conn = self.connect()
        conn.execute("DELETE FROM chunks WHERE note_path = ?", [note_path])
        conn.execute("DELETE FROM links WHERE source_path = ?", [note_path])
        conn.execute("DELETE FROM notes WHERE path = ?", [note_path])

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_note_hash(self, note_path: str) -> str | None:
        conn = self.connect()
        row = conn.execute(
            "SELECT content_hash FROM notes WHERE path = ?", [note_path]
        ).fetchone()
        return row[0] if row else None

    def all_note_paths(self) -> set[str]:
        conn = self.connect()
        return {r[0] for r in conn.execute("SELECT path FROM notes").fetchall()}

    def list_recent(self, n: int = 10) -> list[tuple[str, str, datetime]]:
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT path, title, modified_at
            FROM notes
            ORDER BY modified_at DESC
            LIMIT ?
            """,
            [n],
        ).fetchall()
        return rows

    def read_note(self, note_path: str) -> dict | None:
        conn = self.connect()
        row = conn.execute(
            """
            SELECT path, title, frontmatter, tags, modified_at
            FROM notes WHERE path = ?
            """,
            [note_path],
        ).fetchone()
        if not row:
            return None
        chunks = conn.execute(
            """
            SELECT chunk_index, heading_path, body
            FROM chunks
            WHERE note_path = ?
            ORDER BY chunk_index
            """,
            [note_path],
        ).fetchall()
        return {
            "path": row[0],
            "title": row[1],
            "frontmatter": json.loads(row[2]) if row[2] else {},
            "tags": list(row[3]) if row[3] else [],
            "modified_at": row[4].isoformat() if row[4] else None,
            "chunks": [
                {"chunk_index": c[0], "heading_path": c[1], "body": c[2]} for c in chunks
            ],
        }

    # ------------------------------------------------------------------
    # Search — vector + FTS, fused with reciprocal rank fusion.
    # ------------------------------------------------------------------
    def vector_search(
        self,
        query_embedding: list[float],
        k: int,
        tag: str | None = None,
    ) -> list[SearchHit]:
        conn = self.connect()
        sql = """
            SELECT
                c.note_path, n.title, c.chunk_index, c.heading_path, c.body,
                array_cosine_distance(c.embedding, ?::FLOAT[{dim}]) AS dist
            FROM chunks c
            JOIN notes n ON n.path = c.note_path
        """.format(dim=self.embedding_dim)
        params: list = [query_embedding]
        if tag:
            sql += " WHERE list_contains(n.tags, ?)"
            params.append(tag)
        sql += " ORDER BY dist ASC LIMIT ?"
        params.append(k)
        rows = conn.execute(sql, params).fetchall()
        # Cosine distance ∈ [0, 2]; convert to a rough similarity score for display.
        return [
            SearchHit(
                note_path=r[0],
                title=r[1],
                chunk_index=r[2],
                heading_path=r[3],
                body=r[4],
                score=1.0 - float(r[5]),
            )
            for r in rows
        ]

    def fts_search(self, query: str, k: int, tag: str | None = None) -> list[SearchHit]:
        """Lexical search using DuckDB's FTS extension.

        We rebuild the FTS index lazily; for a personal vault this is cheap.
        """
        conn = self.connect()
        # Build (or rebuild) the FTS index over the chunks table on every query.
        # For a few-thousand-chunk vault this takes milliseconds. If it ever
        # becomes a bottleneck, we can rebuild only after writes.
        try:
            conn.execute(
                "PRAGMA create_fts_index('chunks', 'id', 'body', 'heading_path', overwrite=1)"
            )
        except duckdb.Error:
            # PRAGMA can fail on already-existing index in some versions; ignore.
            pass

        sql = """
            WITH scored AS (
                SELECT id, fts_main_chunks.match_bm25(id, ?) AS score
                FROM chunks
            )
            SELECT
                c.note_path, n.title, c.chunk_index, c.heading_path, c.body,
                s.score
            FROM scored s
            JOIN chunks c ON c.id = s.id
            JOIN notes n ON n.path = c.note_path
            WHERE s.score IS NOT NULL
        """
        params: list = [query]
        if tag:
            sql += " AND list_contains(n.tags, ?)"
            params.append(tag)
        sql += " ORDER BY s.score DESC LIMIT ?"
        params.append(k)
        rows = conn.execute(sql, params).fetchall()
        return [
            SearchHit(
                note_path=r[0],
                title=r[1],
                chunk_index=r[2],
                heading_path=r[3],
                body=r[4],
                score=float(r[5]),
            )
            for r in rows
        ]

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int,
        rrf_k: int,
        tag: str | None = None,
    ) -> list[SearchHit]:
        """Combine vector and FTS results via Reciprocal Rank Fusion."""
        # Pull more than k from each source so fusion has room to work.
        pull = max(k * 4, 20)
        vec_hits = self.vector_search(query_embedding, pull, tag=tag)
        try:
            fts_hits = self.fts_search(query_text, pull, tag=tag)
        except duckdb.Error:
            # FTS may fail if no chunks exist yet; fall back to vector only.
            fts_hits = []

        scores: dict[tuple[str, int], float] = {}
        hit_map: dict[tuple[str, int], SearchHit] = {}
        for ranked, source_hits in [(vec_hits, "vec"), (fts_hits, "fts")]:
            for rank, hit in enumerate(ranked):
                key = (hit.note_path, hit.chunk_index)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                if key not in hit_map:
                    hit_map[key] = hit

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            SearchHit(
                note_path=hit_map[key].note_path,
                title=hit_map[key].title,
                chunk_index=hit_map[key].chunk_index,
                heading_path=hit_map[key].heading_path,
                body=hit_map[key].body,
                score=score,
            )
            for key, score in ordered
        ]

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def find_related(self, note_path: str, k: int = 5) -> list[SearchHit]:
        """Find related notes via:
        1. Outbound + inbound wikilinks (resolved).
        2. Vector similarity to the note's chunks (averaged).
        """
        conn = self.connect()

        linked = conn.execute(
            """
            SELECT DISTINCT target_path FROM links
            WHERE source_path = ? AND target_path IS NOT NULL
            UNION
            SELECT DISTINCT source_path FROM links
            WHERE target_path = ?
            """,
            [note_path, note_path],
        ).fetchall()
        linked_paths = {r[0] for r in linked if r[0] and r[0] != note_path}

        # Average this note's chunk embeddings, then ANN against everything else.
        avg = conn.execute(
            """
            SELECT list_transform(
                       generate_series(1, ?),
                       i -> (SELECT avg(embedding[i]) FROM chunks WHERE note_path = ?)
                   )::FLOAT[{dim}]
            """.format(dim=self.embedding_dim),
            [self.embedding_dim, note_path],
        ).fetchone()
        if not avg or avg[0] is None or avg[0][0] is None:
            return []
        avg_vec = list(avg[0])

        rows = conn.execute(
            """
            SELECT c.note_path, n.title, c.chunk_index, c.heading_path, c.body,
                   array_cosine_distance(c.embedding, ?::FLOAT[{dim}]) AS dist
            FROM chunks c
            JOIN notes n ON n.path = c.note_path
            WHERE c.note_path != ?
            ORDER BY dist ASC
            LIMIT ?
            """.format(dim=self.embedding_dim),
            [avg_vec, note_path, k * 3],
        ).fetchall()

        # Boost linked notes so they surface even if vectors disagree.
        hits: list[SearchHit] = []
        seen_paths: set[str] = set()
        for r in rows:
            path = r[0]
            if path in seen_paths:
                continue
            seen_paths.add(path)
            base_score = 1.0 - float(r[5])
            score = base_score + (0.2 if path in linked_paths else 0.0)
            hits.append(
                SearchHit(
                    note_path=path,
                    title=r[1],
                    chunk_index=r[2],
                    heading_path=r[3],
                    body=r[4],
                    score=score,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
