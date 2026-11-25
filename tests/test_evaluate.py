"""Eval metrics tests — pure functions, no embedder, no DB."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.evaluate import (
    GoldFileError,
    GoldQuery,
    aggregate,
    dedupe_to_notes,
    hit_at_k,
    load_gold,
    reciprocal_rank,
)
from brain.store import SearchHit


def _hit(path: str) -> SearchHit:
    return SearchHit(
        note_path=path, title="", chunk_index=0, heading_path="", body="", score=0.0
    )


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_dedupe_to_notes_preserves_first_occurrence_order():
    hits = [_hit("a.md"), _hit("b.md"), _hit("a.md"), _hit("c.md")]
    assert dedupe_to_notes(hits) == ["a.md", "b.md", "c.md"]


def test_hit_at_k_counts_distinct_notes_not_chunks():
    assert hit_at_k(["a.md", "b.md", "c.md"], expect=["c.md"], k=3) is True
    assert hit_at_k(["a.md", "b.md", "c.md"], expect=["c.md"], k=2) is False


def test_hit_at_k_any_of_expected_counts():
    assert hit_at_k(["a.md"], expect=["z.md", "a.md"], k=1) is True


def test_reciprocal_rank():
    assert reciprocal_rank(["a.md", "b.md"], expect=["b.md"]) == 0.5
    assert reciprocal_rank(["a.md"], expect=["z.md"]) == 0.0


def test_aggregate_overall_and_by_lang():
    rows = [
        {"lang": "en", "hit_at_5": True, "hit_at_10": True, "rr": 1.0, "latency_ms": 10.0},
        {"lang": "es", "hit_at_5": False, "hit_at_10": True, "rr": 0.125, "latency_ms": 30.0},
    ]
    agg = aggregate(rows)
    assert agg["n"] == 2
    assert agg["recall_at_5"] == 0.5
    assert agg["recall_at_10"] == 1.0
    assert agg["mrr_at_10"] == pytest.approx(0.5625)
    assert agg["latency_mean_ms"] == 20.0
    assert agg["by_lang"]["es"]["recall_at_5"] == 0.0
    assert agg["by_lang"]["en"]["recall_at_5"] == 1.0


def test_aggregate_empty_is_zeroes():
    agg = aggregate([])
    assert agg["n"] == 0
    assert agg["by_lang"] == {}


# ----------------------------------------------------------------------
# Gold-set loading
# ----------------------------------------------------------------------


def test_load_gold_parses_entries(tmp_path: Path):
    gold_file = tmp_path / "gold.yaml"
    gold_file.write_text(
        "- query: storage decision\n"
        "  expect: [decisions/storage.md]\n"
        "  tag: project/brain\n"
        "  lang: en\n"
        "- query: apuntes de clase\n"
        "  expect:\n"
        "    - classes/db.md\n"
        "    - classes/db-2.md\n",
        encoding="utf-8",
    )
    gold = load_gold(gold_file)
    assert len(gold) == 2
    assert gold[0] == GoldQuery(
        query="storage decision",
        expect=["decisions/storage.md"],
        tag="project/brain",
        lang="en",
    )
    assert gold[1].lang == "en"  # default
    assert gold[1].tag is None
    assert gold[1].expect == ["classes/db.md", "classes/db-2.md"]


def test_load_gold_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(GoldFileError):
        load_gold(tmp_path / "nope.yaml")


def test_load_gold_raises_on_invalid_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- query: [unclosed\n", encoding="utf-8")
    with pytest.raises(GoldFileError):
        load_gold(bad)


def test_load_gold_raises_on_entry_without_expect(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- query: no expectations\n  expect: []\n", encoding="utf-8")
    with pytest.raises(GoldFileError):
        load_gold(bad)


# ----------------------------------------------------------------------
# Runner + CLI
# ----------------------------------------------------------------------

import json
from datetime import datetime

from brain.evaluate import run_eval
from brain.store import Store


class FakeEmbedder:
    """Maps query text → fixed vector; no model, no network."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed_query(self, text: str) -> list[float]:
        return self.mapping[text]


def _one_hot(i: int, dim: int = 384) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _seed_note(store: Store, path: str, embedding: list[float]) -> None:
    conn = store.connect()
    conn.execute(
        "INSERT INTO notes (path, title, frontmatter, tags, content_hash, modified_at, indexed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [path, path, json.dumps({}), [], "x", datetime.now(), datetime.now()],
    )
    conn.execute(
        "INSERT INTO chunks (id, note_path, chunk_index, heading_path, content, body,"
        " content_hash, char_count, embedding)"
        " VALUES (nextval('chunk_id_seq'), ?, 0, 'H', 'body text', 'body text', ?, 9, ?::FLOAT[])",
        [path, f"h-{path}", embedding],
    )


def test_run_eval_scores_against_seeded_store(tmp_path: Path):
    store = Store(db_path=tmp_path / "t.duckdb", embedding_dim=384)
    store.connect()
    _seed_note(store, "alpha.md", _one_hot(0))
    _seed_note(store, "beta.md", _one_hot(1))
    gold = [
        GoldQuery(query="alpha", expect=["alpha.md"]),
        GoldQuery(query="alpha", expect=["missing.md"]),  # absent path = miss, run completes
    ]
    res = run_eval(store, FakeEmbedder({"alpha": _one_hot(0)}), gold)
    store.close()
    assert res["summary"]["recall_at_5"] == 0.5
    assert res["summary"]["n"] == 2
    assert res["queries"][0]["first_hit_rank"] == 1
    assert res["queries"][1]["first_hit_rank"] is None


def test_cli_eval_errors_cleanly_on_missing_gold(tmp_path: Path):
    from typer.testing import CliRunner

    from brain.cli import app

    result = CliRunner().invoke(app, ["eval", "--gold", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "nope.yaml" in result.output
    assert "Traceback" not in result.output


class StubStore:
    """Records the k passed to hybrid_search."""

    def __init__(self):
        self.calls: list[int] = []

    def hybrid_search(self, query_embedding, query_text, k, rrf_k, tag):
        self.calls.append(k)
        return []


def test_run_eval_matches_production_candidate_pool(monkeypatch):
    from brain.config import settings

    gold = [GoldQuery(query="q", expect=["x.md"])]
    emb = FakeEmbedder({"q": [0.0] * 384})

    stub = StubStore()
    run_eval(stub, emb, gold)
    assert stub.calls == [50]  # no rerank: wide fetch for dedupe headroom

    # With rerank on, production reranks only rerank_candidate_pool
    # candidates — eval must measure the same pipeline.
    monkeypatch.setattr(settings, "rerank_enabled", True)
    stub2 = StubStore()
    run_eval(stub2, emb, gold)
    assert stub2.calls == [settings.rerank_candidate_pool]
