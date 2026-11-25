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
