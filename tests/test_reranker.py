"""Reranker tests — pure-function logic with a fake cross-encoder.

The real cross-encoder loads a 120MB ONNX model on first call (slow),
so unit tests use a `FakeReranker` that scores docs deterministically
based on substring presence. The real model is exercised in the opt-in
integration suite (`tests/test_reranker_e2e.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from brain.reranker import Reranker, rerank_candidates


@dataclass
class _Candidate:
    """Minimal stand-in for SearchHit (avoids importing the real one to
    keep this test isolated from the store layer)."""

    note_path: str
    body: str
    score: float = 0.0


class FakeReranker:
    """Scores docs by counting occurrences of query terms (case-insensitive).

    Deterministic, no model load. Higher score = more matches.
    """

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        terms = [t.lower() for t in query.split() if t]
        return [
            float(sum(doc.lower().count(t) for t in terms))
            for doc in documents
        ]


# Sanity check the protocol — FakeReranker must satisfy it structurally.
def test_fake_reranker_satisfies_protocol():
    r: Reranker = FakeReranker()
    scores = r.rerank("hello world", ["hello hello", "world", "unrelated"])
    assert scores == [2.0, 1.0, 0.0]


def test_rerank_candidates_reorders_by_score():
    """Reranking should sort the candidates by descending rerank score
    and return at most k of them."""
    candidates = [
        _Candidate(note_path="a.md", body="duckdb stores chunks", score=0.10),
        _Candidate(note_path="b.md", body="provenance tracks author=agent", score=0.20),
        _Candidate(note_path="c.md", body="wsl bridge for windows", score=0.30),
    ]
    reranker = FakeReranker()
    out = rerank_candidates(
        reranker=reranker,
        query="provenance author",
        candidates=candidates,
        text_of=lambda c: c.body,
        k=3,
    )
    # 'b.md' has 2 matches (provenance, author), should be #1.
    assert [c.note_path for c in out] == ["b.md", "a.md", "c.md"]


def test_rerank_candidates_respects_k_limit():
    candidates = [
        _Candidate(note_path=f"{i}.md", body=f"doc {i}") for i in range(10)
    ]
    reranker = FakeReranker()
    out = rerank_candidates(
        reranker=reranker, query="doc", candidates=candidates, text_of=lambda c: c.body, k=3
    )
    assert len(out) == 3


def test_rerank_candidates_handles_empty_input():
    out = rerank_candidates(
        reranker=FakeReranker(),
        query="anything",
        candidates=[],
        text_of=lambda c: c.body,
        k=5,
    )
    assert out == []


def test_rerank_candidates_overwrites_score_with_rerank_score():
    """The output's `score` field should reflect the rerank logit so the
    MCP caller sees the post-rerank ranking signal, not the stale RRF score."""
    candidates = [
        _Candidate(note_path="a.md", body="match match", score=99.0),
        _Candidate(note_path="b.md", body="other", score=99.0),
    ]
    out = rerank_candidates(
        reranker=FakeReranker(),
        query="match",
        candidates=candidates,
        text_of=lambda c: c.body,
        k=2,
    )
    assert out[0].note_path == "a.md"
    assert out[0].score == 2.0  # FakeReranker counts 'match' twice
    assert out[1].score == 0.0


def test_rerank_candidates_preserves_tie_order_stable():
    """When two candidates have identical rerank scores, the original
    RRF order is preserved (Python's sort is stable)."""
    candidates = [
        _Candidate(note_path="first.md", body="x", score=0.5),
        _Candidate(note_path="second.md", body="x", score=0.4),
        _Candidate(note_path="third.md", body="x", score=0.3),
    ]
    out = rerank_candidates(
        reranker=FakeReranker(),
        query="nothing",  # no matches → all scored 0.0
        candidates=candidates,
        text_of=lambda c: c.body,
        k=3,
    )
    # All have the same rerank score (0.0), so original order survives.
    assert [c.note_path for c in out] == ["first.md", "second.md", "third.md"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
