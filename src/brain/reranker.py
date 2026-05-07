"""Cross-encoder reranking via fastembed (ONNX, CPU-only, no API key).

Two-stage retrieval pattern: hybrid_search returns ~30 RRF candidates
(cheap bi-encoder + BM25), this module rescores them with a cross-encoder
that processes (query, document) pairs jointly, then returns top k.

Per the v3 design (see `tasks/todo.md`):
- Bi-encoder retrieves; cross-encoder reranks. Joint attention catches
  relevance the cosine-only path misses.
- Default model: `Xenova/ms-marco-MiniLM-L-12-v2` (~120MB, English, fast).
- Output `score` is the rerank logit (higher = more relevant; can be
  negative). NOT comparable to RRF scores from prior versions — note
  this in the MCP tool description so the calling LLM doesn't compare.

Lazy-loaded so importing this module doesn't trigger model download.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder


class Reranker(Protocol):
    """Anything that scores (query, doc) pairs.

    Returned scores must be in the same order as `documents`. Higher
    means more relevant.
    """

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        ...


class FastembedReranker:
    """Cross-encoder backed by fastembed's ONNX runtime."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: TextCrossEncoder | None = None

    def _ensure_loaded(self) -> "TextCrossEncoder":
        if self._model is None:
            # Lazy import — keeps `brain --help` instant.
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._ensure_loaded()
        # fastembed returns Iterable[float] (lazy). Materialize to a list
        # so callers can index, sort, and reuse it.
        return list(model.rerank(query, documents))


def make_reranker(model_name: str) -> Reranker:
    return FastembedReranker(model_name=model_name)


# ----------------------------------------------------------------------
# Generic orchestration helper.
# ----------------------------------------------------------------------

T = TypeVar("T")


def rerank_candidates(
    *,
    reranker: Reranker,
    query: str,
    candidates: list[T],
    text_of: Callable[[T], str],
    k: int,
) -> list[T]:
    """Rerank `candidates` against `query`, return the top `k`.

    The candidates' `score` attribute is overwritten with the rerank
    logit so the MCP caller sees the post-rerank ranking signal (NOT
    the prior RRF score, which is on a different scale and would mix
    confusingly with the new ordering).

    Generic over candidate type via `text_of` — keeps this module
    decoupled from the store layer's `SearchHit` shape.
    """
    if not candidates:
        return []

    documents = [text_of(c) for c in candidates]
    scores = reranker.rerank(query, documents)

    # Overwrite each candidate's `score` field with its rerank logit.
    # The candidates are dataclasses with a mutable `score`; this is
    # idiomatic in this codebase (see SearchHit in store.py).
    for c, s in zip(candidates, scores, strict=True):
        c.score = float(s)

    # Stable sort by descending score; preserves prior order on ties.
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k]
