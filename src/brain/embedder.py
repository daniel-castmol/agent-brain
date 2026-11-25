"""Embeddings via fastembed (ONNX, CPU-only, no API key needed).

Lazy-loaded so importing this module doesn't trigger model download.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding


class Embedder:
    """Thin wrapper around a fastembed TextEmbedding model."""

    def __init__(
        self,
        model_name: str,
        expected_dim: int,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.model_name = model_name
        self.expected_dim = expected_dim
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model: TextEmbedding | None = None

    def _ensure_loaded(self) -> "TextEmbedding":
        if self._model is None:
            # Imported lazily so a `--help` invocation doesn't pay the cost.
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts. Returns one float vector per input."""
        if not texts:
            return []
        model = self._ensure_loaded()
        # fastembed returns a generator of np.ndarrays.
        out: list[list[float]] = []
        for vec in model.embed(texts, batch_size=batch_size):
            out.append(vec.tolist())
        if out and len(out[0]) != self.expected_dim:
            raise ValueError(
                f"Embedding dim mismatch: model returned {len(out[0])}, "
                f"config expected {self.expected_dim}. "
                f"Update BRAIN_EMBEDDING_DIM to match the model."
            )
        return out

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed indexable content, applying the passage prefix.

        The prefix is applied at embed time only — chunk content and
        hashes never contain it.
        """
        return self.embed_batch([self.passage_prefix + t for t in texts], batch_size=batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([self.query_prefix + text])[0]


def make_embedder(
    model_name: str,
    expected_dim: int,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> Embedder:
    return Embedder(
        model_name=model_name,
        expected_dim=expected_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
