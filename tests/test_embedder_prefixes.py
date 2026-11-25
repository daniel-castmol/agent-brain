"""Prefix handling in the embedder — fake model, no download."""

from __future__ import annotations

import numpy as np

from brain.embedder import Embedder, make_embedder


class FakeModel:
    """Records exactly what texts reach the underlying model."""

    def __init__(self):
        self.seen: list[list[str]] = []

    def embed(self, texts, batch_size=32):
        self.seen.append(list(texts))
        for _ in texts:
            yield np.array([0.0, 0.0, 0.0])


def test_embed_query_and_passages_apply_prefixes():
    e = Embedder("x", 3, query_prefix="query: ", passage_prefix="passage: ")
    fake = FakeModel()
    e._model = fake
    e.embed_query("hola")
    e.embed_passages(["a", "b"])
    e.embed_batch(["raw"])  # embed_batch stays raw: hashes/content unaffected
    assert fake.seen == [["query: hola"], ["passage: a", "passage: b"], ["raw"]]


def test_make_embedder_passes_prefixes_through():
    e = make_embedder("x", 3, query_prefix="q: ", passage_prefix="p: ")
    assert e.query_prefix == "q: "
    assert e.passage_prefix == "p: "


def test_empty_prefixes_are_default_noop():
    e = Embedder("x", 3)
    fake = FakeModel()
    e._model = fake
    e.embed_query("hola")
    e.embed_passages(["a"])
    assert fake.seen == [["hola"], ["a"]]
