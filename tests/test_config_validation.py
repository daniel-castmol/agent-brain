"""Settings validation — bad stemmer values fail at startup, not inside
a PRAGMA at first search."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from brain.config import Settings


def test_valid_stemmers_accepted():
    for v in ("porter", "spanish", "english", "none"):
        assert Settings(fts_stemmer=v).fts_stemmer == v


def test_invalid_fts_stemmer_rejected():
    with pytest.raises(ValidationError):
        Settings(fts_stemmer="bogus")


def test_injection_shaped_stemmer_rejected():
    with pytest.raises(ValidationError):
        Settings(fts_stemmer="porter', overwrite=0, stopwords='x")
