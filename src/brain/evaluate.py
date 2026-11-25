"""Retrieval evaluation: gold-set loading and note-level metrics.

Metrics are note-level over *distinct notes*: chunk hits are deduplicated
to their note path before any recall/MRR computation, so five chunks from
two notes count as two results, not five.

This module holds pure functions (no embedder, no DB) plus the runner
that wires them into the live store/embedder (`run_eval`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brain.store import SearchHit


class GoldFileError(Exception):
    """Gold file missing, unparsable, or an entry is malformed."""


@dataclass
class GoldQuery:
    query: str
    expect: list[str]
    tag: str | None = None
    lang: str = "en"


def load_gold(path: Path) -> list[GoldQuery]:
    import yaml

    if not path.exists():
        raise GoldFileError(f"Gold file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise GoldFileError(f"Invalid YAML in {path}: {e}") from e
    if not isinstance(data, list):
        raise GoldFileError(f"Gold file must be a YAML list of entries: {path}")

    out: list[GoldQuery] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or not entry.get("query"):
            raise GoldFileError(f"Entry {i} in {path} has no 'query'")
        expect = entry.get("expect")
        if not isinstance(expect, list) or not expect:
            raise GoldFileError(
                f"Entry {i} ('{entry.get('query')}') needs a non-empty 'expect' list"
            )
        out.append(
            GoldQuery(
                query=str(entry["query"]),
                expect=[str(p) for p in expect],
                tag=entry.get("tag"),
                lang=str(entry.get("lang", "en")),
            )
        )
    return out


def dedupe_to_notes(hits: list[SearchHit]) -> list[str]:
    """Ordered distinct note paths, first occurrence wins."""
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h.note_path not in seen:
            seen.add(h.note_path)
            out.append(h.note_path)
    return out


def hit_at_k(ranked_notes: list[str], expect: list[str], k: int) -> bool:
    expected = set(expect)
    return any(p in expected for p in ranked_notes[:k])


def reciprocal_rank(ranked_notes: list[str], expect: list[str]) -> float:
    expected = set(expect)
    for rank, p in enumerate(ranked_notes, start=1):
        if p in expected:
            return 1.0 / rank
    return 0.0


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    p95_idx = min(n - 1, max(0, round(0.95 * n) - 1))
    return {
        "n": n,
        "recall_at_5": sum(bool(r["hit_at_5"]) for r in rows) / n,
        "recall_at_10": sum(bool(r["hit_at_10"]) for r in rows) / n,
        "mrr_at_10": sum(r["rr"] for r in rows) / n,
        "latency_mean_ms": sum(latencies) / n,
        "latency_p95_ms": latencies[p95_idx],
    }


def aggregate(per_query: list[dict]) -> dict:
    """Overall + per-language metrics from per-query rows.

    Rows carry: lang, hit_at_5, hit_at_10, rr, latency_ms.
    """
    if not per_query:
        return {
            "n": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "latency_mean_ms": 0.0,
            "latency_p95_ms": 0.0,
            "by_lang": {},
        }
    out = _metrics(per_query)
    langs = sorted({r["lang"] for r in per_query})
    out["by_lang"] = {
        lang: _metrics([r for r in per_query if r["lang"] == lang]) for lang in langs
    }
    return out
