"""MCP server.

Exposes the brain to Claude Code / Claude Desktop via stdio transport.
Tool descriptions are written for an LLM to read — they tell Claude WHEN
to use each tool, not just what it does.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from brain.config import settings
from brain.embedder import make_embedder
from brain.store import Store

mcp = FastMCP("brain")

# Lazy singletons — instantiated on first tool call so `--help` is fast.
_store: Store | None = None
_embedder = None


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(db_path=settings.db_path, embedding_dim=settings.embedding_dim)
        _store.connect()
    return _store


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = make_embedder(
            model_name=settings.embedding_model,
            expected_dim=settings.embedding_dim,
        )
    return _embedder


@mcp.tool()
def search_notes(query: str, k: int = 5, tag: str | None = None) -> dict:
    """Search the agent's memory vault for notes semantically related to a query.

    Use this whenever the user references past context, prior decisions, or
    knowledge that could plausibly be in the vault — for example when they
    say "what did we decide about X", "how did I set up Y", "find that note
    on Z", or anytime continuity across sessions matters. The query should
    use the user's own words; conceptual matches work better than exact
    keywords thanks to the underlying embedding model.

    Args:
        query: A natural-language question or topic. Be concrete.
        k: Number of results to return (default 5, max 20).
        tag: Optional frontmatter tag to filter by (e.g. "project/cortex").

    Returns:
        A dict with `hits`, each containing note_path, title, heading_path,
        a body excerpt, and a fused relevance score. Follow up with
        `read_note` on the most relevant path to load full context.
    """
    k = max(1, min(k, 20))
    store = _get_store()
    embedder = _get_embedder()
    query_vec = embedder.embed_query(query)
    hits = store.hybrid_search(
        query_embedding=query_vec,
        query_text=query,
        k=k,
        rrf_k=settings.rrf_k,
        tag=tag,
    )
    return {
        "query": query,
        "hits": [
            {
                "note_path": h.note_path,
                "title": h.title,
                "heading_path": h.heading_path,
                "excerpt": h.body[:400] + ("…" if len(h.body) > 400 else ""),
                "score": round(h.score, 4),
            }
            for h in hits
        ],
    }


@mcp.tool()
def read_note(path: str) -> dict:
    """Read the full content of a note from the vault by its relative path.

    Use this after `search_notes` to load complete context for a relevant
    result, or when the user references a specific note by name. The path
    is vault-relative (e.g. "ideas/agent-memory.md"), exactly as returned
    by `search_notes` or `list_recent`.

    Args:
        path: Vault-relative path of the note to read.

    Returns:
        A dict with the note's title, frontmatter, tags, modified_at
        timestamp, and the full content reassembled from its chunks.
        Returns `{"error": ...}` if the note does not exist.
    """
    store = _get_store()
    note = store.read_note(path)
    if note is None:
        return {"error": f"Note not found: {path}"}
    full_body = "\n\n".join(c["body"] for c in note["chunks"])
    return {
        "path": note["path"],
        "title": note["title"],
        "frontmatter": note["frontmatter"],
        "tags": note["tags"],
        "modified_at": note["modified_at"],
        "content": full_body,
    }


@mcp.tool()
def list_recent(n: int = 10) -> dict:
    """List the most recently modified notes in the vault.

    Use when the user asks about recent work — "what was I working on", "show
    me yesterday's notes", "what's new" — or to orient yourself at the start
    of a session before answering broader questions.

    Args:
        n: How many notes to list (default 10, max 50).

    Returns:
        A dict with `notes`, each containing path, title, and modified_at,
        ordered most-recent first.
    """
    n = max(1, min(n, 50))
    store = _get_store()
    rows = store.list_recent(n)
    return {
        "notes": [
            {
                "path": r[0],
                "title": r[1],
                "modified_at": r[2].isoformat() if r[2] else None,
            }
            for r in rows
        ]
    }


@mcp.tool()
def find_related(path: str, k: int = 5) -> dict:
    """Find notes related to a given note via wikilinks and semantic similarity.

    Use this to discover connected context — "what else have I written about
    this", "what should I read alongside this note" — or when the user wants
    to explore the neighborhood around a specific idea. Combines explicit
    [[wikilinks]] in the source note with averaged-embedding similarity
    against the rest of the vault.

    Args:
        path: Vault-relative path of the seed note.
        k: Number of related notes to return (default 5, max 20).

    Returns:
        A dict with `related`, each containing note_path, title, the
        most-relevant chunk's heading_path and excerpt, and a relevance
        score (linked notes get a small boost).
    """
    k = max(1, min(k, 20))
    store = _get_store()
    hits = store.find_related(path, k=k)
    return {
        "seed_path": path,
        "related": [
            {
                "note_path": h.note_path,
                "title": h.title,
                "heading_path": h.heading_path,
                "excerpt": h.body[:400] + ("…" if len(h.body) > 400 else ""),
                "score": round(h.score, 4),
            }
            for h in hits
        ],
    }


def main() -> None:
    """Run the MCP server over stdio. Called by `brain serve`."""
    mcp.run()


if __name__ == "__main__":
    main()
