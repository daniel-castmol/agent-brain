"""MCP server.

Exposes the brain to Claude Code / Claude Desktop via stdio transport.
Tool descriptions are written for an LLM to read — they tell Claude WHEN
to use each tool, not just what it does.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from brain.config import settings
from brain.embedder import make_embedder
from brain.indexer import index_single_file
from brain.report import build_report
from brain.reranker import make_reranker, rerank_candidates
from brain.store import Store
from brain.writer import (
    InvalidPathError,
    build_note,
    compose_appended_body,
    validate_vault_path,
)

mcp = FastMCP("brain")

# Lazy singletons — instantiated on first tool call so `--help` is fast.
_store: Store | None = None
_embedder = None
_reranker = None


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(db_path=settings.db_path, embedding_dim=settings.embedding_dim)
        _store.connect()
        # Raises EmbeddingConfigMismatchError if the stored vectors were
        # built with a different model/prefix config; the user must run
        # `brain index --rebuild`.
        _store.verify_embedding_config(
            settings.embedding_model, settings.embedding_dim, "", ""
        )
    return _store


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = make_embedder(
            model_name=settings.embedding_model,
            expected_dim=settings.embedding_dim,
        )
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = make_reranker(model_name=settings.rerank_model)
    return _reranker


def _is_agent_authored(frontmatter: dict[str, Any] | None) -> bool:
    """A note is agent-authored if its frontmatter stamps `author: agent`."""
    if not frontmatter:
        return False
    return frontmatter.get("author") == "agent"


@mcp.tool()
def search_notes(query: str, k: int = 5, tag: str | None = None) -> dict:
    """Search the agent's memory vault for notes semantically related to a query.

    Use this whenever the user references past context, prior decisions, or
    knowledge that could plausibly be in the vault — for example when they
    say "what did we decide about X", "how did I set up Y", "find that note
    on Z", or anytime continuity across sessions matters. The query should
    use the user's own words; conceptual matches work better than exact
    keywords thanks to the underlying embedding model.

    When BRAIN_RERANK_ENABLED is true the top RRF candidates are rescored
    by a cross-encoder; in that case `score` is a rerank logit (can be
    negative — higher still means more relevant). Otherwise `score` is the
    fused RRF score. Don't compare scores across calls with different
    rerank settings.

    Args:
        query: A natural-language question or topic. Be concrete.
        k: Number of results to return (default 5, max 20).
        tag: Optional frontmatter tag to filter by (e.g. "project/cortex").

    Returns:
        A dict with `hits`, each containing note_path, title, heading_path,
        a body excerpt, and a relevance score. Follow up with `read_note`
        on the most relevant path to load full context.
    """
    k = max(1, min(k, 20))
    store = _get_store()
    embedder = _get_embedder()
    query_vec = embedder.embed_query(query)

    # If reranking is on, fetch a larger candidate pool so the cross-encoder
    # has something to choose from. Otherwise hybrid_search returns top-k directly.
    fetch_k = settings.rerank_candidate_pool if settings.rerank_enabled else k
    hits = store.hybrid_search(
        query_embedding=query_vec,
        query_text=query,
        k=fetch_k,
        rrf_k=settings.rrf_k,
        tag=tag,
    )

    if settings.rerank_enabled and hits:
        reranker = _get_reranker()
        hits = rerank_candidates(
            reranker=reranker,
            query=query,
            candidates=list(hits),
            text_of=lambda h: f"{h.heading_path}\n\n{h.body}" if h.heading_path else h.body,
            k=k,
        )

    return {
        "query": query,
        "reranked": settings.rerank_enabled,
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


@mcp.tool()
def write_note(
    path: str,
    body: str,
    frontmatter: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Create a new note in the vault, stamped with agent provenance.

    Use this to save a self-contained piece of knowledge worth keeping
    across sessions: a decision and its rationale, a reusable pattern, a
    summary of an investigation, a meeting outcome. Refuses to overwrite
    existing notes — use `append_to_note` to add to an existing file.

    The note is automatically stamped with `author: agent`, `created_at`,
    and the `agent` tag in frontmatter so the user can distinguish
    agent-written content from their own. Path traversal and absolute
    paths are rejected.

    A common pattern is daily session logs at `_agent/sessions/YYYY-MM-DD.md`
    where the agent appends decisions/learnings as they happen.

    Args:
        path: Vault-relative path ending in `.md` (e.g.
            "decisions/2026-05-07-rerank-cutoff.md"). No leading slash, no
            `..` segments.
        body: Markdown body of the note (without frontmatter — the tool
            adds frontmatter itself).
        frontmatter: Optional extra frontmatter fields (e.g. `{"title": ...}`).
            `author`/`created_at`/`tags` are managed by the tool.
        tags: Optional tags to add (in addition to the auto-added `agent`).

    Returns:
        `{"path": ..., "status": "written"}` on success, `{"error": ...}`
        if the path is invalid or the file already exists.
    """
    try:
        np = validate_vault_path(path, settings.vault_dir)
    except InvalidPathError as e:
        return {"error": str(e)}

    if np.abs.exists():
        return {"error": f"Note already exists, use append_to_note: {np.rel}"}

    text = build_note(body, frontmatter=frontmatter, tags=tags)
    np.abs.parent.mkdir(parents=True, exist_ok=True)
    np.abs.write_text(text, encoding="utf-8")

    index_single_file(settings, _get_embedder(), _get_store(), np.rel)
    return {"path": np.rel, "status": "written"}


@mcp.tool()
def append_to_note(path: str, content: str, heading: str | None = None) -> dict:
    """Append content to an existing note.

    Use this to add to a session log, log a follow-up under an existing
    decision, or extend a working document without overwriting it. If
    `heading` is given, the content is inserted under that heading (the
    heading is created if missing). If omitted:

    - For agent-authored notes (`author: agent` in frontmatter), content
      is appended at the end with a `### YYYY-MM-DD HH:MM` timestamp.
    - For user-authored notes, content is inserted under `## Agent notes`
      so the user's own content is never silently mutated.

    Args:
        path: Vault-relative path of the existing note.
        content: New content to add (plain markdown, no frontmatter).
        heading: Optional ATX heading like `"## Decisions"` to insert under.

    Returns:
        `{"path": ..., "status": "appended"}` on success, `{"error": ...}`
        if the path is invalid or the file does not exist.
    """
    try:
        np = validate_vault_path(path, settings.vault_dir)
    except InvalidPathError as e:
        return {"error": str(e)}

    if not np.abs.exists():
        return {"error": f"Note not found, use write_note to create: {np.rel}"}

    existing = np.abs.read_text(encoding="utf-8")

    # Detect authorship from existing frontmatter.
    store = _get_store()
    note_meta = store.read_note(np.rel)
    is_agent = _is_agent_authored(note_meta["frontmatter"] if note_meta else None)

    new_text = compose_appended_body(
        existing,
        content,
        heading=heading,
        is_agent_authored=is_agent,
    )
    np.abs.write_text(new_text, encoding="utf-8")

    index_single_file(settings, _get_embedder(), store, np.rel)
    return {"path": np.rel, "status": "appended"}


@mcp.tool()
def vault_report() -> dict:
    """Summarize the agent's memory vault — counts, authorship mix, recency.

    Use when the user asks about the state of their memory ("what's in the
    vault", "how much have I written", "what has the agent been logging"),
    when starting a session and you want a one-shot orientation, or to
    detect self-reference pollution (large agent-authored share, lots of
    recent activity but few user-authored notes).

    Returns:
        A dict with three sections:
        - `vault`: notes, chunks, links, total_chars, oldest/newest note,
          age_days
        - `authorship`: agent_authored, user_authored, mixed_authored
          (user-authored notes that have an `## Agent notes` section)
        - `activity`: notes modified in last 24h / 7d / 30d, plus the 5
          most-recently-modified notes
    """
    return build_report(_get_store())


def main() -> None:
    """Run the MCP server over stdio. Called by `brain serve`."""
    mcp.run()


if __name__ == "__main__":
    main()
