# brain — Claude Code project context

Local markdown-based memory layer for AI agents, exposed via MCP. The vault
is the source of truth, DuckDB is a disposable index, and a stdio MCP server
lets agents query the whole thing.

## Status

v1 shipped — fresh agent-only vault, English-only embeddings, four read-only
tools. Verified end-to-end: `brain init` → `brain index` → `brain status`
runs clean. 9/9 unit tests pass.

```
notes:  N    chunks: M    links: K
vault:  ~/brain-vault         (BRAIN_VAULT_DIR)
db:     ~/.brain/db.duckdb    (BRAIN_DB_DIR + BRAIN_DB_FILENAME)
model:  BAAI/bge-small-en-v1.5  (384d)
```

## Architecture

Bronze → silver → gold. Same medallion pattern used in CORTEX/Connect dbt
work, applied to agent memory.

- **Bronze:** `~/brain-vault/**/*.md` — markdown files, only durable layer.
- **Silver:** DuckDB tables `notes` (frontmatter + tags + hash), `chunks`
  (heading-aware splits), `links` (resolved wikilinks). Rebuildable from
  bronze.
- **Gold:** cosine top-k over `chunks.embedding` + DuckDB FTS BM25, fused
  with reciprocal rank fusion. Served via MCP.

## Stack

| layer       | choice                            | rationale                                |
| ----------- | --------------------------------- | ---------------------------------------- |
| storage     | DuckDB 1.5+                       | one file, SQL, no daemon                 |
| vectors     | `FLOAT[384]` ARRAY column         | brute-force is fine at vault scale       |
| embeddings  | `bge-small-en-v1.5` via fastembed | ONNX, CPU, ~30 MB, no API key            |
| FTS         | DuckDB `fts` extension            | hybrid retrieval                         |
| protocol    | MCP `FastMCP`, stdio              | drops into Claude Code via `claude mcp add` |
| package mgr | `uv`                              | locked, project-scoped                   |

## Layout

```
src/brain/
  config.py    pydantic-settings, BRAIN_* env vars
  parser.py    frontmatter, headings, wikilinks, tags
  chunker.py   section-aware splits, heading breadcrumb prefix
  embedder.py  fastembed lazy-loader, dim validation
  store.py     DuckDB schema, hybrid_search via RRF, find_related
  indexer.py   walk → diff via SHA256 → re-embed only changed files
  server.py    4 FastMCP tools: search_notes, read_note, list_recent, find_related
  cli.py       typer CLI: init, index (--watch), serve, search, status
tests/         pure-function tests for parser + chunker
```

## Working conventions

Global `~/.claude/CLAUDE.md` rules apply (plan mode for non-trivial changes,
surgical fixes, subagents for complex tasks, log work in `tasks/todo.md`,
log lessons in `tasks/lessons.md`). Project-specific:

- **Run tests before any commit.** `uv run pytest` is fast (<1s for the
  pure-function tests). Don't break the 9 green tests.
- **Markdown is canonical.** Never store data only in DuckDB. If a piece of
  state matters, it lives in a markdown file in the vault.
- **Heading breadcrumb in `chunk.content`.** Each embedded chunk starts with
  its heading path (`# Architecture > Storage`) before the body. This is
  load-bearing for retrieval quality — don't strip it.
- **Incremental sync via SHA256.** `notes.content_hash` and
  `chunks.content_hash` skip re-embedding unchanged content. Don't break
  the hash flow when adding new fields.
- **Embedding dim must match the model.** `embedder.py` validates
  `BRAIN_EMBEDDING_DIM` against what the model returns and raises if they
  disagree. Switching models means updating both.
- **Lazy-load expensive imports.** `brain --help` should be instant. Don't
  import `fastembed` at module top level outside `embedder.py`.

## Where things live

- Tests: `tests/`
- DB schema: `src/brain/store.py` → `SCHEMA_SQL`
- MCP tool descriptions: `src/brain/server.py` — these are LLM-facing and
  shape *when* agents call each tool. Edit the docstrings carefully.
- Roadmap: `tasks/todo.md`
- Running list of lessons hit during dev: `tasks/lessons.md`

## Quick reference

```bash
uv sync                       # install deps
uv run brain init             # create vault and db dirs
uv run brain index            # full incremental sync
uv run brain index --watch    # sync, then watch for changes (debounced 2s)
uv run brain serve            # run MCP server over stdio
uv run brain search "query"   # debug search from CLI
uv run brain status           # config + counts
uv run pytest                 # 9 tests, <1s
```

## Wire into Claude Code

```bash
claude mcp add --scope user --transport stdio brain \
  -- uv --directory $(pwd) run brain serve
claude mcp list   # confirm "brain: connected"
```

`--scope user` makes brain available across all projects. After this, any
Claude Code session can call `search_notes`, `read_note`, `list_recent`,
`find_related`.
