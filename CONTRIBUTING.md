# Contributing

Thanks for your interest! `brain` is small and intentionally simple, so
contributions should aim to keep it that way.

## Setup

```bash
git clone https://github.com/<your-handle>/brain.git
cd brain
uv sync
uv run pytest
```

Tests should pass before you start.

## Project shape

```
src/brain/      core library
  config.py     env-var settings
  parser.py     markdown → Note (frontmatter, headings, wikilinks, tags)
  chunker.py    Note → Chunks (heading-aware splits)
  embedder.py   fastembed wrapper
  store.py      DuckDB schema, hybrid search, find_related
  indexer.py    walk vault → diff via SHA256 → upsert
  server.py     FastMCP server (4 tools)
  cli.py        typer CLI

tests/          pure-function tests (no DB, no model)
vault-seed/     example notes documenting the project itself
tasks/          working roadmap (todo.md) + lessons (lessons.md)
```

## Conventions

- **Markdown is the source of truth.** Anything stored only in DuckDB
  is wrong. State that matters lives in `~/brain-vault/*.md`.
- **Lazy-load expensive imports.** `brain --help` should be instant.
  Anything heavy (`fastembed`, etc.) only imports inside the function
  that actually uses it.
- **Keep the embedding-dim guard.** `embedder.py` validates the model's
  output dim against `BRAIN_EMBEDDING_DIM`. Don't remove this — silent
  dim mismatches corrupt the index.
- **Heading breadcrumb prefixes are load-bearing.** Each chunk's content
  starts with `# Path > To > Heading`. This is what lets the embedding
  model disambiguate similar headings across notes. Don't strip it.

## Running tests

```bash
uv run pytest                  # all tests
uv run pytest tests/test_parser.py -v
```

The parser and chunker tests are pure functions and run in milliseconds.

## Pull requests

1. Open an issue first for anything non-trivial — happy to discuss
   design before you spend time.
2. Keep PRs focused. One idea per PR.
3. Add tests for new functionality.
4. Update [`tasks/todo.md`](./tasks/todo.md) and
   [`tasks/lessons.md`](./tasks/lessons.md) when relevant.
5. Don't add dependencies casually. The current dep list is short on
   purpose.

## What's a good first contribution?

Look at `tasks/todo.md` — anything in the **v3 — retrieval polish**
section is bounded enough to be a clean PR. Per-chunk SHA reuse and
case-insensitive wikilink resolution are both well-scoped.
