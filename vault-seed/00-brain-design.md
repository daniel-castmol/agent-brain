---
tags: [project/brain, design, architecture]
created: 2026-05-06
---

# Brain — design rationale

A local markdown-based memory layer for AI agents, exposed via MCP. This
note captures the v1 design decisions so future Claude Code sessions can
search and recall the *why* behind the implementation.

## Problem

Out of the box, AI coding agents forget everything between sessions.
`CLAUDE.md` gives static context but no write-back, no semantic recall
over months of work, and no way for the agent to surface "I think we
discussed this back in March" with actual receipts. The cloud-managed
memory products (Mem0, Zep, etc.) require trusting a third party with
personal data and lock you into their retrieval API. The goal of brain
is the smallest useful version of persistent agent memory: a vault you
control, indexed by something you can `cat`, served to agents over a
standard protocol.

## Architectural pattern

Same medallion pattern from data engineering, applied to agent memory:

- **Bronze:** `~/brain-vault/**/*.md`. The only durable layer. Plain
  markdown, version-controllable, portable across AI tools, editable by
  hand.
- **Silver:** DuckDB tables — `notes` (metadata + tags + content hash),
  `chunks` (heading-aware splits), `links` (resolved wikilinks).
  Rebuildable from bronze via `brain index`.
- **Gold:** cosine top-k over `chunks.embedding` (FLOAT[384]) plus
  DuckDB FTS BM25, fused with reciprocal rank fusion. Served via four
  MCP tools.

The disposability of silver and gold is the most important property.
Blow away `~/.brain/db.duckdb`, run `brain index`, and the system is
back. Same property as dbt building gold tables from bronze sources.

## Stack decisions

### DuckDB over SQLite + sqlite-vec

Both work. DuckDB won because (a) it's already my daily-driver analytics
engine, (b) ARRAY columns + `array_cosine_distance` give brute-force
top-k for free, and (c) the FTS extension is built in. Acknowledged
risk: the `vss` extension's HNSW persistence is experimental — but we
don't enable HNSW in v1 because brute-force is fine at personal-vault
scale. Markdown stays as the source of truth so any DB corruption is
recoverable.

### Local embeddings via fastembed

`bge-small-en-v1.5` — 384 dimensions, ~30MB ONNX model, runs on CPU. No
API key, no network call after the one-time model download, works
offline. Multilingual support deferred (would be `bge-m3`, 1024d) since
v1 is English-only by choice. Switching models means updating both the
fastembed model name and the DuckDB FLOAT[N] column type.

### MCP over stdio, not HTTP

stdio is what Claude Code expects for local servers — `claude mcp add
--transport stdio` spawns the server as a child process and pipes
JSON-RPC over stdin/stdout. HTTP is for remote/cloud servers. v1 is
local-only, so stdio.

### `uv` for everything

Project-scoped venvs, lockfile, fast resolver. `uv run brain index`
works without activating anything.

## Quality choices that matter

### Heading breadcrumb prefix in chunk content

Every chunk gets `# Path > To > Section\n\n` prepended before the body
gets embedded. This is load-bearing for retrieval — the embedding model
needs hierarchy context to disambiguate "Storage" under "Architecture"
from "Storage" under "Personal Finance". Cheap to add, large effect on
retrieval quality.

### Hybrid retrieval via reciprocal rank fusion

Vector search alone misses queries with specific terms ("FastMCP",
"bge-small"). FTS alone misses semantic matches ("how do I store
embeddings"). RRF combines them: each retrieval source contributes
1/(k+rank), summed across sources. The chunk that's top-3 in *both*
sources beats the chunk that's top-1 in only one.

### Incremental sync via SHA256

`notes.content_hash = sha256(file_bytes)`. On each `brain index` run,
files whose hash matches the stored value get skipped entirely — no
parsing, no embedding, no DB writes. At a few hundred notes the
difference is the difference between "instant" and "wait for the
embedding model to load".

## Constraints kept in mind

- **No surprise costs.** Everything runs locally. The only network call
  is the one-time fastembed model download from HuggingFace.
- **No vendor lock-in.** Plain markdown means the vault is portable to
  any other tool that reads text. The DuckDB index is throwaway.
- **No rebuild required for routine edits.** File-watcher mode
  (`brain index --watch`) re-syncs only changed files, debounced 2s.
- **Fast `--help`.** Lazy-load fastembed and DuckDB so the CLI feels
  snappy even for trivial commands.

## What's next

See `tasks/todo.md` in the project repo for the full roadmap.
Highlights: write-back tools (`write_note`, `append_to_note`), a
`SKILL.md` that teaches Claude Code *when* to call each tool, HNSW once
the vault grows, and an AWS port using S3 + S3 Vectors + Lambda.

## Related

- [[Brain — session log 2026-05-06]] — the conversation that produced v1
- Project repo: https://github.com/daniel-castmol/agent-brain
