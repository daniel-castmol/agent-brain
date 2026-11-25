# brain

**Local markdown-based memory layer for AI agents, exposed via MCP.**

A folder of `.md` files is the source of truth. A DuckDB sidecar — embeddings,
full-text search, and a wikilink graph — is a disposable, fully-rebuildable
index. An MCP server gives Claude Code, Claude Desktop, or any other
MCP-compatible client semantic + lexical search, full-note reads, and
graph-based "find related" — all running locally, no API keys, no cloud.

```text
~/brain-vault/                       ← markdown is the source of truth
       │
       ▼
  parser → chunker → embedder → DuckDB (notes, chunks, links)
                                       │
                                       ▼
                                 MCP server (stdio)
                                 ├─ search_notes      (+ optional rerank)
                                 ├─ read_note
                                 ├─ list_recent
                                 ├─ find_related
                                 ├─ write_note         ← v2
                                 ├─ append_to_note     ← v2
                                 └─ vault_report       ← v2
                                       │
                                       ▼
                              Claude Code / Desktop
```

## Why

Out of the box, AI coding agents forget everything between sessions.
`CLAUDE.md` gets you static context, but no write-back, no semantic recall
over months of work, and no way for the agent to surface "I think we
discussed this back in March" with actual receipts. Cloud-managed memory
products want you to trust them with personal data and lock you into their
retrieval API.

`brain` is the smallest useful version of persistent agent memory: a vault
you control, indexed by something you can `cat`, served to agents over a
standard protocol.

## Properties

- **Local-first.** No API keys. No cloud account. After a one-time
  ~30 MB embedding-model download, fully offline.
- **Markdown is canonical.** The vault is the only durable layer.
  Everything else rebuilds from it. No vendor lock-in — your notes stay
  portable to any tool that reads text.
- **Hybrid retrieval.** Vector cosine similarity + DuckDB BM25 full-text
  search, fused with reciprocal rank fusion. Better than either alone.
- **Heading-aware chunks.** Each chunk's heading path
  (`# Architecture > Storage`) gets prefixed before embedding, so the
  model has hierarchy context.
- **Incremental sync.** SHA256 per file; unchanged files skip the whole
  parse/embed/write pipeline.
- **MCP-native.** Drops straight into Claude Code via `claude mcp add`,
  or Claude Desktop via `claude_desktop_config.json`.

## Architecture

A medallion-style pipeline borrowed from data engineering, applied to
agent memory:

| Layer  | What                                                           | Durable? |
| ------ | -------------------------------------------------------------- | -------- |
| Bronze | `~/brain-vault/**/*.md` — markdown, version-controllable       | yes      |
| Silver | DuckDB tables: `notes`, `chunks` (FLOAT[384] embeddings), `links` | no, rebuildable |
| Gold   | Cosine top-k + FTS BM25 fused via RRF, optional cross-encoder rerank, served via 7 MCP tools | no |

Disposability of silver and gold is the most important property. Blow away
`~/.brain/db.duckdb`, run `brain index`, and the system is back. Same shape
as dbt rebuilding gold tables from bronze sources.

## Stack

| Layer       | Choice                            | Why                                       |
| ----------- | --------------------------------- | ----------------------------------------- |
| Storage     | DuckDB                            | One file, SQL, fast, no daemon            |
| Vectors     | DuckDB `FLOAT[384]` ARRAY column  | Brute-force is fine at vault scale; HNSW available later via `vss` |
| Embeddings  | `bge-small-en-v1.5` via fastembed | ONNX, CPU, ~30 MB, no API key             |
| FTS         | DuckDB `fts` extension            | Hybrid retrieval                          |
| Protocol    | MCP, stdio transport              | Drops into Claude Code / Desktop directly |
| Package mgr | `uv`                              | Fast, lockfile, project-scoped            |

## Quickstart

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/daniel-castmol/agent-brain.git
cd agent-brain
uv sync                   # install dependencies into .venv
uv run brain init         # create ~/brain-vault and ~/.brain
uv run brain index        # initial sync (downloads the embedding model first time)
uv run brain status       # confirm counts
```

Add a note to test:

```bash
mkdir -p ~/brain-vault/inbox
cat > ~/brain-vault/inbox/first-note.md <<'EOF'
---
tags: [test]
---

# First Note

Testing the brain memory system. We're using DuckDB for storage and
bge-small-en-v1.5 for embeddings.

## Next Steps

- Wire into Claude Code
- Add more notes
EOF

uv run brain index
uv run brain search "duckdb storage"
```

For full setup including Claude Code and Claude Desktop integration,
see [INSTALL.md](./INSTALL.md).

## Live indexing

```bash
uv run brain index --watch
```

Edits to `.md` files in the vault trigger an incremental re-sync (debounced
2s). Only files whose SHA256 changed get re-parsed and re-embedded.

## Configuration

All settings come from environment variables prefixed `BRAIN_` (or a `.env`
file in the project root). See [`.env.example`](./.env.example) for the
full list. The defaults work out of the box.

## Privacy

Everything runs locally. The vault, the DuckDB index, and all embeddings
live entirely on your machine. The only network call is the one-time
embedding-model download from HuggingFace.

When you query brain through Claude (Code or Desktop), the *retrieved
note excerpts* become part of the prompt sent to Anthropic for that
request — same as any text you paste into a Claude conversation. brain
itself never sends anything anywhere.

## MCP tools

The MCP server exposes seven tools. Read tools (1–4) work on a v1 install;
write tools (5–6) and the report tool (7) are v2.

| Tool             | Purpose                                                          |
| ---------------- | ---------------------------------------------------------------- |
| `search_notes`   | Hybrid (vector + BM25) search, optional cross-encoder rerank     |
| `read_note`      | Load a note's full content from its vault-relative path          |
| `list_recent`    | Most recently modified notes                                     |
| `find_related`   | Wikilink + averaged-embedding neighborhood of a seed note        |
| `write_note`     | Create a new note, stamped with `author: agent` provenance       |
| `append_to_note` | Append to an existing note (under a heading or `## Agent notes`) |
| `vault_report`   | Counts, authorship mix, recent activity                          |

## Optional cross-encoder rerank

Two-stage retrieval: hybrid_search returns ~25 candidates (cheap), a
cross-encoder rescores them jointly (more accurate, ~100ms, ~120MB
one-time download). Off by default — flip on once your vault is big
enough that hybrid alone is noisy:

```bash
export BRAIN_RERANK_ENABLED=true
# Optional: pick a different cross-encoder
export BRAIN_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
```

## Roadmap

**v3 — retrieval polish.** Per-chunk SHA reuse, HNSW once vault scale
justifies it, better wikilink resolution, tag namespace queries.

**v4 — cloud port.** S3 + S3 Vectors + Lambda mirror of the local stack
for users who want a shareable cloud version. Same MCP surface, swappable
backend.

See [`tasks/todo.md`](./tasks/todo.md) for the working roadmap.

## Testing

```bash
uv run pytest                       # 36 unit tests, <2s
uv run pytest -m integration        # +3 e2e tests (downloads ~120MB cross-encoder)
```

## License

[MIT](./LICENSE).
