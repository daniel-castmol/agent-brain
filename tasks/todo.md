# brain — todo

## Now

- [ ] Wire brain into Claude Code: `claude mcp add --scope user --transport stdio brain -- uv --directory <abs> run brain serve`
- [ ] Verify with `claude mcp list` and `/mcp` inside a session — should show `brain: connected`
- [ ] Drop the seed design note (`vault-seed/00-brain-design.md`) into `~/brain-vault/` and re-index, so brain has something real to search
- [ ] Sanity-check inside Claude Code: ask "what did we decide about the storage layer for brain?" — should call `search_notes`

## v2 — write-back ✅ shipped

- [x] `write_note(path, body, frontmatter=None, tags=None)` MCP tool
  - Validates path is inside vault, refuses traversal/absolute paths
  - Refuses to overwrite — use `append_to_note` for existing notes
  - Auto-stamps `author: agent`, `created_at`, and `agent` tag
  - Triggers single-file re-index via `index_single_file`
- [x] `append_to_note(path, content, heading=None)` MCP tool
  - Explicit heading → insert under it (creates if missing)
  - Agent-authored note → append at end with `### YYYY-MM-DD HH:MM`
  - User-authored note → insert under `## Agent notes`
- [x] Cross-encoder reranker wired into `search_notes`
  - Off by default (`BRAIN_RERANK_ENABLED=true` to opt in)
  - Two-stage: hybrid_search top-N, then cross-encoder rescoring
- [x] `vault_report()` MCP tool + `brain report` CLI subcommand
  - Counts (notes/chunks/links), authorship mix, recent-activity bins
- [x] `SKILL.md` (Claude Code skill) teaches *when* to use each tool

### v2 known issues / migration notes

- **Foreign key drop in `chunks` schema.** v1 declared `chunks.note_path
  REFERENCES notes(path)`; DuckDB doesn't support UPDATE on FK-referenced
  rows, so the FK is dropped in v2. Existing v1 databases with the FK
  will fail on first re-index of an existing note. Workaround:
  `rm ~/.brain/db.duckdb && uv run brain index`. Vault is the truth; DB
  is rebuildable.

## v3 — retrieval polish

- [ ] Per-chunk SHA reuse: when a file changes, only re-embed chunks whose
      `content_hash` actually changed. Currently re-embeds the whole file.
- [ ] Turn on HNSW once vault > ~5k chunks: `INSTALL vss; CREATE INDEX ... USING HNSW`. Note WAL persistence caveat — keep markdown as truth.
- [ ] Better wikilink resolution: handle nested folders (e.g. `[[projects/cortex/spec]]`), case-sensitivity flag.
- [ ] Tag namespace queries: `tag = "project/cortex"` should also match `tag = "project/cortex/phase1"`.
- [ ] Filtered HNSW (ACORN-1 fork) if filtered queries become a bottleneck.

## v4 — cloud port

- [ ] CDK stack mirroring local arch:
  - S3 versioned bucket for markdown (BRONZE)
  - S3 ObjectCreated event → SQS → Lambda parser
  - Bedrock embed (Cohere multilingual? or stay with Titan v2)
  - S3 Vectors index (GOLD) — metadata filterable by tag/project/since
  - DynamoDB for graph triples (SILVER) — single-table design
  - API Gateway + Lambda for MCP Streamable HTTP transport
- [ ] CLI flag `brain serve --remote` that proxies to the cloud stack so the
      same Claude Code config works locally or against the cloud.

## Maybe / parking lot

- [ ] Obsidian plugin to surface brain hits inline while writing
- [ ] Multi-vault support (work vault vs personal vault, scoped per agent)
- [ ] Read-only "shared" mode — point brain at a teammate's published vault snapshot
- [ ] Backfill from existing `digital-brain` vault (read-only) into a separate index, exposed as a second MCP server
