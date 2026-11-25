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

## v3 — eval harness + multilingual readiness ✅ shipped (code)

Spec: `docs/superpowers/specs/2026-11-25-brain-v3-multilingual-eval-design.md`

- [x] `brain eval` — gold-set harness (note-level recall@5/@10, MRR@10,
      per-language breakdown, latency); example at `evals/example-gold.yaml`
- [x] Embedding-config guard: `meta` table stamps model/dim/prefixes;
      mismatch → hard error pointing at `brain index --rebuild`
- [x] `brain index --rebuild` — sanctioned migration path for model swaps
- [x] Per-chunk SHA reuse: only changed chunks get re-embedded
      (`chunks_reused` in index stats)
- [x] FTS index rebuilt only after writes (dirty flag), not per query
- [x] `BRAIN_QUERY_PREFIX` / `BRAIN_PASSAGE_PREFIX` (e5-style models) and
      `BRAIN_FTS_STEMMER` (spanish/none for mixed vaults)

### v3 — operational steps (gated on vault content)

- [ ] Rebuild the vault through normal use; grow `<vault>/_evals/gold.yaml`
      alongside it (~20 queries, ≥⅓ Spanish)
- [ ] Baseline eval on bge-small-en-v1.5, then the multilingual bake-off
      (rerank off), pick winner by recall@5, flip defaults, `--rebuild`
- [ ] Re-run eval with winner; record before/after numbers (Substack)

## Parked retrieval polish

- [ ] Turn on HNSW once vault > ~5k chunks: `INSTALL vss; CREATE INDEX ... USING HNSW`. Note WAL persistence caveat — keep markdown as truth.
- [ ] Better wikilink resolution: handle nested folders (e.g. `[[projects/cortex/spec]]`), case-sensitivity flag.
- [ ] Tag namespace queries: `tag = "project/cortex"` should also match `tag = "project/cortex/phase1"`.
- [ ] Filtered HNSW (ACORN-1 fork) if filtered queries become a bottleneck.
- [ ] Multilingual cross-encoder reranker (current default is English-only MS MARCO)

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
