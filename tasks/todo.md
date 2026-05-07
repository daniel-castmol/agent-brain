# brain — todo

## Now

- [ ] Wire brain into Claude Code: `claude mcp add --scope user --transport stdio brain -- uv --directory <abs> run brain serve`
- [ ] Verify with `claude mcp list` and `/mcp` inside a session — should show `brain: connected`
- [ ] Drop the seed design note (`vault-seed/00-brain-design.md`) into `~/brain-vault/` and re-index, so brain has something real to search
- [ ] Sanity-check inside Claude Code: ask "what did we decide about the storage layer for brain?" — should call `search_notes`

## v2 — write-back

- [ ] `write_note(path, frontmatter, body)` MCP tool
  - Default destination `<vault>/_agent/sessions/YYYY-MM-DD.md` so user-curated content stays clean
  - Validates path is inside vault, refuses traversal
  - Triggers re-index of just that file
- [ ] `append_to_note(path, content, heading=None)` MCP tool
  - Appends under a specific heading or at end of file
- [ ] Ship a `SKILL.md` (Claude Code skill) that teaches *when* to use each tool
  - Search at the start of any session referencing past work
  - Always `read_note` after a search hit before answering — excerpts aren't full context
  - Append to today's session log when concluding a non-trivial decision
- [ ] Decide on session-log format. Probably:
  ```
  ---
  date: 2026-05-06
  tags: [session]
  ---
  # Session: <topic>
  ## Context
  ## Decisions
  ## Next
  ```

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
