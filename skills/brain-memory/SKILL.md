---
name: brain-memory
description: >
  Personal memory layer for the user — markdown vault at ~/brain-vault served via the
  brain MCP server (stdio). Seven tools: search_notes, read_note, list_recent,
  find_related, write_note, append_to_note, vault_report. Use whenever the user references past
  work, prior decisions, or knowledge that might be in the vault, OR when the user
  wants to capture something worth keeping across sessions. Triggers on phrases like
  "what did we decide about", "remember when", "find that note on", "what was I
  working on", "save this to my vault", "write this as a note", "add this under",
  "append to my note on", "log this in brain", or any continuity-across-sessions
  intent. The user edits this vault in Obsidian; provenance matters — the agent's
  writes must always be distinguishable from their own.
---

# Brain memory layer

This is the user's personal memory layer — a markdown vault at `~/brain-vault`
that the brain MCP server reads and writes. Same memory across Claude Code
(WSL), Claude Code (IDE), and Claude Desktop.

The vault is **mixed-authorship**: the user writes notes by hand in Obsidian,
and the agent writes notes via the MCP tools. The agent's writes are always
stamped with `author: agent` frontmatter and tagged `agent` so the user can
tell at a glance what came from them vs from us.

## Tools at your disposal

| Tool | When to use |
|------|-------------|
| `search_notes(query, k=5, tag=None)` | At the start of any session that references past work, or when the user asks "what did we…", "have I written about…", "find that note on…". Use natural-language queries — embeddings handle conceptual matches. The `score` in results is a fused RRF rank score by default; when reranking is enabled (`BRAIN_RERANK_ENABLED=true`, off by default) it is a cross-encoder logit instead: scores >+2 are strong, 0–1 are marginal, negative means the chunk doesn't actually answer the query. Either way, higher means more relevant — use it to gauge whether to trust a hit before quoting from it. |
| `read_note(path)` | After a `search_notes` hit, before answering. Excerpts in search results are not enough context. |
| `list_recent(n=10)` | When the user asks "what was I working on", "show me yesterday's notes", or to orient yourself early in a session. |
| `find_related(path, k=5)` | When the user wants to explore a topic — "what else is connected to this", "what should I read alongside". |
| `vault_report()` | When the user asks for a status/dashboard/summary of their vault — "give me a brain report", "how's the vault looking", "what have I been writing". Format the dict result as a readable report (markdown tables work well); don't dump it raw. |
| `write_note(path, body, frontmatter=None, tags=None)` | When the user explicitly says save/write/note this, or after a substantial decision/research that's worth preserving. Stamps `author: agent` automatically. **Refuses to overwrite** existing notes — use `append_to_note` for those. |
| `append_to_note(path, content, heading=None)` | When adding to an existing note — daily session log, ongoing topic note, or contributing to one of the user's hand-written notes. Routing rules below. |

## Read patterns

**Search at the top of any session that references past context.** If the
user says "let's continue with the brain roadmap" or "what did we decide
about embeddings", *first* call `search_notes`, *then* `read_note` on the
top hit, *then* answer. Don't answer from training-data assumptions when
the vault may have ground truth.

**Always `read_note` before quoting.** Search returns 400-char excerpts
that may cut mid-sentence. Don't paraphrase from an excerpt — read the
full note first.

## Write patterns

### Path conventions

Use the user's existing folder structure when possible. Browse with
`list_recent(20)` first if you're unsure where a note belongs.

Common roots:
- `research/<theme>.md` — research findings, deep-dives, comparisons
- `projects/<project>/<topic>.md` — project-scoped notes (specs, designs, refs)
- `projects/<project>/sessions/YYYY-MM-DD.md` — session log scoped to one project
- `sessions/YYYY-MM-DD.md` — cross-project / general session log (rare)
- `decisions/<topic>.md` — standalone architectural / product decisions
- `ideas/<idea>.md` — early-stage thoughts

**Default for session logs: per-project.** A session is almost always about one project. Use the flat `sessions/` only when the work genuinely spans projects (e.g. tooling/setup affecting many).

Don't invent new top-level folders without asking. Subdirectories are
created as needed.

### Provenance — the rule

**Every agent-created note** is auto-stamped:
```yaml
---
author: agent
created_at: <ISO timestamp>
tags: [agent, ...]
---
```

You don't add this manually — `write_note` does it. Don't override `author`
in the `frontmatter=` argument; it's enforced.

**When appending to a note the user wrote** (no `author: agent` in its
frontmatter) and you don't pass an explicit `heading`, the content lands
under `## Agent notes` with a `### YYYY-MM-DD HH:MM` timestamp subheading.
This protects their hand-written notes from being silently mutated — your
contribution sits in a clearly labeled section they can fold or delete.

**When appending to an agent-authored note** with no explicit `heading`,
the content goes straight at the end of the file (no marker — it's
already ours).

**When the user gives an explicit heading** (e.g. "add this under
## Decisions"), the routing above is bypassed and the content goes
under that heading.

### When to write_note vs append_to_note

- **`write_note`** — new file. Captures something genuinely new: a research
  result, a fresh decision, a topic that doesn't have a home yet.
- **`append_to_note`** — adds to an existing file. Logs a follow-up,
  extends a topic note, or contributes to one of the user's notes.

Quick rule: if the path you'd write to already exists, use
`append_to_note`. `write_note` will return an error if the path is taken
— that's intentional, to protect existing notes from being clobbered.

### Anti-patterns

- ❌ Don't `write_note` for a one-line scratch — append it to today's
  session log instead (`sessions/YYYY-MM-DD.md`).
- ❌ Don't write notes "just in case" — only when the content has clear
  future-session value. Ephemeral conversation belongs in conversation.
- ❌ Don't fabricate paths or topic structure. If unsure, search/list
  first or ask.
- ❌ Don't write a note then immediately read it back to "verify" —
  `write_note` returns `{"status": "written"}` on success. The round-trip
  is already proven by tests.

## Example flows

**User: "What did we decide about the storage layer for brain?"**
1. `search_notes("storage layer brain decision")` → top hit is `decisions/brain-storage.md`
2. `read_note("decisions/brain-storage.md")` → full content
3. Answer from the note's actual decision text, citing the path.

**User: "Save this conversation about LLM ranking research as a note under research/llm-ranking"**
1. Compose the body from the conversation.
2. `write_note("research/llm-ranking.md", body=..., tags=["research/llm"])` →
   gets stamped `author: agent`, tagged `agent` + `research/llm`.

**User: "Append our conclusion to projects/cortex/spec.md"**
1. `append_to_note("projects/cortex/spec.md", content=...)` →
   if the file is user-authored (which a spec usually is), the conclusion
   lands under `## Agent notes` with today's timestamp.

**User: "Add a 'Decisions' entry to today's session log"**
1. `append_to_note("sessions/2026-11-25.md", content=..., heading="## Decisions")` →
   explicit heading; goes straight under that section regardless of authorship.

## Operational notes

- The DuckDB index at `~/.brain/db.duckdb` is a derived artifact —
  rebuildable from the vault via `brain index`. Markdown is the truth.
- Writes synchronously re-index the changed file (~200ms after model warm-up).
  No need to wait or re-poll.
- The first tool call after server restart loads the embedder model
  (~30 MB, takes a beat). Subsequent calls are fast.
- If a tool returns `{"error": ...}` instead of the expected fields, the
  call failed for the reason given — don't retry without addressing it.
