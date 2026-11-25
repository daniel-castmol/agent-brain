# brain v3 — the whole-life vault: eval harness + multilingual retrieval

Date: 2026-11-25
Status: draft, pending approval

## Intent and why

The vision for brain is **one knowledge base for the user's whole
intellectual life** — projects, class notes, book notes, investigations —
co-written by human and agent, so the collaboration compounds across
months. Two things block that today:

1. **Embeddings are English-only** (`bge-small-en-v1.5`). The user's
   class/book notes include Spanish; semantic recall on them will be poor
   or random. The existing `digital-brain` Obsidian vault can't be
   meaningfully merged in until this is fixed.
2. **Retrieval quality is unmeasured.** There is no way to tell whether a
   model swap, chunk-size change, or the cross-encoder rerank makes search
   better or worse. Every future retrieval change is a guess.

v3 fixes both, in that order: build the measuring stick first, then use it
to choose the multilingual model, then migrate and backfill.

## Goals

1. **`brain eval`** — a retrieval evaluation harness run against the real
   index, driven by a user-curated gold set of queries → expected notes.
   Reports recall@5, recall@10, MRR@10, per-query hit/miss, and mean
   search latency.
2. **Multilingual embeddings** — swap the embedding model to one that
   handles English + Spanish, chosen by gold-set numbers, not vibes.
   Full re-index migration path (`brain index --rebuild`).
3. **Scale work pulled forward** (the vault gets much bigger after
   backfill):
   - Per-chunk SHA reuse: when a file changes, re-embed only chunks whose
     `content_hash` changed; reuse stored embeddings for matching hashes.
   - Persistent FTS index: rebuild only after writes (dirty flag), not on
     every query (today: `store.py` `fts_search` rebuilds per query).
   - FTS stemmer becomes configurable (`BRAIN_FTS_STEMMER`) for mixed
     English/Spanish content; default decided by eval numbers.
4. **Backfill the existing `digital-brain` vault** into `~/brain-vault`
   so there is one vault.

## Non-goals (v4+)

- Memory lifecycle: consolidation, decay, supersede semantics.
- HNSW / `vss` extension (brute-force cosine is fine ≤ ~100k chunks, and
  vss WAL persistence is still risky; `brain eval` latency numbers will
  tell us when this changes).
- Cloud port, Obsidian plugin, multi-vault scoping.
- Automatic memory extraction from conversations (brain stays editorial).

## Anti-goals

- Anything that makes the DuckDB index non-disposable. `rm db.duckdb &&
  brain index` must always fully recover the system.
- Any network dependency at query or index time (local-first is the
  product). Translating Spanish notes to English at index time was
  considered and rejected for this reason.
- New English-only assumptions anywhere.

## Constraints

- Models must be available through **fastembed** (ONNX, CPU). No
  PyTorch/sentence-transformers dependency.
- Keep `brain --help` instant (lazy imports).
- Markdown stays canonical; the gold set must not leak the user's private
  note paths into the public repo.
- Don't break the SHA256 incremental-sync flow or the heading-breadcrumb
  chunk prefix.

## Approaches considered

1. **Eval-first, data-driven model swap (chosen).** Build `brain eval` +
   gold set, benchmark 2–3 fastembed multilingual candidates on the
   user's own vault, pick the winner, migrate. Slightly slower to the
   finish line, but every later retrieval change inherits the harness,
   and the model choice is defensible in public (great Substack material).
2. **Straight swap to the strongest multilingual model (bge-m3, 1024d).**
   Fastest path, but ~2 GB model, slower CPU inference, a schema dim
   change, and no evidence it actually helps on this vault. Rejected as
   flying blind.
3. **Keep English embeddings, translate Spanish content at index time.**
   Adds an LLM/network dependency and a shadow copy of the vault's
   meaning. Violates local-first and markdown-as-truth. Rejected.

## Design

### Part 1 — eval harness (build first)

**Gold set.** YAML file, default location `<vault>/_evals/gold.yaml`
(inside the vault: private by default, versioned with the user's own
vault backups; the indexer ignores non-`.md` files). The repo ships a
small example gold set under `evals/example-gold.yaml` keyed to
`vault-seed/`, used by an integration test. Entry shape:

```yaml
- query: "what did we decide about the storage layer"
  expect:                 # hit = any of these paths in top-k
    - decisions/brain-storage.md
  tag: null               # optional tag filter passed to search
  lang: en                # informational; lets eval report per-language
```

**CLI.** `brain eval [--gold PATH] [--k 5] [--json]`:
- Runs the full production search path (`hybrid_search`, plus rerank when
  `BRAIN_RERANK_ENABLED` is set) per query.
- Metrics: recall@5, recall@10, MRR@10, per-language breakdown, mean and
  p95 query latency. All metrics are **note-level over distinct notes**:
  chunk hits are deduplicated to their note path first, and recall@k is
  computed over the top-k *distinct notes* (so five chunks from two notes
  count as two results, not five).
- Human output: rich table with per-query ✓/✗ and the rank of the first
  correct hit. `--json` emits machine-readable results so two runs (e.g.
  different `BRAIN_EMBEDDING_MODEL`) can be diffed.
- Pure metric functions live in a new `src/brain/evaluate.py` with unit
  tests; the module name avoids colliding with the `evals/` data dir.

**Target gold set size:** start with ~20 queries written by the user
(mix: project decisions, class topics, book notes; at least ⅓ Spanish
once backfill lands). Growing it is an ongoing habit, not a one-time task.

### Part 2 — multilingual model selection + migration

**Candidates** (final list = whatever current fastembed supports at
implementation time; verify with `TextEmbedding.list_supported_models()`):

| Model | Dim | Size | Notes |
|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~470 MB | Same dim as today — no schema change |
| `intfloat/multilingual-e5-large` (or `-small` if supported) | 1024 | ~2.2 GB | Needs `query:`/`passage:` prefixes |
| `BAAI/bge-m3` (if supported) | 1024 | ~2.2 GB | Strongest on paper |

**Selection procedure:** the bake-off runs **after** the digital-brain
backfill (see build order), so the vault and the gold set are genuinely
bilingual when the model is chosen. Index the merged vault once per
candidate (separate `BRAIN_DB_FILENAME` per run), run `brain eval --json`
on each, compare recall@5 / MRR / latency. Decision rule: smallest model
within 2 points of the best recall@5 wins. The losing numbers go in the
Substack post.

**Reranker is out of the bake-off.** The default cross-encoder
(`Xenova/ms-marco-MiniLM-L-6-v2`) is English-only MS MARCO; with rerank
on it could bury Spanish hits and make the embedder look guilty. So:
rerank stays disabled during model selection, and v3 documents the
default reranker as English-only. After the embedder is chosen, eval is
re-run with rerank on vs. off; if fastembed supports a multilingual
cross-encoder (e.g. `BAAI/bge-reranker-v2-m3`), evaluating it the same
way is an optional follow-up, not a v3 commitment.

**Prefix support.** Some models (e5 family) need asymmetric prefixes.
`embedder.py` gains `BRAIN_QUERY_PREFIX` / `BRAIN_PASSAGE_PREFIX`
settings (default empty). fastembed already applies bge's query
instruction in `query_embed`; verify per candidate and only add our own
prefixes where fastembed doesn't.

**Migration.** New flag `brain index --rebuild`: drops `notes`, `chunks`,
`links` (and the FTS index), recreates the schema at the configured dim,
then runs a full index. This replaces "rm the db file by hand" in the
docs. README + INSTALL get a short "switching embedding models" section.

**Embedding-config guard (new `meta` table).** The dim-validation check
in `embedder.py` is not enough: a same-dim model swap (e.g. to
multilingual-MiniLM, also 384d) or a prefix change would silently mix
embedding spaces — and per-chunk reuse (Part 3) would make the stale
vectors immortal. So the store records the active embedding config
(model name, dim, query/passage prefixes) in a one-row `meta` table,
written at index time. On connect, a mismatch between `meta` and the
current settings is a **hard error** telling the user to run
`brain index --rebuild`. Per-chunk embedding reuse is only valid when
`meta` matches, which the guard guarantees.

**FTS stemmer.** New setting `BRAIN_FTS_STEMMER` (default `porter`,
documented values incl. `spanish`, `none`). Passed into
`create_fts_index`. Whether the default changes for mixed vaults is
decided by the eval numbers, not guessed.

### Part 3 — scale work

- **Per-chunk embedding reuse.** In `indexer`, when a file's note-hash
  changed: chunk it, then `SELECT content_hash, embedding FROM chunks
  WHERE note_path = ?`, embed only chunks whose hash isn't in that map,
  and reuse stored vectors for the rest. (Chunk content includes the
  heading breadcrumb, so the hash already captures heading moves.)
- **FTS persistence.** `Store` keeps a dirty flag: set on any chunk
  write/delete, checked in `fts_search`; the index is (re)built only when
  dirty or missing. Startup treats the index as dirty once, to stay
  correct after an external process modified the db.
- No change to brute-force cosine. `brain eval` latency output is the
  tripwire for revisiting HNSW.

### Part 4 — backfill `digital-brain`

Copy (never move) the old vault's markdown into `~/brain-vault/`,
preserving mtimes (`cp -rp`) so `list_recent` stays honest. Non-markdown
assets (images, PDFs) are ignored by the indexer; broken Obsidian-isms
(dataview blocks, callouts) index as plain text, which is acceptable.
Open questions for the user below (location, target folder layout,
whether anything stays out). After backfill: `brain index --rebuild`
with the chosen model, then re-run `brain eval`.

## Known limitation to document (not fix) in v3

DuckDB allows one writer process. Running `brain index --watch` while the
MCP server is serving (or two MCP clients at once) can hit lock errors.
v3 documents the rule — "either `--watch` or `serve`, not both; external
edits get picked up by the next `brain index` run" — in README
troubleshooting. A proper fix (single writer daemon or read-only server
connections with write hand-off) is v4 territory.

## Success criteria

- `brain eval` runs the full gold set against the merged vault in < 60 s
  and the numbers are reproducible across runs.
- A Spanish-language query retrieves the expected Spanish note in top-5
  (measured, not anecdotal) after the model swap.
- Per-chunk reuse: editing one section of a large note re-embeds only the
  changed chunks (observable in `IndexStats`, which gains
  `chunks_reused`).
- FTS no longer rebuilds per query; repeated `search_notes` calls show it.
- All existing unit tests stay green; new pure functions (metrics, chunk
  diffing) get unit tests; model-dependent paths get `integration`-marked
  tests.

## What's left to the builder

- Exact YAML schema details and eval output formatting.
- How `--rebuild` confirms before dropping (flag `--yes` vs. interactive
  prompt in TTY).
- Whether reuse of per-chunk embeddings happens via content_hash lookup
  per note (described above) or a global hash→embedding cache table.
- Internal structure of `evaluate.py`.

## Build order

Backfill comes **before** the bake-off — the model must be chosen on the
bilingual vault it will actually serve, with Spanish queries in the gold
set.

1. Eval harness + example gold set + unit tests (pure metrics).
2. FTS dirty-flag persistence (small, immediately measurable win).
3. Embedding-config `meta` table + guard, `--rebuild`, prefix settings,
   stemmer setting.
4. Per-chunk SHA reuse (+ `chunks_reused` stat) — lands after the guard,
   so stale vectors can never survive a model/prefix change.
5. Backfill digital-brain into the vault; index with the current English
   model.
6. User writes the real gold set (~20 queries, ≥⅓ Spanish); baseline
   numbers on the current English model recorded — this is the "before"
   picture.
7. Model bake-off (rerank off), pick winner, flip config defaults/docs,
   final `--rebuild`.
8. Re-run eval with the winner (and rerank on/off); record final
   numbers — the before/after for the Substack post.
