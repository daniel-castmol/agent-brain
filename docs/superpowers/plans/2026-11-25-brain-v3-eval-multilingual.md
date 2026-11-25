# brain v3 — Eval Harness + Multilingual Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measurable retrieval quality (`brain eval`), safe embedding-model migration (meta-table guard + `--rebuild`), and the scale/multilingual plumbing (FTS dirty flag, per-chunk embedding reuse, prefix + stemmer settings).

**Architecture:** A new pure-function module `evaluate.py` computes note-level retrieval metrics from a YAML gold set and is wired into the CLI. `store.py` gains a one-row embedding-config `meta` table whose mismatch is a hard error, a dirty-flag-gated FTS index, and a hash→embedding lookup that lets the indexer skip re-embedding unchanged chunks. `config.py`/`embedder.py` gain query/passage prefixes and an FTS stemmer setting.

**Tech Stack:** Python 3.11, DuckDB (fts extension), fastembed, typer/rich, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-brain-v3-multilingual-eval-design.md`

## Global Constraints

- Markdown is canonical; the DuckDB index must stay fully rebuildable (`rm db.duckdb` or `--rebuild` + `brain index` recovers everything).
- Never import `fastembed` at module top level outside `embedder.py`; `brain --help` stays instant.
- Don't break the SHA256 incremental flow or the heading-breadcrumb chunk prefix.
- `uv run pytest -m "not integration"` must pass before every commit; tests that load real models get `@pytest.mark.integration`.
- No network calls at query/index time beyond fastembed's one-time model download.
- The model bake-off itself (spec build-order steps 6–8) is operational, not code: this plan ends when the tooling for it exists.

## Review Focus

1. Missing or malformed gold YAML → `brain eval` exits with a one-line error naming the file, not a traceback. (Test in Task 2.)
2. A gold entry expecting a path absent from the index → counts as a miss; the run completes. (Test in Task 2.)
3. Empty vault / zero chunks → `fts_search` and `brain eval` return empty results, no crash on FTS index creation. (Test in Task 3.)
4. Pre-v3 database file (no `meta` rows) → first `verify_embedding_config` stamps the current config instead of raising. (Test in Task 4.)
5. Same-dim model swap or prefix change without `--rebuild` → hard error with a message naming `brain index --rebuild`. (Tests in Task 4.)

---

### Task 1: Eval metrics — pure functions

**Files:**
- Create: `src/brain/evaluate.py`
- Create: `tests/test_evaluate.py`
- Modify: `pyproject.toml` (add `pyyaml>=6.0` to dependencies)

**Interfaces:**
- Consumes: `SearchHit` from `brain.store` (fields: `note_path, title, chunk_index, heading_path, body, score`).
- Produces (all in `brain/evaluate.py`):
  - `@dataclass GoldQuery: query: str; expect: list[str]; tag: str | None = None; lang: str = "en"`
  - `load_gold(path: Path) -> list[GoldQuery]` — raises `GoldFileError(Exception)` on missing file, invalid YAML, or an entry without `query`/non-empty `expect`.
  - `dedupe_to_notes(hits: list[SearchHit]) -> list[str]` — ordered distinct `note_path`s.
  - `hit_at_k(ranked_notes: list[str], expect: list[str], k: int) -> bool` — any expected path within the first k distinct notes.
  - `reciprocal_rank(ranked_notes: list[str], expect: list[str]) -> float` — 1/rank of first expected note (1-based), 0.0 if absent.
  - `aggregate(per_query: list[dict]) -> dict` — takes dicts with keys `lang, hit_at_5, hit_at_10, rr, latency_ms`; returns `{"recall_at_5", "recall_at_10", "mrr_at_10", "n", "latency_mean_ms", "latency_p95_ms", "by_lang": {lang: {...same metrics...}}}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py — representative assertions (cover each function)
def test_dedupe_to_notes_preserves_first_occurrence_order():
    hits = [_hit("a.md"), _hit("b.md"), _hit("a.md"), _hit("c.md")]
    assert dedupe_to_notes(hits) == ["a.md", "b.md", "c.md"]

def test_hit_at_k_counts_distinct_notes_not_chunks():
    assert hit_at_k(["a.md", "b.md", "c.md"], expect=["c.md"], k=3) is True
    assert hit_at_k(["a.md", "b.md", "c.md"], expect=["c.md"], k=2) is False

def test_reciprocal_rank():
    assert reciprocal_rank(["a.md", "b.md"], expect=["b.md"]) == 0.5
    assert reciprocal_rank(["a.md"], expect=["z.md"]) == 0.0

def test_aggregate_overall_and_by_lang():
    rows = [
        {"lang": "en", "hit_at_5": True, "hit_at_10": True, "rr": 1.0, "latency_ms": 10.0},
        {"lang": "es", "hit_at_5": False, "hit_at_10": True, "rr": 0.125, "latency_ms": 30.0},
    ]
    agg = aggregate(rows)
    assert agg["recall_at_5"] == 0.5 and agg["n"] == 2
    assert agg["by_lang"]["es"]["recall_at_5"] == 0.0

def test_load_gold_parses_entries(tmp_path):  # writes a 2-entry yaml, checks fields + lang default "en"
def test_load_gold_raises_on_missing_file_and_bad_entry(tmp_path):  # pytest.raises(GoldFileError) ×2
```

`_hit(path)` is a local helper constructing `SearchHit(note_path=path, title="", chunk_index=0, heading_path="", body="", score=0.0)`.

- [ ] **Step 2: Run tests, verify they fail** — `uv run pytest tests/test_evaluate.py -q`, expected: import error / failures.
- [ ] **Step 3: Implement `src/brain/evaluate.py`** (pure functions above; `import yaml` inside `load_gold` is fine — PyYAML is light). Add `pyyaml>=6.0` to `pyproject.toml` deps, run `uv sync`.
- [ ] **Step 4: Run tests, verify pass** — `uv run pytest tests/test_evaluate.py -q`.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: eval metrics module (note-level recall/MRR, gold-set loader)"`

### Task 2: Eval runner + `brain eval` CLI + example gold set

**Files:**
- Modify: `src/brain/evaluate.py` (add runner)
- Modify: `src/brain/cli.py` (add `eval` command)
- Create: `evals/example-gold.yaml`
- Test: `tests/test_evaluate.py` (extend)

**Interfaces:**
- Consumes: `Store.hybrid_search(query_embedding, query_text, k, rrf_k, tag)`, `Embedder.embed_query(text)`, Task 1 functions, `settings` (`rerank_enabled`, `rerank_candidate_pool`, `rrf_k`), `rerank_candidates`/`make_reranker` from `brain.reranker` (same wiring as `server.search_notes`).
- Produces: `run_eval(store, embedder, gold: list[GoldQuery], k_values: tuple[int, int] = (5, 10)) -> dict` — per query: fetch 50 chunk hits via the production search path (rerank applied when `settings.rerank_enabled`), `dedupe_to_notes`, score with Task 1 functions, time each query with `time.perf_counter()`; returns `{"summary": aggregate(...), "queries": [{query, lang, first_hit_rank, hit_at_5, hit_at_10, latency_ms}]}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_eval_scores_against_seeded_store(tmp_path):
    # Seed store via test_report.py's direct-SQL pattern: two notes whose chunk
    # embeddings are one-hot vectors; FakeEmbedder.embed_query returns the
    # one-hot vector matching the note named in the query.
    res = run_eval(store, FakeEmbedder(), [GoldQuery(query="alpha", expect=["alpha.md"]),
                                           GoldQuery(query="alpha", expect=["missing.md"])])
    assert res["summary"]["recall_at_5"] == 0.5          # Review Focus 2: absent path = miss, run completes
    assert res["queries"][0]["first_hit_rank"] == 1

def test_cli_eval_errors_cleanly_on_missing_gold(tmp_path):
    # typer.testing.CliRunner: brain eval --gold /nonexistent.yaml
    # exit_code != 0, output contains the path, no traceback
```

- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement `run_eval` in `evaluate.py`** (lazy-import reranker bits only when `settings.rerank_enabled`).
- [ ] **Step 4: Implement `brain eval` in `cli.py`:** `eval(gold: Path = typer.Option(settings.vault_dir / "_evals" / "gold.yaml", "--gold"), raw: bool = typer.Option(False, "--json"))` (no `--k`: the report always covers @5 and @10). Catches `GoldFileError` → `console.print` + `raise typer.Exit(1)`. Human output: rich table (query, lang, first-hit rank or ✗, latency) + summary block; `--json` prints `run_eval`'s dict. Reuses `_make_store`/`_make_embedder`.
- [ ] **Step 5: Write `evals/example-gold.yaml`:** 3 entries keyed to `vault-seed/` notes (e.g. query "why duckdb over sqlite" → expect `00-brain-design.md`), with a header comment explaining the format and that the real gold set lives at `<vault>/_evals/gold.yaml`.
- [ ] **Step 6: Run full suite** — `uv run pytest -m "not integration" -q`, all pass.
- [ ] **Step 7: Commit** — `git commit -m "feat: brain eval command with gold-set runner and example gold set"`

### Task 3: FTS dirty flag — stop rebuilding the index per query

**Files:**
- Modify: `src/brain/store.py`
- Test: `tests/test_store_fts.py` (create)

**Interfaces:**
- Produces: `Store._fts_dirty: bool` (initialized `True` at construction so a fresh connection always rebuilds once — covers external writers); set to `True` by `replace_chunks`, `delete_chunks_for`, `delete_note`; `fts_search` runs `PRAGMA create_fts_index(..., overwrite=1)` only when `self._fts_dirty`, then sets it `False`. Public behavior of `fts_search`/`hybrid_search` unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_fts_search_clears_dirty_flag_and_skips_rebuild(store):
    _insert_note(store, path="a.md", chunks_with_headings=["H"])  # reuse test_report pattern
    store.fts_search("body", k=5)
    assert store._fts_dirty is False
    store.fts_search("body", k=5)                   # second call: no PRAGMA (flag still False)

def test_write_marks_dirty_and_new_content_searchable(store):
    ...; store.fts_search("body", 5); store.replace_chunks("a.md", [chunk], [[0.0]*384])
    assert store._fts_dirty is True
    hits = store.fts_search("<new chunk term>", 5)   # finds content added after first search
    assert hits and hits[0].note_path == "a.md"

def test_fts_search_on_empty_store_returns_empty(store):   # Review Focus 3
    assert store.fts_search("anything", 5) == []
```

- [ ] **Step 2: Run, verify fail** (the dirty-flag asserts fail against current per-query rebuild).
- [ ] **Step 3: Implement** in `store.py`; drop the now-redundant try/except around the PRAGMA only if the empty-store test still passes.
- [ ] **Step 4: Run full suite, pass.**
- [ ] **Step 5: Commit** — `git commit -m "perf: rebuild FTS index only after writes (dirty flag)"`

### Task 4: Embedding-config meta table, guard, and `brain index --rebuild`

**Files:**
- Modify: `src/brain/store.py` (schema + methods), `src/brain/cli.py` (`--rebuild` flag, guard call), `src/brain/server.py` (guard call in `_get_store`)
- Test: `tests/test_store_meta.py` (create)

**Interfaces:**
- Produces (in `brain.store`):
  - `SCHEMA_SQL` gains `CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR);`
  - `class EmbeddingConfigMismatchError(Exception)` — message must name the differing field(s), both values, and the fix: `brain index --rebuild`.
  - `Store.verify_embedding_config(model: str, dim: int, query_prefix: str, passage_prefix: str) -> None` — reads the `embedding_config` key (JSON value); absent → writes current config (Review Focus 4); present & equal → no-op; present & different → raises.
  - `Store.rebuild_schema() -> None` — `DROP TABLE IF EXISTS` chunks, links, notes, meta; `DROP SEQUENCE IF EXISTS chunk_id_seq`; re-execute `SCHEMA_SQL`; set `_fts_dirty = True`.
- Consumed by: `cli._make_store` and `server._get_store` call `verify_embedding_config(settings.embedding_model, settings.embedding_dim, settings.query_prefix, settings.passage_prefix)` right after `connect()` (until Task 5 lands, pass `""` for both prefixes); `brain index --rebuild [--yes]` calls `rebuild_schema()` first (`typer.confirm` unless `--yes`), then indexes.

- [ ] **Step 1: Write the failing tests**

```python
def test_verify_stamps_config_when_absent(store):          # fresh db → no raise, config stored
def test_verify_noop_when_config_matches(store):
def test_verify_raises_on_model_change_same_dim(store):    # Review Focus 5: bge-small → MiniLM, both 384
    with pytest.raises(EmbeddingConfigMismatchError, match="--rebuild"):
def test_verify_raises_on_prefix_change(store):            # Review Focus 5
def test_rebuild_schema_drops_everything_and_recreates(store):
    # insert a note, rebuild, counts are 0, verify_embedding_config stamps fresh
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement store methods + wire guard into `cli._make_store` and `server._get_store` + add `--rebuild`/`--yes` options to `cli.index`.**
- [ ] **Step 4: Run full suite, pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: embedding-config guard (meta table) and brain index --rebuild"`

### Task 5: Prefix and stemmer settings

**Files:**
- Modify: `src/brain/config.py`, `src/brain/embedder.py`, `src/brain/store.py`, `src/brain/indexer.py`, `src/brain/cli.py`, `src/brain/server.py`, `.env.example`
- Test: `tests/test_embedder_prefixes.py` (create), `tests/test_store_fts.py` (extend)

**Interfaces:**
- `Settings` gains: `query_prefix: str = ""`, `passage_prefix: str = ""`, `fts_stemmer: str = "porter"` (env: `BRAIN_QUERY_PREFIX`, `BRAIN_PASSAGE_PREFIX`, `BRAIN_FTS_STEMMER`).
- `Embedder.__init__` gains `query_prefix: str = ""`, `passage_prefix: str = ""`; `make_embedder(model_name, expected_dim, query_prefix="", passage_prefix="")` passes them through. New method `embed_passages(texts: list[str]) -> list[list[float]]` = `embed_batch` with `passage_prefix` prepended to each text; `embed_query` prepends `query_prefix`. `embed_batch` stays raw (hashes in `chunker` are unaffected — prefixes are applied at embed time only, never stored in `chunk.content`).
- `Store.__init__` gains `fts_stemmer: str = "porter"`; the FTS PRAGMA becomes `create_fts_index('chunks', 'id', 'body', 'heading_path', stemmer='<value>', overwrite=1)`.
- Call-site updates: `indexer.index_vault`/`index_single_file` switch `embedder.embed_batch([c.content ...])` → `embedder.embed_passages(...)`; `cli._make_store`/`_make_embedder` and `server._get_store`/`_get_embedder` pass the new settings; Task 4's guard calls drop the `""` placeholders.

- [ ] **Step 1: Write the failing tests**

```python
def test_embed_query_and_passages_apply_prefixes():
    e = Embedder("x", 3, query_prefix="query: ", passage_prefix="passage: ")
    e._model = FakeModel()            # .embed(texts, batch_size=...) records texts, yields np.array([0.,0.,0.]) each
    e.embed_query("hola"); e.embed_passages(["a", "b"]); e.embed_batch(["raw"])
    assert FakeModel.seen == [["query: hola"], ["passage: a", "passage: b"], ["raw"]]

def test_fts_stemmer_none_matches_exact_spanish_token(store_with_stemmer_none):
    # chunk body "configuración de embeddings"; fts_search("configuración") hits it
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** config fields, embedder changes, store stemmer parameter, all call-site updates, and `.env.example` entries (document that changing model or prefixes requires `brain index --rebuild`, and `spanish`/`none` as stemmer values for mixed vaults).
- [ ] **Step 4: Run full suite, pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: query/passage prefix support and configurable FTS stemmer"`

### Task 6: Per-chunk embedding reuse

**Files:**
- Modify: `src/brain/store.py`, `src/brain/indexer.py`
- Test: `tests/test_indexer_reuse.py` (create)

**Interfaces:**
- `Store.get_chunk_embeddings(note_path: str) -> dict[str, list[float]]` — `content_hash → embedding` for the note's current chunks.
- `IndexStats` gains `chunks_reused: int = 0`; the `cli.index` summary line prints it.
- In `indexer`, new helper `_embed_with_reuse(store, note_path, chunks, embedder) -> tuple[list[list[float]], int]`: fetch the existing hash→embedding map **before** `delete_chunks_for`, call `embedder.embed_passages` only for chunks whose `content_hash` isn't in the map, return (embeddings aligned with `chunks`, reused count). Used by both `index_vault` and `index_single_file`. Safe because Task 4's guard guarantees stored vectors match the active embedding config.

- [ ] **Step 1: Write the failing test**

```python
def test_changed_file_reembeds_only_changed_chunks(tmp_path):
    # Real parser/chunker on a tmp vault; CountingFakeEmbedder returns [hash-derived]*384
    # and counts texts passed to embed_passages.
    # 1) write note with sections A, B, C; index_vault → 3 embedded
    # 2) edit only section B's text; index_vault again
    stats = index_vault(settings, fake_embedder, store)
    assert fake_embedder.embedded_count == 1          # only B
    assert stats.chunks_reused == 2                   # A and C reused
    # reused vectors survive: SELECT embedding for A's hash unchanged
```

- [ ] **Step 2: Run, verify fail** (current code re-embeds all 3).
- [ ] **Step 3: Implement** `get_chunk_embeddings`, `_embed_with_reuse`, wire into both index paths, extend `IndexStats` + CLI summary line.
- [ ] **Step 4: Run full suite, pass.**
- [ ] **Step 5: Commit** — `git commit -m "perf: reuse stored embeddings for unchanged chunks on re-index"`

### Task 7: Docs — backup, single-writer rule, model switching, eval

**Files:**
- Modify: `README.md`, `INSTALL.md`, `CLAUDE.md`, `tasks/todo.md`

**Interfaces:** none (prose only). Content each doc must gain:

- [ ] **Step 1: README** — new "Back up your vault" section (private `git init` in `~/brain-vault` from day one; the DuckDB index stays out of git — disposable); "Switching embedding models" section (`BRAIN_EMBEDDING_MODEL` + `BRAIN_EMBEDDING_DIM` + prefixes, then `brain index --rebuild`); `brain eval` added to the MCP-tools/CLI documentation with a 3-line gold.yaml example; roadmap section updated to v3 status.
- [ ] **Step 2: INSTALL troubleshooting** — "database is locked" entry: one writer at a time — don't run `index --watch` while the MCP server is in use; run `brain index` after Obsidian editing sessions instead.
- [ ] **Step 3: CLAUDE.md** — quick-reference gains `brain eval` and `--rebuild`; new env vars listed; layout line for `evaluate.py`. `tasks/todo.md`: tick the shipped v3 items, note bake-off is gated on vault content.
- [ ] **Step 4: Run full suite one final time** — `uv run pytest -m "not integration" -q`, all pass.
- [ ] **Step 5: Commit** — `git commit -m "docs: v3 — eval, rebuild, vault backup, single-writer guidance"`
