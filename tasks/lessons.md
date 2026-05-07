# brain — lessons

Append-only log. Each entry: short title, then 1–3 sentences of what was
learned and why it mattered. Newest at top.

---

## 2026-05-06 — Embedding dim is load-bearing config

`bge-small-en-v1.5` returns 384-dim vectors. The DuckDB column type
`FLOAT[384]` is fixed at table-creation time, so changing models means
either dropping the table or running a migration. `embedder.py` validates
this at runtime and raises if `BRAIN_EMBEDDING_DIM` disagrees with the
model — keep that guard in place.

## 2026-05-06 — DuckDB VSS persistence is experimental

The `vss` extension's HNSW index has known issues with WAL recovery on
unexpected shutdown — corruption / data loss possible. We deliberately
shipped v1 without HNSW (brute-force cosine works at vault scale anyway)
and kept markdown as the source of truth so any DB corruption is
recoverable by re-running `brain index`.

## 2026-05-06 — Heading breadcrumb in chunk content matters

Each chunk gets `# Path > To > Section\n\n` prepended before the body
text gets embedded. This was a deliberate quality choice — the embedding
model needs context to disambiguate "Storage" under "Architecture" from
"Storage" under "Personal Finance". Don't strip it during refactors.

## 2026-05-06 — Lazy-load fastembed

Importing `fastembed` at module top level triggers the ONNX runtime load
(~60MB model on disk, takes a beat). `brain --help` and `brain status`
should be instant, so `embedder.py` only imports inside `_ensure_loaded()`.
Same pattern when adding any new heavy dependency.

## 2026-05-06 — Wikilink resolution is two-pass

Can't resolve `[[Note Name]]` to a vault path during the first parse pass
because the target file might not have been parsed yet. `indexer.py`
collects raw wikilink targets per file in pass 1, then resolves against
the full title→path map in pass 2. Don't try to inline this — it'll break
on circular references and forward links.
