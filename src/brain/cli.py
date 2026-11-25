"""brain CLI.

Commands:
    brain init                  Create the vault directory and DB.
    brain index                 One-shot incremental sync.
    brain index --watch         Sync, then keep watching for changes.
    brain serve                 Run the MCP server (stdio).
    brain search "<query>"      Debug search from the command line.
    brain status                Show counts and config.
    brain report                Show a vault health report.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from brain.config import settings
from brain.embedder import make_embedder
from brain.indexer import index_vault
from brain.report import build_report
from brain.store import EmbeddingConfigMismatchError, Store

app = typer.Typer(no_args_is_help=True, help="Local markdown memory layer for AI agents.")
console = Console()


def _make_store(verify: bool = True) -> Store:
    settings.ensure_dirs()
    store = Store(
        db_path=settings.db_path,
        embedding_dim=settings.embedding_dim,
        fts_stemmer=settings.fts_stemmer,
    )
    store.connect()
    if verify:
        try:
            store.verify_embedding_config(
                settings.embedding_model,
                settings.embedding_dim,
                settings.query_prefix,
                settings.passage_prefix,
            )
        except EmbeddingConfigMismatchError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1) from e
    return store


def _make_embedder():
    return make_embedder(
        model_name=settings.embedding_model,
        expected_dim=settings.embedding_dim,
        query_prefix=settings.query_prefix,
        passage_prefix=settings.passage_prefix,
    )


@app.command()
def init() -> None:
    """Create the vault and DB directories, write a starter README in the vault."""
    settings.ensure_dirs()
    starter = settings.vault_dir / "README.md"
    if not starter.exists():
        starter.write_text(
            "---\n"
            "tags: [meta]\n"
            "---\n\n"
            "# Brain Vault\n\n"
            "This is the agent's memory vault. Drop markdown notes anywhere\n"
            "under this directory and run `brain index` to ingest them.\n\n"
            "## Conventions\n\n"
            "- Use frontmatter `tags` for filterable metadata.\n"
            "- Use `[[wikilinks]]` to connect notes.\n"
            "- Headings (`##`, `###`) define chunk boundaries.\n",
            encoding="utf-8",
        )
    console.print(f"[green]✓[/green] Vault: {settings.vault_dir}")
    console.print(f"[green]✓[/green] DB:    {settings.db_path}")


@app.command()
def index(
    watch: bool = typer.Option(False, "--watch", "-w", help="Keep watching after initial sync."),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Drop the index and re-embed the whole vault (required after changing the embedding model or prefixes).",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the --rebuild confirmation."),
) -> None:
    """Sync the vault into DuckDB. Add --watch to keep it live."""
    if rebuild:
        store = _make_store(verify=False)
        if not yes:
            typer.confirm(
                f"Drop the index at {settings.db_path} and re-embed the whole vault?",
                abort=True,
            )
        store.rebuild_schema()
        store.verify_embedding_config(
            settings.embedding_model,
            settings.embedding_dim,
            settings.query_prefix,
            settings.passage_prefix,
        )
        console.print("[yellow]Index dropped — re-embedding from scratch.[/yellow]")
    else:
        store = _make_store()
    embedder = _make_embedder()

    def _run_once() -> None:
        with console.status("[cyan]Indexing vault…"):
            stats = index_vault(settings, embedder, store)
        console.print(
            f"[green]✓[/green] scanned={stats.scanned} new={stats.new} "
            f"updated={stats.updated} unchanged={stats.unchanged} "
            f"deleted={stats.deleted} chunks_embedded={stats.chunks_embedded}"
        )

    _run_once()
    if not watch:
        return

    console.print(f"[cyan]Watching {settings.vault_dir} for changes (Ctrl-C to stop)…")

    class _Handler(FileSystemEventHandler):
        def __init__(self):
            self._dirty = False
            self._last_event = 0.0

        def on_any_event(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return
            if not str(event.src_path).endswith(".md"):
                return
            self._dirty = True
            self._last_event = time.monotonic()

    handler = _Handler()
    observer = Observer()
    observer.schedule(handler, str(settings.vault_dir), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1.0)
            # Debounce: re-index ~2s after the last filesystem event.
            if handler._dirty and (time.monotonic() - handler._last_event) > 2.0:
                handler._dirty = False
                _run_once()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


@app.command()
def serve() -> None:
    """Run the MCP server over stdio. Configure your MCP client to launch this."""
    # Imported here so `brain --help` doesn't pay the import cost.
    from brain.server import main as _run_server

    _run_server()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(5, "-k", help="Top-k results."),
    tag: str | None = typer.Option(None, "--tag", help="Filter by frontmatter tag."),
) -> None:
    """One-shot search from the command line, for debugging."""
    store = _make_store()
    embedder = _make_embedder()
    query_vec = embedder.embed_query(query)
    hits = store.hybrid_search(
        query_embedding=query_vec,
        query_text=query,
        k=k,
        rrf_k=settings.rrf_k,
        tag=tag,
    )
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("score", justify="right", style="green", width=8)
    table.add_column("path", style="white")
    table.add_column("heading", style="dim")
    table.add_column("excerpt", style="white", overflow="fold")
    for h in hits:
        excerpt = h.body[:160].replace("\n", " ") + ("…" if len(h.body) > 160 else "")
        table.add_row(f"{h.score:.3f}", h.note_path, h.heading_path, excerpt)
    console.print(table)


@app.command()
def eval(
    gold: Path | None = typer.Option(
        None, "--gold", help="Gold-set YAML (default: <vault>/_evals/gold.yaml)."
    ),
    raw: bool = typer.Option(False, "--json", help="Output raw JSON results."),
) -> None:
    """Evaluate retrieval quality against a gold set of queries.

    Reports note-level recall@5, recall@10, MRR@10, per-language
    breakdown, and query latency. See evals/example-gold.yaml for the
    gold-set format.
    """
    from brain.evaluate import GoldFileError, load_gold, run_eval

    gold_path = gold or (settings.vault_dir / "_evals" / "gold.yaml")
    try:
        gold_queries = load_gold(gold_path)
    except GoldFileError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from e

    store = _make_store()
    embedder = _make_embedder()
    with console.status("[cyan]Running eval…"):
        res = run_eval(store, embedder, gold_queries)

    if raw:
        console.print_json(data=res)
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("query", overflow="fold")
    table.add_column("lang", width=4)
    table.add_column("first hit", justify="right")
    table.add_column("ms", justify="right")
    for q in res["queries"]:
        rank = f"#{q['first_hit_rank']}" if q["first_hit_rank"] else "[red]✗[/red]"
        table.add_row(q["query"], q["lang"], rank, f"{q['latency_ms']:.0f}")
    console.print(table)

    s = res["summary"]
    summary = Table(show_header=False, box=None)
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("queries", str(s["n"]))
    summary.add_row("recall@5", f"{s['recall_at_5']:.2%}")
    summary.add_row("recall@10", f"{s['recall_at_10']:.2%}")
    summary.add_row("MRR@10", f"{s['mrr_at_10']:.3f}")
    summary.add_row("latency mean/p95", f"{s['latency_mean_ms']:.0f} / {s['latency_p95_ms']:.0f} ms")
    for lang, m in s["by_lang"].items():
        summary.add_row(f"recall@5 [{lang}]", f"{m['recall_at_5']:.2%} (n={m['n']})")
    console.print(summary)


@app.command()
def status() -> None:
    """Print config and DB counts."""
    store = _make_store()
    conn = store.connect()
    notes_n = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    chunks_n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    links_n = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]

    table = Table(show_header=False, box=None)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("vault_dir", str(settings.vault_dir))
    table.add_row("db_path", str(settings.db_path))
    table.add_row("model", settings.embedding_model)
    table.add_row("dim", str(settings.embedding_dim))
    table.add_row("notes", str(notes_n))
    table.add_row("chunks", str(chunks_n))
    table.add_row("links", str(links_n))
    console.print(table)


@app.command()
def report(
    raw: bool = typer.Option(False, "--json", help="Output the raw JSON report."),
) -> None:
    """Show a vault health report — counts, authorship mix, recent activity."""
    store = _make_store()
    data = build_report(store)

    if raw:
        console.print_json(data=data)
        return

    v = data["vault"]
    a = data["authorship"]
    act = data["activity"]

    summary = Table(show_header=False, box=None)
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("notes", str(v["notes"]))
    summary.add_row("chunks", str(v["chunks"]))
    summary.add_row("links", str(v["links"]))
    summary.add_row("total chars", f"{v['total_chars']:,}")
    summary.add_row("vault age", f"{v['age_days']} days")
    if v["oldest_note"]:
        summary.add_row("oldest", v["oldest_note"]["path"])
    if v["newest_note"]:
        summary.add_row("newest", v["newest_note"]["path"])
    summary.add_row("agent-authored", str(a["agent_authored"]))
    summary.add_row("user-authored", str(a["user_authored"]))
    summary.add_row("mixed (agent in user)", str(a["mixed_authored"]))
    summary.add_row("modified 24h", str(act["modified_24h"]))
    summary.add_row("modified 7d", str(act["modified_7d"]))
    summary.add_row("modified 30d", str(act["modified_30d"]))
    console.print(summary)

    if act["recent_notes"]:
        console.print("\n[bold cyan]Recent[/bold cyan]")
        recent = Table(show_header=True, header_style="bold cyan", box=None)
        recent.add_column("modified_at", style="dim")
        recent.add_column("path")
        recent.add_column("title", style="white")
        for n in act["recent_notes"]:
            recent.add_row(n["modified_at"] or "—", n["path"], n["title"] or "")
        console.print(recent)


if __name__ == "__main__":
    app()
