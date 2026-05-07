# Installing brain

Step-by-step setup for macOS, Linux, and Windows (via WSL). Wires brain
into Claude Code and/or Claude Desktop so your AI agent has persistent
memory.

Total time: ~10 minutes, plus one-time ~30 MB model download.

---

## 1. Prerequisites

You need:

- **Python 3.11 or newer.** Check with `python3 --version`.
- **`uv`.** Fast Python package manager from Astral.

### Installing `uv`

**macOS / Linux / WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal, or run `source $HOME/.local/bin/env` to load
it in the current shell.

**Windows (PowerShell, native — only if you're not using WSL):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```

> **Windows note.** If your daily dev workflow is in WSL (Ubuntu, etc.),
> install `uv` *inside WSL*, not on the Windows side. The rest of this
> guide assumes WSL for Windows users — paths and tooling are simpler.

---

## 2. Get the code

```bash
git clone https://github.com/<your-handle>/brain.git
cd brain
```

If you don't have git, you can download a tarball from the GitHub Releases
page or the green **Code** → **Download ZIP** button.

---

## 3. Install dependencies and initialize

```bash
uv sync                   # creates .venv/, installs everything from uv.lock
uv run brain init         # creates ~/brain-vault and ~/.brain
```

Expected output:
```
✓ Vault: /home/<you>/brain-vault
✓ DB:    /home/<you>/.brain/db.duckdb
```

`init` also drops a starter `README.md` into your new vault explaining the
conventions (frontmatter tags, wikilinks, headings).

---

## 4. First indexing run

```bash
uv run brain index
```

The first run downloads the `bge-small-en-v1.5` embedding model
(~30 MB) into `~/.cache/fastembed/`. Every run after this is offline.

Expected output:
```
✓ scanned=1 new=1 updated=0 unchanged=0 deleted=0 chunks_embedded=2
```

Verify everything:
```bash
uv run brain status
```

---

## 5. Smoke test from the CLI

Add a real note:

```bash
cat > ~/brain-vault/test.md <<'EOF'
---
tags: [test]
---

# Test Note

This is a quick test of search. Brain uses DuckDB for storage,
bge-small-en-v1.5 for embeddings, and reciprocal rank fusion to combine
vector and keyword search.
EOF

uv run brain index
uv run brain search "how does search work"
```

You should see your test note come back with a relevance score. If yes,
the indexing pipeline is healthy. Time to wire it into Claude.

---

## 6. Wire into Claude Code

Claude Code is the terminal-based agent. Pick this if you want brain
available in any project from any directory.

```bash
claude mcp add --scope user --transport stdio brain \
  -- uv --directory $(pwd) run brain serve
```

`--scope user` makes brain available across all projects (not just this
one). `$(pwd)` resolves to the absolute path of your `brain` clone.

Verify:
```bash
claude mcp list
```

You should see `brain: connected` in the output.

Open Claude Code anywhere and try:

> What did we decide about storage layer choices?

If brain is wired up, Claude Code will call `search_notes`, find your
test note, and answer with what's actually in the vault.

---

## 7. Wire into Claude Desktop

Claude Desktop is the chat app. Pick this if you mostly want
chat-style queries against your vault.

### Find the config file

| OS               | Path                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| macOS            | `~/Library/Application Support/Claude/claude_desktop_config.json`    |
| Linux            | `~/.config/Claude/claude_desktop_config.json`                        |
| Windows (MSIX)   | `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\claude_desktop_config.json` |

> **Windows MSIX gotcha.** The path above looks weird because Claude
> Desktop is an MSIX-packaged app — its `%APPDATA%\Claude` is virtualized
> under `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\`. Don't use
> the unvirtualized path; it'll work in some places and silently not in
> others.

If the file doesn't exist, create it.

### macOS / Linux config

Replace `<absolute-path-to-brain>` with the output of `pwd` from your
brain clone.

```json
{
  "mcpServers": {
    "brain": {
      "command": "uv",
      "args": [
        "--directory", "<absolute-path-to-brain>",
        "run", "brain", "serve"
      ]
    }
  }
}
```

If `uv` isn't in Claude Desktop's PATH (likely if you installed it via
the curl script), use the absolute path. Find it with `which uv` and
substitute it for `"uv"` above.

### Windows / WSL config

Claude Desktop runs on Windows, but brain runs in WSL. We bridge with
`wsl.exe`. Find your distro name first:

```powershell
wsl -l -v
```

Use the exact name (e.g. `Ubuntu`, `Ubuntu-22.04`). Then inside WSL find
the `uv` binary:

```bash
which uv
# probably /home/<you>/.local/bin/uv
```

Then in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "brain": {
      "command": "wsl.exe",
      "args": [
        "-d", "Ubuntu",
        "--cd", "/home/<you>/path/to/brain",
        "--",
        "/home/<you>/.local/bin/uv", "run", "brain", "serve"
      ]
    }
  }
}
```

### Restart Claude Desktop

**Fully quit** — system-tray icon → Quit, not just close the window.
MSIX caches aggressively. On Windows you can also force-quit from
PowerShell:

```powershell
taskkill /IM Claude.exe /F
```

Reopen. Start a new chat. Look at the tools indicator — you should see
4 brain tools (search_notes, read_note, list_recent, find_related).

### Sanity test

In a new Claude Desktop chat:

> What's in my brain vault?

Claude Desktop should call `list_recent` and show you your test note.

---

## 8. (Optional) Live indexing

For ongoing use, keep brain re-syncing as you edit:

```bash
uv run brain index --watch
```

Edits to `.md` files trigger incremental re-sync 2 seconds after you
save. Only the files whose contents actually changed get re-embedded.

You can leave this running in a tmux pane / terminal tab, or wrap it in
a systemd user service / launchd agent if you want it to survive
reboots.

---

## Troubleshooting

### `uv: command not found` after install

Restart your shell, or `source $HOME/.local/bin/env`. The installer adds
`~/.local/bin` to your PATH, but the change doesn't apply to the current
shell.

### MCP server shows "disconnected" / no brain tools appear

The server is crashing on startup. Run the exact command Claude is
running, manually:

```bash
# What you'd see in Claude Code:
uv --directory /path/to/brain run brain serve

# What you'd see from Claude Desktop on Windows:
wsl.exe -d Ubuntu --cd /home/you/brain -- /home/you/.local/bin/uv run brain serve
```

If it works, it'll hang waiting for stdin (that's correct — MCP server is
alive, waiting for JSON-RPC). Hit Ctrl-C to exit. If you get an error
instead, that's the actual problem to fix — usually wrong path, wrong
distro name, or `uv` not at the location you specified.

### Server logs

Claude Desktop logs MCP server output to:

| OS               | Path                                                            |
| ---------------- | --------------------------------------------------------------- |
| macOS            | `~/Library/Logs/Claude/mcp-server-brain.log`                    |
| Linux            | `~/.config/Claude/logs/mcp-server-brain.log`                    |
| Windows (MSIX)   | `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\logs\mcp-server-brain.log` |

### Embedding model download fails

Network issue or HuggingFace rate limit. Retry. If you're behind a
restrictive proxy, set `HF_HUB_ENDPOINT` to a mirror or pre-download
the model manually into `~/.cache/fastembed/`.

### `brain init` says vault already exists

Safe to ignore. `init` is idempotent — it won't overwrite an existing
vault or DB.

### I want a different vault location

Set `BRAIN_VAULT_DIR` in a `.env` file in the project root, or as a
shell env var. See [`.env.example`](./.env.example) for everything
configurable.

### I want to nuke everything and start over

```bash
rm -rf ~/.brain                 # the DuckDB index (rebuildable)
rm -rf ~/brain-vault            # YOUR NOTES — be sure
uv run brain init
uv run brain index
```

The vault is the only thing that's not rebuildable. Back it up if it
matters.

---

## Updating to a new version

```bash
cd /path/to/brain
git pull
uv sync
uv run brain index
```

If a new version changes the embedding model (rare), you'll need to
re-index from scratch:

```bash
rm ~/.brain/db.duckdb
uv run brain index
```

Your vault is never touched.

---

## Uninstalling

```bash
# Remove from Claude Code:
claude mcp remove brain

# Remove from Claude Desktop:
# delete the "brain" entry from claude_desktop_config.json

# Remove the install:
rm -rf /path/to/brain          # the source code
rm -rf ~/.brain                # the DuckDB index
rm -rf ~/.cache/fastembed      # the embedding model

# Your vault stays unless you explicitly delete it:
# rm -rf ~/brain-vault
```
