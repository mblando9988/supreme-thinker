# Think MCP (Go)

This is a dependency-light Go implementation of a local Think MCP server.

The Go server is a dependency-free local MCP server. It communicates over newline-delimited JSON-RPC on stdin/stdout, persists a structured reasoning ledger as JSON, and records hash-only integrity telemetry. It does not fine-tune or modify the underlying LLM.

## Requirements

Required:

- Go 1.26 or newer
- A filesystem with permission to create the configured ledger and lock files

Optional semantic-recall backend:

- Python 3
- Locally installed PyTorch and Transformers
- A complete local BGE-small or MiniLM model snapshot

The Go binary has no Node.js, npm, Claude Agent SDK, Anthropic SDK, MCP SDK, or external Go module dependency. The Python helper is optional and also works without ML packages by using its dependency-free lexical scorer.

## Build and run

From this directory:

```bash
go build -o think-mcp main.go
./think-mcp
```

The server is an MCP stdio process. Do not write logs to stdout; stdout is reserved for JSON-RPC. Runtime errors go to stderr.

## MCP client configuration

Point the MCP client directly at the compiled binary. Use an absolute path in the client configuration, replacing the placeholder with the location on your machine:

```json
{
  "mcpServers": {
    "think": {
      "command": "<PROJECT_ROOT>/think-mcp",
      "env": {
        "THINK_STORE": "<PROJECT_ROOT>/thoughts.json",
        "THINK_INTEGRITY_LOG": "<PROJECT_ROOT>/integrity.jsonl"
      }
    }
  }
}
```

If the environment variables are omitted, the server places `thoughts.json` and `integrity.jsonl` beside the executable. `THINK_STORE` and `THINK_INTEGRITY_LOG` are recommended so runtime state is explicit and easy to back up.

### Codex reload and probe

Codex snapshots its available MCP tools when a turn starts. A running desktop turn may not hot-load a newly added MCP server, so this project includes a reload command that rebuilds the binary, reinstalls the global Codex MCP entry, and tests every one of the nine tools over real stdio JSON-RPC:

```bash
scripts/codex_reload_think_mcp.sh
```

To also launch a fresh non-interactive Codex run and verify that Codex can see the tool names after reload:

```bash
scripts/codex_reload_think_mcp.sh --codex-probe
```

The deterministic server probe can be run directly without touching Codex config:

```bash
python3 scripts/probe_think_mcp.py --binary ./think-mcp
```

The reload command configures the MCP server named `think` with explicit state paths:

- `THINK_STORE=<PROJECT_ROOT>/thoughts.json`
- `THINK_INTEGRITY_LOG=<PROJECT_ROOT>/integrity.jsonl`

After running it, start a new Codex task or reload the desktop app to make the freshly discovered tools available in the model's tool list.

## Tools

The server exposes nine MCP tools. Recall uses the local semantic backend described below when it is available:

| Tool | Purpose |
| --- | --- |
| `think` | Create a structured one-shot breakdown, stored as explicitly unverified. |
| `get_thoughts` | Browse the persistent ledger by id, status, or recency. |
| `think_open` | Pre-register a claim, alternatives, prior belief, and tests. |
| `think_conclude` | Close an open claim with per-test evidence and a posterior. |
| `think_recall` | Find similar prior claims using lexical similarity plus optional local semantic recall. |
| `think_verify` | Record a verification claim that includes sources and an independent invariant. |
| `think_isolate` | Record a controlled failure-isolation analysis. |
| `think_discover` | Record a structured cross-domain discovery hypothesis. |
| `think_calibrate` | Summarize the number of ledger records available for calibration. |

## Integrity gate

Every tool call passes through a dispatcher-level integrity check before execution and a postflight event after execution.

The current Go gate:

- blocks unsupported high confidence without evidence;
- flags strongly similar prior records with opposing belief values;
- blocks obvious lexical goal drift between `input` and `meaning`;
- records only a short SHA-256 claim digest, tool, decision, finding codes, and failure state.

### Optional Python semantic recall

The Go server remains authoritative. When `semantic_recall.py` is present beside the executable, it is enabled by default: Go sends each recall query and the current ledger candidates to a short-lived local Python process. The helper first tries a complete, already-cached local Transformers encoder (`BAAI/bge-small-en-v1.5`, then `sentence-transformers/all-MiniLM-L6-v2`) using mean pooling and normalized cosine similarity. It sets Transformers/Hugging Face offline mode and loads only local snapshot directories; it never downloads or calls a cloud service. If no complete model or ML runtime is available, it uses the dependency-free token/character scorer. Go combines the helper score with lexical similarity using the stronger score.

If Python, the local ML runtime, or the cached model is missing, the helper fails, or the timeout expires, Go silently falls back to lexical similarity. Semantic recall is advisory: it can improve paraphrase matching, but it cannot independently prove that two claims are equivalent and never blocks a claim by itself. The model is loaded lazily per helper process, so the first request may be slower.

To disable the bundled helper, set `THINK_SEMANTIC_SIDECAR` explicitly to an empty value. To point to another helper, set it to that script's path.

#### Local model selection

Model selection is strictly local:

1. `THINK_SEMANTIC_MODEL`, when set, must point to a complete local snapshot directory.
2. Otherwise the helper searches the configured cache roots and selects the first complete snapshot by model-family priority:
   - `BAAI/bge-small-en-v1.5`
   - `sentence-transformers/all-MiniLM-L6-v2`

   Within a model family, snapshot directories are checked in sorted directory-name order; the helper does not contact a registry to determine which snapshot is newest.
3. A complete snapshot must contain `config.json`, `tokenizer.json`, and either `model.safetensors` or `pytorch_model.bin`.
4. A remote model ID such as `BAAI/bge-small-en-v1.5` is not accepted as `THINK_SEMANTIC_MODEL`; use a local directory path.

The helper forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` before importing Transformers. It passes `local_files_only=True` to model/tokenizer loading. No package installation, model download, cloud request, or remote model resolution is performed.

For deterministic dependency-free operation, set `THINK_SEMANTIC_BACKEND=lexical`. Any other model-loading failure falls back to the same lexical scorer. The helper is intentionally short-lived: Go starts at most one fresh Python process per tool call, shared between the integrity gate and the prior-work note. That process loads the model once, so each request can include a cold model load and is covered by the default five-second Go timeout.

This is structural checking, not mind-reading or proof that an LLM intended to deceive. Lexical drift and similarity checks can miss paraphrases and can produce false positives.

## Staged evidence rules

`think_open` refuses a posterior so the belief is registered before testing. `think_conclude` requires:

- an existing open breakdown;
- evidence for every declared test, with no duplicate or unknown test names;
- a `ran`, `observed`, and `verdict` field for each evidence item;
- an artifact for every adversarial test;
- regular, non-symlink artifacts;
- artifact modification after the breakdown was opened;
- the observed excerpt inside each supplied artifact;
- a posterior between `0.01` and `0.99`, with a basic direction check against counted pass/fail verdicts.

Artifact checks establish what was observed at conclusion time. They do not prove that a claimed command generated the file. `think_verify` records the supplied sources and invariant; it does not independently execute or evaluate them. Artifact paths are not restricted to a workspace in this lightweight variant.

## Persistence and locking

Ledger writes use an atomic exclusive lock file with an owner token, stale-owner recovery for dead owners, and atomic replacement of the JSON store. Appends and conclusions perform their read/modify/write transaction while holding the lock, preventing concurrent writers from overwriting one another.

The lock timeout and stale-owner window are configurable:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `THINK_STORE` | executable directory + `thoughts.json` | Ledger path |
| `THINK_INTEGRITY_LOG` | executable directory + `integrity.jsonl` | Hash-only telemetry path |
| `THINK_MIN_ELAPSED_MS` | `20000` | Minimum open-to-conclude interval |
| `THINK_STORE_LOCK_TIMEOUT_MS` | `10000` | Lock acquisition timeout |
| `THINK_STORE_LOCK_STALE_MS` | `60000` | Dead-owner recovery threshold |
| `THINK_SEMANTIC_SIDECAR` | executable directory + `semantic_recall.py` | Local semantic-recall helper; empty disables it |
| `THINK_SEMANTIC_MODEL` | unset | Complete local model snapshot; unset enables cache discovery |
| `THINK_SEMANTIC_BACKEND` | `model` | Set to `lexical` to disable model loading |
| `THINK_SEMANTIC_CACHE` | unset | Optional local Hugging Face cache root searched before the standard cache; it must contain `models--.../snapshots/...` directories |
| `THINK_SEMANTIC_TIMEOUT_MS` | `5000` | Maximum sidecar time per recall request, including cold model load |

## Tests and proof

Run the Go test suite and race detector:

```bash
go test ./...
go test -race ./...
```

The tests prove:

- nine-tool registration;
- valid tool schemas for strict MCP clients, including no `required:null` fields;
- every advertised tool is callable through the JSON-RPC dispatcher;
- real subprocess MCP stdio responses, including notification handling;
- integrity blocking for unsupported confidence and lexical drift;
- staged conclusion with artifact/excerpt checks;
- concurrent append preservation and lock cleanup;
- hash-only integrity logging;
- absence of copied Node package files, `node_modules`, TypeScript output, and Node test harnesses;
- Python sidecar protocol, local model discovery/loading, embedding paraphrase ranking, explicit lexical mode, remote-identifier rejection, offline configuration, and Go lexical fallback.

Run the Python helper tests separately:

```bash
python3 -m unittest semantic_recall_test.py
```

## Scope and limitations

This is a dependency-light Go replacement, not a claim of byte-for-byte parity with the original TypeScript implementation. The Go binary has no external module dependencies. The optional Python helper uses already-installed PyTorch/Transformers when available, but remains usable with its dependency-free lexical scorer and never installs packages or downloads models. It has simpler implementations of verification, discovery, and calibration than the original. The original remains available as the reference implementation.

It is a ledger, integrity gate, recall, verification, and calibration foundation—not reinforcement learning or LLM fine-tuning. A later experience graph can consume the ledger without reintroducing the Claude Agent SDK into this runtime.
