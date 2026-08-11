#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_BIN="${CODEX_BIN:-/opt/homebrew/bin/codex}"
SERVER_NAME="${THINK_MCP_NAME:-think}"
STORE="${THINK_STORE:-$ROOT/thoughts.json}"
INTEGRITY_LOG="${THINK_INTEGRITY_LOG:-$ROOT/integrity.jsonl}"
RUN_CODEX_PROBE=0

usage() {
  cat <<'USAGE'
Usage: scripts/codex_reload_think_mcp.sh [--codex-probe]

Rebuild the local think MCP server, reinstall its Codex MCP entry, and run the
full stdio tool smoke probe. This reloads configuration for new Codex processes.
Already-running Codex turns keep their existing tool snapshot until restarted.

Options:
  --codex-probe   Launch a fresh non-interactive Codex run and verify it sees
                  all 9 think tools. This uses the configured Codex model.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codex-probe)
      RUN_CODEX_PROBE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$CODEX_BIN" ]]; then
  echo "missing Codex binary: $CODEX_BIN" >&2
  exit 1
fi

cd "$ROOT"
go build -o "$ROOT/think-mcp" "$ROOT/main.go"

if "$CODEX_BIN" mcp get "$SERVER_NAME" >/dev/null 2>&1; then
  "$CODEX_BIN" mcp remove "$SERVER_NAME" >/dev/null
fi

"$CODEX_BIN" mcp add "$SERVER_NAME" \
  --env "THINK_STORE=$STORE" \
  --env "THINK_INTEGRITY_LOG=$INTEGRITY_LOG" \
  -- "$ROOT/think-mcp" >/dev/null

"$CODEX_BIN" mcp get "$SERVER_NAME" >/dev/null
python3 "$ROOT/scripts/probe_think_mcp.py" --binary "$ROOT/think-mcp"

if [[ "$RUN_CODEX_PROBE" -eq 1 ]]; then
  probe_output="$(mktemp "${TMPDIR:-/tmp}/think-codex-probe.XXXXXX")"
  "$CODEX_BIN" exec --json --skip-git-repo-check -C "$ROOT" \
    "Do not modify files. If MCP tools from a server named think are available, list only their exact tool names. Otherwise say NO_THINK_TOOLS." \
    >"$probe_output"

  for tool in think get_thoughts think_open think_conclude think_recall think_verify think_isolate think_discover think_calibrate; do
    if ! grep -q "$tool" "$probe_output"; then
      echo "fresh Codex probe did not report $tool" >&2
      echo "probe output: $probe_output" >&2
      exit 1
    fi
  done
  rm -f "$probe_output"
fi

echo "think MCP reloaded for new Codex processes"
echo "server: $ROOT/think-mcp"
echo "store: $STORE"
echo "integrity log: $INTEGRITY_LOG"
