# Think MCP (Go)

A dependency-light, standalone Go implementation of a local Think MCP server. It communicates via newline-delimited JSON-RPC on `stdin`/`stdout`, persists a structured reasoning ledger as JSON, and records hash-only integrity telemetry without modifying the underlying LLM.

## Requirements

- **Required:** Go 1.26+ and filesystem permissions for ledger/lock files.
- **Optional Semantic-Recall Backend:** BGE-small or MiniLM model

*Note: The Go binary has zero external dependencies (no Node.js, npm, or SDK modules).*

## Build and Run

```bash
go build -o think-mcp main.go
./think-mcp
