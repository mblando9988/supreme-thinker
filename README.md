# supreme-thinker

Think MCP (Go)
A dependency-light, standalone Go implementation of a local Think MCP server. It communicates via newline-delimited JSON-RPC on stdin/stdout, persists a structured reasoning ledger as JSON, and records hash-only integrity telemetry without modifying the underlying LLM.

Requirements
Required: Go 1.26+ and filesystem permissions for ledger/lock files.
Optional Semantic-Recall Backend: BGE-small or MiniLM model
Note: The Go binary has zero external dependencies (no Node.js, npm, or SDK modules).

Build and Run
go build -o think-mcp main.go
./think-mcp

# Photos
<img width="779" height="215" alt="Screenshot 2026-08-10 at 11 33 39 PM" src="https://github.com/user-attachments/assets/6979f625-b75e-483d-815c-0df1e339bd62" />

<img width="787" height="197" alt="Screenshot 2026-08-10 at 11 33 02 PM" src="https://github.com/user-attachments/assets/d27912d3-ea19-4ae5-8479-7f979dd8057a" />

<img width="1586" height="354" alt="image" src="https://github.com/user-attachments/assets/17dd1c94-2506-451c-ab96-610ff9ee05fe" />
