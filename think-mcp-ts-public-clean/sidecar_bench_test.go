package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// benchConfig wires a fake sidecar that sleeps loadMs to simulate a cold local
// model load and appends one line to counter per invocation, then replies with
// deterministic scores. One row is seeded so the sidecar actually launches.
func benchConfig(tb testing.TB, loadMs int) (Config, string) {
	d := tb.TempDir()
	counter := filepath.Join(d, "runs.txt")
	sidecar := filepath.Join(d, "sleep_sidecar.py")
	script := fmt.Sprintf("import sys, json, time\ntime.sleep(%d / 1000.0)\nline = sys.stdin.readline()\nreq = json.loads(line)\nwith open(%q, 'a') as f:\n    f.write('run\\n')\nscores = [{'id': c['id'], 'score': 0.5} for c in req.get('candidates', [])]\nprint(json.dumps({'id': 'recall', 'ok': True, 'scores': scores, 'backend': 'bench'}))\nsys.stdout.flush()\n", loadMs, counter)
	if err := os.WriteFile(sidecar, []byte(script), 0755); err != nil {
		tb.Fatal(err)
	}
	cfg := Config{Store: filepath.Join(d, "thoughts.json"), IntegrityLog: filepath.Join(d, "integrity.jsonl"), LockTimeout: 5 * time.Second, LockStale: time.Minute, MinElapsed: 0, SemanticSidecar: sidecar, SemanticTimeout: 10 * time.Second}
	if _, err := appendStore(cfg.Store, StoreEntry{"ts": time.Now().UTC().Format(time.RFC3339Nano), "status": "oneshot", "input": "The renderer uses a blue gradient.", "meaning": "The landing page has visual styling.", "posterior": map[string]any{"p": 0.5, "because": "mixed evidence"}}, cfg); err != nil {
		tb.Fatal(err)
	}
	return cfg, counter
}

func runThinkCall(b *testing.B, cfg Config) {
	args := validBreakdown()
	args["posterior"] = map[string]any{"p": 0.5, "because": "The evidence remains mixed and more testing is required."}
	payload, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": map[string]any{"name": "think", "arguments": args}})
	var out strings.Builder
	if err := serve(strings.NewReader(string(payload)+"\n"), &out, cfg); err != nil {
		b.Fatal(err)
	}
	if !strings.Contains(out.String(), "Breakdown") {
		b.Fatalf("unexpected response: %s", out.String())
	}
}

func BenchmarkThinkWithSidecarLoad(b *testing.B) {
	cfg, counter := benchConfig(b, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runThinkCall(b, cfg)
	}
	b.StopTimer()
	body, _ := os.ReadFile(counter)
	runs := len(strings.Split(strings.TrimSpace(string(body)), "\n"))
	b.ReportMetric(float64(runs), "sidecar-runs")
}
