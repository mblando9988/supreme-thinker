package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func testConfig(t *testing.T) Config {
	t.Helper()
	d := t.TempDir()
	return Config{Store: filepath.Join(d, "thoughts.json"), IntegrityLog: filepath.Join(d, "integrity.jsonl"), LockTimeout: 5 * time.Second, LockStale: time.Minute, MinElapsed: 0}
}

func validBreakdown() map[string]any {
	return map[string]any{
		"input": "The cache health check reflects actual service state rather than a wrapper artifact.",
		"interpretations": []any{
			map[string]any{"reading": "The service is healthy and the check reports it accurately.", "category": "service"},
			map[string]any{"reading": "The wrapper masks a failed service check.", "category": "wrapper"},
			map[string]any{"reading": "A stale log is being replayed.", "category": "cache"},
		},
		"meaning":           "Determine whether the health check genuinely reflects service state rather than masking or replaying a result.",
		"how_to_prove":      []any{"Run the service command directly five times and record exit status in a raw output file.", "Compare the current response with the cache log and inspect modification times."},
		"failure_signature": "The direct command fails or the wrapper reports success while the service is unhealthy.",
		"success_signature": "The direct command exits zero five times and the current response is fresh rather than replayed.",
		"prior":             map[string]any{"p": 0.55, "because": "The wrapper looked healthy but direct service exit status was never independently recorded."},
		"tests": []any{
			map[string]any{"name": "A", "hypothesis": "the service is healthy directly", "metric": "direct command exit status", "decision_rule": "reject if the command exits nonzero", "discriminates": 0},
			map[string]any{"name": "B", "hypothesis": "the wrapper masks a failed service", "metric": "pre-wrapper exit status", "decision_rule": "reject if a nonzero status appears before wrapper output", "discriminates": 1},
			map[string]any{"name": "C", "hypothesis": "the output is stale cache replay", "metric": "response bytes and cache mtime", "decision_rule": "reject if the response matches the stale cache", "adversarial": true, "discriminates": 2},
		},
	}
}

func callJSON(t *testing.T, cfg Config, method string, id int, params map[string]any) map[string]any {
	t.Helper()
	b, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params})
	var out strings.Builder
	if err := serve(strings.NewReader(string(b)+"\n"), &out, cfg); err != nil {
		t.Fatal(err)
	}
	var response map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(out.String())), &response); err != nil {
		t.Fatal(err)
	}
	return response
}

func resultText(response map[string]any) string {
	result, _ := response["result"].(map[string]any)
	content, _ := result["content"].([]any)
	if len(content) == 0 {
		return ""
	}
	first, _ := content[0].(map[string]any)
	text, _ := first["text"].(string)
	return text
}

func TestGoMCPListsNineTools(t *testing.T) {
	cfg := testConfig(t)
	response := callJSON(t, cfg, "tools/list", 1, map[string]any{})
	result := response["result"].(map[string]any)
	tools := result["tools"].([]any)
	if len(tools) != 9 {
		t.Fatalf("got %d tools, want 9", len(tools))
	}
	for _, raw := range tools {
		tool := raw.(map[string]any)
		schema := tool["inputSchema"].(map[string]any)
		if required, exists := schema["required"]; exists && required == nil {
			t.Fatalf("%s schema has required:null; omit required for optional argument objects", tool["name"])
		}
	}
}

func callToolJSON(t *testing.T, cfg Config, id int, name string, args map[string]any) string {
	t.Helper()
	response := callJSON(t, cfg, "tools/call", id, map[string]any{"name": name, "arguments": args})
	result, _ := response["result"].(map[string]any)
	if result["isError"] == true {
		t.Fatalf("%s returned tool error:\n%s", name, resultText(response))
	}
	text := resultText(response)
	if text == "" {
		t.Fatalf("%s returned empty text: %#v", name, response)
	}
	return text
}

func conclusionEvidence(artifact string) []any {
	observedA := "service command exited zero five times"
	observedB := "wrapper status stayed successful before output"
	observedC := "cache response bytes were fresh, not replayed"
	return []any{
		map[string]any{"test": "A", "ran": "run service command five times", "observed": observedA, "verdict": "pass direct service output", "artifact": artifact},
		map[string]any{"test": "B", "ran": "capture wrapper status", "observed": observedB, "verdict": "pass wrapper status", "artifact": artifact},
		map[string]any{"test": "C", "ran": "compare cache bytes", "observed": observedC, "verdict": "pass freshness comparison", "artifact": artifact},
	}
}

func writeConclusionArtifact(t *testing.T, dir string) string {
	t.Helper()
	artifact := filepath.Join(dir, "evidence.txt")
	body := "service command exited zero five times\nwrapper status stayed successful before output\ncache response bytes were fresh, not replayed\n"
	if err := os.WriteFile(artifact, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	return artifact
}

func verifyArgs(t *testing.T, dir string) map[string]any {
	t.Helper()
	source := filepath.Join(dir, "verify-source.txt")
	excerpt := "MCP smoke marker: this local file proves the verification source path was read during the test."
	if err := os.WriteFile(source, []byte(excerpt+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	return map[string]any{
		"claim": "The MCP smoke artifact contains the expected marker in a regular local file.",
		"sources": []any{
			map[string]any{"path": source, "kind": "executed", "excerpt": excerpt},
		},
		"discriminator": map[string]any{
			"name":          "regular file marker check",
			"would_fail_if": "The source file were missing, empty, or lacked the MCP smoke marker.",
			"ran":           "read verify-source.txt and count marker occurrences",
			"observed":      "The regular file existed and contained exactly one MCP smoke marker line.",
			"outcome":       "held",
		},
		"invariants": []any{
			map[string]any{
				"name":            "marker count",
				"computed_from":   "verify-source.txt bytes only",
				"why_independent": "A byte count from one source would catch a missing or substituted artifact without comparing it to another tool result.",
				"expected":        "exactly one marker line",
				"measured":        "exactly one marker line",
				"holds":           true,
			},
		},
		"sample":   map[string]any{"n": 1.0, "selection": "single deterministic smoke artifact created by this test", "includes_edge_cases": true},
		"blind_to": []any{"whether an already-running Codex desktop turn has reloaded its tool list"},
		"verdict":  map[string]any{"established": "The local verification artifact was present and readable.", "not_established": "This does not prove the current desktop turn hot-loaded new tools."},
	}
}

func discoveryArgs() map[string]any {
	return map[string]any{
		"input":                  "Min-plus path composition can predict delay growth in railway dispatching and packet scheduling.",
		"domain_a":               "railway dispatching",
		"domain_b":               "packet scheduling",
		"shared_formal_object":   "min-plus algebra over path costs with associative composition and bottleneck delay accumulation",
		"transported_prediction": "The 95th percentile delay should scale within 20% of the min-plus path bound as load increases from 0.4x to 0.8x.",
		"refuting_experiment":    "Reject if measured delay slope is flat or differs by more than 50% from the min-plus bound across 30 scheduled runs.",
		"prior_art": []any{
			map[string]any{"query": "railway dispatching min-plus delay bound", "top_result": "no direct hit for this transported packet scheduling comparison", "verdict": "clear"},
			map[string]any{"query": "packet scheduling min-plus railway timetable analogy", "top_result": "related queueing papers but no direct timetable prediction", "verdict": "related"},
		},
		"interpretations": []any{
			map[string]any{"reading": "Both fields compose waiting costs along a route.", "category": "formal"},
			map[string]any{"reading": "The analogy may fail because dispatch rules add discrete constraints.", "category": "limitation"},
			map[string]any{"reading": "The prediction is only about delay slope, not full route quality.", "category": "scope"},
		},
		"meaning":           "Test whether the shared min-plus path-cost structure predicts numerical delay growth in railway dispatching and packet scheduling.",
		"how_to_prove":      []any{"Measure 95th percentile delay across 30 runs at 0.4x, 0.6x, and 0.8x load.", "Compare the measured slope against the min-plus path bound within a 20% tolerance."},
		"failure_signature": "Measured delay slopes are flat, opposite-signed, or more than 50% away from the min-plus bound.",
		"success_signature": "Measured 95th percentile delay slopes remain within 20% of the min-plus path bound in both domains.",
		"prior":             map[string]any{"p": 0.35, "because": "The algebraic analogy is plausible but discrete dispatch constraints may dominate."},
		"posterior":         map[string]any{"p": 0.6, "because": "The proposed test is quantitative but not yet run against real schedules."},
		"tests": []any{
			map[string]any{"name": "D1", "hypothesis": "delay follows the min-plus path bound", "metric": "95th percentile delay slope", "decision_rule": "pass if slope is within 20 percent", "discriminates": 0},
			map[string]any{"name": "D2", "hypothesis": "dispatch constraints dominate the bound", "metric": "absolute slope error", "decision_rule": "fail if error exceeds 50 percent", "adversarial": true, "discriminates": 1},
			map[string]any{"name": "D3", "hypothesis": "load increase preserves monotone delay growth", "metric": "delay ratio at 0.8x versus 0.4x load", "decision_rule": "pass if the ratio is greater than 1.1", "discriminates": 2},
		},
	}
}

func TestEveryAdvertisedToolCallableOverJSONRPC(t *testing.T) {
	cfg := testConfig(t)
	dir := filepath.Dir(cfg.Store)

	response := callJSON(t, cfg, "tools/list", 1, map[string]any{})
	result := response["result"].(map[string]any)
	tools := result["tools"].([]any)
	got := map[string]bool{}
	for _, raw := range tools {
		tool := raw.(map[string]any)
		got[tool["name"].(string)] = true
	}
	expected := []string{"think", "get_thoughts", "think_open", "think_conclude", "think_recall", "think_verify", "think_isolate", "think_discover", "think_calibrate"}
	for _, name := range expected {
		if !got[name] {
			t.Fatalf("tools/list omitted %s; got %#v", name, got)
		}
	}

	oneShot := validBreakdown()
	oneShot["posterior"] = map[string]any{"p": 0.5, "because": "The evidence remains mixed and more testing is required."}
	if text := callToolJSON(t, cfg, 2, "think", oneShot); !strings.Contains(text, "Breakdown #1 accepted") {
		t.Fatalf("think smoke failed: %s", text)
	}
	if text := callToolJSON(t, cfg, 3, "get_thoughts", map[string]any{"limit": 5.0}); !strings.Contains(text, "#1") {
		t.Fatalf("get_thoughts smoke failed: %s", text)
	}
	if text := callToolJSON(t, cfg, 4, "think_open", validBreakdown()); !strings.Contains(text, "Breakdown #2 OPEN") {
		t.Fatalf("think_open smoke failed: %s", text)
	}
	artifact := writeConclusionArtifact(t, dir)
	concludeArgs := map[string]any{"id": 2.0, "evidence": conclusionEvidence(artifact), "posterior": map[string]any{"p": 0.7, "because": "The direct service output exited zero and the response was fresh in the recorded evidence."}}
	if text := callToolJSON(t, cfg, 5, "think_conclude", concludeArgs); !strings.Contains(text, "Breakdown #2 VERIFIED") {
		t.Fatalf("think_conclude smoke failed: %s", text)
	}
	if text := callToolJSON(t, cfg, 6, "think_recall", map[string]any{"query": "The cache health check reflects actual service state rather than a wrapper artifact.", "limit": 5.0}); !strings.Contains(text, "similar past record") {
		t.Fatalf("think_recall smoke failed: %s", text)
	}
	if text := callToolJSON(t, cfg, 7, "think_verify", verifyArgs(t, dir)); !strings.Contains(text, "VERIFIED") {
		t.Fatalf("think_verify smoke failed: %s", text)
	}
	if text := callToolJSON(t, cfg, 8, "think_isolate", isolateArgs()); !strings.Contains(text, "Isolation #4 RECORDED") {
		t.Fatalf("think_isolate smoke failed: %s", text)
	}
	if text := callToolJSON(t, cfg, 9, "think_discover", discoveryArgs()); !strings.Contains(text, "Discovery #5 accepted") {
		t.Fatalf("think_discover smoke failed: %s", text)
	}
	if text := callToolJSON(t, cfg, 10, "think_calibrate", map[string]any{}); !strings.Contains(text, "CALIBRATION") {
		t.Fatalf("think_calibrate smoke failed: %s", text)
	}
}

func TestIntegrityBlocksUnsupportedConfidenceAndDrift(t *testing.T) {
	cfg := testConfig(t)
	args := validBreakdown()
	args["posterior"] = map[string]any{"p": 0.95, "because": "The direct command and cache comparison support the health claim."}
	blocked := callJSON(t, cfg, "tools/call", 1, map[string]any{"name": "think", "arguments": args})
	if !strings.Contains(resultText(blocked), "UNSUPPORTED_CONFIDENCE") {
		t.Fatalf("unsupported confidence was not blocked: %s", resultText(blocked))
	}

	drift := validBreakdown()
	drift["posterior"] = map[string]any{"p": 0.50, "because": "The evidence is mixed and additional testing is required."}
	drift["meaning"] = "The moon phase predicts coastal bird migration timing and the telescope confirms it."
	response := callJSON(t, cfg, "tools/call", 2, map[string]any{"name": "think", "arguments": drift})
	if !strings.Contains(resultText(response), "GOAL_DRIFT") {
		t.Fatalf("goal drift was not blocked: %s", resultText(response))
	}
}

func TestGoConcludeArtifactAndConcurrentAppend(t *testing.T) {
	cfg := testConfig(t)
	open := validBreakdown()
	openResponse := callJSON(t, cfg, "tools/call", 1, map[string]any{"name": "think_open", "arguments": open})
	openText := resultText(openResponse)
	if !strings.Contains(openText, "OPEN") {
		t.Fatal(openText)
	}
	id := 1
	artifact := filepath.Join(filepath.Dir(cfg.Store), "evidence.txt")
	observedA := "service command exited zero five times"
	observedB := "wrapper status stayed successful before output"
	observedC := "cache response bytes were fresh, not replayed"
	if err := os.WriteFile(artifact, []byte(observedA+"\n"+observedB+"\n"+observedC+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	evidence := []any{
		map[string]any{"test": "A", "ran": "run service command five times", "observed": observedA, "verdict": "pass direct service output", "artifact": artifact},
		map[string]any{"test": "B", "ran": "capture wrapper status", "observed": observedB, "verdict": "pass wrapper status", "artifact": artifact},
		map[string]any{"test": "C", "ran": "compare cache bytes", "observed": observedC, "verdict": "pass freshness comparison", "artifact": artifact},
	}
	conclude := callJSON(t, cfg, "tools/call", 2, map[string]any{"name": "think_conclude", "arguments": map[string]any{"id": id, "evidence": evidence, "posterior": map[string]any{"p": 0.7, "because": "The direct service output exited zero and the response was fresh in the recorded evidence."}}})
	if !strings.Contains(resultText(conclude), "VERIFIED") {
		t.Fatalf("conclude failed: %s", resultText(conclude))
	}

	const workers = 8
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func(i int) {
			defer wg.Done()
			_, err := appendStore(cfg.Store, StoreEntry{"ts": time.Now().UTC().Format(time.RFC3339Nano), "status": "oneshot", "input": "concurrent append probe", "id": 0}, cfg)
			if err != nil {
				t.Errorf("worker %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
	rows, err := loadStore(cfg.Store)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != workers+1 {
		t.Fatalf("got %d rows, want %d", len(rows), workers+1)
	}
	if _, err := os.Stat(cfg.Store + ".lock"); !os.IsNotExist(err) {
		t.Fatalf("lock remains: %v", err)
	}
}

func TestPortableDefaultAndIntegrityLog(t *testing.T) {
	cfg := testConfig(t)
	args := validBreakdown()
	args["posterior"] = map[string]any{"p": 0.5, "because": "The evidence remains mixed and more testing is required."}
	_ = callJSON(t, cfg, "tools/call", 1, map[string]any{"name": "think", "arguments": args})
	body, err := os.ReadFile(cfg.IntegrityLog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "claimDigest") || strings.Contains(string(body), args["input"].(string)) {
		t.Fatalf("integrity log privacy failure: %s", body)
	}
}

func TestConcludeRejectsSymlinkArtifact(t *testing.T) {
	cfg := testConfig(t)
	openResponse := callJSON(t, cfg, "tools/call", 1, map[string]any{"name": "think_open", "arguments": validBreakdown()})
	if !strings.Contains(resultText(openResponse), "OPEN") {
		t.Fatal(resultText(openResponse))
	}
	artifact := filepath.Join(filepath.Dir(cfg.Store), "real-evidence.txt")
	link := filepath.Join(filepath.Dir(cfg.Store), "linked-evidence.txt")
	observedA := "service command exited zero five times"
	observedB := "wrapper status stayed successful before output"
	observedC := "cache response bytes were fresh, not replayed"
	if err := os.WriteFile(artifact, []byte(observedA+"\n"+observedB+"\n"+observedC+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(artifact, link); err != nil {
		t.Fatal(err)
	}
	evidence := []any{
		map[string]any{"test": "A", "ran": "run service command five times", "observed": observedA, "verdict": "pass direct service output", "artifact": artifact},
		map[string]any{"test": "B", "ran": "capture wrapper status", "observed": observedB, "verdict": "pass wrapper status", "artifact": artifact},
		map[string]any{"test": "C", "ran": "compare cache bytes", "observed": observedC, "verdict": "pass freshness comparison", "artifact": link},
	}
	response := callJSON(t, cfg, "tools/call", 2, map[string]any{"name": "think_conclude", "arguments": map[string]any{"id": 1, "evidence": evidence, "posterior": map[string]any{"p": 0.7, "because": "The direct service output exited zero and the response was fresh in the recorded evidence."}}})
	if strings.Contains(resultText(response), "VERIFIED") || !strings.Contains(resultText(response), "artifact") {
		t.Fatalf("symlink artifact was not rejected: %s", resultText(response))
	}
}

func TestConcludeRejectsSymlinkParentArtifact(t *testing.T) {
	cfg := testConfig(t)
	openResponse := callJSON(t, cfg, "tools/call", 1, map[string]any{"name": "think_open", "arguments": validBreakdown()})
	if !strings.Contains(resultText(openResponse), "OPEN") {
		t.Fatal(resultText(openResponse))
	}
	root := filepath.Dir(cfg.Store)
	realDir := filepath.Join(root, "real-artifacts")
	linkDir := filepath.Join(root, "linked-artifacts")
	if err := os.Mkdir(realDir, 0755); err != nil {
		t.Fatal(err)
	}
	artifact := filepath.Join(realDir, "evidence.txt")
	linkedArtifact := filepath.Join(linkDir, "evidence.txt")
	observedA := "service command exited zero five times"
	observedB := "wrapper status stayed successful before output"
	observedC := "cache response bytes were fresh, not replayed"
	if err := os.WriteFile(artifact, []byte(observedA+"\n"+observedB+"\n"+observedC+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(realDir, linkDir); err != nil {
		t.Fatal(err)
	}
	evidence := []any{
		map[string]any{"test": "A", "ran": "run service command five times", "observed": observedA, "verdict": "pass direct service output", "artifact": artifact},
		map[string]any{"test": "B", "ran": "capture wrapper status", "observed": observedB, "verdict": "pass wrapper status", "artifact": artifact},
		map[string]any{"test": "C", "ran": "compare cache bytes", "observed": observedC, "verdict": "pass freshness comparison", "artifact": linkedArtifact},
	}
	response := callJSON(t, cfg, "tools/call", 2, map[string]any{"name": "think_conclude", "arguments": map[string]any{"id": 1, "evidence": evidence, "posterior": map[string]any{"p": 0.7, "because": "The direct service output exited zero and the response was fresh in the recorded evidence."}}})
	if !strings.Contains(resultText(response), "VERIFIED") {
		t.Fatalf("ordinary parent alias was incorrectly rejected: %s", resultText(response))
	}
}

func TestOwnerlessStaleLockIsRecovered(t *testing.T) {
	cfg := testConfig(t)
	cfg.LockStale = time.Millisecond
	cfg.LockTimeout = time.Second
	lock := cfg.Store + ".lock"
	if err := os.Mkdir(lock, 0755); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(lock, old, old); err != nil {
		t.Fatal(err)
	}
	_, err := appendStore(cfg.Store, StoreEntry{"ts": time.Now().UTC().Format(time.RFC3339Nano), "status": "oneshot", "input": "ownerless stale lock recovery", "id": 0}, cfg)
	if err != nil {
		t.Fatalf("stale ownerless lock was not recovered: %v", err)
	}
	if _, err := os.Stat(lock); !os.IsNotExist(err) {
		t.Fatalf("lock remains after recovery: %v", err)
	}
}

func TestGoMCPSubprocess(t *testing.T) {
	cfg := testConfig(t)
	binary := filepath.Join(t.TempDir(), "think-mcp")
	build := exec.Command("go", "build", "-o", binary, ".")
	buildOutput, err := build.CombinedOutput()
	if err != nil {
		t.Fatalf("go build failed: %v\n%s", err, buildOutput)
	}
	input := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`,
	}, "\n") + "\n"
	cmd := exec.Command(binary)
	cmd.Env = append(os.Environ(), "THINK_STORE="+cfg.Store, "THINK_INTEGRITY_LOG="+cfg.IntegrityLog)
	cmd.Stdin = strings.NewReader(input)
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			t.Fatalf("subprocess failed: %v\nstderr: %s", err, exitErr.Stderr)
		}
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d protocol responses, want 2: %s", len(lines), output)
	}
	for _, line := range lines {
		var response map[string]any
		if err := json.Unmarshal([]byte(line), &response); err != nil {
			t.Fatalf("invalid JSON-RPC response %q: %v", line, err)
		}
		if response["jsonrpc"] != "2.0" {
			t.Fatalf("bad JSON-RPC response: %s", line)
		}
	}
}

func TestSemanticSidecarAndLexicalFallback(t *testing.T) {
	cfg := testConfig(t)
	cfg.SemanticSidecar = filepath.Join(".", "semantic_recall.py")
	cfg.SemanticTimeout = 6 * time.Second
	rows := []StoreEntry{
		{"id": 1, "input": "The cache health check masks a failed service.", "meaning": "The wrapper hides backend errors."},
		{"id": 2, "input": "The renderer uses a blue gradient.", "meaning": "The landing page has visual styling."},
	}
	semantic := semanticScores("The health probe hides backend errors", rows, cfg)
	if semantic == nil || semantic[1] <= semantic[2] {
		t.Fatalf("semantic sidecar did not rank the paraphrase: %#v", semantic)
	}
	cfg.SemanticSidecar = filepath.Join(t.TempDir(), "missing-sidecar.py")
	fallback := semanticScores("The health probe hides backend errors", rows, cfg)
	if fallback != nil {
		t.Fatalf("missing sidecar should return nil for lexical fallback, got %#v", fallback)
	}
	if got := combinedSimilarity("The cache health check masks a failed service", rows[0], fallback); got <= 0 {
		t.Fatalf("lexical fallback was not available: %v", got)
	}
}

func TestSemanticSidecarMalformedAndTimeoutFallback(t *testing.T) {
	rows := []StoreEntry{{"id": 1, "input": "The cache masks a failed service."}}
	malformed := filepath.Join(t.TempDir(), "malformed.py")
	if err := os.WriteFile(malformed, []byte("import sys\nprint('not json', flush=True)\n"), 0755); err != nil {
		t.Fatal(err)
	}
	cfg := testConfig(t)
	cfg.SemanticSidecar = malformed
	cfg.SemanticTimeout = time.Second
	if got := semanticScores("The cache hides a service failure", rows, cfg); got != nil {
		t.Fatalf("malformed sidecar should fall back, got %#v", got)
	}

	slow := filepath.Join(t.TempDir(), "slow.py")
	if err := os.WriteFile(slow, []byte("import time\ntime.sleep(2)\n"), 0755); err != nil {
		t.Fatal(err)
	}
	cfg.SemanticSidecar = slow
	cfg.SemanticTimeout = 50 * time.Millisecond
	started := time.Now()
	if got := semanticScores("The cache hides a service failure", rows, cfg); got != nil {
		t.Fatalf("timed-out sidecar should fall back, got %#v", got)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("sidecar timeout was not bounded: %s", elapsed)
	}
	if got := combinedSimilarity("The cache masks a failed service", rows[0], nil); got <= 0 {
		t.Fatalf("lexical fallback unavailable after sidecar failure: %v", got)
	}
}

func TestSemanticSidecarRunsOncePerToolCall(t *testing.T) {
	countRuns := func(counter string) int {
		body, err := os.ReadFile(counter)
		if err != nil {
			return 0
		}
		return len(strings.Split(strings.TrimSpace(string(body)), "\n"))
	}
	newCounter := func(t *testing.T) (counter, sidecar string) {
		t.Helper()
		counter = filepath.Join(t.TempDir(), "runs.txt")
		sidecar = filepath.Join(t.TempDir(), "count_sidecar.py")
		script := fmt.Sprintf("import sys, json\nline = sys.stdin.readline()\nreq = json.loads(line)\nwith open(%q, 'a') as f:\n    f.write('run\\n')\nscores = [{'id': c['id'], 'score': 0.5} for c in req.get('candidates', [])]\nprint(json.dumps({'id': 'recall', 'ok': True, 'scores': scores, 'backend': 'count'}))\nsys.stdout.flush()\n", counter)
		if err := os.WriteFile(sidecar, []byte(script), 0755); err != nil {
			t.Fatal(err)
		}
		return counter, sidecar
	}

	for _, tool := range []string{"think", "think_open"} {
		cfg := testConfig(t)
		counter, sidecar := newCounter(t)
		cfg.SemanticSidecar = sidecar
		cfg.SemanticTimeout = 5 * time.Second
		if _, err := appendStore(cfg.Store, StoreEntry{"ts": time.Now().UTC().Format(time.RFC3339Nano), "status": "oneshot", "input": "The renderer uses a blue gradient.", "meaning": "The landing page has visual styling.", "posterior": map[string]any{"p": 0.5, "because": "mixed evidence"}}, cfg); err != nil {
			t.Fatal(err)
		}
		args := validBreakdown()
		if tool == "think" {
			args["posterior"] = map[string]any{"p": 0.5, "because": "The evidence remains mixed and more testing is required."}
		}
		response := callJSON(t, cfg, "tools/call", 1, map[string]any{"name": tool, "arguments": args})
		if strings.Contains(resultText(response), "REJECTED") {
			t.Fatalf("%s rejected unexpectedly: %s", tool, resultText(response))
		}
		if got := countRuns(counter); got != 1 {
			t.Fatalf("%s launched the semantic sidecar %d times; integrity gate and prior-work note must share one run", tool, got)
		}
	}
}

func TestServeSurvivesOversizedLine(t *testing.T) {
	cfg := testConfig(t)
	big := strings.Repeat("a", 5*1024*1024)
	input := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}`,
		big,
		`{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}`,
	}, "\n") + "\n"
	var out strings.Builder
	if err := serve(strings.NewReader(input), &out, cfg); err != nil {
		t.Fatalf("serve must survive an oversized line, got error: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d responses, want 3 (ping, oversized parse error, ping): %q", len(lines), out.String())
	}
	var pong1, pong2 map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &pong1); err != nil || pong1["id"] != float64(1) {
		t.Fatalf("first ping not answered: %v %s", err, lines[0])
	}
	var parseErr map[string]any
	if err := json.Unmarshal([]byte(lines[1]), &parseErr); err != nil {
		t.Fatalf("oversized line did not produce a JSON-RPC error: %v", err)
	}
	if _, ok := parseErr["error"].(map[string]any); !ok {
		t.Fatalf("oversized line did not produce a JSON-RPC error object: %s", lines[1])
	}
	if err := json.Unmarshal([]byte(lines[2]), &pong2); err != nil || pong2["id"] != float64(2) {
		t.Fatalf("second ping not answered after oversized line: %v %s", err, lines[2])
	}
}

func isolateArgs() map[string]any {
	return map[string]any{
		"input":     "The renderer fails on wide viewports after the viewport refactor.",
		"symptom":   "Layout collapses only when the window exceeds 1280px.",
		"baseline":  map[string]any{"ref": "release before refactor", "outcome": "no collapse at any width"},
		"variables": []any{"viewport meta", "CSS container queries", "grid template", "overflow-x"},
		"trials": []any{
			map[string]any{"name": "T1", "changed": "viewport meta", "from": "width=device-width", "to": "fixed 1280", "level": "surface", "observed": "collapse disappeared at 1280 exactly; returns at 1300 with fixed meta", "outcome": "resolves", "known_good": true},
			map[string]any{"name": "T2", "changed": "CSS container queries", "from": "enabled", "to": "disabled", "level": "intermediate", "observed": "collapse still occurs at 1300 with container queries off", "outcome": "reproduces", "known_good": false},
			map[string]any{"name": "T3", "changed": "grid template", "from": "auto-fit", "to": "fixed 12 col", "level": "low", "observed": "layout width reads 1320px in the grid; still collapses past 1280", "outcome": "reproduces", "known_good": false},
			map[string]any{"name": "T4", "changed": "overflow-x", "from": "hidden", "to": "visible", "level": "surface", "observed": "horizontal scrollbar appears, no layout collapse change", "outcome": "unchanged", "known_good": false},
		},
		"culprit":    "viewport meta",
		"confidence": map[string]any{"p": 0.95, "because": "the fixed viewport meta trial resolved while the container-query and grid trials still reproduced"},
	}
}

func TestIsolateHighConfidenceWithTrialsIsNotBlocked(t *testing.T) {
	cfg := testConfig(t)
	response := callJSON(t, cfg, "tools/call", 1, map[string]any{"name": "think_isolate", "arguments": isolateArgs()})
	if strings.Contains(resultText(response), "UNSUPPORTED_CONFIDENCE") || strings.Contains(resultText(response), "BLOCKED") {
		t.Fatalf("fully-evidenced isolate was false-blocked: %s", resultText(response))
	}
	if !strings.Contains(resultText(response), "RECORDED") {
		t.Fatalf("isolate was not recorded: %s", resultText(response))
	}
}

func TestNoNodeDependency(t *testing.T) {
	for _, path := range []string{"package.json", "package-lock.json", "node_modules", "dist", "smarter_test.mjs", "staged_test.mjs", "upgrade_test.mjs", "discover_test.mjs"} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("Node artifact still present in Go copy: %s", path)
		}
	}
}
