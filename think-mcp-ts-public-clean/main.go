package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const serverVersion = "2.0.0-go"

var stopWords = map[string]bool{"a": true, "an": true, "the": true, "to": true, "of": true, "in": true, "on": true, "and": true, "or": true, "for": true, "with": true, "by": true, "is": true, "are": true, "be": true, "this": true, "that": true, "it": true, "as": true, "at": true, "from": true, "than": true, "then": true, "so": true, "under": true, "over": true, "not": true, "no": true, "if": true, "when": true, "more": true, "less": true, "most": true, "our": true, "we": true, "you": true, "they": true}

var wordRE = regexp.MustCompile(`[a-z0-9]+`)

type Config struct {
	Store           string
	IntegrityLog    string
	LockTimeout     time.Duration
	LockStale       time.Duration
	MinElapsed      time.Duration
	SemanticSidecar string
	SemanticTimeout time.Duration
}

type StoreEntry map[string]any

type RPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Method  string          `json:"method"`
	Params  map[string]any  `json:"params"`
}

type RPCResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id,omitempty"`
	Result  any    `json:"result,omitempty"`
	Error   any    `json:"error,omitempty"`
}

type ToolResult struct {
	Content []map[string]string `json:"content"`
	IsError bool                `json:"isError,omitempty"`
}

type Finding struct{ Code, Message string }
type IntegrityCheck struct {
	EpisodeID string
	Tool      string
	Decision  string
	Findings  []Finding
	Digest    string
}

func main() {
	cfg := loadConfig()
	if err := serve(os.Stdin, os.Stdout, cfg); err != nil {
		fmt.Fprintln(os.Stderr, "think-mcp:", err)
		os.Exit(1)
	}
}

func loadConfig() Config {
	exe, _ := os.Executable()
	root := filepath.Dir(exe)
	if filepath.Base(root) == "bin" {
		root = filepath.Dir(root)
	}
	if root == "." || root == "" {
		root, _ = os.Getwd()
	}
	semanticSidecar := filepath.Join(root, "semantic_recall.py")
	if value, present := os.LookupEnv("THINK_SEMANTIC_SIDECAR"); present {
		semanticSidecar = value
	}
	return Config{
		Store:           envOr("THINK_STORE", filepath.Join(root, "thoughts.json")),
		IntegrityLog:    envOr("THINK_INTEGRITY_LOG", filepath.Join(root, "integrity.jsonl")),
		LockTimeout:     time.Duration(envDuration("THINK_STORE_LOCK_TIMEOUT_MS", 10000)) * time.Millisecond,
		LockStale:       time.Duration(envDuration("THINK_STORE_LOCK_STALE_MS", 60000)) * time.Millisecond,
		MinElapsed:      time.Duration(envDuration("THINK_MIN_ELAPSED_MS", 20000)) * time.Millisecond,
		SemanticSidecar: semanticSidecar,
		SemanticTimeout: time.Duration(envDuration("THINK_SEMANTIC_TIMEOUT_MS", 5000)) * time.Millisecond,
	}
}

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
func envDuration(k string, fallback int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
		return v
	}
	return fallback
}

const maxRequestBytes = 4 * 1024 * 1024

var errRequestTooLong = errors.New("request line exceeds 4 MiB")

// readRequestLine reads one newline-delimited line. A line longer than limit is
// discarded (drained to its newline) so the stream stays in sync and the server
// survives oversized input instead of dying. It returns io.EOF when no more data
// remains.
func readRequestLine(r *bufio.Reader, limit int) (string, error) {
	var line []byte
	for {
		frag, err := r.ReadSlice('\n')
		line = append(line, frag...)
		if err == bufio.ErrBufferFull {
			if len(line) > limit {
				for {
					if _, err := r.ReadSlice('\n'); err != bufio.ErrBufferFull {
						break
					}
				}
				return "", errRequestTooLong
			}
			continue
		}
		if err == io.EOF && len(line) == 0 {
			return "", io.EOF
		}
		return strings.TrimRight(string(line), "\r\n"), err
	}
}

func serve(in io.Reader, out io.Writer, cfg Config) error {
	r := bufio.NewReaderSize(in, 64*1024)
	enc := json.NewEncoder(out)
	for {
		line, err := readRequestLine(r, maxRequestBytes)
		if err == io.EOF {
			return nil
		}
		if err == errRequestTooLong {
			_ = enc.Encode(RPCResponse{JSONRPC: "2.0", ID: nil, Error: map[string]any{"code": -32700, "message": "Parse error: request line exceeds 4 MiB"}})
			continue
		}
		if err != nil {
			return err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var req RPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			_ = enc.Encode(RPCResponse{JSONRPC: "2.0", ID: nil, Error: map[string]any{"code": -32700, "message": "Parse error"}})
			continue
		}
		// MCP notifications do not receive responses.
		if req.ID == nil || string(req.ID) == "null" {
			continue
		}
		resp := dispatch(req, cfg)
		if err := enc.Encode(resp); err != nil {
			return err
		}
	}
}

func dispatch(req RPCRequest, cfg Config) RPCResponse {
	id := any(nil)
	if len(req.ID) > 0 {
		_ = json.Unmarshal(req.ID, &id)
	}
	switch req.Method {
	case "initialize":
		return ok(id, map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{"tools": map[string]any{}}, "serverInfo": map[string]string{"name": "think", "version": serverVersion}})
	case "ping":
		return ok(id, map[string]any{})
	case "tools/list":
		return ok(id, map[string]any{"tools": toolDefinitions()})
	case "tools/call":
		name, _ := req.Params["name"].(string)
		args, _ := req.Params["arguments"].(map[string]any)
		if args == nil {
			args = map[string]any{}
		}
		return callTool(id, name, args, cfg)
	default:
		return fail(id, -32601, "Method not found: "+req.Method)
	}
}

func ok(id any, result any) RPCResponse { return RPCResponse{JSONRPC: "2.0", ID: id, Result: result} }
func fail(id any, code int, msg string) RPCResponse {
	return RPCResponse{JSONRPC: "2.0", ID: id, Error: map[string]any{"code": code, "message": msg}}
}
func toolError(id any, msg string) RPCResponse {
	b, _ := json.Marshal(ToolResult{Content: []map[string]string{{"type": "text", "text": msg}}, IsError: true})
	var v any
	_ = json.Unmarshal(b, &v)
	return ok(id, v)
}
func toolOK(id any, text string) RPCResponse {
	b, _ := json.Marshal(ToolResult{Content: []map[string]string{{"type": "text", "text": text}}})
	var v any
	_ = json.Unmarshal(b, &v)
	return ok(id, v)
}

func schemaObject(required []string, properties map[string]any) map[string]any {
	schema := map[string]any{"type": "object", "properties": properties, "additionalProperties": false}
	if required != nil {
		schema["required"] = required
	}
	return schema
}

func toolDefinitions() []map[string]any {
	str := map[string]any{"type": "string"}
	num := map[string]any{"type": "number"}
	integer := map[string]any{"type": "integer"}
	boolean := map[string]any{"type": "boolean"}
	strs := map[string]any{"type": "array", "items": str}
	interp := schemaObject([]string{"reading", "category"}, map[string]any{"reading": str, "category": str})
	test := schemaObject([]string{"name", "hypothesis", "metric", "decision_rule", "discriminates"}, map[string]any{"name": str, "hypothesis": str, "metric": str, "decision_rule": str, "adversarial": boolean, "discriminates": integer})
	belief := schemaObject([]string{"p", "because"}, map[string]any{"p": num, "because": str})
	common := map[string]any{"input": str, "interpretations": map[string]any{"type": "array", "items": interp}, "meaning": str, "how_to_prove": strs, "failure_signature": str, "success_signature": str, "prior": belief, "posterior": belief, "tests": map[string]any{"type": "array", "items": test}}
	openCommon := map[string]any{"input": str, "interpretations": map[string]any{"type": "array", "items": interp}, "meaning": str, "how_to_prove": strs, "failure_signature": str, "success_signature": str, "prior": belief, "tests": map[string]any{"type": "array", "items": test}, "scope": strs}
	evidence := schemaObject([]string{"test", "ran", "observed", "verdict"}, map[string]any{"test": str, "ran": str, "observed": str, "verdict": str, "artifact": str})
	conclude := schemaObject([]string{"id", "evidence", "posterior"}, map[string]any{"id": integer, "evidence": map[string]any{"type": "array", "items": evidence}, "posterior": belief})
	priorArt := schemaObject([]string{"query", "top_result", "verdict"}, map[string]any{"query": str, "top_result": str, "verdict": str})
	invariant := schemaObject([]string{"name", "computed_from", "why_independent", "expected", "measured", "holds"}, map[string]any{"name": str, "computed_from": str, "why_independent": str, "expected": str, "measured": str, "holds": boolean})
	source := schemaObject([]string{"path", "kind", "excerpt"}, map[string]any{"path": str, "kind": str, "excerpt": str})
	trial := schemaObject([]string{"name", "changed", "from", "to", "level", "observed", "outcome"}, map[string]any{"name": str, "changed": str, "from": str, "to": str, "known_good": boolean, "level": str, "observed": str, "outcome": str})
	comparison := schemaObject([]string{"a", "b"}, map[string]any{"a": str, "b": str, "agreement": str})
	return []map[string]any{
		{"name": "think", "description": "Create a structured, falsifiable one-shot breakdown.", "inputSchema": schemaObject([]string{"input", "interpretations", "meaning", "how_to_prove", "failure_signature", "success_signature", "prior", "posterior", "tests"}, common)},
		{"name": "get_thoughts", "description": "Browse the persistent breakdown ledger.", "inputSchema": schemaObject(nil, map[string]any{"id": integer, "limit": integer, "status": str, "query": str, "full": boolean})},
		{"name": "think_open", "description": "Pre-register a claim and tests before evidence exists.", "inputSchema": schemaObject([]string{"input", "interpretations", "meaning", "how_to_prove", "failure_signature", "success_signature", "prior", "tests"}, openCommon)},
		{"name": "think_conclude", "description": "Conclude an opened breakdown with evidence and a posterior.", "inputSchema": conclude},
		{"name": "think_recall", "description": "Find similar prior claims and belief conflicts.", "inputSchema": schemaObject([]string{"query"}, map[string]any{"query": str, "limit": integer, "min_sim": num})},
		{"name": "think_verify", "description": "Establish a claim with artifacts and an independent invariant.", "inputSchema": schemaObject([]string{"claim", "sources", "discriminator", "invariants", "sample", "blind_to", "verdict"}, map[string]any{"claim": str, "sources": map[string]any{"type": "array", "items": source}, "comparison": comparison, "discriminator": schemaObject([]string{"would_fail_if", "ran", "observed", "outcome"}, map[string]any{"name": str, "would_fail_if": str, "ran": str, "observed": str, "outcome": str}), "invariants": map[string]any{"type": "array", "items": invariant}, "sample": schemaObject([]string{"n", "selection", "includes_edge_cases"}, map[string]any{"n": num, "selection": str, "includes_edge_cases": boolean}), "blind_to": strs, "verdict": schemaObject([]string{"established", "not_established"}, map[string]any{"established": str, "not_established": str})})},
		{"name": "think_isolate", "description": "Diagnose a failure with controlled one-variable trials.", "inputSchema": schemaObject([]string{"input", "symptom", "baseline", "variables", "trials", "culprit", "confidence"}, map[string]any{"input": str, "symptom": str, "baseline": schemaObject([]string{"ref", "outcome"}, map[string]any{"ref": str, "outcome": str}), "variables": strs, "trials": map[string]any{"type": "array", "items": trial}, "culprit": str, "confidence": belief})},
		{"name": "think_discover", "description": "Evaluate a quantitative, falsifiable cross-domain theory.", "inputSchema": schemaObject([]string{"input", "domain_a", "domain_b", "shared_formal_object", "transported_prediction", "refuting_experiment", "prior_art", "interpretations", "meaning", "how_to_prove", "failure_signature", "success_signature", "prior", "posterior", "tests"}, map[string]any{"input": str, "domain_a": str, "domain_b": str, "shared_formal_object": str, "transported_prediction": str, "refuting_experiment": str, "prior_art": map[string]any{"type": "array", "items": priorArt}, "interpretations": map[string]any{"type": "array", "items": interp}, "meaning": str, "how_to_prove": strs, "failure_signature": str, "success_signature": str, "prior": belief, "posterior": belief, "tests": map[string]any{"type": "array", "items": test}})},
		{"name": "think_calibrate", "description": "Summarize belief calibration and adversarial outcomes.", "inputSchema": schemaObject(nil, map[string]any{})},
	}
}

func schemaErrors(value any, schema map[string]any, path string) []string {
	actual := "null"
	switch value.(type) {
	case map[string]any:
		actual = "object"
	case []any:
		actual = "array"
	case string:
		actual = "string"
	case bool:
		actual = "boolean"
	case float64:
		actual = "number"
	}
	typeName, _ := schema["type"].(string)
	if typeName == "object" {
		obj, ok := value.(map[string]any)
		if !ok {
			return []string{fmt.Sprintf("%s must be an object (got %s)", path, actual)}
		}
		out := []string{}
		if req, ok := schema["required"].([]string); ok {
			for _, k := range req {
				if _, exists := obj[k]; !exists {
					out = append(out, path+"."+k+" is required")
				}
			}
		}
		props, _ := schema["properties"].(map[string]any)
		if schema["additionalProperties"] == false {
			for k := range obj {
				if _, exists := props[k]; !exists {
					out = append(out, path+"."+k+" is not allowed")
				}
			}
		}
		for k, child := range props {
			if v, exists := obj[k]; exists {
				if childMap, ok := child.(map[string]any); ok {
					out = append(out, schemaErrors(v, childMap, path+"."+k)...)
				}
			}
		}
		return out
	}
	if typeName == "array" {
		arr, ok := value.([]any)
		if !ok {
			return []string{fmt.Sprintf("%s must be an array (got %s)", path, actual)}
		}
		if item, ok := schema["items"].(map[string]any); ok {
			out := []string{}
			for i, v := range arr {
				out = append(out, schemaErrors(v, item, fmt.Sprintf("%s[%d]", path, i))...)
			}
			return out
		}
		return nil
	}
	valid := (typeName == "string" && actual == "string") || (typeName == "boolean" && actual == "boolean") || (typeName == "number" && actual == "number") || (typeName == "integer" && actual == "number" && value.(float64) == math.Trunc(value.(float64)))
	if !valid && typeName != "" {
		return []string{fmt.Sprintf("%s must be a %s (got %s)", path, typeName, actual)}
	}
	return nil
}

func schemaFor(name string) map[string]any {
	for _, t := range toolDefinitions() {
		if t["name"] == name {
			return t["inputSchema"].(map[string]any)
		}
	}
	return nil
}

func callTool(id any, name string, args map[string]any, cfg Config) RPCResponse {
	if name == "think_open" {
		if _, exists := args["posterior"]; exists {
			return toolError(id, "Schema validation failed:\n- posterior is not allowed at open — run tests, then call think_conclude with the posterior.")
		}
	}
	if errors := schemaErrors(args, schemaFor(name), "arguments"); len(errors) > 0 {
		return toolError(id, "Schema validation failed:\n- "+strings.Join(errors, "\n- "))
	}
	check, semantic := integrityBefore(name, args, cfg)
	if check.Decision == "BLOCK" || check.Decision == "REQUIRE_VERIFY" {
		integrityAfter(check, true, cfg)
		return toolError(id, integrityBlock(check))
	}
	var text string
	var err error
	switch name {
	case "think":
		text, err = toolThink(args, cfg, semantic)
	case "get_thoughts":
		text, err = toolGet(args, cfg)
	case "think_calibrate":
		text, err = toolCalibrate(cfg)
	case "think_open":
		text, err = toolOpen(args, cfg, semantic)
	case "think_conclude":
		text, err = toolConcludeAtomic(args, cfg)
	case "think_recall":
		text, err = toolRecall(args, cfg)
	case "think_verify":
		text, err = toolVerify(args, cfg)
	case "think_isolate":
		text, err = toolIsolate(args, cfg, semantic)
	case "think_discover":
		text, err = toolDiscover(args, cfg)
	default:
		return fail(id, -32602, "Unknown tool: "+name)
	}
	integrityAfter(check, err != nil, cfg)
	if err != nil {
		return toolError(id, err.Error()+integrityNote(check))
	}
	return toolOK(id, text+integrityNote(check))
}

func integrityBefore(tool string, args map[string]any, cfg Config) (IntegrityCheck, map[int]float64) {
	claim := claimFor(args, cfg)
	digest := shortHash(claim)
	c := IntegrityCheck{EpisodeID: fmt.Sprintf("%d", time.Now().UnixNano()), Tool: tool, Decision: "CLEAR", Digest: digest}
	p, hasP := probability(args, cfg)
	hasEvidence := len(asSlice(args["evidence"])) > 0 || asString(args["artifact"]) != "" || len(asSlice(args["trials"])) > 0
	if hasP && p >= .9 && p <= 1 && !hasEvidence && tool != "think_verify" && tool != "think_open" {
		c.Findings = append(c.Findings, Finding{"UNSUPPORTED_CONFIDENCE", fmt.Sprintf("p=%g has no recorded evidence; use think_open → think_conclude or independent verification.", p)})
		c.Decision = "REQUIRE_VERIFY"
	}
	var semantic map[int]float64
	if claim != "" && hasP {
		if rows, _ := loadStore(cfg.Store); rows != nil {
			semantic = semanticScores(claim, rows, cfg)
			for _, row := range rows {
				old, ok := finalP(row)
				sim := combinedSimilarity(claim, row, semantic)
				if !ok || sim < .65 {
					continue
				}
				if (old <= .35 && p >= .65) || (old >= .65 && p <= .35) {
					c.Findings = append(c.Findings, Finding{"CONTRADICTION", fmt.Sprintf("strongly similar record #%v concluded p=%g while this call holds p=%g; reconcile the changed evidence.", row["id"], old, p)})
					if c.Decision == "CLEAR" {
						c.Decision = "WARN"
					}
					break
				}
			}
		}
	}
	input, meaning := asString(args["input"]), asString(args["meaning"])
	if input != "" && meaning != "" && sharedContent(input, meaning) == 0 {
		c.Findings = append(c.Findings, Finding{"GOAL_DRIFT", "meaning shares no topical terms with input; the breakdown changed subject before it started."})
		c.Decision = "BLOCK"
	}
	recordIntegrity(c, "preflight", false, cfg)
	return c, semantic
}
func integrityAfter(c IntegrityCheck, failed bool, cfg Config) {
	recordIntegrity(c, "postflight", failed, cfg)
}
func integrityBlock(c IntegrityCheck) string {
	parts := []string{}
	for _, f := range c.Findings {
		parts = append(parts, f.Code+": "+f.Message)
	}
	return "Integrity gate BLOCKED (" + joinCodes(c.Findings) + "): " + strings.Join(parts, " | ")
}
func integrityNote(c IntegrityCheck) string {
	if len(c.Findings) == 0 {
		return ""
	}
	parts := []string{}
	for _, f := range c.Findings {
		parts = append(parts, f.Code+": "+f.Message)
	}
	return "\nINTEGRITY " + c.Decision + ": " + strings.Join(parts, " | ")
}
func joinCodes(fs []Finding) string {
	a := []string{}
	for _, f := range fs {
		a = append(a, f.Code)
	}
	return strings.Join(a, ", ")
}
func recordIntegrity(c IntegrityCheck, phase string, failed bool, cfg Config) {
	e := map[string]any{"event": "integrity", "phase": phase, "episodeId": c.EpisodeID, "tool": c.Tool, "decision": c.Decision, "findingCodes": codes(c.Findings), "claimDigest": c.Digest, "failed": failed, "at": time.Now().UTC().Format(time.RFC3339Nano)}
	b, err := json.Marshal(e)
	if err != nil {
		return
	}
	_ = os.MkdirAll(filepath.Dir(cfg.IntegrityLog), 0755)
	_ = withLock(cfg.IntegrityLog, cfg, func() error {
		f, err := os.OpenFile(cfg.IntegrityLog, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = f.Write(append(b, '\n'))
		return err
	})
}
func codes(fs []Finding) []string {
	a := []string{}
	for _, f := range fs {
		a = append(a, f.Code)
	}
	return a
}
func shortHash(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:])[:16] }
func probability(a map[string]any, cfg Config) (float64, bool) {
	for _, k := range []string{"posterior", "confidence", "prior"} {
		if b, ok := a[k].(map[string]any); ok {
			if p, ok := b["p"].(float64); ok {
				return p, true
			}
		}
	}
	if id, ok := asInt(a["id"]); ok {
		if rows, _ := loadStore(cfg.Store); rows != nil {
			for _, r := range rows {
				if x, ok := asInt(r["id"]); ok && x == id {
					if b, ok := r["posterior"].(map[string]any); ok {
						if p, ok := b["p"].(float64); ok {
							return p, true
						}
					}
				}
			}
		}
	}
	return 0, false
}
func claimFor(a map[string]any, cfg Config) string {
	for _, k := range []string{"input", "claim", "symptom", "query"} {
		if s := asString(a[k]); s != "" {
			return s
		}
	}
	if id, ok := asInt(a["id"]); ok {
		if rows, _ := loadStore(cfg.Store); rows != nil {
			for _, r := range rows {
				if x, ok := asInt(r["id"]); ok && x == id {
					return asString(r["input"])
				}
			}
		}
	}
	return ""
}

func toolThink(a map[string]any, cfg Config, semantic map[int]float64) (string, error) {
	violations, warnings := enforceBreakdown(a, true)
	if len(violations) > 0 {
		return "", errors.New("REJECTED — incomplete/gamed breakdown. Fix and call think again:\n- " + strings.Join(violations, "\n- "))
	}
	e := copyMap(a)
	e["id"] = 0
	e["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
	e["status"] = "oneshot"
	entry, err := appendStore(cfg.Store, e, cfg)
	if err != nil {
		return "", err
	}
	warnNote := ""
	if len(warnings) > 0 {
		warnNote = "\n⚠ not blocking, but look:\n- " + strings.Join(warnings, "\n- ")
	}
	recallNote := priorWorkNote(cfg.Store, asString(entry["input"]), asIntDefault(entry["id"]), asFloatPtr(entry["posterior"]), semantic, cfg)
	return fmt.Sprintf("Breakdown #%d accepted: %d interpretations, %d proofs, %d tests (%d adversarial). Now act on it.\nstatus: oneshot (UNVERIFIED — no test was actually run or checked). If you will ACT on this claim (config edit, deploy, irreversible step), re-stake it via think_open → run the tests → think_conclude.%s%s", asIntDefault(entry["id"]), len(asSlice(entry["interpretations"])), len(asSlice(entry["how_to_prove"])), len(asSlice(entry["tests"])), countAdversarial(asSlice(entry["tests"])), warnNote, recallNote), nil
}
func toolOpen(a map[string]any, cfg Config, semantic map[int]float64) (string, error) {
	if _, ok := a["posterior"]; ok {
		return "", errors.New("posterior is not allowed at open — that is the point. Run the pre-registered tests first, then think_conclude with evidence.")
	}
	violations, warnings := enforceBreakdown(a, false)
	if len(violations) > 0 {
		return "", errors.New("REJECTED — incomplete/gamed breakdown. Fix and call think_open again:\n- " + strings.Join(violations, "\n- "))
	}
	e := copyMap(a)
	e["id"] = 0
	e["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
	e["openedAt"] = e["ts"]
	e["status"] = "open"
	entry, err := appendStore(cfg.Store, e, cfg)
	if err != nil {
		return "", err
	}
	warnNote := ""
	if len(warnings) > 0 {
		warnNote = "\n⚠ " + strings.Join(warnings, "\n⚠ ")
	}
	staleNote := staleOpenNote(cfg.Store, asIntDefault(entry["id"]))
	tests := asSlice(entry["tests"])
	scope := asStringSlice(entry["scope"])
	scopeNote := ""
	if len(scope) > 0 {
		scopeNote = ", scope: " + strings.Join(scope, ", ")
	}
	testNames := testNames(tests)
	recallNote := priorWorkNote(cfg.Store, asString(entry["input"]), asIntDefault(entry["id"]), asFloatPtr(entry["prior"]), semantic, cfg)
	return fmt.Sprintf("Breakdown #%d OPEN at %s — %d tests pre-registered (%d adversarial)%s.\nNow RUN each test for real. Save each test's raw output to a file (artifact).\nThen call think_conclude with id=%d, one evidence item per test (%s), and your posterior.\nAdversarial evidence MUST carry an artifact file created after this moment; the observed excerpt must appear verbatim in it.\nConclude before %ds elapse is refused.%s%s%s", asIntDefault(entry["id"]), asString(entry["openedAt"]), len(tests), countAdversarial(tests), scopeNote, asIntDefault(entry["id"]), testNames, int(cfg.MinElapsed.Seconds()), warnNote, staleNote, recallNote), nil
}
func countAdversarial(tests []any) int {
	count := 0
	for _, raw := range tests {
		if t, ok := raw.(map[string]any); ok && t["adversarial"] == true {
			count++
		}
	}
	return count
}
func testNames(tests []any) string {
	out := []string{}
	for _, raw := range tests {
		if t, ok := raw.(map[string]any); ok {
			out = append(out, asString(t["name"]))
		}
	}
	return strings.Join(out, ", ")
}
func findTestByName(tests []any, name string) (map[string]any, bool) {
	for _, raw := range tests {
		if t, ok := raw.(map[string]any); ok && asString(t["name"]) == name {
			return t, true
		}
	}
	return nil, false
}
func asStringSlice(v any) []string {
	out := []string{}
	for _, raw := range asSlice(v) {
		if s, ok := raw.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
func asFloatPtr(v any) *float64 {
	if m, ok := v.(map[string]any); ok {
		if p, ok := m["p"].(float64); ok {
			return &p
		}
	}
	return nil
}
func staleOpenNote(path string, exclude int) string {
	rows, err := loadStore(path)
	if err != nil {
		return ""
	}
	stale := []string{}
	for _, r := range rows {
		if asIntDefault(r["id"]) == exclude || asString(r["status"]) != "open" {
			continue
		}
		t, err := time.Parse(time.RFC3339Nano, asString(r["openedAt"]))
		if err == nil && time.Since(t) > 24*time.Hour {
			stale = append(stale, fmt.Sprintf("#%d", asIntDefault(r["id"])))
		}
	}
	if len(stale) == 0 {
		return ""
	}
	return "\n⚠ stale OPEN breakdowns never concluded: " + strings.Join(stale, ", ") + " — claims you promised to test and abandoned. Conclude them or note why they died (think_calibrate lists them)."
}
func priorWorkNote(storePath, query string, exclude int, currentP *float64, semantic map[int]float64, cfg Config) string {
	if len(strings.TrimSpace(query)) < 10 {
		return ""
	}
	rows, err := loadStore(storePath)
	if err != nil {
		return ""
	}
	if semantic == nil {
		semantic = semanticScores(query, rows, cfg)
	}
	type hit struct {
		row StoreEntry
		sim float64
	}
	hits := []hit{}
	for _, row := range rows {
		if asIntDefault(row["id"]) == exclude {
			continue
		}
		sim := combinedSimilarity(query, row, semantic)
		if sim >= .35 {
			hits = append(hits, hit{row: row, sim: sim})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].sim == hits[j].sim {
			return asIntDefault(hits[i].row["id"]) < asIntDefault(hits[j].row["id"])
		}
		return hits[i].sim > hits[j].sim
	})
	if len(hits) > 3 {
		hits = hits[:3]
	}
	if len(hits) == 0 {
		return ""
	}
	lines := []string{}
	contradictions := []string{}
	for _, h := range hits {
		marker := "~"
		if h.sim >= .65 {
			marker = "≈"
		}
		lines = append(lines, fmt.Sprintf("  %s sim %.2f %s", marker, h.sim, rowSummary(h.row)))
		if currentP == nil || h.sim < .65 {
			continue
		}
		oldP, ok := finalP(h.row)
		if !ok {
			continue
		}
		if (oldP <= .35 && *currentP >= .65) || (oldP >= .65 && *currentP <= .35) {
			contradictions = append(contradictions, fmt.Sprintf("  ⚠ CONTRADICTION: #%d concluded p=%g on a strongly-similar claim (sim %.2f) while you now hold p=%g — reconcile before acting (get_thoughts {id: %d}). Something changed, or one of the two beliefs is wrong.", asIntDefault(h.row["id"]), oldP, h.sim, *currentP, asIntDefault(h.row["id"])))
		}
	}
	result := "\n◆ PRIOR WORK — similar past records:\n" + strings.Join(lines, "\n")
	if len(contradictions) > 0 {
		result += "\n" + strings.Join(contradictions, "\n")
	}
	return result
}
func toolConcludeAtomic(a map[string]any, cfg Config) (string, error) {
	id, ok := asInt(a["id"])
	if !ok {
		return "", errors.New("id must be an integer")
	}
	var result string
	err := withLock(cfg.Store, cfg, func() error {
		rows, err := loadStore(cfg.Store)
		if err != nil {
			return err
		}
		idx := -1
		for i, row := range rows {
			if x, _ := asInt(row["id"]); x == id {
				idx = i
				break
			}
		}
		if idx < 0 {
			return fmt.Errorf("REJECTED — no breakdown with id %d. Open one with think_open first.", id)
		}
		row := rows[idx]
		status := asString(row["status"])
		if status == "verified" {
			return fmt.Errorf("REJECTED — breakdown #%d is already concluded (verified at %s).", id, asString(row["concludedAt"]))
		}
		if status != "open" {
			if status == "" {
				status = "legacy one-shot"
			}
			return fmt.Errorf("REJECTED — breakdown #%d is a %s record, not an open staged breakdown.", id, status)
		}
		opened, _ := time.Parse(time.RFC3339Nano, asString(row["openedAt"]))
		if !opened.IsZero() && time.Since(opened) < cfg.MinElapsed {
			elapsed := int(time.Since(opened).Seconds())
			floor := int(cfg.MinElapsed.Seconds())
			return fmt.Errorf("REJECTED — only %ds since think_open. Pre-registered tests cannot have run yet (floor %ds). Run the tests, save their output, then conclude.", elapsed, floor)
		}
		post, ok := a["posterior"].(map[string]any)
		if !ok {
			return errors.New("posterior must be an object")
		}
		p, _ := post["p"].(float64)
		if p < .01 || p > .99 {
			return errors.New("posterior must be between 0.01 and 0.99")
		}
		evidence := asSlice(a["evidence"])
		tests := asSlice(row["tests"])
		declared := map[string]bool{}
		adversarial := map[string]bool{}
		for _, raw := range tests {
			if t, ok := raw.(map[string]any); ok {
				name := asString(t["name"])
				declared[name] = true
				adversarial[name] = t["adversarial"] == true
			}
		}
		violations := []string{}
		warnings := []string{}
		seen := map[string]bool{}
		observedTexts := []string{}
		ranTexts := []string{}
		failures, successes := 0, 0
		for i, raw := range evidence {
			e, ok := raw.(map[string]any)
			if !ok {
				return errors.New("each evidence item must be an object")
			}
			name := strings.TrimSpace(asString(e["test"]))
			ran := strings.TrimSpace(asString(e["ran"]))
			observed := asString(e["observed"])
			verdictText := strings.TrimSpace(asString(e["verdict"]))
			if !declared[name] {
				violations = append(violations, fmt.Sprintf("evidence[%d].test = %q does not match any pre-registered test (%s).", i, name, testNames(tests)))
				continue
			}
			if seen[name] {
				violations = append(violations, fmt.Sprintf("evidence[%d] duplicates test %q — one evidence item per test.", i, name))
			}
			seen[name] = true
			for key, value := range map[string]string{"ran": ran, "observed": observed, "verdict": verdictText} {
				if len(value) > 8000 {
					violations = append(violations, fmt.Sprintf("evidence[%d].%s exceeds 8000 chars.", i, key))
				}
			}
			if len(ran) < 8 || lowEffort(ran, 2) {
				violations = append(violations, fmt.Sprintf("evidence[%d].ran is too thin — give the exact command/tool call you executed.", i))
			}
			if len(strings.TrimSpace(observed)) < 30 || lowEffort(observed, 5) {
				violations = append(violations, fmt.Sprintf("evidence[%d].observed is too thin (≥30 chars of verbatim output required).", i))
			}
			if len(verdictText) < 10 || len(content(verdictText)) < 2 {
				violations = append(violations, fmt.Sprintf("evidence[%d].verdict is too thin — state which decision_rule branch the observation hit.", i))
			}
			observedTexts = append(observedTexts, observed)
			ranTexts = append(ranTexts, norm(ran))
			if t, ok := findTestByName(tests, name); ok {
				testTerms := map[string]bool{}
				for _, term := range content(asString(t["hypothesis"]) + " " + asString(t["metric"]) + " " + asString(t["decision_rule"])) {
					testTerms[term] = true
				}
				evTerms := content(observed + " " + verdictText)
				engages := false
				for _, term := range evTerms {
					if testTerms[term] {
						engages = true
						break
					}
				}
				if len(testTerms) > 0 && len(evTerms) > 0 && !engages {
					violations = append(violations, fmt.Sprintf("evidence[%d] does not engage test %q — observed/verdict share no terms with its hypothesis/metric/decision_rule.", i, name))
				}
			}
			if len(ran) < 8 || len(verdictText) < 2 || len(strings.TrimSpace(observed)) < 30 {
				continue
			}
			verdict := strings.ToLower(asString(e["verdict"]))
			if strings.Contains(verdict, "fail") || strings.Contains(verdict, "reject") || strings.Contains(verdict, "error") || strings.Contains(verdict, "false") {
				failures++
			} else if strings.Contains(verdict, "pass") || strings.Contains(verdict, "success") || strings.Contains(verdict, "true") {
				successes++
			}
			artifact := asString(e["artifact"])
			if adversarial[name] && artifact == "" {
				violations = append(violations, fmt.Sprintf("evidence[%d] evidences the ADVERSARIAL test %q but has no artifact — the refutation attempt must leave a raw-output file created after open.", i, name))
			}
			if artifact != "" {
				st, body, err := readArtifactNoFollow(artifact)
				if err != nil {
					violations = append(violations, fmt.Sprintf("evidence[%d].artifact could not be read (%v).", i, err))
					continue
				}
				if st.Size() < 20 {
					violations = append(violations, fmt.Sprintf("evidence[%d].artifact is %d bytes — too small to be real test output.", i, st.Size()))
					continue
				}
				if !opened.IsZero() && st.ModTime().Before(opened.Add(-2*time.Second)) {
					violations = append(violations, fmt.Sprintf("evidence[%d].artifact predates the open (%s < openedAt %s) — evidence files must be produced by tests run AFTER think_open.", i, st.ModTime().UTC().Format(time.RFC3339Nano), asString(row["openedAt"])))
					continue
				}
				needle := norm(observed)
				if len(needle) > 800 {
					needle = needle[:800]
				}
				if len(needle) >= 15 && !strings.Contains(norm(string(body)), needle) {
					violations = append(violations, fmt.Sprintf("evidence[%d].observed is not present in artifact %s — the excerpt must be verbatim from the file.", i, artifact))
				}
				e["artifact_meta"] = map[string]any{"mtime": st.ModTime().UTC().Format(time.RFC3339Nano), "size": st.Size(), "sha256": fmt.Sprintf("%x", sha256.Sum256(body))}
			}
		}
		for _, raw := range tests {
			if t, ok := raw.(map[string]any); ok {
				name := asString(t["name"])
				if !seen[name] {
					violations = append(violations, fmt.Sprintf("test %q has no evidence — every pre-registered test must be evidenced (that is the contract you signed at open).", name))
				}
			}
		}
		for _, i := range nearDupes(observedTexts, .9) {
			violations = append(violations, fmt.Sprintf("evidence[%d].observed is nearly identical to an earlier item — different tests must show their own output, not one pasted result.", i))
		}
		for i, ran := range ranTexts {
			if ran == "" {
				continue
			}
			for j := 0; j < i; j++ {
				if ran == ranTexts[j] {
					warnings = append(warnings, fmt.Sprintf("evidence[%d].ran is the identical command as evidence[%d] — if one run evidences two tests, each must read a DIFFERENT observable from its output.", i, j))
					break
				}
			}
		}
		if len(violations) > 0 {
			return fmt.Errorf("REJECTED — conclude does not close #%d (it stays OPEN; rerun tests and retry):\n- %s", id, strings.Join(violations, "\n- "))
		}
		if failures > successes && p > 0.65 {
			return errors.New("posterior is too favorable for evidence containing more failures than successes")
		}
		if successes > failures && p < 0.35 {
			return errors.New("posterior is too unfavorable for evidence containing more successes than failures")
		}
		row["posterior"] = post
		row["evidence"] = evidence
		row["concludedAt"] = time.Now().UTC().Format(time.RFC3339Nano)
		row["status"] = "verified"
		rows[idx] = row
		b, _ := json.MarshalIndent(rows, "", "  ")
		tmp := fmt.Sprintf("%s.%d.tmp", cfg.Store, os.Getpid())
		if err := os.WriteFile(tmp, b, 0644); err != nil {
			return err
		}
		if err := os.Rename(tmp, cfg.Store); err != nil {
			_ = os.Remove(tmp)
			return err
		}
		attested := 0
		for _, raw := range evidence {
			if e, ok := raw.(map[string]any); ok {
				if meta, ok := e["artifact_meta"].(map[string]any); ok && len(meta) > 0 {
					attested++
				}
			}
		}
		postP, _ := post["p"].(float64)
		priorP, _ := row["prior"].(map[string]any)
		priorValue, _ := priorP["p"].(float64)
		warnNote := ""
		if len(warnings) > 0 {
			warnNote = "\n⚠ " + strings.Join(warnings, "\n⚠ ")
		}
		result = fmt.Sprintf("Breakdown #%d VERIFIED: %d/%d tests evidenced, %d artifact-attested, prior %g → posterior %g (swing %.2f). Recorded at %s.%s", id, len(evidence), len(tests), attested, priorValue, postP, postP-priorValue, asString(row["concludedAt"]), warnNote)
		return nil
	})
	return result, err
}

func toolGet(a map[string]any, cfg Config) (string, error) {
	rows, err := loadStore(cfg.Store)
	if err != nil {
		return "", err
	}
	if len(rows) == 0 {
		return "No breakdowns recorded yet.", nil
	}
	if id, ok := asInt(a["id"]); ok {
		for _, r := range rows {
			if x, _ := asInt(r["id"]); x == id {
				b, _ := json.MarshalIndent(r, "", "  ")
				return string(b), nil
			}
		}
		return fmt.Sprintf("No record with id %d.", id), nil
	}
	status := asString(a["status"])
	query := strings.TrimSpace(asString(a["query"]))
	semantic := semanticScores(query, rows, cfg)
	filtered := make([]StoreEntry, 0, len(rows))
	for _, r := range rows {
		if status != "" && asString(r["status"]) != status {
			continue
		}
		if query != "" && combinedSimilarity(query, r, semantic) < .2 {
			continue
		}
		filtered = append(filtered, r)
	}
	if len(filtered) == 0 {
		if status != "" {
			return "No records with status " + status + ".", nil
		}
		return "No records matched the query.", nil
	}
	if a["full"] == true {
		b, _ := json.MarshalIndent(filtered, "", "  ")
		return string(b), nil
	}
	limit := 10
	if x, ok := asInt(a["limit"]); ok && x > 0 {
		limit = x
	}
	if limit > 25 {
		limit = 25
	}
	start := len(filtered) - limit
	if start < 0 {
		start = 0
	}
	out := []string{}
	for _, r := range filtered[start:] {
		out = append(out, rowSummary(r))
	}
	return strings.Join(out, "\n"), nil
}
func toolRecall(a map[string]any, cfg Config) (string, error) {
	q := asString(a["query"])
	rows, _ := loadStore(cfg.Store)
	semantic := semanticScores(q, rows, cfg)
	hits := []struct {
		r StoreEntry
		s float64
	}{}
	for _, r := range rows {
		if s := combinedSimilarity(q, r, semantic); s >= .35 {
			hits = append(hits, struct {
				r StoreEntry
				s float64
			}{r, s})
		}
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].s > hits[j].s })
	if len(hits) > 5 {
		hits = hits[:5]
	}
	if len(hits) == 0 {
		return "No past record is similar to this query.", nil
	}
	out := []string{fmt.Sprintf("%d similar past record(s):", len(hits))}
	for _, h := range hits {
		out = append(out, fmt.Sprintf("  sim %.2f %s", h.s, rowSummary(h.r)))
	}
	return strings.Join(out, "\n"), nil
}

var verifyDiffRE = regexp.MustCompile(`(?i)\b(diff|difference|delta|compar\w*|match\w*|agree\w*|versus|\bvs\.?\b|both|against each other|between them|consistent with)\b`)
var verifyFillerRE = regexp.MustCompile(`(?i)^(it (is|'s) independent|because it (is|'s) independent|independent|obviously|clearly|by definition|self.?evident|trust me|standard|well.?known)\.?$`)

func verifyValidate(a map[string]any) ([]string, []string) {
	v, warn := []string{}, []string{}
	claim := strings.TrimSpace(asString(a["claim"]))
	if len(claim) < 15 || len(content(claim)) < 3 {
		v = append(v, fmt.Sprintf("\"claim\" must be a real claim (>=15 chars of content), got %q.", claim))
	}
	sources := asSlice(a["sources"])
	if len(sources) == 0 {
		v = append(v, "\"sources\" must have >=1 artifact you actually read or ran. A claim from memory is not a claim.")
	}
	if len(sources) > 24 {
		v = append(v, "\"sources\" capped at 24.")
	}
	for i, raw := range sources {
		s, _ := raw.(map[string]any)
		p := strings.TrimSpace(asString(s["path"]))
		kind := strings.TrimSpace(asString(s["kind"]))
		ex := strings.TrimSpace(asString(s["excerpt"]))
		if p == "" {
			v = append(v, fmt.Sprintf("sources[%d].path is required.", i))
		} else if _, err := os.Stat(p); err != nil {
			v = append(v, fmt.Sprintf("sources[%d].path does not exist on disk: %s. Cite a real artifact.", i, p))
		}
		if kind != "read" && kind != "executed" {
			v = append(v, fmt.Sprintf("sources[%d].kind must be \"read\" or \"executed\" (got %q).", i, kind))
		}
		if len(ex) < 30 {
			v = append(v, fmt.Sprintf("sources[%d].excerpt must be >=30 chars quoted verbatim from the artifact.", i))
		}
	}
	if len(sources) > 0 && !hasSourceKind(sources, "executed") {
		warn = append(warn, "No source has kind:\"executed\" — every source was read, none was run. Reading tells you what it claims; running tells you what it does.")
	}
	d, dok := a["discriminator"].(map[string]any)
	if !dok {
		v = append(v, "\"discriminator\" is required: the case that WOULD have failed if the claim were false.")
	} else {
		wf := strings.TrimSpace(asString(d["would_fail_if"]))
		obs := strings.TrimSpace(asString(d["observed"]))
		if len(wf) < 20 || lowEffort(wf, 4) {
			v = append(v, "\"discriminator.would_fail_if\" must state concretely what a FALSE claim would have produced here.")
		}
		if strings.TrimSpace(asString(d["ran"])) == "" {
			v = append(v, "\"discriminator.ran\" must be the exact command/tool you executed.")
		}
		if len(obs) < 30 {
			v = append(v, "\"discriminator.observed\" must be >=30 chars of real output.")
		}
		outcome := strings.TrimSpace(asString(d["outcome"]))
		if outcome != "held" && outcome != "failed" {
			v = append(v, "\"discriminator.outcome\" must be \"held\" or \"failed\".")
		}
	}
	invs := asSlice(a["invariants"])
	cmp, hasCmp := a["comparison"].(map[string]any)
	sides := []string{}
	if hasCmp {
		if x := strings.TrimSpace(asString(cmp["a"])); x != "" {
			sides = append(sides, x)
		}
		if x := strings.TrimSpace(asString(cmp["b"])); x != "" {
			sides = append(sides, x)
		}
	}
	if len(invs) == 0 {
		v = append(v, "\"invariants\" must have >=1 property that holds INDEPENDENT of the comparison. An invariant is computed from ONE side alone and must hold from first principles.")
	}
	if len(invs) > 24 {
		v = append(v, "\"invariants\" capped at 24.")
	}
	independent := 0
	failedNames := []string{}
	for i, raw := range invs {
		inv, _ := raw.(map[string]any)
		name := strings.TrimSpace(asString(inv["name"]))
		from := strings.TrimSpace(asString(inv["computed_from"]))
		why := strings.TrimSpace(asString(inv["why_independent"]))
		if name == "" {
			v = append(v, fmt.Sprintf("invariants[%d].name is required.", i))
		}
		if from == "" {
			v = append(v, fmt.Sprintf("invariants[%d].computed_from is required — name the ONE side this is computed from.", i))
		}
		if strings.TrimSpace(asString(inv["expected"])) == "" {
			v = append(v, fmt.Sprintf("invariants[%d].expected is required (what the property must equal).", i))
		}
		if strings.TrimSpace(asString(inv["measured"])) == "" {
			v = append(v, fmt.Sprintf("invariants[%d].measured is required (what you actually got).", i))
		}
		if _, ok := inv["holds"].(bool); !ok {
			v = append(v, fmt.Sprintf("invariants[%d].holds must be a boolean.", i))
		} else if inv["holds"] == false {
			failedNames = append(failedNames, name)
		}
		namesBoth := len(sides) == 2 && strings.Contains(norm(from), norm(sides[0])) && strings.Contains(norm(from), norm(sides[1]))
		if namesBoth {
			v = append(v, fmt.Sprintf("invariants[%d].computed_from names BOTH %q and %q — that is a cross-check, not an invariant. Compute it from one side alone.", i, sides[0], sides[1]))
		} else if verifyDiffRE.MatchString(from) {
			v = append(v, fmt.Sprintf("invariants[%d].computed_from reads as a comparison (%q). An invariant is a property of ONE output, not a relation between two.", i, from))
		} else if from != "" {
			independent++
		}
		if len(why) < 25 || verifyFillerRE.MatchString(why) || lowEffort(why, 4) {
			v = append(v, fmt.Sprintf("invariants[%d].why_independent must explain why this property holds regardless of who computed it — what class of shared error it would catch.", i))
		}
	}
	if len(invs) > 0 && independent == 0 {
		v = append(v, "No invariant survived independence checking — every one was a relation between the two things being compared. This verdict rests entirely on a cross-check.")
	}
	if len(failedNames) > 0 {
		warn = append(warn, "Invariant(s) FAILED: "+strings.Join(failedNames, ", ")+". A failing invariant falsifies the side it was computed from — do not report this as established.")
	}
	sample, sok := a["sample"].(map[string]any)
	if !sok {
		v = append(v, "\"sample\" is required: { n, selection, includes_edge_cases }.")
	} else {
		n, nok := sample["n"].(float64)
		if !nok || math.IsNaN(n) || math.IsInf(n, 0) || n < 1 {
			v = append(v, "\"sample.n\" must be the number of cases actually tested.")
		} else if n == 1 {
			warn = append(warn, "sample.n = 1. One agreeing case is not evidence of a general claim — it is one point. Scale it or narrow the claim to that case.")
		}
		if len(strings.TrimSpace(asString(sample["selection"]))) < 15 {
			v = append(v, "\"sample.selection\" must say how cases were chosen — hand-picked convenient values are a ritual, not a test.")
		}
		if sample["includes_edge_cases"] != true {
			warn = append(warn, "sample.includes_edge_cases is not true. Random-only sampling misses the boundaries where conventions actually diverge (identity, zeros, sign flips, degenerate configs).")
		}
	}
	blind := asSlice(a["blind_to"])
	if len(blind) == 0 {
		v = append(v, "\"blind_to\" must list >=1 thing this test structurally CANNOT detect. Every test has a blind spot; an unstated one becomes an overclaim.")
	}
	if len(blind) > 24 {
		v = append(v, "\"blind_to\" capped at 24.")
	}
	for i, raw := range blind {
		s := asString(raw)
		if len(s) < 15 || lowEffort(s, 3) {
			v = append(v, fmt.Sprintf("blind_to[%d] must be a concrete blind spot.", i))
		}
	}
	vd, vok := a["verdict"].(map[string]any)
	if !vok {
		v = append(v, "\"verdict\" is required: { established, not_established }.")
	} else {
		established, notEstablished := strings.TrimSpace(asString(vd["established"])), strings.TrimSpace(asString(vd["not_established"]))
		if len(established) < 15 {
			v = append(v, "\"verdict.established\" must state precisely what the evidence licenses.")
		}
		if len(notEstablished) < 15 {
			v = append(v, "\"verdict.not_established\" is required — what the evidence does NOT license.")
		}
		if norm(established) == norm(notEstablished) {
			v = append(v, "\"verdict.established\" and \"verdict.not_established\" are identical.")
		}
	}
	return v, warn
}
func hasSourceKind(sources []any, kind string) bool {
	for _, raw := range sources {
		if s, ok := raw.(map[string]any); ok && strings.TrimSpace(asString(s["kind"])) == kind {
			return true
		}
	}
	return false
}

func toolVerify(a map[string]any, cfg Config) (string, error) {
	violations, warnings := verifyValidate(a)
	if len(violations) > 0 {
		return "", errors.New("REJECTED — this verdict is not supported by its evidence:\n- " + strings.Join(violations, "\n- "))
	}
	invs := asSlice(a["invariants"])
	okCount := 0
	for _, raw := range invs {
		if inv, ok := raw.(map[string]any); ok && inv["holds"] == true {
			okCount++
		}
	}
	claim := strings.TrimSpace(asString(a["claim"]))
	verdict, _ := a["verdict"].(map[string]any)
	e := StoreEntry{"id": 0, "ts": time.Now().UTC().Format(time.RFC3339Nano), "status": "verify", "kind": "verify", "input": claim, "verdict": map[string]any{"established": asString(verdict["established"]), "not_established": asString(verdict["not_established"])}, "invariants": summarizeInvariants(invs), "sample_n": asMapValue(a["sample"], "n"), "blind_to": a["blind_to"]}
	entry, err := appendStore(cfg.Store, e, cfg)
	if err != nil {
		return "", err
	}
	lines := []string{fmt.Sprintf("VERIFIED (recorded as #%d) — %q", asIntDefault(entry["id"]), claim), "", fmt.Sprintf("  independent invariants: %d/%d hold", okCount, len(invs))}
	for _, raw := range invs {
		inv, _ := raw.(map[string]any)
		state := "FAIL"
		if inv["holds"] == true {
			state = "PASS"
		}
		lines = append(lines, fmt.Sprintf("    %s  %s  [from: %s]  expected %s / measured %s", state, asString(inv["name"]), asString(inv["computed_from"]), asString(inv["expected"]), asString(inv["measured"])))
	}
	d, _ := a["discriminator"].(map[string]any)
	sample, _ := a["sample"].(map[string]any)
	lines = append(lines, "", fmt.Sprintf("  discriminator: %s — %s", asString(d["name"]), asString(d["outcome"])), fmt.Sprintf("    would have failed if: %s", asString(d["would_fail_if"])), fmt.Sprintf("  sample: n=%v, edge cases: %s", sample["n"], yesNo(sample["includes_edge_cases"])), fmt.Sprintf("  sources: %s", sourceSummary(a["sources"])), "", "  ESTABLISHED:     "+asString(verdict["established"]), "  NOT ESTABLISHED: "+asString(verdict["not_established"]), "  BLIND TO:")
	for _, raw := range asSlice(a["blind_to"]) {
		lines = append(lines, "    - "+asString(raw))
	}
	if len(warnings) > 0 {
		lines = append(lines, "", "  WARNINGS:")
		for _, w := range warnings {
			lines = append(lines, "    ! "+w)
		}
	}
	lines = append(lines, "", "Report the ESTABLISHED line. Do not let it imply the NOT ESTABLISHED line — that is the failure this gate exists to stop.")
	return strings.Join(lines, "\n") + priorWorkNote(cfg.Store, claim, asIntDefault(entry["id"]), nil, nil, cfg), nil
}
func summarizeInvariants(invs []any) []any {
	out := []any{}
	for _, raw := range invs {
		if i, ok := raw.(map[string]any); ok {
			out = append(out, map[string]any{"name": i["name"], "holds": i["holds"]})
		}
	}
	return out
}
func asMapValue(v any, key string) any {
	if m, ok := v.(map[string]any); ok {
		return m[key]
	}
	return nil
}
func yesNo(v any) string {
	if v == true {
		return "yes"
	}
	return "NO"
}
func sourceSummary(v any) string {
	out := []string{}
	for _, raw := range asSlice(v) {
		if s, ok := raw.(map[string]any); ok {
			p := asString(s["path"])
			out = append(out, asString(s["kind"])+":"+filepath.Base(p))
		}
	}
	return strings.Join(out, ", ")
}

var isolateObservableRE = regexp.MustCompile(`(?i)\b(error|errors|exit|nonzero|zero|crash\w*|exception|stack|trace|log|logs?|nan|inf|null|http|status|code|file|files|mtime|byte\w*|pixel\w*|warning|blocked|rejected|timeout|identical|match\w*|differ\w*|output|prints?|returns?|value|values|latency|ms|seconds?|count|rate|assert\w*|segfault|oom|signal)\b`)
var isolateQuantRE = regexp.MustCompile(`(?i)\d|%|nan|inf|null|zero|nonzero|exit|status|=`)

func toolIsolate(a map[string]any, cfg Config, semantic map[int]float64) (string, error) {
	violations, warnings := []string{}, []string{}
	input := strings.TrimSpace(asString(a["input"]))
	if len(input) < 15 || len(content(input)) < 3 {
		violations = append(violations, "\"input\" is too thin — state the actual problem (≥15 chars, ≥3 content words).")
	}
	symptom := strings.TrimSpace(asString(a["symptom"]))
	if len(symptom) < 10 {
		violations = append(violations, "\"symptom\" is too thin — describe the surface failure you first saw.")
	}
	baseline, baselineOK := a["baseline"].(map[string]any)
	if !baselineOK || len(strings.TrimSpace(asString(baseline["ref"]))) < 3 || len(strings.TrimSpace(asString(baseline["outcome"]))) < 3 {
		violations = append(violations, "\"baseline\" must be { ref, outcome } — the known-good state you vary from and its result.")
	}
	vars := asStringSlice(a["variables"])
	if len(vars) < 2 || len(vars) > 12 {
		violations = append(violations, fmt.Sprintf("\"variables\" must list 2..12 candidate factors (got %d).", len(vars)))
	}
	seenVars := map[string]bool{}
	for _, name := range vars {
		key := norm(name)
		if seenVars[key] {
			violations = append(violations, "\"variables\" has duplicates — each candidate factor must be distinct.")
			break
		}
		seenVars[key] = true
	}
	trials := asSlice(a["trials"])
	if len(trials) < 2 || len(trials) > 20 {
		violations = append(violations, fmt.Sprintf("\"trials\" must have 2..20 entries (got %d).", len(trials)))
	}
	lowConcrete, knownGood := 0, 0
	observedTexts := []string{}
	changedPerTrial := []string{}
	for i, raw := range trials {
		t, _ := raw.(map[string]any)
		changed := strings.TrimSpace(asString(t["changed"]))
		from, to := asString(t["from"]), asString(t["to"])
		level := strings.ToLower(strings.TrimSpace(asString(t["level"])))
		outcome := strings.ToLower(strings.TrimSpace(asString(t["outcome"])))
		observed := asString(t["observed"])
		for key, value := range map[string]string{"observed": observed, "from": from, "to": to} {
			if len(value) > 8000 {
				violations = append(violations, fmt.Sprintf("trials[%d].%s exceeds 8000 chars.", i, key))
			}
		}
		if changed == "" {
			violations = append(violations, fmt.Sprintf("trials[%d].changed is empty — every trial must flip exactly one variable.", i))
		} else if !seenVars[norm(changed)] {
			violations = append(violations, fmt.Sprintf("trials[%d].changed = %q is not one of the declared variables — a controlled trial flips a DECLARED variable, nothing else.", i, changed))
		}
		changedPerTrial = append(changedPerTrial, norm(changed))
		if strings.TrimSpace(from) != "" && norm(from) == norm(to) {
			violations = append(violations, fmt.Sprintf("trials[%d] has from == to — nothing was actually changed.", i))
		}
		if level != "surface" && level != "intermediate" && level != "low" {
			violations = append(violations, fmt.Sprintf("trials[%d].level must be surface | intermediate | low (got %q).", i, level))
		}
		if outcome != "reproduces" && outcome != "resolves" && outcome != "unchanged" {
			violations = append(violations, fmt.Sprintf("trials[%d].outcome must be reproduces | resolves | unchanged (got %q).", i, outcome))
		}
		if len(strings.TrimSpace(observed)) < 20 || lowEffort(observed, 4) {
			violations = append(violations, fmt.Sprintf("trials[%d].observed is too thin — record the actual signal you read (≥20 chars).", i))
		}
		if level == "low" {
			concrete := isolateObservableRE.MatchString(observed) || isolateQuantRE.MatchString(observed)
			restates := symptom != "" && (cloneText(observed, symptom) || jaccard(observed, symptom) >= .8)
			if concrete && !restates {
				lowConcrete++
			} else if restates {
				violations = append(violations, fmt.Sprintf("trials[%d] claims level \"low\" but just restates the surface symptom — a low-level read must show a deeper signal (log line, exit code, NaN, byte/pixel value, measurement).", i))
			} else {
				violations = append(violations, fmt.Sprintf("trials[%d] claims level \"low\" but observed has no concrete machine signal — cite the actual low-level observable.", i))
			}
		}
		if t["known_good"] == true {
			knownGood++
			if outcome != "resolves" {
				warnings = append(warnings, fmt.Sprintf("trials[%d] is a known-good swap but did not resolve — if the trusted equivalent didn't fix it, the suspect may not be the cause.", i))
			}
		}
		observedTexts = append(observedTexts, observed)
	}
	for _, i := range nearDupes(observedTexts, .9) {
		violations = append(violations, fmt.Sprintf("trials[%d].observed is nearly identical to another trial — each controlled trial must show its OWN result, not one pasted output.", i))
	}
	if lowConcrete < 1 {
		violations = append(violations, "no trial reads the LOW level with a concrete signal — at least one trial must go beneath the surface symptom (log/exit code/measurement), or you're diagnosing on appearances.")
	}
	if knownGood < 1 {
		violations = append(violations, "no known-good swap — at least one trial must replace the suspect with a trusted known-good equivalent (known_good:true) to confirm the culprit.")
	}
	culprit := strings.TrimSpace(asString(a["culprit"]))
	if !seenVars[norm(culprit)] {
		violations = append(violations, fmt.Sprintf("\"culprit\" = %q is not one of the declared variables — the identified cause must be a declared variable you actually varied.", culprit))
	} else {
		culpritTrials, otherTrials := []map[string]any{}, []map[string]any{}
		for _, raw := range trials {
			t, _ := raw.(map[string]any)
			changed := norm(asString(t["changed"]))
			if changed == norm(culprit) {
				culpritTrials = append(culpritTrials, t)
			} else if seenVars[changed] {
				otherTrials = append(otherTrials, t)
			}
		}
		culpritResolves, otherReproduces, culpritKnownGood := false, false, false
		for _, t := range culpritTrials {
			outcome := strings.ToLower(strings.TrimSpace(asString(t["outcome"])))
			if outcome == "resolves" {
				culpritResolves = true
			}
			if t["known_good"] == true && outcome == "resolves" {
				culpritKnownGood = true
			}
		}
		for _, t := range otherTrials {
			if strings.ToLower(strings.TrimSpace(asString(t["outcome"]))) == "reproduces" {
				otherReproduces = true
			}
		}
		if len(culpritTrials) == 0 {
			violations = append(violations, fmt.Sprintf("no trial flips the culprit %q — you cannot name a cause you never isolated in its own single-variable trial.", culprit))
		}
		if len(culpritTrials) > 0 && !culpritResolves {
			violations = append(violations, fmt.Sprintf("flipping %q alone never RESOLVED the symptom in any trial — an unconfirmed suspect is not a root cause.", culprit))
		}
		if len(otherTrials) == 0 {
			violations = append(violations, "every trial flips the culprit — you need at least one trial changing a DIFFERENT variable that still REPRODUCES, or you haven't shown the difference is exactly one variable.")
		} else if !otherReproduces {
			violations = append(violations, "no non-culprit trial reproduced the symptom — if changing other variables also 'fixed' it, the culprit is not isolated (everything looks like the cause). Add a control that holds the culprit and still fails.")
		}
		if !culpritKnownGood {
			warnings = append(warnings, fmt.Sprintf("the known-good swap did not target the culprit %q (or didn't resolve) — the strongest confirmation swaps the SUSPECT itself for a trusted equivalent.", culprit))
		}
	}
	confidence, confidenceOK := a["confidence"].(map[string]any)
	if !confidenceOK {
		violations = append(violations, "\"confidence\" must be an object { p: 0..1, because: \"…\" }.")
	} else {
		p, pok := confidence["p"].(float64)
		why := strings.TrimSpace(asString(confidence["because"]))
		if !pok || math.IsNaN(p) || math.IsInf(p, 0) || p < .01 || p > .99 {
			violations = append(violations, fmt.Sprintf("\"confidence.p\" must be a probability in [0.01, 0.99] — a single isolation never earns certainty (got %v).", confidence["p"]))
		}
		if len(why) < 15 || len(content(why)) < 3 {
			violations = append(violations, "\"confidence.because\" is too thin — cite the trial that isolated it (≥15 chars, ≥3 content words).")
		}
		trialTerms := map[string]bool{}
		for _, term := range content(strings.Join(observedTexts, " ") + " " + culprit) {
			trialTerms[term] = true
		}
		cited := content(why)
		shared := false
		for _, term := range cited {
			if trialTerms[term] {
				shared = true
				break
			}
		}
		if len(trialTerms) > 0 && len(cited) > 0 && !shared {
			violations = append(violations, "\"confidence.because\" shares no terms with the trials/culprit — cite the discriminating trial that isolated the cause.")
		}
	}
	if len(violations) > 0 {
		return "", errors.New("REJECTED — this is not a controlled isolation. Fix and call think_isolate again:\n- " + strings.Join(violations, "\n- "))
	}
	e := copyMap(a)
	e["id"] = 0
	e["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
	e["status"] = "isolated"
	entry, err := appendStore(cfg.Store, e, cfg)
	if err != nil {
		return "", err
	}
	warnNote := ""
	if len(warnings) > 0 {
		warnNote = "\n⚠ " + strings.Join(warnings, "\n⚠ ")
	}
	recallNote := priorWorkNote(cfg.Store, input+" "+culprit, asIntDefault(entry["id"]), asFloatPtr(a["confidence"]), semantic, cfg)
	confidenceP := ""
	if confidenceOK {
		confidenceP = fmt.Sprintf("%v", confidence["p"])
	}
	return fmt.Sprintf("Isolation #%d RECORDED: culprit = %q (confidence %s).\n%d controlled trials, %d known-good swap(s), %d low-level read(s).\nConfirmed by one-variable difference: flipping %q alone resolved while another single-variable change still reproduced.%s%s", asIntDefault(entry["id"]), culprit, confidenceP, len(trials), knownGood, lowConcrete, culprit, warnNote, recallNote), nil
}

var discoveryBioAI = regexp.MustCompile(`(?i)\b(biolog\w*|genom\w*|genetic\w*|gene expression|protein\w*|proteom\w*|\bcell(s|ular)?\b|neuro\w*|neural net\w*|synap\w*|organism\w*|molecular biolog\w*|evolutionary biolog\w*|ecolog\w*|metabol\w*|enzyme\w*|\bDNA\b|\bRNA\b|immun\w*|deep learning|machine learning|\bLLM\b|large language model|transformer network|artificial intelligence|\bA\.?I\.?\b|\bML\b|reinforcement learning|neural network|backpropagation)\b`)
var discoveryQuant = regexp.MustCompile(`(?i)\d|%|percent|fold|exponent|scal(e|es|ing)|proportional|∝|∼|~|≥|≤|>=|<=|>|<|=|\bx\b|orders? of magnitude|power[- ]law|linear\w*|exponential\w*|logarithm\w*|slope|rate|ratio|coefficient|within|per\b`)
var discoveryRefute = regexp.MustCompile(`(?i)\b(reject|refut\w*|disappear\w*|vanish\w*|fails?|fail to|no (correlation|relationship|effect|change|signal|dependence)|instead|rather than|would falsify|falsif\w*|breaks?|does not|doesn'?t|do not|not hold|null result|absent|flat|unchanged|contradict\w*|rules? out)\b`)

func discoveryValidation(a map[string]any) []string {
	v := []string{}
	str := func(key string) string { return strings.TrimSpace(asString(a[key])) }
	da, db := str("domain_a"), str("domain_b")
	sfo, pred, exp := str("shared_formal_object"), str("transported_prediction"), str("refuting_experiment")
	for _, item := range []struct{ key, value string }{{"domain_a", da}, {"domain_b", db}} {
		if len(item.value) < 3 {
			v = append(v, fmt.Sprintf("\"%s\" is missing (name a real field, ≥3 chars).", item.key))
		} else if len(item.value) > 8000 {
			v = append(v, fmt.Sprintf("\"%s\" exceeds 8000 chars.", item.key))
		} else if discoveryBioAI.MatchString(item.value) {
			v = append(v, fmt.Sprintf("\"%s\" (%q) is a biology or AI/ML domain — this tool excludes both by constraint. Pick a domain outside bio and AI.", item.key, truncate(item.value, 40)))
		}
	}
	if da != "" && db != "" && cloneText(da, db) {
		v = append(v, "\"domain_a\" and \"domain_b\" are the same domain — a cross-domain connection needs two genuinely different fields.")
	}
	if len(sfo) < 25 {
		v = append(v, "\"shared_formal_object\" is too thin (need ≥25 chars naming the actual shared math/structure, e.g. \"product of random transfer matrices / positive Lyapunov exponent\").")
	} else if len(sfo) > 8000 {
		v = append(v, "\"shared_formal_object\" exceeds 8000 chars.")
	} else if lowEffort(sfo, 4) || len(content(sfo)) < 3 {
		v = append(v, "\"shared_formal_object\" looks like filler — name a concrete formal object shared by both domains.")
	} else if cloneText(sfo, da) || cloneText(sfo, db) {
		v = append(v, "\"shared_formal_object\" just restates a domain name — it must be the transportable formalism itself, not the field it came from.")
	}
	if sfo != "" && discoveryBioAI.MatchString(sfo) {
		v = append(v, fmt.Sprintf("\"shared_formal_object\" pulls in bio/AI vocabulary (%q) — keep the shared structure outside those domains.", discoveryBioAI.FindString(sfo)))
	}
	if len(pred) < 20 {
		v = append(v, "\"transported_prediction\" is too thin (need ≥20 chars).")
	} else if len(pred) > 8000 {
		v = append(v, "\"transported_prediction\" exceeds 8000 chars.")
	} else if !discoveryQuant.MatchString(pred) {
		v = append(v, "\"transported_prediction\" is not quantitative — a discovery must predict a number, scaling law, exponent, or comparator, not a qualitative \"it will affect X\".")
	}
	if len(exp) < 25 {
		v = append(v, "\"refuting_experiment\" is too thin (need ≥25 chars naming the measurement).")
	} else if len(exp) > 8000 {
		v = append(v, "\"refuting_experiment\" exceeds 8000 chars.")
	} else {
		if !discoveryQuant.MatchString(exp) {
			v = append(v, "\"refuting_experiment\" names no measurable quantity — say what you would measure numerically.")
		}
		if !discoveryRefute.MatchString(exp) {
			v = append(v, "\"refuting_experiment\" does not state the outcome that would FALSIFY the theory (e.g. \"reject if decay is power-law not exponential\") — an experiment with no failing outcome is not a falsification.")
		}
	}
	pa := asSlice(a["prior_art"])
	if len(pa) < 2 {
		v = append(v, "\"prior_art\" needs ≥2 recorded searches (query + top_result + verdict) — the novelty claim must be checked against the literature, not asserted.")
	}
	for i, raw := range pa {
		p, ok := raw.(map[string]any)
		if !ok {
			v = append(v, fmt.Sprintf("prior_art[%d] must be { query, top_result, verdict }.", i))
			continue
		}
		q, result, verdict := strings.TrimSpace(asString(p["query"])), strings.TrimSpace(asString(p["top_result"])), strings.ToLower(strings.TrimSpace(asString(p["verdict"])))
		if len(q) < 5 {
			v = append(v, fmt.Sprintf("prior_art[%d].query is missing or too short (record the actual search you ran).", i))
		}
		if result == "" {
			v = append(v, fmt.Sprintf("prior_art[%d].top_result is missing (what did the search return — a title/finding, or \"no direct hit\").", i))
		}
		if verdict != "clear" && verdict != "related" && verdict != "subsumed" {
			v = append(v, fmt.Sprintf("prior_art[%d].verdict must be one of clear | related | subsumed (got %q).", i, asString(p["verdict"])))
		}
	}
	return v
}
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
func discoveryNovelty(pa []any) string {
	hasSubsumed, allClear := false, len(pa) > 0
	for _, raw := range pa {
		if p, ok := raw.(map[string]any); ok {
			verdict := strings.ToLower(strings.TrimSpace(asString(p["verdict"])))
			if verdict == "subsumed" {
				hasSubsumed = true
			}
			if verdict != "clear" {
				allClear = false
			}
		}
	}
	if hasSubsumed {
		return "SUBSUMED — a prior-art search found this already exists; not novel."
	}
	if allClear {
		return "CLEAR (round-1) — no prior art found; novelty survives the searches run, not yet exhaustively cleared."
	}
	return "RELATED — adjacent work exists; the specific transported prediction needs a deeper subsumption check before claiming novelty."
}
func toolDiscover(a map[string]any, cfg Config) (string, error) {
	baseViolations, warnings := enforce(a, true)
	violations := append(baseViolations, discoveryValidation(a)...)
	if len(violations) > 0 {
		return "", errors.New("REJECTED — incomplete/gamed discovery. Fix and call think_discover again:\n- " + strings.Join(violations, "\n- "))
	}
	pa := asSlice(a["prior_art"])
	novelty := discoveryNovelty(pa)
	e := copyMap(a)
	e["id"] = 0
	e["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
	e["status"] = "oneshot"
	e["kind"] = "discovery"
	e["novelty"] = novelty
	entry, err := appendStore(cfg.Store, e, cfg)
	if err != nil {
		return "", err
	}
	warnNote := ""
	if len(warnings) > 0 {
		warnNote = fmt.Sprintf("\n⚠ %d low-overlap warning(s) (not blocking).", len(warnings))
	}
	post := asMapValue(a["posterior"], "p")
	return fmt.Sprintf("Discovery #%d accepted — all gates passed.\n  domains: %s  ×  %s\n  shared object: %s\n  novelty: %s\n  plausibility (posterior.p): %v\n  tests: %d (%d adversarial), prior-art searches: %d%s", asIntDefault(entry["id"]), asString(entry["domain_a"]), asString(entry["domain_b"]), truncate(asString(entry["shared_formal_object"]), 90), novelty, post, len(asSlice(entry["tests"])), countAdversarial(asSlice(entry["tests"])), len(pa), warnNote), nil
}
func pct(n, d int) string {
	if d == 0 {
		return "–"
	}
	return fmt.Sprintf("%d%%", int(math.Round(float64(100*n)/float64(d))))
}
func toolCalibrate(cfg Config) (string, error) {
	rows, err := loadStore(cfg.Store)
	if err != nil {
		return "", err
	}
	if len(rows) == 0 {
		return "Store is empty — no calibration data yet.", nil
	}
	byStatus := map[string]int{}
	statusOrder := []string{}
	scored := []StoreEntry{}
	for _, r := range rows {
		status := asString(r["status"])
		if status == "" {
			if asString(r["kind"]) == "discovery" {
				status = "discovery"
			} else {
				status = "legacy"
			}
		}
		if _, seen := byStatus[status]; !seen {
			statusOrder = append(statusOrder, status)
		}
		byStatus[status]++
		if _, ok := finalP(r); ok {
			if _, hasPrior := asFloat(r["prior"]); hasPrior {
				scored = append(scored, r)
			}
		}
	}
	swings := []float64{}
	for _, r := range scored {
		post, _ := finalP(r)
		prior, _ := asFloat(r["prior"])
		swings = append(swings, post-prior)
	}
	abs := append([]float64(nil), swings...)
	sort.Float64s(abs)
	meanAbs, medianAbs := 0.0, 0.0
	for _, x := range abs {
		meanAbs += math.Abs(x)
	}
	if len(abs) > 0 {
		meanAbs /= float64(len(abs))
		// Match the TypeScript reference: for an even count, use the upper middle value.
		medianAbs = abs[len(abs)/2]
	}
	raised, lowered, frozen, extreme := 0, 0, 0, 0
	for _, swing := range swings {
		if swing > 0 {
			raised++
		} else if swing < 0 {
			lowered++
		} else {
			frozen++
		}
	}
	for _, r := range scored {
		if p, ok := finalP(r); ok && (p >= .9 || p <= .1) {
			extreme++
		}
	}
	advPass, advFail, advAmb := 0, 0, 0
	for _, r := range rows {
		if asString(r["status"]) != "verified" {
			continue
		}
		tests, evidence := asSlice(r["tests"]), asSlice(r["evidence"])
		advNames := map[string]bool{}
		for _, raw := range tests {
			if t, ok := raw.(map[string]any); ok && t["adversarial"] == true {
				advNames[asString(t["name"])] = true
			}
		}
		for _, raw := range evidence {
			if e, ok := raw.(map[string]any); ok && advNames[asString(e["test"])] {
				switch verdictDirection(asString(e["verdict"])) {
				case "pass":
					advPass++
				case "fail":
					advFail++
				default:
					advAmb++
				}
			}
		}
	}
	stale := []StoreEntry{}
	now := time.Now()
	for _, r := range rows {
		if asString(r["status"]) == "open" {
			if t, err := time.Parse(time.RFC3339Nano, asString(r["openedAt"])); err == nil && now.Sub(t) > 24*time.Hour {
				stale = append(stale, r)
			}
		}
	}
	start := asString(rows[0]["ts"])
	end := asString(rows[len(rows)-1]["ts"])
	if len(start) >= 10 {
		start = start[:10]
	}
	if len(end) >= 10 {
		end = end[:10]
	}
	statusParts := make([]string, 0, len(statusOrder))
	for _, key := range statusOrder {
		statusParts = append(statusParts, fmt.Sprintf("%s %d", key, byStatus[key]))
	}
	lines := []string{fmt.Sprintf("CALIBRATION over %d records (%s → %s)", len(rows), start, end), "", "  status: " + strings.Join(statusParts, " · "), "", fmt.Sprintf("  belief swings (%d rows with prior+posterior):", len(scored)), fmt.Sprintf("    mean |Δp| %.2f · median |Δp| %.2f · raised %s · lowered %s · unmoved %s", meanAbs, medianAbs, pct(raised, len(scored)), pct(lowered, len(scored)), pct(frozen, len(scored))), fmt.Sprintf("    posterior extreme (≤0.1 or ≥0.9): %s", pct(extreme, len(scored))), fmt.Sprintf("  adversarial verdicts on verified rows: pass %d · FAIL %d · ambiguous %d", advPass, advFail, advAmb)}
	if len(stale) > 0 {
		lines = append(lines, "", "  STALE OPENS (staked >24h ago, never concluded — dead pre-registrations):")
		for _, r := range stale {
			t, _ := time.Parse(time.RFC3339Nano, asString(r["openedAt"]))
			days := int(math.Round(now.Sub(t).Hours() / 24))
			lines = append(lines, fmt.Sprintf("    %s  (%dd old)", rowSummary(r), days))
		}
		lines = append(lines, "    A stale open is a claim you promised to test and abandoned — conclude it or note why it died.")
	}
	concluded := raised + lowered + frozen
	advice := []string{}
	if concluded >= 8 && float64(raised)/float64(concluded) >= .8 {
		advice = append(advice, fmt.Sprintf("evidence raised belief in %s of concluded breakdowns — refutation attempts are rarely biting. Make adversarial tests attack the load-bearing assumption, not a side detail.", pct(raised, concluded)))
	}
	if concluded >= 8 && advFail == 0 && advPass >= 5 {
		advice = append(advice, fmt.Sprintf("no adversarial test has EVER returned a failing verdict (%d passes) — either every claim staked was true, or the refuters are soft. Stake riskier claims or sharpen the refuters.", advPass))
	}
	if len(scored) >= 8 && float64(extreme)/float64(len(scored)) >= .5 {
		advice = append(advice, fmt.Sprintf("%s of posteriors are ≥0.9 or ≤0.1 — beliefs are landing at the rails. Check the evidence really licenses that much certainty each time.", pct(extreme, len(scored))))
	}
	if len(scored) >= 8 && meanAbs < .08 {
		advice = append(advice, fmt.Sprintf("mean |Δp| is %.2f — tests are barely moving belief. Pre-register tests that could actually change your mind, or the ritual is decorative.", meanAbs))
	}
	if len(advice) > 0 {
		lines = append(lines, "", "  ADVICE (computed from the ledger):")
		for _, item := range advice {
			lines = append(lines, "    → "+item)
		}
	}
	recent := rows
	if len(recent) > 8 {
		recent = recent[len(recent)-8:]
	}
	lines = append(lines, "", "  recent:")
	for _, r := range recent {
		lines = append(lines, "    "+rowSummary(r))
	}
	return strings.Join(lines, "\n"), nil
}

func norm(s string) string {
	return strings.TrimSpace(strings.ToLower(strings.Join(wordRE.FindAllString(strings.ToLower(s), -1), " ")))
}
func content(s string) []string {
	out := []string{}
	for _, w := range strings.Fields(norm(s)) {
		if len(w) > 2 && !stopWords[w] {
			out = append(out, w)
		}
	}
	return out
}
func lowEffort(s string, minDistinct int) bool {
	words := strings.Fields(norm(s))
	uniq := map[string]bool{}
	for _, w := range words {
		uniq[w] = true
	}
	return len(uniq) < minDistinct || (len(words) > 0 && float64(len(uniq))/float64(len(words)) < .4)
}
func cloneText(a, b string) bool { return norm(a) != "" && norm(a) == norm(b) }
func jaccard(a, b string) float64 {
	A, B := map[string]bool{}, map[string]bool{}
	for _, w := range strings.Fields(norm(a)) {
		A[w] = true
	}
	for _, w := range strings.Fields(norm(b)) {
		B[w] = true
	}
	if len(A) == 0 || len(B) == 0 {
		return 0
	}
	inter := 0
	for w := range A {
		if B[w] {
			inter++
		}
	}
	return float64(inter) / float64(len(A)+len(B)-inter)
}
func nearDupes(list []string, threshold float64) []int {
	out := []int{}
	for i := range list {
		for j := 0; j < i; j++ {
			if jaccard(list[i], list[j]) >= threshold {
				out = append(out, i)
				break
			}
		}
	}
	return out
}

var verdictPassLeadRE = regexp.MustCompile(`(?i)^["'\s]*(pass\w*|verified|confirmed|holds?|survive[sd]?|met)\b`)
var verdictFailLeadRE = regexp.MustCompile(`(?i)^["'\s]*(fail\w*|refut\w*|reject\w*|falsif\w*|disprov\w*|blocked)\b`)
var verdictPassRE = regexp.MustCompile(`(?i)\b(pass(?:es|ed)?|survive[sd]?|holds?|held|confirmed|verified|satisfied|succeed\w*)\b`)
var verdictFailRE = regexp.MustCompile(`(?i)\b(fail(?:s|ed)?|refuted|falsified|disproved|regressed|did not (?:hold|pass|survive)|does not (?:hold|pass|survive))\b`)

func verdictDirection(s string) string {
	t := strings.TrimSpace(s)
	if t == "" {
		return "ambiguous"
	}
	if verdictPassLeadRE.MatchString(t) {
		return "pass"
	}
	if verdictFailLeadRE.MatchString(t) {
		return "fail"
	}
	p := verdictPassRE.MatchString(t)
	f := verdictFailRE.MatchString(t)
	if p && !f {
		return "pass"
	}
	if f && !p {
		return "fail"
	}
	return "ambiguous"
}

var quantRE = regexp.MustCompile(`(?i)\d|%|percent|fold|≥|≤|>=|<=|>|<|=|\bx\b|times|at least|at most|no more than|no less than|more than|less than|fewer than|greater than|within`)
var vagueRE = regexp.MustCompile(`(?i)\b(looks?|look|feels?|feel|seems?|seem|noticeabl\w*|somewhat|fairly|pretty|kind of|a bit|good enough|help(s)? much|better than before)\b`)
var reducePctRE = regexp.MustCompile(`(?i)\b(fall|falls|fell|drop|drops|reduc\w*|decreas\w*|lower\w*|cut|cuts|shrink\w*)\b[^.]{0,40}?(\d+(?:\.\d+)?)\s*(?:percent|%)`)
var boundedOneRE = regexp.MustCompile(`(?i)\b(correlation(?: coefficient)?|pearson|spearman|r²|r2|cosine similarity|probability)\b[^.]{0,30}?(\d+(?:\.\d+)?)`)
var refuteRE = regexp.MustCompile(`(?i)\b(reject|refut\w*|disappear\w*|vanish\w*|fails?|fail to|no (improvement|gain|effect|change|benefit)|worse|regress\w*|negat\w*|falsif\w*|breaks?|does not|doesn'?t|do not|not hold|null result|absent|flat|unchanged)\b`)
var observableRE = regexp.MustCompile(`(?i)\b(error|errors|exit|exits?|nonzero|zero|crash\w*|exception|stack|trace|log|logs|http|status|screenshot|file|files|mtime|byte\w*|pixel\w*|warning|blocked|rejected|accepted|verified|present|absent|exists|missing|empty|timeout|identical|match\w*|differ\w*|output|prints?|returns?)\b`)

func dupes(list []string) []int {
	seen := map[string]bool{}
	out := []int{}
	for i, s := range list {
		key := norm(s)
		if seen[key] {
			out = append(out, i)
		} else {
			seen[key] = true
		}
	}
	return out
}
func unfalsifiable(s string) bool { return vagueRE.MatchString(s) && !quantRE.MatchString(s) }
func impossiblePct(s string) (float64, bool) {
	m := reducePctRE.FindStringSubmatch(s)
	if len(m) == 0 {
		return 0, false
	}
	n, err := strconv.ParseFloat(m[2], 64)
	return n, err == nil && n > 100
}
func outOfUnitRange(s string) (string, float64, bool) {
	m := boundedOneRE.FindStringSubmatch(s)
	if len(m) == 0 {
		return "", 0, false
	}
	n, err := strconv.ParseFloat(m[2], 64)
	if err != nil || n <= 1 {
		return "", 0, false
	}
	end := m[0]
	if strings.Contains(strings.ToLower(end), "%") || strings.Contains(strings.ToLower(end), "percent") {
		return "", 0, false
	}
	return m[1], n, true
}
func normalizeValidatorList(items []string) []string {
	for i := range items {
		items[i] = normalizeValidatorText(items[i])
	}
	return items
}

func enforce(a map[string]any, wantPosterior bool) ([]string, []string) {
	v, warn := []string{}, []string{}
	lenOf := func(x any) int { return len(asSlice(x)) }
	if lenOf(a["interpretations"]) > 12 {
		v = append(v, fmt.Sprintf("\\\"interpretations\\\" has %d items (max 12).", lenOf(a["interpretations"])))
	}
	if lenOf(a["how_to_prove"]) > 15 {
		v = append(v, fmt.Sprintf("\\\"how_to_prove\\\" has %d items (max 15).", lenOf(a["how_to_prove"])))
	}
	if lenOf(a["tests"]) > 20 {
		v = append(v, fmt.Sprintf("\\\"tests\\\" has %d items (max 20) — a breakdown is not a dump; keep it focused.", lenOf(a["tests"])))
	}
	if len(v) > 0 {
		return normalizeValidatorList(v), normalizeValidatorList(warn)
	}
	over := func(k, s string, cap int) {
		if len(s) > cap {
			v = append(v, fmt.Sprintf("\\\"%s\\\" exceeds %d chars (%d).", k, cap, len(s)))
		}
	}
	checkStr := func(k string, min, minWords int) {
		s := strings.TrimSpace(asString(a[k]))
		if len(s) < min {
			v = append(v, fmt.Sprintf("\\\"%s\\\" is missing or too thin (need ≥%d chars, got %d).", k, min, len(s)))
			return
		}
		over(k, s, 8000)
		if lowEffort(s, minWords) {
			v = append(v, fmt.Sprintf("\\\"%s\\\" looks like filler/repetition (need ≥%d distinct words of real content).", k, minWords))
		} else if len(content(s)) < (minWords+1)/2 {
			v = append(v, fmt.Sprintf("\\\"%s\\\" is almost all filler words (need ≥%d content words, not stopwords).", k, (minWords+1)/2))
		}
	}
	input := strings.TrimSpace(asString(a["input"]))
	if len(input) < 15 {
		v = append(v, "\\\"input\\\" is missing or too short (need ≥15 chars).")
	} else {
		over("input", input, 20000)
		if lowEffort(input, 4) {
			v = append(v, "\\\"input\\\" looks like filler.")
		}
	}
	interps := asSlice(a["interpretations"])
	if len(interps) < 2 {
		v = append(v, fmt.Sprintf("\\\"interpretations\\\" needs ≥2 distinct readings/categories (got %d).", len(interps)))
	}
	readings := []string{}
	for i, raw := range interps {
		m, _ := raw.(map[string]any)
		r, c := strings.TrimSpace(asString(m["reading"])), strings.TrimSpace(asString(m["category"]))
		readings = append(readings, r)
		if r == "" || c == "" {
			v = append(v, fmt.Sprintf("interpretations[%d] needs non-empty \\\"reading\\\" and \\\"category\\\".", i))
		} else if lowEffort(r, 3) {
			v = append(v, fmt.Sprintf("interpretations[%d].reading looks like filler.", i))
		}
	}
	for _, i := range dupes(readings) {
		v = append(v, fmt.Sprintf("interpretations[%d] duplicates an earlier reading — make them genuinely distinct.", i))
	}
	for _, i := range nearDupes(readings, .8) {
		v = append(v, fmt.Sprintf("interpretations[%d] is a reworded paraphrase of an earlier reading (same word set) — give a genuinely different interpretation.", i))
	}
	checkStr("meaning", 20, 6)
	proof := asSlice(a["how_to_prove"])
	if len(proof) < 2 {
		v = append(v, fmt.Sprintf("\\\"how_to_prove\\\" needs ≥2 concrete methods (got %d).", len(proof)))
	}
	proofStrs := []string{}
	for i, raw := range proof {
		s := asString(raw)
		proofStrs = append(proofStrs, s)
		if len(strings.TrimSpace(s)) < 12 || lowEffort(s, 4) {
			v = append(v, fmt.Sprintf("how_to_prove[%d] is too vague (need a concrete, checkable method).", i))
		}
	}
	for _, i := range dupes(proofStrs) {
		v = append(v, fmt.Sprintf("how_to_prove[%d] duplicates an earlier method.", i))
	}
	for _, i := range nearDupes(proofStrs, .8) {
		v = append(v, fmt.Sprintf("how_to_prove[%d] is a reworded paraphrase of an earlier method — use a genuinely different mechanism.", i))
	}
	checkStr("failure_signature", 20, 6)
	checkStr("success_signature", 20, 6)
	tests := asSlice(a["tests"])
	if len(tests) < 3 {
		v = append(v, fmt.Sprintf("\\\"tests\\\" needs ≥3 variations (got %d).", len(tests)))
	}
	killed := map[int]bool{}
	adv := 0
	refute := refuteRE
	for i, raw := range tests {
		t, _ := raw.(map[string]any)
		for _, k := range []string{"name", "hypothesis", "metric", "decision_rule"} {
			if strings.TrimSpace(asString(t[k])) == "" {
				v = append(v, fmt.Sprintf("tests[%d] is missing \\\"%s\\\".", i, k))
			}
		}
		if lowEffort(asString(t["hypothesis"]), 3) {
			v = append(v, fmt.Sprintf("tests[%d].hypothesis looks like filler.", i))
		}
		dr := asString(t["decision_rule"])
		if unfalsifiable(dr) {
			v = append(v, fmt.Sprintf("tests[%d].decision_rule is unfalsifiable (\"%.40s…\") — give a measurable threshold/comparator, not a perception.", i, strings.TrimSpace(dr)))
		}
		if n, ok := impossiblePct(dr); ok {
			v = append(v, fmt.Sprintf("tests[%d].decision_rule is numerically impossible — a reduction of %g%% exceeds 100%%.", i, n))
		}
		if what, n, ok := outOfUnitRange(dr); ok {
			v = append(v, fmt.Sprintf("tests[%d].decision_rule is out of range — %s cannot be %g (bounded by 1).", i, what, n))
		}
		if strings.TrimSpace(dr) != "" && !quantRE.MatchString(dr) && !observableRE.MatchString(dr) && !unfalsifiable(dr) {
			warn = append(warn, fmt.Sprintf("tests[%d].decision_rule has no measurable anchor (no number, comparator, or named observable like an error/exit code/log/file) — it may be unfalsifiable in practice; anchor it to something the world can contradict.", i))
		}
		d, ok := asInt(t["discriminates"])
		if !ok {
			v = append(v, fmt.Sprintf("tests[%d].discriminates must be the 0-based index of the interpretation this test would ELIMINATE — a test that distinguishes nothing is decorative.", i))
		} else if d < 0 || d >= len(interps) {
			v = append(v, fmt.Sprintf("tests[%d].discriminates = %d is out of range (0..%d).", i, d, len(interps)-1))
		} else {
			killed[d] = true
		}
		if t["adversarial"] == true {
			adv++
			if !refute.MatchString(asString(t["decision_rule"])) {
				v = append(v, "the adversarial test's DECISION_RULE must encode a reject/refute path (e.g. \\\"reject if …\\\").")
			}
		}
	}
	if len(tests) > 0 && adv == 0 {
		v = append(v, "at least one test must set \\\"adversarial\\\": true (a variant that tries to REFUTE the claim).")
	}
	for _, i := range dupes(testBodies(tests)) {
		v = append(v, fmt.Sprintf("tests[%d] is a near-duplicate (same hypothesis+metric) — vary the mechanism, don't just rename.", i))
	}
	for _, i := range nearDupes(testHypotheses(tests), .8) {
		v = append(v, fmt.Sprintf("tests[%d].hypothesis is a reworded paraphrase of an earlier test — vary the mechanism.", i))
	}
	if len(tests) > 0 && len(killed) < 2 {
		v = append(v, fmt.Sprintf("the tests do not discriminate between interpretations — they bear on %d reading(s), need ≥2. At least two interpretations must EACH have a test that could kill them, or the tests are decorative.", len(killed)))
	}
	if cloneText(asString(a["meaning"]), input) {
		v = append(v, "\\\"meaning\\\" is a verbatim copy of \\\"input\\\" — interpret it, don't echo it.")
	}
	if cloneText(asString(a["failure_signature"]), input) || cloneText(asString(a["failure_signature"]), asString(a["meaning"])) {
		v = append(v, "\\\"failure_signature\\\" just echoes the input/meaning — describe the actual failure.")
	}
	if cloneText(asString(a["success_signature"]), input) || cloneText(asString(a["success_signature"]), asString(a["meaning"])) {
		v = append(v, "\\\"success_signature\\\" just echoes the input/meaning — describe the actual success.")
	}
	if jaccard(asString(a["failure_signature"]), asString(a["success_signature"])) >= .85 {
		v = append(v, "\\\"failure_signature\\\" and \\\"success_signature\\\" are nearly identical — they must describe OPPOSITE outcomes.")
	}
	prior, priorOK := a["prior"].(map[string]any)
	if !priorOK {
		v = append(v, "\\\"prior\\\" must be an object { p: 0..1, because: \\\"…\\\" }.")
	} else {
		p, ok := prior["p"].(float64)
		if !ok || math.IsNaN(p) || p < 0 || p > 1 {
			v = append(v, fmt.Sprintf("\\\"prior.p\\\" must be a probability in [0,1] (got %v).", prior["p"]))
		}
		if len(content(asString(prior["because"]))) < 3 || len(strings.TrimSpace(asString(prior["because"]))) < 15 {
			v = append(v, "\\\"prior.because\\\" is too thin — state the actual reasoning (≥15 chars, ≥3 content words).")
		}
		if ok && (p < .05 || p > .95) {
			v = append(v, fmt.Sprintf("\\\"prior.p\\\" = %v is dogmatic — a pre-registered belief must be UPDATABLE by the tests.", p))
		}
	}
	if wantPosterior {
		post, ok := a["posterior"].(map[string]any)
		if !ok {
			v = append(v, "\\\"posterior\\\" must be an object { p: 0..1, because: \\\"…\\\" }.")
		} else {
			p, pok := post["p"].(float64)
			if !pok || math.IsNaN(p) || p < 0 || p > 1 {
				v = append(v, fmt.Sprintf("\\\"posterior.p\\\" must be a probability in [0,1] (got %v).", post["p"]))
			}
			if len(content(asString(post["because"]))) < 3 || len(strings.TrimSpace(asString(post["because"]))) < 15 {
				v = append(v, "\\\"posterior.because\\\" is too thin — state the actual reasoning (≥15 chars, ≥3 content words).")
			}
			if pok && (p < .01 || p > .99) {
				v = append(v, fmt.Sprintf("\\\"posterior.p\\\" = %v is absolute certainty — finite tests never earn p outside 0.01–0.99.", p))
			}
			if priorOK && cloneText(asString(post["because"]), asString(prior["because"])) {
				v = append(v, "\\\"posterior.because\\\" just repeats \\\"prior.because\\\" — say what the TESTS changed, not the same sentence twice.")
			}
		}
	}
	for i := range v {
		v[i] = normalizeValidatorText(v[i])
	}
	for i := range warn {
		warn[i] = normalizeValidatorText(warn[i])
	}
	return normalizeValidatorList(v), normalizeValidatorList(warn)
}

func normalizeValidatorText(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, `\"`, `"`), `\n`, "\n")
}

func testBodies(tests []any) []string {
	out := make([]string, 0, len(tests))
	for _, raw := range tests {
		t, _ := raw.(map[string]any)
		out = append(out, asString(t["hypothesis"])+"|"+asString(t["metric"]))
	}
	return out
}
func testHypotheses(tests []any) []string {
	out := make([]string, 0, len(tests))
	for _, raw := range tests {
		t, _ := raw.(map[string]any)
		out = append(out, asString(t["hypothesis"]))
	}
	return out
}
func enforceBreakdown(a map[string]any, posterior bool) ([]string, []string) {
	return enforce(a, posterior)
}

func readArtifactNoFollow(path string) (os.FileInfo, []byte, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, nil, err
	}
	// Resolve normal platform aliases such as macOS /var -> /private/var
	// only for the parent directory, then apply O_NOFOLLOW to the final
	// artifact component. This preserves portability while preventing a
	// final-file symlink from being followed.
	parent, base := filepath.Dir(abs), filepath.Base(abs)
	resolvedParent, err := filepath.EvalSymlinks(parent)
	if err != nil {
		return nil, nil, err
	}
	resolved := filepath.Join(resolvedParent, base)
	f, err := os.OpenFile(resolved, os.O_RDONLY|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if !st.Mode().IsRegular() {
		return st, nil, errors.New("artifact must be a regular file")
	}
	const maxArtifactBytes = 4 * 1024 * 1024
	body, err := io.ReadAll(io.LimitReader(f, maxArtifactBytes+1))
	if err != nil {
		return st, nil, err
	}
	if len(body) > maxArtifactBytes {
		return st, nil, errors.New("artifact exceeds 4 MiB read limit")
	}
	return st, body, nil
}

func loadStore(path string) ([]StoreEntry, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return []StoreEntry{}, nil
	}
	if err != nil {
		return nil, err
	}
	var rows []StoreEntry
	if err = json.Unmarshal(b, &rows); err != nil {
		return nil, fmt.Errorf("invalid store: %w", err)
	}
	return rows, nil
}
func appendStore(path string, entry StoreEntry, cfg Config) (StoreEntry, error) {
	var out StoreEntry
	err := withLock(path, cfg, func() error {
		rows, e := loadStore(path)
		if e != nil {
			return e
		}
		max := 0
		for _, r := range rows {
			if x, _ := asInt(r["id"]); x > max {
				max = x
			}
		}
		entry["id"] = max + 1
		rows = append(rows, entry)
		out = entry
		b, _ := json.MarshalIndent(rows, "", "  ")
		tmp := fmt.Sprintf("%s.%d.tmp", path, os.Getpid())
		if e = os.WriteFile(tmp, b, 0644); e != nil {
			return e
		}
		return os.Rename(tmp, path)
	})
	return out, err
}
func withLock(path string, cfg Config, fn func() error) error {
	lock := path + ".lock"
	if err := os.MkdirAll(filepath.Dir(lock), 0755); err != nil {
		return err
	}
	token := fmt.Sprintf("%d-%d", os.Getpid(), time.Now().UnixNano())
	start := time.Now()
	for {
		f, err := os.OpenFile(lock, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		if err == nil {
			if _, err := f.WriteString(token); err != nil {
				_ = f.Close()
				_ = os.Remove(lock)
				return err
			}
			if err := f.Sync(); err != nil {
				_ = f.Close()
				_ = os.Remove(lock)
				return err
			}
			_ = f.Close()
			defer func() {
				if owner, readErr := os.ReadFile(lock); readErr == nil && string(owner) == token {
					_ = os.Remove(lock)
				}
			}()
			return fn()
		}
		if !os.IsExist(err) {
			return err
		}
		if time.Since(start) > cfg.LockTimeout {
			return errors.New("store busy")
		}
		if st, statErr := os.Stat(lock); statErr == nil && time.Since(st.ModTime()) > cfg.LockStale {
			owner, ownerErr := os.ReadFile(lock)
			if ownerErr == nil {
				pid := 0
				_, _ = fmt.Sscanf(string(owner), "%d-", &pid)
				alive := false
				if pid > 0 {
					if _, findErr := os.FindProcess(pid); findErr == nil {
						alive = syscall.Kill(pid, 0) == nil
					}
				}
				if !alive {
					_ = os.Remove(lock)
				}
			} else if os.IsNotExist(ownerErr) || errors.Is(ownerErr, syscall.EISDIR) {
				// Recover an ownerless lock after the stale grace period. The
				// exclusive file creation above prevents a new owner from being
				// mistaken for the process that is currently executing.
				_ = os.RemoveAll(lock)
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
}

type cappedOutput struct {
	buf bytes.Buffer
	max int
}

func (w *cappedOutput) Write(p []byte) (int, error) {
	if w.buf.Len()+len(p) > w.max {
		return 0, errors.New("semantic sidecar output exceeded limit")
	}
	return w.buf.Write(p)
}

func semanticScores(query string, rows []StoreEntry, cfg Config) map[int]float64 {
	if strings.TrimSpace(query) == "" || strings.TrimSpace(cfg.SemanticSidecar) == "" || len(rows) == 0 {
		return nil
	}
	if _, err := os.Stat(cfg.SemanticSidecar); err != nil {
		return nil
	}
	candidates := make([]map[string]any, 0, len(rows))
	for _, row := range rows {
		candidates = append(candidates, map[string]any{
			"id":   asIntDefault(row["id"]),
			"text": asString(row["input"]) + " " + asString(row["meaning"]),
		})
	}
	request := map[string]any{"id": "recall", "query": query, "candidates": candidates}
	payload, err := json.Marshal(request)
	if err != nil {
		return nil
	}
	timeout := cfg.SemanticTimeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "python3", cfg.SemanticSidecar)
	cmd.Stdin = bytes.NewReader(append(payload, '\n'))
	var output cappedOutput
	output.max = 1 << 20
	cmd.Stdout = &output
	err = cmd.Run()
	if err != nil || ctx.Err() != nil {
		return nil
	}
	var response struct {
		ID     string `json:"id"`
		OK     bool   `json:"ok"`
		Scores []struct {
			ID    int     `json:"id"`
			Score float64 `json:"score"`
		} `json:"scores"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(output.buf.Bytes()), &response); err != nil || !response.OK || response.ID != "recall" {
		return nil
	}
	result := make(map[int]float64, len(response.Scores))
	for _, item := range response.Scores {
		if item.Score >= 0 && item.Score <= 1 && !math.IsNaN(item.Score) && !math.IsInf(item.Score, 0) {
			result[item.ID] = item.Score
		}
	}
	return result
}

func combinedSimilarity(query string, row StoreEntry, semantic map[int]float64) float64 {
	lexical := lexicalSim(query, asString(row["input"])+" "+asString(row["meaning"]))
	if semantic == nil {
		return lexical
	}
	if score, ok := semantic[asIntDefault(row["id"])]; ok && score > lexical {
		return score
	}
	return lexical
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
func asSlice(v any) []any {
	if x, ok := v.([]any); ok {
		return x
	}
	return nil
}
func asFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, math.IsNaN(x) == false && math.IsInf(x, 0) == false
	case float32:
		f := float64(x)
		return f, math.IsNaN(f) == false && math.IsInf(f, 0) == false
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case json.Number:
		f, err := strconv.ParseFloat(string(x), 64)
		return f, err == nil && !math.IsNaN(f) && !math.IsInf(f, 0)
	case map[string]any:
		return asFloat(x["p"])
	default:
		return 0, false
	}
}
func asInt(v any) (int, bool) {
	switch x := v.(type) {
	case int:
		return x, true
	case float64:
		if x == math.Trunc(x) {
			return int(x), true
		}
	case json.Number:
		i, _ := strconv.Atoi(string(x))
		return i, i != 0
	}
	return 0, false
}
func asIntDefault(v any) int { x, _ := asInt(v); return x }
func copyMap(a map[string]any) map[string]any {
	b := map[string]any{}
	for k, v := range a {
		b[k] = v
	}
	return b
}
func words(s string) []string {
	out := []string{}
	for _, w := range wordRE.FindAllString(strings.ToLower(s), -1) {
		if len(w) > 2 && !stopWords[w] {
			out = append(out, w)
		}
	}
	return out
}
func sharedContent(a, b string) int {
	right := map[string]bool{}
	for _, w := range words(b) {
		right[w] = true
	}
	n := 0
	seen := map[string]bool{}
	for _, w := range words(a) {
		if !seen[w] && right[w] {
			n++
		}
		seen[w] = true
	}
	return n
}
func lexicalSim(a, b string) float64 {
	aa := map[string]bool{}
	bb := map[string]bool{}
	for _, w := range words(a) {
		aa[w] = true
	}
	for _, w := range words(b) {
		bb[w] = true
	}
	if len(aa) == 0 || len(bb) == 0 {
		return 0
	}
	inter := 0
	for w := range aa {
		if bb[w] {
			inter++
		}
	}
	return float64(inter) / float64(len(aa)+len(bb)-inter)
}
func finalP(r StoreEntry) (float64, bool) {
	if b, ok := r["posterior"].(map[string]any); ok {
		if p, ok := b["p"].(float64); ok {
			return p, true
		}
	}
	if b, ok := r["confidence"].(map[string]any); ok {
		if p, ok := b["p"].(float64); ok {
			return p, true
		}
	}
	return 0, false
}
func rowSummary(r StoreEntry) string {
	p := ""
	if x, ok := finalP(r); ok {
		p = fmt.Sprintf(" p %.2f", x)
	}
	return fmt.Sprintf("#%d [%s]%s :: %.110s", asIntDefault(r["id"]), asString(r["status"]), p, strings.ReplaceAll(asString(r["input"]), "\n", " "))
}
