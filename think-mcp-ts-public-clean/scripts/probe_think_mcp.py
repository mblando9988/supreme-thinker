#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


EXPECTED_TOOLS = [
    "think",
    "get_thoughts",
    "think_open",
    "think_conclude",
    "think_recall",
    "think_verify",
    "think_isolate",
    "think_discover",
    "think_calibrate",
]


def rpc(binary, store, integrity_log, requests):
    env = os.environ.copy()
    env.update(
        {
            "THINK_STORE": str(store),
            "THINK_INTEGRITY_LOG": str(integrity_log),
            "THINK_MIN_ELAPSED_MS": "1",
            "THINK_SEMANTIC_BACKEND": "lexical",
        }
    )
    payload = "\n".join(json.dumps(item, separators=(",", ":")) for item in requests) + "\n"
    proc = subprocess.run(
        [str(binary)],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"server exited {proc.returncode}\nstderr:\n{proc.stderr}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    responses = [json.loads(line) for line in lines]
    if len(responses) != len([r for r in requests if "id" in r]):
        raise AssertionError(f"expected one response per request id, got {len(responses)}:\n{proc.stdout}")
    return responses


def call_tool(binary, store, integrity_log, rpc_id, name, arguments):
    response = rpc(
        binary,
        store,
        integrity_log,
        [
            {"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        ],
    )[0]
    if "error" in response:
        raise AssertionError(f"{name} JSON-RPC error: {response['error']}")
    result = response.get("result", {})
    content = result.get("content") or []
    text = content[0].get("text", "") if content else ""
    if result.get("isError"):
        raise AssertionError(f"{name} tool error:\n{text}")
    if not text:
        raise AssertionError(f"{name} returned no text: {response}")
    return text


def valid_breakdown(include_posterior=True):
    args = {
        "input": "The cache health check reflects actual service state rather than a wrapper artifact.",
        "interpretations": [
            {"reading": "The service is healthy and the check reports it accurately.", "category": "service"},
            {"reading": "The wrapper masks a failed service check.", "category": "wrapper"},
            {"reading": "A stale log is being replayed.", "category": "cache"},
        ],
        "meaning": "Determine whether the health check genuinely reflects service state rather than masking or replaying a result.",
        "how_to_prove": [
            "Run the service command directly five times and record exit status in a raw output file.",
            "Compare the current response with the cache log and inspect modification times.",
        ],
        "failure_signature": "The direct command fails or the wrapper reports success while the service is unhealthy.",
        "success_signature": "The direct command exits zero five times and the current response is fresh rather than replayed.",
        "prior": {
            "p": 0.55,
            "because": "The wrapper looked healthy but direct service exit status was never independently recorded.",
        },
        "tests": [
            {
                "name": "A",
                "hypothesis": "the service is healthy directly",
                "metric": "direct command exit status",
                "decision_rule": "reject if the command exits nonzero",
                "discriminates": 0,
            },
            {
                "name": "B",
                "hypothesis": "the wrapper masks a failed service",
                "metric": "pre-wrapper exit status",
                "decision_rule": "reject if a nonzero status appears before wrapper output",
                "discriminates": 1,
            },
            {
                "name": "C",
                "hypothesis": "the output is stale cache replay",
                "metric": "response bytes and cache mtime",
                "decision_rule": "reject if the response matches the stale cache",
                "adversarial": True,
                "discriminates": 2,
            },
        ],
    }
    if include_posterior:
        args["posterior"] = {"p": 0.5, "because": "The evidence remains mixed and more testing is required."}
    return args


def write_evidence(path):
    path.write_text(
        "service command exited zero five times\n"
        "wrapper status stayed successful before output\n"
        "cache response bytes were fresh, not replayed\n",
        encoding="utf-8",
    )


def conclusion_evidence(path):
    return [
        {
            "test": "A",
            "ran": "run service command five times",
            "observed": "service command exited zero five times",
            "verdict": "pass direct service output",
            "artifact": str(path),
        },
        {
            "test": "B",
            "ran": "capture wrapper status",
            "observed": "wrapper status stayed successful before output",
            "verdict": "pass wrapper status",
            "artifact": str(path),
        },
        {
            "test": "C",
            "ran": "compare cache bytes",
            "observed": "cache response bytes were fresh, not replayed",
            "verdict": "pass freshness comparison",
            "artifact": str(path),
        },
    ]


def verify_args(workdir):
    source = workdir / "verify-source.txt"
    excerpt = "MCP smoke marker: this local file proves the verification source path was read during the test."
    source.write_text(excerpt + "\n", encoding="utf-8")
    return {
        "claim": "The MCP smoke artifact contains the expected marker in a regular local file.",
        "sources": [{"path": str(source), "kind": "executed", "excerpt": excerpt}],
        "discriminator": {
            "name": "regular file marker check",
            "would_fail_if": "The source file were missing, empty, or lacked the MCP smoke marker.",
            "ran": "read verify-source.txt and count marker occurrences",
            "observed": "The regular file existed and contained exactly one MCP smoke marker line.",
            "outcome": "held",
        },
        "invariants": [
            {
                "name": "marker count",
                "computed_from": "verify-source.txt bytes only",
                "why_independent": "A byte count from one source would catch a missing or substituted artifact without comparing it to another tool result.",
                "expected": "exactly one marker line",
                "measured": "exactly one marker line",
                "holds": True,
            }
        ],
        "sample": {"n": 1, "selection": "single deterministic smoke artifact created by this test", "includes_edge_cases": True},
        "blind_to": ["whether an already-running Codex desktop turn has reloaded its tool list"],
        "verdict": {
            "established": "The local verification artifact was present and readable.",
            "not_established": "This does not prove the current desktop turn hot-loaded new tools.",
        },
    }


def isolate_args():
    return {
        "input": "The renderer fails on wide viewports after the viewport refactor.",
        "symptom": "Layout collapses only when the window exceeds 1280px.",
        "baseline": {"ref": "release before refactor", "outcome": "no collapse at any width"},
        "variables": ["viewport meta", "CSS container queries", "grid template", "overflow-x"],
        "trials": [
            {
                "name": "T1",
                "changed": "viewport meta",
                "from": "width=device-width",
                "to": "fixed 1280",
                "level": "surface",
                "observed": "collapse disappeared at 1280 exactly; returns at 1300 with fixed meta",
                "outcome": "resolves",
                "known_good": True,
            },
            {
                "name": "T2",
                "changed": "CSS container queries",
                "from": "enabled",
                "to": "disabled",
                "level": "intermediate",
                "observed": "collapse still occurs at 1300 with container queries off",
                "outcome": "reproduces",
                "known_good": False,
            },
            {
                "name": "T3",
                "changed": "grid template",
                "from": "auto-fit",
                "to": "fixed 12 col",
                "level": "low",
                "observed": "layout width reads 1320px in the grid; still collapses past 1280",
                "outcome": "reproduces",
                "known_good": False,
            },
            {
                "name": "T4",
                "changed": "overflow-x",
                "from": "hidden",
                "to": "visible",
                "level": "surface",
                "observed": "horizontal scrollbar appears, no layout collapse change",
                "outcome": "unchanged",
                "known_good": False,
            },
        ],
        "culprit": "viewport meta",
        "confidence": {
            "p": 0.95,
            "because": "the fixed viewport meta trial resolved while the container-query and grid trials still reproduced",
        },
    }


def discovery_args():
    return {
        "input": "Min-plus path composition can predict delay growth in railway dispatching and packet scheduling.",
        "domain_a": "railway dispatching",
        "domain_b": "packet scheduling",
        "shared_formal_object": "min-plus algebra over path costs with associative composition and bottleneck delay accumulation",
        "transported_prediction": "The 95th percentile delay should scale within 20% of the min-plus path bound as load increases from 0.4x to 0.8x.",
        "refuting_experiment": "Reject if measured delay slope is flat or differs by more than 50% from the min-plus bound across 30 scheduled runs.",
        "prior_art": [
            {
                "query": "railway dispatching min-plus delay bound",
                "top_result": "no direct hit for this transported packet scheduling comparison",
                "verdict": "clear",
            },
            {
                "query": "packet scheduling min-plus railway timetable analogy",
                "top_result": "related queueing papers but no direct timetable prediction",
                "verdict": "related",
            },
        ],
        "interpretations": [
            {"reading": "Both fields compose waiting costs along a route.", "category": "formal"},
            {"reading": "The analogy may fail because dispatch rules add discrete constraints.", "category": "limitation"},
            {"reading": "The prediction is only about delay slope, not full route quality.", "category": "scope"},
        ],
        "meaning": "Test whether the shared min-plus path-cost structure predicts numerical delay growth in railway dispatching and packet scheduling.",
        "how_to_prove": [
            "Measure 95th percentile delay across 30 runs at 0.4x, 0.6x, and 0.8x load.",
            "Compare the measured slope against the min-plus path bound within a 20% tolerance.",
        ],
        "failure_signature": "Measured delay slopes are flat, opposite-signed, or more than 50% away from the min-plus bound.",
        "success_signature": "Measured 95th percentile delay slopes remain within 20% of the min-plus path bound in both domains.",
        "prior": {"p": 0.35, "because": "The algebraic analogy is plausible but discrete dispatch constraints may dominate."},
        "posterior": {"p": 0.6, "because": "The proposed test is quantitative but not yet run against real schedules."},
        "tests": [
            {
                "name": "D1",
                "hypothesis": "delay follows the min-plus path bound",
                "metric": "95th percentile delay slope",
                "decision_rule": "pass if slope is within 20 percent",
                "discriminates": 0,
            },
            {
                "name": "D2",
                "hypothesis": "dispatch constraints dominate the bound",
                "metric": "absolute slope error",
                "decision_rule": "fail if error exceeds 50 percent",
                "adversarial": True,
                "discriminates": 1,
            },
            {
                "name": "D3",
                "hypothesis": "load increase preserves monotone delay growth",
                "metric": "delay ratio at 0.8x versus 0.4x load",
                "decision_rule": "pass if the ratio is greater than 1.1",
                "discriminates": 2,
            },
        ],
    }


def require_contains(name, text, needle):
    if needle not in text:
        raise AssertionError(f"{name} did not contain {needle!r}:\n{text}")


def main():
    parser = argparse.ArgumentParser(description="Probe the compiled think MCP server over real stdio JSON-RPC.")
    parser.add_argument("--binary", default=str(Path(__file__).resolve().parents[1] / "think-mcp"))
    parser.add_argument("--store")
    parser.add_argument("--integrity-log")
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        raise SystemExit(f"missing binary: {binary}")

    with tempfile.TemporaryDirectory(prefix="think-mcp-probe-") as tmp:
        workdir = Path(tmp)
        store = Path(args.store).resolve() if args.store else workdir / "thoughts.json"
        integrity_log = Path(args.integrity_log).resolve() if args.integrity_log else workdir / "integrity.jsonl"

        responses = rpc(
            binary,
            store,
            integrity_log,
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe", "version": "1"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ],
        )
        tools = responses[-1]["result"]["tools"]
        names = [tool["name"] for tool in tools]
        if names != EXPECTED_TOOLS:
            raise AssertionError(f"tool names changed:\nexpected {EXPECTED_TOOLS}\ngot      {names}")
        for tool in tools:
            schema = tool["inputSchema"]
            if "required" in schema and schema["required"] is None:
                raise AssertionError(f"{tool['name']} emitted required:null")

        require_contains("think", call_tool(binary, store, integrity_log, 3, "think", valid_breakdown(True)), "Breakdown #1 accepted")
        require_contains("get_thoughts", call_tool(binary, store, integrity_log, 4, "get_thoughts", {"limit": 5}), "#1")
        require_contains("think_open", call_tool(binary, store, integrity_log, 5, "think_open", valid_breakdown(False)), "Breakdown #2 OPEN")

        time.sleep(0.05)
        evidence = workdir / "evidence.txt"
        write_evidence(evidence)
        require_contains(
            "think_conclude",
            call_tool(
                binary,
                store,
                integrity_log,
                6,
                "think_conclude",
                {
                    "id": 2,
                    "evidence": conclusion_evidence(evidence),
                    "posterior": {
                        "p": 0.7,
                        "because": "The direct service output exited zero and the response was fresh in the recorded evidence.",
                    },
                },
            ),
            "Breakdown #2 VERIFIED",
        )
        require_contains(
            "think_recall",
            call_tool(
                binary,
                store,
                integrity_log,
                7,
                "think_recall",
                {"query": "The cache health check reflects actual service state rather than a wrapper artifact.", "limit": 5},
            ),
            "similar past record",
        )
        require_contains("think_verify", call_tool(binary, store, integrity_log, 8, "think_verify", verify_args(workdir)), "VERIFIED")
        require_contains("think_isolate", call_tool(binary, store, integrity_log, 9, "think_isolate", isolate_args()), "Isolation #4 RECORDED")
        require_contains("think_discover", call_tool(binary, store, integrity_log, 10, "think_discover", discovery_args()), "Discovery #5 accepted")
        require_contains("think_calibrate", call_tool(binary, store, integrity_log, 11, "think_calibrate", {}), "CALIBRATION")

        print("think MCP probe passed")
        print("\n".join(names))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
