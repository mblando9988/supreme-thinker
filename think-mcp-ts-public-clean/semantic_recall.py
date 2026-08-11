#!/usr/bin/env python3
"""Optional offline recall helper for think-mcp.

Protocol: one JSON request per line, one JSON response per line.
Go remains the ledger and integrity authority. This process never downloads,
resolves, or contacts a remote model registry: only local model directories
are accepted and Transformers is forced into offline mode.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Set these before importing Transformers so a cached model cannot trigger a
# metadata or weight lookup on the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "if", "in", "is", "it", "of", "on", "or", "the", "to", "with",
}
MAX_TEXT_CHARS = 4096
MAX_TOKENS = 256
MAX_CANDIDATES = 256

# Ordered by the intended quality/size tradeoff. The directory names are only
# cache keys; discovery still requires a complete local snapshot.
KNOWN_CACHE_KEYS = (
    "models--BAAI--bge-small-en-v1.5",
    "models--sentence-transformers--all-MiniLM-L6-v2",
)

_MODEL_ATTEMPTED = False
_MODEL: Any = None
_TOKENIZER: Any = None
_MODEL_LABEL = ""


def normalize(text: object) -> str:
    return " ".join(TOKEN_RE.findall(str(text or "").lower()))


def tokens(text: str) -> list[str]:
    return [word for word in normalize(text).split() if word not in STOP and len(word) > 2]


def features(text: str) -> Counter[str]:
    """Dependency-free fallback using word and character features."""
    clean = normalize(text)
    result: Counter[str] = Counter()
    for word in tokens(clean):
        result[f"w:{word}"] += 2.0
    padded = f"  {clean}  "
    for size in (3, 4, 5):
        for i in range(max(0, len(padded) - size + 1)):
            result[f"c:{padded[i:i + size]}"] += 1.0
    return result


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    dot = sum(left[key] * right[key] for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("THINK_SEMANTIC_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key, "").strip()
        if value:
            root = Path(value).expanduser()
            roots.append(root / "hub" if key == "HF_HOME" else root)
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _complete_snapshot(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "tokenizer.json").is_file()
        and any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    )


def discover_model_path() -> Path | None:
    """Return a complete local snapshot, never a remote model identifier."""
    explicit = os.environ.get("THINK_SEMANTIC_MODEL", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if _complete_snapshot(path) else None

    for root in _cache_roots():
        for key in KNOWN_CACHE_KEYS:
            snapshots = root / key / "snapshots"
            if not snapshots.is_dir():
                continue
            for snapshot in sorted(snapshots.iterdir()):
                if _complete_snapshot(snapshot):
                    return snapshot
    return None


def _model_label(path: Path) -> str:
    for key in KNOWN_CACHE_KEYS:
        if key in path.parts:
            return key.removeprefix("models--").replace("--", "/")
    return path.name


def _load_local_model() -> bool:
    global _MODEL_ATTEMPTED, _MODEL, _TOKENIZER, _MODEL_LABEL
    if _MODEL_ATTEMPTED:
        return _MODEL is not None
    _MODEL_ATTEMPTED = True

    if os.environ.get("THINK_SEMANTIC_BACKEND", "model").strip().lower() in {"lexical", "none", "off"}:
        return False
    path = discover_model_path()
    if path is None:
        return False

    try:
        # Imports stay lazy so lexical-only installations do not need the ML
        # stack merely to start the MCP helper.
        import torch
        from transformers import AutoModel, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        _MODEL = AutoModel.from_pretrained(str(path), local_files_only=True)
        _MODEL.eval()
        _MODEL_LABEL = _model_label(path)
        return True
    except Exception:
        # A broken/incomplete local model must never take recall down. Keep the
        # failure private and use the deterministic dependency-free scorer.
        _MODEL = None
        _TOKENIZER = None
        _MODEL_LABEL = ""
        return False


def embedding_backend() -> str:
    return f"offline-transformers:{_MODEL_LABEL}" if _load_local_model() else "offline-char-token"


def _embedding_scores(query: str, rows: list[tuple[object, str]]) -> list[dict[str, object]] | None:
    if not _load_local_model():
        return None
    try:
        import torch
        texts = [query[:MAX_TEXT_CHARS]] + [text[:MAX_TEXT_CHARS] for _, text in rows]
        encoded = _TOKENIZER(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = _MODEL(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
            scores = torch.mv(vectors[1:], vectors[0])
        scored = [
            {"id": candidate_id, "score": round(max(0.0, min(1.0, float(score))), 6)}
            for (candidate_id, _), score in zip(rows, scores)
        ]
        scored.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return scored
    except Exception:
        return None


def _lexical_scores(query: str, rows: list[tuple[object, str]]) -> list[dict[str, object]]:
    query_features = features(query[:MAX_TEXT_CHARS])
    vectors: list[tuple[object, Counter[str]]] = []
    document_frequency: Counter[str] = Counter()
    for candidate_id, text in rows:
        vector = features(text[:MAX_TEXT_CHARS])
        vectors.append((candidate_id, vector))
        document_frequency.update(vector.keys())

    document_count = max(1, len(vectors))

    def weighted(vector: Counter[str]) -> Counter[str]:
        return Counter({
            key: value * math.log((1.0 + document_count) / (1.0 + document_frequency[key])) + 1.0
            for key, value in vector.items()
        })

    query_weighted = weighted(query_features)
    scored = [
        {"id": candidate_id, "score": round(cosine(query_weighted, weighted(vector)), 6)}
        for candidate_id, vector in vectors
    ]
    scored.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return scored


def score_query(query: str, candidates: list[object]) -> list[dict[str, object]]:
    rows: list[tuple[object, str]] = []
    for candidate in candidates[:MAX_CANDIDATES]:
        if isinstance(candidate, dict):
            rows.append((candidate.get("id"), str(candidate.get("text") or "")))
    return _embedding_scores(query, rows) or _lexical_scores(query, rows)


def handle(request: object) -> dict[str, object]:
    if not isinstance(request, dict):
        return {"ok": False, "error": "request must be an object"}
    request_id = request.get("id")
    query = request.get("query")
    candidates = request.get("candidates")
    if not isinstance(query, str) or not isinstance(candidates, list):
        return {"id": request_id, "ok": False, "error": "query and candidates are required"}
    return {
        "id": request_id,
        "ok": True,
        "scores": score_query(query, candidates),
        "backend": embedding_backend(),
    }


def _reset_backend_for_tests() -> None:
    global _MODEL_ATTEMPTED, _MODEL, _TOKENIZER, _MODEL_LABEL
    _MODEL_ATTEMPTED = False
    _MODEL = None
    _TOKENIZER = None
    _MODEL_LABEL = ""


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle(request)
        except Exception as exc:  # keep the line protocol alive for one bad request
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
