#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import semantic_recall


class SemanticRecallTests(unittest.TestCase):
    def tearDown(self):
        semantic_recall._reset_backend_for_tests()
        os.environ.pop("THINK_SEMANTIC_BACKEND", None)
        os.environ.pop("THINK_SEMANTIC_MODEL", None)

    def test_local_model_is_discoverable_without_remote_identifier(self):
        path = semantic_recall.discover_model_path()
        if path is None:
            self.skipTest("no complete cached BGE/MiniLM snapshot is present")
        self.assertTrue(path.is_absolute())
        self.assertTrue((path / "config.json").is_file())
        self.assertTrue((path / "tokenizer.json").is_file())
        self.assertTrue(
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )

    def test_cached_embedding_backend_ranks_paraphrase(self):
        if semantic_recall.discover_model_path() is None:
            self.skipTest("no complete cached embedding model is present")
        scores = semantic_recall.score_query(
            "The health probe hides backend errors",
            [
                {"id": 1, "text": "The cache health check masks a failed service."},
                {"id": 2, "text": "The renderer uses a blue gradient in the landing page."},
            ],
        )
        self.assertEqual(scores[0]["id"], 1)
        self.assertGreater(scores[0]["score"], scores[1]["score"])
        self.assertTrue(semantic_recall.embedding_backend().startswith("offline-transformers:"))

    def test_remote_identifier_is_rejected_without_network(self):
        os.environ["THINK_SEMANTIC_MODEL"] = "BAAI/bge-small-en-v1.5"
        self.assertIsNone(semantic_recall.discover_model_path())
        self.assertEqual(semantic_recall.embedding_backend(), "offline-char-token")

    def test_explicit_lexical_mode_never_loads_model(self):
        os.environ["THINK_SEMANTIC_BACKEND"] = "lexical"
        self.assertEqual(semantic_recall.embedding_backend(), "offline-char-token")
        scores = semantic_recall.score_query(
            "stale cache output",
            [{"id": 4, "text": "The cache replays stale output."}],
        )
        self.assertEqual(scores[0]["id"], 4)

    def test_json_lines_process_reports_backend(self):
        process = subprocess.run(
            [sys.executable, "semantic_recall.py"],
            input=json.dumps({
                "id": "test",
                "query": "stale cache output",
                "candidates": [{"id": 4, "text": "The cache replays stale output."}],
            }) + "\n",
            text=True,
            capture_output=True,
            check=True,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        )
        response = json.loads(process.stdout)
        self.assertTrue(response["ok"])
        self.assertEqual(response["id"], "test")
        self.assertEqual(response["scores"][0]["id"], 4)
        self.assertIn(response["backend"].split(":")[0], {"offline-transformers", "offline-char-token"})


if __name__ == "__main__":
    unittest.main()
