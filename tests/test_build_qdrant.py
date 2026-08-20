from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

RUNTIME_DEPENDENCIES = ("torch", "qdrant_client", "sentence_transformers", "transformers")
DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(package) is not None for package in RUNTIME_DEPENDENCIES
)

if DEPENDENCIES_AVAILABLE:
    from rag_db.build_qdrant import BuildStats, iter_entities, make_search_text


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "RAG runtime dependencies are not installed")
class BuildQdrantTests(unittest.TestCase):
    def test_make_search_text_accepts_string(self):
        self.assertEqual(make_search_text("  경복궁 근정전  "), "경복궁 근정전")

    def test_make_search_text_joins_list(self):
        self.assertEqual(make_search_text([" 경복궁 ", "", "근정전"]), "경복궁 근정전")

    def test_iter_entities_accepts_string_and_list_search_terms(self):
        entities = [
            {"doc_id": "string", "search_terms": "경복궁 근정전", "image_path": None},
            {"doc_id": "list", "search_terms": ["경복궁", "근정전"], "image_path": []},
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "entities.jsonl"
            input_path.write_text(
                "".join(json.dumps(entity, ensure_ascii=False) + "\n" for entity in entities),
                encoding="utf-8",
            )
            stats = BuildStats()

            loaded = list(iter_entities(input_path, stats))

        self.assertEqual([entity["doc_id"] for entity in loaded], ["string", "list"])
        self.assertEqual(stats.lines_read, 2)
        self.assertEqual(stats.invalid_entities, 0)
        self.assertEqual(loaded[0]["image_path"], [])


if __name__ == "__main__":
    unittest.main()
