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

    def test_iter_entities_normalizes_supported_search_terms_and_image_paths(self):
        entities = [
            {
                "doc_id": "strings",
                "search_terms": "경복궁 근정전",
                "image_path": "images/palace.jpg",
            },
            {
                "doc_id": "lists",
                "search_terms": ["경복궁", "근정전"],
                "image_path": ["images/palace.jpg"],
            },
            {"doc_id": "null-image", "search_terms": "경복궁", "image_path": None},
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "entities.jsonl"
            input_path.write_text(
                "".join(json.dumps(entity, ensure_ascii=False) + "\n" for entity in entities),
                encoding="utf-8",
            )
            stats = BuildStats()

            loaded = list(iter_entities(input_path, stats))

        self.assertEqual(
            [entity["doc_id"] for entity in loaded],
            ["strings", "lists", "null-image"],
        )
        self.assertEqual(stats.lines_read, 3)
        self.assertEqual(stats.invalid_entities, 0)
        self.assertEqual(loaded[0]["image_path"], ["images/palace.jpg"])
        self.assertEqual(loaded[1]["image_path"], ["images/palace.jpg"])
        self.assertEqual(loaded[2]["image_path"], [])

    def test_iter_entities_rejects_unsupported_image_path_type(self):
        entity = {"doc_id": "invalid", "search_terms": "경복궁", "image_path": 3}

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "entities.jsonl"
            input_path.write_text(json.dumps(entity) + "\n", encoding="utf-8")
            stats = BuildStats()

            loaded = list(iter_entities(input_path, stats))

        self.assertEqual(loaded, [])
        self.assertEqual(stats.lines_read, 1)
        self.assertEqual(stats.invalid_entities, 1)


if __name__ == "__main__":
    unittest.main()
