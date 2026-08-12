from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

RUNTIME_DEPENDENCIES = ("torch", "qdrant_client", "sentence_transformers", "transformers")
DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(package) is not None for package in RUNTIME_DEPENDENCIES
)

if DEPENDENCIES_AVAILABLE:
    from rag_db.infer_with_rag import (
        Candidate,
        QdrantRetriever,
        build_answer_feature,
        load_ocr_index,
        map_test_ocr,
        normalize_exact_text,
        parse_search_terms,
    )


def _payload(doc_id: str, title: str, description: str = "본문") -> dict:
    return {
        "doc_id": doc_id,
        "source": "test",
        "title": title,
        "search_terms": [title],
        "description": description,
        "image_path": [],
    }


class _FakeEncoders:
    def embed_text(self, text):
        return [float(len(text))]

    def embed_image(self, image):
        return [1.0]


class _FakeQdrant:
    def __init__(self):
        self.query_calls = []
        self.payloads = [
            _payload("exact", "경복궁"),
            _payload("semantic", "근정전"),
            _payload("image", "광화문"),
            _payload("both", "궁궐"),
        ]

    def scroll(self, **kwargs):
        points = [
            SimpleNamespace(
                payload={"doc_id": payload["doc_id"], "title": payload["title"]}
            )
            for payload in self.payloads
        ]
        return points, None

    def retrieve(self, **kwargs):
        requested = set(kwargs["ids"])
        from rag_db.build_qdrant import deterministic_point_id

        return [
            SimpleNamespace(payload=payload)
            for payload in self.payloads
            if deterministic_point_id(payload["doc_id"]) in requested
        ]

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        if kwargs["offset"]:
            return SimpleNamespace(points=[])
        if kwargs["using"] == "text":
            points = [
                SimpleNamespace(payload=self.payloads[1], score=0.91),
                SimpleNamespace(payload=self.payloads[3], score=0.95),
            ]
        else:
            points = [
                SimpleNamespace(payload=self.payloads[2], score=0.92),
                SimpleNamespace(payload=self.payloads[3], score=0.97),
            ]
        return SimpleNamespace(points=points)


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "RAG inference dependencies are not installed",
)
class RagInferenceTest(unittest.TestCase):
    def test_search_term_parser_accepts_fence_and_deduplicates(self) -> None:
        result = parse_search_terms('```json\n[" 경복궁 ", "근정전", "경복궁", 3]\n```')
        self.assertEqual(result, ["경복궁", "근정전"])
        self.assertEqual(parse_search_terms("not json"), [])

    def test_exact_normalization_is_unicode_and_whitespace_only(self) -> None:
        self.assertEqual(normalize_exact_text("  경복궁\n 본전  "), "경복궁 본전")
        self.assertNotEqual(normalize_exact_text("경복궁"), normalize_exact_text("경복궁터"))

    def test_ocr_mapping_uses_split_and_image_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ocr.jsonl"
            rows = [
                {"split": "train", "image_name": "same.jpg", "ocr_text": "훈련"},
                {"split": "test", "image_name": "same.jpg", "ocr_text": "테스트"},
            ]
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            records = [
                {
                    "metadata": {"split": "test"},
                    "model_input": {"image_name": "same.jpg"},
                }
            ]
            self.assertEqual(map_test_ocr(records, load_ocr_index(path)), ["테스트"])

    def test_retrieval_exact_semantic_image_and_doc_id_fusion(self) -> None:
        client = _FakeQdrant()
        retriever = QdrantRetriever(client, "collection", _FakeEncoders(), 0.9, 100)
        image = Image.new("RGB", (4, 4))
        candidates = retriever.retrieve([" 경복궁 ", "관련어"], image)

        # Exact matches bypass semantic lookup for that term. The image query is a
        # one-row matrix because the collection's image field is a multivector.
        self.assertEqual(len(client.query_calls), 2)
        self.assertEqual(client.query_calls[0]["using"], "text")
        self.assertEqual(client.query_calls[0]["score_threshold"], 0.9)
        self.assertEqual(client.query_calls[1]["using"], "image")
        self.assertEqual(client.query_calls[1]["query"], [[1.0]])

        self.assertEqual([item.doc_id for item in candidates], ["exact", "both", "image"])
        both = candidates[1]
        self.assertAlmostEqual(both.final_score, 1.92)
        self.assertEqual(candidates[0].text_score, 2.0)

    def test_answer_prompt_omits_empty_rag_section(self) -> None:
        sample = {"question_form": "SA", "image": Image.new("RGB", (4, 4))}
        no_rag = build_answer_feature(sample, "질문", [], "OCR", [])
        self.assertNotIn("RAG 참고정보", no_rag["conversation"][-1]["content"])

        candidate = Candidate("doc", _payload("doc", "제목", "실제 설명"))
        with_rag = build_answer_feature(sample, "질문", [], "OCR", [candidate])
        content = with_rag["conversation"][-1]["content"]
        self.assertIn("RAG 참고정보:\n실제 설명", content)
        self.assertNotIn("제목", content)
        self.assertIn("정답이 아닐 수 있습니다", with_rag["conversation"][0]["content"])


if __name__ == "__main__":
    unittest.main()
