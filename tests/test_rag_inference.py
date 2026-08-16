from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
from rag_db.augmentation import CombinedVQADataset, RagAugmentedDataset
from rag_db.prompts import Candidate, build_answer_feature, build_search_feature

RUNTIME_DEPENDENCIES = ("torch", "qdrant_client", "sentence_transformers", "transformers")
DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(package) is not None for package in RUNTIME_DEPENDENCIES
)

if DEPENDENCIES_AVAILABLE:
    from rag_db.infer_with_rag import (
        LocalImageIndex,
        QdrantRetriever,
        normalize_exact_text,
        parse_search_terms,
        truncate_kanana_encoding,
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


class RagPromptAndDatasetTest(unittest.TestCase):
    def test_search_prompt_uses_only_image_and_question(self) -> None:
        sample = {"image": Image.new("RGB", (4, 4))}
        feature = build_search_feature(sample, "경복궁의 건물은 무엇인가?")
        content = feature["conversation"][-1]["content"]
        self.assertIn("경복궁의 건물은 무엇인가?", content)
        self.assertEqual(feature["conversation"][1]["content"], "<image>")

    def test_answer_prompt_omits_empty_rag_section(self) -> None:
        sample = {"question_form": "SA", "image": Image.new("RGB", (4, 4))}
        no_rag = build_answer_feature(sample, "질문", [], [])
        self.assertNotIn("RAG 참고정보", no_rag["conversation"][-1]["content"])

        candidate = Candidate("doc", _payload("doc", "제목", "실제 설명"))
        with_rag = build_answer_feature(sample, "질문", [], [candidate])
        content = with_rag["conversation"][-1]["content"]
        self.assertIn("RAG 참고정보:\n실제 설명", content)
        self.assertNotIn("제목", content)
        self.assertIn("정답이 아닐 수 있습니다", with_rag["conversation"][0]["content"])

    def test_answer_prompt_caps_rag_text_for_training(self) -> None:
        sample = {"question_form": "SA", "image": Image.new("RGB", (4, 4))}
        candidate = Candidate("doc", _payload("doc", "제목", "가나다라마바사"))
        feature = build_answer_feature(
            sample, "질문", [], [candidate], max_rag_chars=4
        )
        self.assertIn("RAG 참고정보:\n가나다라", feature["conversation"][-1]["content"])

    def test_rag_dataset_preserves_answer_and_adds_context(self) -> None:
        class Dataset:
            records = [
                {"metadata": {"question_form": "SA"}},
                {"metadata": {"question_form": "LA"}},
            ]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {
                    "question_id": str(index),
                    "question_form": self.records[index]["metadata"]["question_form"],
                    "question": f"질문 {index}",
                    "options": [],
                    "answer": f"정답 {index}",
                    "image": Image.new("RGB", (4, 4)),
                    "conversation": [],
                }

        combined = CombinedVQADataset([Dataset(), Dataset()])
        self.assertEqual(len(combined), 4)
        self.assertEqual(combined[2]["question_id"], "0")

        candidate = Candidate("doc", _payload("doc", "제목", "검색 본문"))
        augmented = RagAugmentedDataset(Dataset(), [[candidate], []])
        first = augmented[0]
        self.assertEqual(first["answer"], "정답 0")
        self.assertIn("RAG 참고정보:\n검색 본문", first["conversation"][-1]["content"])
        self.assertNotIn("RAG 참고정보", augmented[1]["conversation"][-1]["content"])


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "RAG inference dependencies are not installed",
)
class RagInferenceTest(unittest.TestCase):
    def test_search_term_parser_accepts_fence_and_deduplicates(self) -> None:
        result = parse_search_terms('```json\n[" 경복궁 ", "근정전", "경복궁", 3]\n```')
        self.assertEqual(result, ["경복궁", "근정전"])
        self.assertEqual(parse_search_terms("[]"), [])

    def test_search_term_parser_recovers_common_kanana_outputs(self) -> None:
        self.assertEqual(parse_search_terms("피부과"), ["피부과"])
        self.assertEqual(
            parse_search_terms("[의복명, 전통문화]"), ["의복명", "전통문화"]
        )
        self.assertEqual(
            parse_search_terms('{"건축물명": "종묘 정전", "원래 건축물": "정전"}'),
            ["종묘 정전", "정전"],
        )
        self.assertEqual(
            parse_search_terms('{search_terms: "불국사", 보조어: 석가탑}'),
            ["불국사", "석가탑"],
        )

    def test_exact_normalization_is_unicode_and_whitespace_only(self) -> None:
        self.assertEqual(normalize_exact_text("  경복궁\n 본전  "), "경복궁 본전")
        self.assertNotEqual(normalize_exact_text("경복궁"), normalize_exact_text("경복궁터"))

    def test_kanana_encoding_truncation_preserves_image_and_generation_suffix(self) -> None:
        import torch

        text_encoding = {
            "input_ids": torch.tensor([10, 11, -1, -1, 20, 21, 22, 30, 31]),
            "attention_mask": torch.ones(9, dtype=torch.long),
            "seq_length": 9,
        }
        changed = truncate_kanana_encoding(
            text_encoding, max_length=7, generation_suffix_length=2
        )

        self.assertTrue(changed)
        self.assertEqual(
            text_encoding["input_ids"].tolist(), [11, -1, -1, 20, 21, 30, 31]
        )
        self.assertEqual(text_encoding["attention_mask"].tolist(), [1] * 7)
        self.assertEqual(text_encoding["seq_length"], 7)

    def test_kanana_encoding_below_limit_is_unchanged(self) -> None:
        import torch

        text_encoding = {
            "input_ids": torch.tensor([10, -1, 20, 30]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "seq_length": 4,
        }
        self.assertFalse(
            truncate_kanana_encoding(
                text_encoding, max_length=4, generation_suffix_length=1
            )
        )
        self.assertEqual(text_encoding["input_ids"].tolist(), [10, -1, 20, 30])

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
        image_filter = client.query_calls[1]["query_filter"]
        self.assertEqual(image_filter.must[0].has_vector, "image")

        self.assertEqual([item.doc_id for item in candidates], ["exact", "both", "image"])
        both = candidates[1]
        self.assertAlmostEqual(both.final_score, 1.92)
        self.assertEqual(candidates[0].text_score, 2.0)

    def test_local_image_index_uses_max_sim_and_threshold(self) -> None:
        first = _payload("first", "첫째")
        second = _payload("second", "둘째")
        index = LocalImageIndex(
            payloads=[first, second],
            vectors=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.6],
                ],
                dtype=np.float32,
            ),
            owners=np.asarray([0, 0, 1], dtype=np.int32),
        )
        results = index.search([1.0, 0.0], 0.9)
        self.assertEqual(results, [(first, 1.0)])



if __name__ == "__main__":
    unittest.main()
