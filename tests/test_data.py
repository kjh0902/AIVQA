from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aivqa.data import (
    GenerationCollator,
    KananaVQADataset,
    TrainCollator,
    extract_sa_constraints,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def batch_encode_collate(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        lengths = []
        for sample in samples:
            self.assert_sample(sample)
            has_answer = sample["conv"][-1]["role"] == "assistant"
            lengths.append(6 if has_answer else 5)

        width = max(lengths)
        input_ids = np.zeros((len(samples), width), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = np.arange(1, length + 1)
            input_ids[row, 2] = -1
            attention_mask[row, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": np.ones((len(samples), 2), dtype=np.float32),
            "image_metas": {"vision_grid_thw": np.ones((len(samples), 3))},
        }

    @staticmethod
    def assert_sample(sample) -> None:
        if set(sample) != {"image", "conv"}:
            raise TypeError("processor sample must contain image and conv")
        if len(sample["image"]) != 1 or not isinstance(sample["image"][0], Image.Image):
            raise TypeError("processor sample must contain one PIL image")
        for message in sample["conv"]:
            if not isinstance(message["content"], str):
                raise TypeError("conversation content must be text")


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "train").mkdir()
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(
            self.root / "train" / "mc.jpg", format="JPEG"
        )
        Image.new("RGBA", (8, 8), color=(0, 255, 0, 128)).save(
            self.root / "train" / "sa.jpg", format="TIFF"
        )
        Image.new("RGB", (8, 8), color=(0, 0, 255)).save(
            self.root / "train" / "la.jpg", format="JPEG"
        )
        self.original_tiff_bytes = (self.root / "train" / "sa.jpg").read_bytes()
        records = [
            {
                "metadata": {
                    "question_id": "1",
                    "split": "train",
                    "question_form": "MC",
                },
                "model_input": {
                    "image_name": "mc.jpg",
                    "question": "정답을 고르세요.",
                    "options": ["1) 첫째", "2) 둘째"],
                },
                "model_output": {"answer": "2"},
            },
            {
                "metadata": {
                    "question_id": "2",
                    "split": "train",
                    "question_form": "SA",
                },
                "model_input": {
                    "image_name": "sa.jpg",
                    "question": (
                        "사진 속 창살에서 찾을 수 있는 도형 중 정사각형 외의 도형을 "
                        "찾아 4음절로 답하시오."
                    ),
                    "options": [],
                },
                "model_output": {"answer": "정답"},
            },
            {
                "metadata": {
                    "question_id": "3",
                    "split": "train",
                    "question_form": "LA",
                },
                "model_input": {
                    "image_name": "la.jpg",
                    "question": "내용을 서술하세요.",
                    "options": [],
                },
                "model_output": {"answer": "서술형 정답입니다."},
            },
        ]
        self.json_path = self.root / "annotations.json"
        self.json_path.write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        self.dataset = KananaVQADataset(self.json_path, dataset_root=self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_mc_options_are_appended(self) -> None:
        sample = self.dataset[0]
        self.assertIn("객관식 (MC)", sample["formatted_question"])
        self.assertIn("선택지:\n1) 첫째\n2) 둘째", sample["formatted_question"])
        self.assertEqual(sample["question"], "정답을 고르세요.")
        self.assertEqual(sample["options"], ["1) 첫째", "2) 둘째"])
        self.assertEqual(sample["answer"], "2")
        self.assertEqual(
            [message["role"] for message in sample["conversation"]],
            ["system", "user", "user"],
        )
        self.assertEqual(sample["conversation"][1]["content"], "<image>")

    def test_empty_options_are_not_rendered_and_source_image_is_unchanged(self) -> None:
        sample = self.dataset[1]
        formatted_question = sample["formatted_question"]
        self.assertNotIn("선택지:", formatted_question)
        self.assertNotIn("[]", formatted_question)

        image = sample["image"]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 255, 0))
        self.assertEqual(
            (self.root / "train" / "sa.jpg").read_bytes(), self.original_tiff_bytes
        )

    def test_question_form_specific_instructions_reach_both_collators(self) -> None:
        samples = [self.dataset[index] for index in range(3)]
        prompts = [sample["conversation"][0]["content"] for sample in samples]

        self.assertIn("오름차순", prompts[0])
        self.assertIn("이 문제는 단답형입니다.", prompts[1])
        self.assertIn("- 요구 음절 수: 4음절", prompts[1])
        self.assertIn("최종 답변이 위 조건을 만족하는지", prompts[1])
        self.assertIn("250자 이내의 한 문단", prompts[2])
        self.assertEqual(len(set(prompts)), 3)

        for collator_type in (TrainCollator, GenerationCollator):
            processor = _FakeProcessor()
            collator_type(processor)(samples)
            collated_prompts = [
                sample["conv"][0]["content"] for sample in processor.calls[0][0]
            ]
            self.assertEqual(collated_prompts, prompts)

    def test_sa_constraint_parser_supports_all_requested_units(self) -> None:
        question = (
            "첫 답은 2음절, 둘째 답은 3어절로 쓰고, 이유 4가지를 답하시오. "
            "이칭 5개를 나열하시오. 마지막으로 6답하시오."
        )
        self.assertEqual(
            extract_sa_constraints(question),
            [
                ("음절 수", "2음절"),
                ("어절 수", "3어절"),
                ("답변 개수", "4가지"),
                ("답변 개수", "5개"),
                ("답 수", "6답"),
            ],
        )

    def test_sa_constraint_parser_ignores_descriptive_object_counts(self) -> None:
        question = (
            "사진 속 문구는 물체의 개수가 2개임을 뜻한다. "
            "4개의 그릇 중 하나의 이름을 3음절로 답하시오."
        )
        self.assertEqual(
            extract_sa_constraints(question),
            [("음절 수", "3음절")],
        )

    def test_sa_constraint_parser_supports_korean_count_words(self) -> None:
        self.assertEqual(
            extract_sa_constraints("식품의 이름을 두 음절로 답하시오."),
            [("음절 수", "두 음절")],
        )

    def test_collator_uses_native_sample_shape_and_reuses_image(self) -> None:
        sample = self.dataset[0]
        processor = _FakeProcessor()
        TrainCollator(processor)([sample])
        processor_sample = processor.calls[0][0][0]
        self.assertIs(processor_sample["image"][0], sample["image"])
        self.assertEqual(set(processor_sample), {"image", "conv"})

    def test_train_collator_adds_answer_and_masks_prompt_and_image_tokens(self) -> None:
        processor = _FakeProcessor()
        batch = TrainCollator(processor)([self.dataset[0]])
        full_conversation = processor.calls[0][0][0]["conv"]
        self.assertEqual(
            full_conversation[-1], {"role": "assistant", "content": "2"}
        )
        np.testing.assert_array_equal(
            batch["labels"], [[-100, -100, -100, -100, -100, 6]]
        )
        self.assertEqual(processor.calls[0][1]["padding_side"], "right")
        self.assertEqual(processor.calls[0][1]["max_length"], 2048)

    def test_generation_collator_excludes_answer_and_left_pads(self) -> None:
        processor = _FakeProcessor()
        batch = GenerationCollator(processor)([self.dataset[0]])
        conversation = processor.calls[0][0][0]["conv"]
        self.assertEqual(
            [message["role"] for message in conversation],
            ["system", "user", "user"],
        )
        self.assertNotIn("labels", batch)
        self.assertTrue(processor.calls[0][1]["add_generation_prompt"])
        self.assertEqual(processor.calls[0][1]["padding_side"], "left")

    def test_train_collator_rejects_missing_answer(self) -> None:
        processor = _FakeProcessor()
        sample = self.dataset[0]
        sample["answer"] = None
        with self.assertRaisesRegex(ValueError, "Training sample 1"):
            TrainCollator(processor)([sample])


if __name__ == "__main__":
    unittest.main()
