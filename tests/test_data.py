from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aivqa.data import GenerationCollator, QwenVQADataset, TrainCollator


class _FakeTokenizer:
    padding_side = "right"


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        is_batch = bool(conversations and isinstance(conversations[0], list))
        batch = conversations if is_batch else [conversations]
        lengths = [5 if chat[-1]["role"] == "assistant" else 4 for chat in batch]
        width = max(lengths)
        input_ids = np.zeros((len(batch), width), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = np.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "train").mkdir()
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(
            self.root / "train" / "mc.jpg", format="JPEG"
        )
        # Deliberately store TIFF bytes under a .jpg name to reproduce the source data.
        Image.new("RGBA", (8, 8), color=(0, 255, 0, 128)).save(
            self.root / "train" / "sa.jpg", format="TIFF"
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
                    "question": "무엇인가요?",
                    "options": [],
                },
                "model_output": {"answer": "정답"},
            },
        ]
        self.json_path = self.root / "annotations.json"
        self.json_path.write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        self.dataset = QwenVQADataset(self.json_path, dataset_root=self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_mc_options_are_appended(self) -> None:
        sample = self.dataset[0]
        self.assertIn("객관식 (MC)", sample["formatted_question"])
        self.assertIn("선택지:\n1) 첫째\n2) 둘째", sample["formatted_question"])
        self.assertEqual(sample["answer"], "2")
        self.assertEqual([message["role"] for message in sample["messages"]], ["system", "user"])

    def test_empty_options_are_not_rendered(self) -> None:
        sample = self.dataset[1]
        formatted_question = sample["formatted_question"]
        self.assertNotIn("선택지:", formatted_question)
        self.assertNotIn("[]", formatted_question)

        image = sample["messages"][1]["content"][0]["image"]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 255, 0))
        self.assertEqual(
            (self.root / "train" / "sa.jpg").read_bytes(), self.original_tiff_bytes
        )

    def test_message_copy_reuses_in_memory_image(self) -> None:
        sample = self.dataset[0]
        original_image = sample["messages"][1]["content"][0]["image"]
        processor = _FakeProcessor()
        TrainCollator(processor)([sample])
        collated_image = processor.calls[0][0][0][1]["content"][0]["image"]
        self.assertIs(collated_image, original_image)

    def test_train_collator_adds_answer_and_masks_prompt(self) -> None:
        processor = _FakeProcessor()
        batch = TrainCollator(processor)([self.dataset[0]])
        full_conversation = processor.calls[0][0][0]
        self.assertEqual(full_conversation[-1], {"role": "assistant", "content": "2"})
        np.testing.assert_array_equal(batch["labels"], [[-100, -100, -100, -100, 5]])

    def test_generation_collator_excludes_answer(self) -> None:
        processor = _FakeProcessor()
        batch = GenerationCollator(processor)([self.dataset[0]])
        conversation = processor.calls[0][0][0]
        self.assertEqual([message["role"] for message in conversation], ["system", "user"])
        self.assertNotIn("labels", batch)
        self.assertTrue(processor.calls[0][1]["add_generation_prompt"])

    def test_train_collator_rejects_missing_answer(self) -> None:
        processor = _FakeProcessor()
        sample = self.dataset[0]
        sample["answer"] = None
        with self.assertRaisesRegex(ValueError, "Training sample 1"):
            TrainCollator(processor)([sample])


if __name__ == "__main__":
    unittest.main()
