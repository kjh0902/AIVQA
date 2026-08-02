from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from paddleocr_single_image_test import (
    DETECTION_MODEL,
    RECOGNITION_MODEL,
    TEXT_DET_LIMIT_SIDE_LEN,
    TEXT_DET_LIMIT_TYPE,
    build_ocr_models,
    extract_detection_results,
    load_original_rgb,
    recognize_text_results,
    save_visualization,
)


class SingleImageOcrTest(unittest.TestCase):
    def test_builds_requested_detection_and_korean_recognition_models(self) -> None:
        detection_calls = []
        recognition_calls = []

        class FakeTextDetection:
            def __init__(self, **kwargs) -> None:
                detection_calls.append(kwargs)

        class FakeTextRecognition:
            def __init__(self, **kwargs) -> None:
                recognition_calls.append(kwargs)

        fake_module = types.SimpleNamespace(
            TextDetection=FakeTextDetection,
            TextRecognition=FakeTextRecognition,
        )
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            detector, recognizer = build_ocr_models("cpu")

        self.assertIsInstance(detector, FakeTextDetection)
        self.assertIsInstance(recognizer, FakeTextRecognition)
        self.assertEqual(detection_calls[0]["model_name"], DETECTION_MODEL)
        self.assertEqual(detection_calls[0]["limit_type"], TEXT_DET_LIMIT_TYPE)
        self.assertEqual(
            detection_calls[0]["limit_side_len"], TEXT_DET_LIMIT_SIDE_LEN
        )
        self.assertEqual(recognition_calls[0]["model_name"], RECOGNITION_MODEL)
        self.assertEqual(os.environ["FLAGS_use_mkldnn"], "0")

    def test_keeps_each_detection_and_confidence_independent(self) -> None:
        payload = {
            "dt_polys": [
                [[10, 10], [40, 8], [42, 20], [12, 22]],
                [[60, 30], [90, 30], [90, 40], [60, 40]],
            ],
            "dt_scores": [0.91, 0.82],
        }

        entries = extract_detection_results(payload, (100, 80))

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["bbox"], [10, 8, 42, 22])
        self.assertEqual(entries[0]["detection_confidence"], 0.91)
        self.assertEqual(entries[1]["detection_confidence"], 0.82)

    def test_recognizes_each_detection_from_lossless_temporary_image(self) -> None:
        class FakeRecognizer:
            def predict(self, image_paths, batch_size=1):
                self.image_paths = image_paths
                self.batch_size = batch_size
                self.paths_existed_during_prediction = all(
                    Path(path).is_file() for path in image_paths
                )
                return [
                    {"res": {"rec_text": "첫 번째", "rec_score": 0.97}},
                    {"res": {"rec_text": "두 번째", "rec_score": 0.88}},
                ]

        entries = [
            {
                "index": 1,
                "polygon": [[10, 10], [50, 10], [50, 25], [10, 25]],
                "text": "",
                "detection_confidence": 0.91,
                "recognition_confidence": None,
            },
            {
                "index": 2,
                "polygon": [[60, 30], [100, 30], [100, 45], [60, 45]],
                "text": "",
                "detection_confidence": 0.82,
                "recognition_confidence": None,
            },
        ]
        recognizer = FakeRecognizer()

        results = recognize_text_results(
            recognizer, Image.new("RGB", (120, 80), color="white"), entries
        )

        self.assertTrue(recognizer.paths_existed_during_prediction)
        self.assertEqual(recognizer.batch_size, 1)
        self.assertEqual(results[0]["text"], "첫 번째")
        self.assertEqual(results[0]["recognition_confidence"], 0.97)
        self.assertEqual(results[1]["text"], "두 번째")
        self.assertEqual(results[1]["recognition_confidence"], 0.88)

    def test_applies_exif_rotation_and_converts_to_original_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "rotated.jpg"
            image = Image.new("L", (40, 20), color=128)
            exif = Image.Exif()
            exif[274] = 6
            image.save(image_path, exif=exif)

            loaded = load_original_rgb(image_path)

        self.assertEqual(loaded.size, (20, 40))
        self.assertEqual(loaded.mode, "RGB")

    def test_saves_polygon_and_text_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "visualization.png"
            image = Image.new("RGB", (120, 80), color="white")
            entries = [
                {
                    "index": 1,
                    "polygon": [[10, 20], [90, 20], [90, 40], [10, 40]],
                    "text": "OCR",
                }
            ]

            save_visualization(image, entries, output_path)

            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as visualized:
                self.assertEqual(visualized.size, image.size)
                self.assertNotEqual(visualized.getpixel((10, 20)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
