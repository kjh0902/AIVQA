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
    build_ocr_pipeline,
    extract_text_results,
    load_original_rgb,
    save_visualization,
)


class SingleImageOcrTest(unittest.TestCase):
    def test_builds_requested_detection_and_korean_recognition_pipeline(self) -> None:
        calls = []

        class FakePaddleOCR:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

        fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            pipeline = build_ocr_pipeline("cpu")

        self.assertIsInstance(pipeline, FakePaddleOCR)
        self.assertEqual(calls[0]["text_detection_model_name"], DETECTION_MODEL)
        self.assertEqual(calls[0]["text_recognition_model_name"], RECOGNITION_MODEL)
        self.assertFalse(calls[0]["use_doc_orientation_classify"])
        self.assertFalse(calls[0]["use_doc_unwarping"])
        self.assertFalse(calls[0]["use_textline_orientation"])
        self.assertEqual(calls[0]["text_rec_score_thresh"], 0.0)
        self.assertEqual(os.environ["FLAGS_use_mkldnn"], "0")

    def test_keeps_each_detection_independent_and_attaches_recognition(self) -> None:
        payload = {
            "dt_polys": [
                [[10, 10], [40, 8], [42, 20], [12, 22]],
                [[60, 30], [90, 30], [90, 40], [60, 40]],
            ],
            "dt_scores": [0.91, 0.82],
            "rec_polys": [
                [[60, 30], [90, 30], [90, 40], [60, 40]],
                [[10, 10], [40, 8], [42, 20], [12, 22]],
            ],
            "rec_texts": ["두 번째", "첫 번째"],
            "rec_scores": [0.88, 0.97],
        }

        entries = extract_text_results(payload, (100, 80))

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["bbox"], [10, 8, 42, 22])
        self.assertEqual(entries[0]["text"], "첫 번째")
        self.assertEqual(entries[0]["detection_confidence"], 0.91)
        self.assertEqual(entries[0]["recognition_confidence"], 0.97)
        self.assertEqual(entries[1]["text"], "두 번째")

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
