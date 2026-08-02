from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aivqa.text_crops import (
    MULTI_IMAGE_PROMPT_PREFIX,
    DetectedTextBox,
    TextCropDataset,
    add_text_crops_to_feature,
    build_text_detector,
    detect_text_crops,
    expand_bbox_if_within_limit,
    select_text_group_bboxes,
)


class _DetectionResult:
    def __init__(self, polygons, scores) -> None:
        self.json = {"res": {"dt_polys": polygons, "dt_scores": scores}}


class _FakeDetector:
    def __init__(self, polygons, scores) -> None:
        self.polygons = polygons
        self.scores = scores
        self.calls = []

    def predict(self, image_path, batch_size=1):
        self.calls.append((image_path, batch_size))
        return [_DetectionResult(self.polygons, self.scores)]


def _polygon(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class TextCropTest(unittest.TestCase):
    def test_detector_uses_requested_cpu_configuration(self) -> None:
        calls = []

        class FakeTextDetection:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

        fake_module = types.SimpleNamespace(TextDetection=FakeTextDetection)
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            detector = build_text_detector()

        self.assertIsInstance(detector, FakeTextDetection)
        self.assertEqual(
            calls,
            [
                {
                    "model_name": "PP-OCRv5_server_det",
                    "device": "cpu",
                    "enable_mkldnn": False,
                }
            ],
        )
        self.assertEqual(os.environ["FLAGS_use_mkldnn"], "0")

    def test_nearby_boxes_are_merged_and_only_three_groups_selected(self) -> None:
        boxes = [
            DetectedTextBox((10, 10, 40, 20), 0.9),
            DetectedTextBox((45, 10, 80, 20), 0.8),
            DetectedTextBox((10, 60, 30, 75), 0.7),
            DetectedTextBox((100, 60, 125, 75), 0.7),
            DetectedTextBox((170, 60, 195, 75), 0.7),
        ]

        groups = select_text_group_bboxes(boxes, (220, 100))

        self.assertEqual(len(groups), 3)
        self.assertIn((10, 10, 80, 20), groups)

    def test_bbox_expands_only_within_pixel_limit(self) -> None:
        bbox = (20, 20, 120, 70)

        self.assertEqual(
            expand_bbox_if_within_limit(bbox, (200, 100), max_pixels=10_000),
            (12, 16, 128, 74),
        )
        self.assertEqual(
            expand_bbox_if_within_limit(bbox, (200, 100), max_pixels=5_000),
            bbox,
        )

    def test_detection_crops_without_resizing(self) -> None:
        detector = _FakeDetector(
            [_polygon(10, 10, 40, 20), _polygon(45, 10, 80, 20)],
            [0.9, 0.8],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

            crops = detect_text_crops(detector, image_path, max_pixels=10_000)

        self.assertEqual(detector.calls, [(str(image_path), 1)])
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].size, (82, 18))

    def test_no_detection_keeps_original_feature(self) -> None:
        feature = {"messages": []}
        self.assertIs(add_text_crops_to_feature(feature, []), feature)

    def test_crops_follow_full_image_and_prompt_is_prefixed(self) -> None:
        full_image = Image.new("RGB", (20, 20))
        crop = Image.new("RGB", (10, 10))
        feature = {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": full_image},
                        {"type": "text", "text": "question"},
                    ],
                },
            ]
        }

        augmented = add_text_crops_to_feature(feature, [crop])
        content = augmented["messages"][1]["content"]

        self.assertEqual([item["type"] for item in content], ["image", "image", "text"])
        self.assertIs(content[0]["image"], full_image)
        self.assertIs(content[1]["image"], crop)
        self.assertEqual(
            content[2]["text"], f"{MULTI_IMAGE_PROMPT_PREFIX}\n\nquestion"
        )
        self.assertEqual(feature["messages"][1]["content"][1]["text"], "question")

    def test_dataset_wrapper_falls_back_to_full_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (20, 20)).save(image_path)
            feature = {
                "image_path": str(image_path),
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.new("RGB", (20, 20))},
                            {"type": "text", "text": "question"},
                        ],
                    },
                ],
            }
            wrapped = TextCropDataset(
                [feature], _FakeDetector([], []), max_pixels=1024
            )

            result = wrapped[0]

        self.assertIs(result, feature)

if __name__ == "__main__":
    unittest.main()
