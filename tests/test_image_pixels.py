from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from analyze_image_pixels import (
    analyze_split,
    create_output_dir,
    read_display_dimensions,
    summarize_images,
    write_tables_and_reports,
)


class ImagePixelAnalysisTest(unittest.TestCase):
    def test_exif_rotation_uses_display_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "rotated.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (10, 20)).save(image_path, exif=exif)

            width, height, orientation = read_display_dimensions(image_path)

            self.assertEqual((width, height), (20, 10))
            self.assertEqual(orientation, 6)

    def test_split_analysis_summary_and_file_encodings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "outputs"
            frames = []
            for split, size in zip(
                ("train", "validation", "test"),
                ((10, 10), (20, 10), (40, 20)),
            ):
                (root / split).mkdir()
                Image.new("RGB", size).save(root / split / f"{split}.png")
                json_path = root / f"{split}.json"
                json_path.write_text(
                    json.dumps(
                        [
                            {
                                "metadata": {
                                    "question_id": split,
                                    "split": split,
                                },
                                "model_input": {"image_name": f"{split}.png"},
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                frames.append(analyze_split(split, json_path, root, max_pixels=200))

            details = pd.concat(frames, ignore_index=True)
            summary = summarize_images(details, max_pixels=200)
            output_dir = create_output_dir(
                output_root, datetime(2026, 7, 28, 12, 34, 56)
            )
            paths = write_tables_and_reports(
                details, summary, output_dir, min_pixels=64, max_pixels=200
            )

            overall = summary[summary["split"] == "all"].iloc[0]
            self.assertEqual(overall["image_count"], 3)
            self.assertEqual(overall["pixel_count_min"], 100)
            self.assertEqual(overall["pixel_count_p50"], 200)
            self.assertEqual(overall["pixel_count_max"], 800)
            self.assertEqual(overall["above_max_pixels_count"], 1)
            self.assertAlmostEqual(overall["above_max_pixels_percent"], 100 / 3)
            self.assertEqual(len(paths), 4)
            self.assertTrue((output_dir / "image_pixel_details.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertFalse((output_dir / "image_pixel_summary.json").read_bytes().startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
