from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from debug_text_crops import save_debug_bundle


class DebugTextCropsTest(unittest.TestCase):
    def test_saves_original_crops_and_question_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "source.jpg"
            output_dir = root / "output"
            output_dir.mkdir()
            Image.new("RGB", (100, 80), color="white").save(image_path)
            crop = Image.new("RGB", (30, 20), color="black")
            record = {
                "model_input": {
                    "question": "사진 속 글자는 무엇인가?",
                    "options": [],
                }
            }
            feature = {"question_id": "0149", "image_path": str(image_path)}

            save_debug_bundle(output_dir, "test", record, feature, [crop])

            self.assertTrue((output_dir / "original.png").is_file())
            self.assertTrue((output_dir / "crop_1.png").is_file())
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["question_id"], "0149")
            self.assertEqual(metadata["question"], "사진 속 글자는 무엇인가?")
            self.assertEqual(metadata["crop_files"], ["crop_1.png"])


if __name__ == "__main__":
    unittest.main()
