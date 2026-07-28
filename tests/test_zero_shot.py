from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_zero_shot import (
    ZERO_SHOT_PREDICTIONS_NAME,
    parse_args,
    validate_args,
)
from train_lora import parse_args as parse_training_args


class ZeroShotUtilitiesTest(unittest.TestCase):
    def test_generation_defaults_match_training_inference(self) -> None:
        with patch("sys.argv", ["generate_zero_shot.py"]):
            zero_shot_args = parse_args()
        with patch("sys.argv", ["train_lora.py"]):
            training_args = parse_training_args()

        self.assertEqual(zero_shot_args.model_id, training_args.model_id)
        self.assertEqual(zero_shot_args.eval_batch_size, training_args.eval_batch_size)
        self.assertEqual(zero_shot_args.max_new_tokens, training_args.max_new_tokens)
        self.assertEqual(zero_shot_args.min_pixels, training_args.min_pixels)
        self.assertEqual(zero_shot_args.max_pixels, training_args.max_pixels)
        self.assertEqual(zero_shot_args.dtype, training_args.dtype)
        self.assertEqual(
            zero_shot_args.attn_implementation,
            training_args.attn_implementation,
        )
        self.assertTrue(ZERO_SHOT_PREDICTIONS_NAME.endswith("_zero_shot.json"))

    def test_invalid_zero_shot_arguments_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_json = Path(temp_dir) / "test.json"
            test_json.write_text("[]", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "generate_zero_shot.py",
                    "--test-json",
                    str(test_json),
                    "--eval-batch-size",
                    "0",
                ],
            ):
                args = parse_args()
            with self.assertRaisesRegex(ValueError, "eval-batch-size"):
                validate_args(args)


if __name__ == "__main__":
    unittest.main()
