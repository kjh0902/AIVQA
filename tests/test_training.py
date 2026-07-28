from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from train_lora import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    configure_image_pixel_limits,
    create_run_output_dir,
    find_adapter_target_modules,
    parse_args,
    save_test_predictions,
)


class _FakeImageProcessor:
    size = {"shortest_edge": 1, "longest_edge": 2}


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = _FakeImageProcessor()


class TrainingUtilitiesTest(unittest.TestCase):
    def test_shorter_training_defaults(self) -> None:
        with patch("sys.argv", ["train_lora.py"]):
            args = parse_args()

        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.num_workers, 2)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.warmup_ratio, 0.10)
        self.assertEqual(args.early_stopping_patience, 2)
        self.assertEqual(args.gradient_accumulation_steps, 8)

    def test_each_run_gets_a_unique_timestamped_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs" / "qwen3_vl_lora"
            started_at = datetime(2026, 7, 27, 12, 34, 56)

            first = create_run_output_dir(output_root, started_at)
            second = create_run_output_dir(output_root, started_at)

            self.assertEqual(first, output_root / "run_20260727_123456")
            self.assertEqual(second, output_root / "run_20260727_123456_01")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_default_pixel_budget_is_applied_to_processor(self) -> None:
        processor = _FakeProcessor()
        configure_image_pixel_limits(
            processor, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
        )
        self.assertEqual(
            processor.image_processor.size,
            {
                "shortest_edge": 64 * 32 * 32,
                "longest_edge": 1024 * 32 * 32,
            },
        )

    def test_invalid_pixel_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            configure_image_pixel_limits(_FakeProcessor(), 200, 100)

    def test_exactly_144_language_attention_targets_are_selected(self) -> None:
        names = [
            f"model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in range(36)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        names.extend(
            [
                "model.visual.blocks.0.attn.q_proj",
                "model.visual.merger.linear_fc1",
                "lm_head",
                "model.language_model.layers.0.mlp.up_proj",
            ]
        )
        targets = find_adapter_target_modules(names)
        self.assertEqual(len(targets), 144)
        self.assertFalse(any("visual" in target for target in targets))
        self.assertFalse(any("lm_head" in target for target in targets))

    def test_missing_decoder_projection_raises(self) -> None:
        names = [
            f"model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in range(36)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        with self.assertRaises(RuntimeError):
            find_adapter_target_modules(names[:-1])

    def test_test_predictions_preserve_source_and_existing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "test.json"
            output = root / "predictions.json"
            original = [
                {"metadata": {"question_id": "1"}, "model_output": None},
                {
                    "metadata": {"question_id": "2"},
                    "model_output": {"confidence": 0.5},
                },
            ]
            source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

            save_test_predictions(source, ["첫째", "둘째"], output)

            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), original)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([item["metadata"]["question_id"] for item in result], ["1", "2"])
            self.assertEqual(result[0]["model_output"], {"answer": "첫째"})
            self.assertEqual(
                result[1]["model_output"], {"confidence": 0.5, "answer": "둘째"}
            )

    def test_source_json_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "test.json"
            source.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                save_test_predictions(source, [], source)


if __name__ == "__main__":
    unittest.main()
