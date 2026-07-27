from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from train_lora import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    configure_image_pixel_limits,
    find_adapter_target_modules,
    save_test_predictions,
)


class _FakeImageProcessor:
    size = {"shortest_edge": 1, "longest_edge": 2}


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = _FakeImageProcessor()


class TrainingUtilitiesTest(unittest.TestCase):
    def test_default_pixel_budget_is_applied_to_processor(self) -> None:
        processor = _FakeProcessor()
        configure_image_pixel_limits(
            processor, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
        )
        self.assertEqual(
            processor.image_processor.size,
            {
                "shortest_edge": 64 * 32 * 32,
                "longest_edge": 512 * 32 * 32,
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
