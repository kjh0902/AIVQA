from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from train_lora import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    EXPECTED_DECODER_LAYERS,
    IMAGE_COMPRESSION_FACTOR,
    MODEL_ID,
    _verify_only_llm_adapters_are_trainable,
    configure_image_pixel_limits,
    create_run_output_dir,
    find_adapter_target_modules,
    parse_args,
    save_best_adapter,
    save_test_predictions,
)


class _FakeImageProcessor:
    size = {"min_pixels": 1, "max_pixels": 2}


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = _FakeImageProcessor()


class TrainingUtilitiesTest(unittest.TestCase):
    def test_kanana_training_defaults_are_16gb_friendly(self) -> None:
        with patch("sys.argv", ["train_lora.py"]):
            args = parse_args()

        self.assertEqual(args.model_id, MODEL_ID)
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.train_batch_size, 1)
        self.assertEqual(args.eval_batch_size, 1)
        self.assertEqual(args.max_length, 2048)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertTrue(args.gradient_checkpointing)
        self.assertFalse(args.load_in_4bit)

    def test_each_run_gets_a_unique_timestamped_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs" / "kanana_1_5_v_3b_lora"
            started_at = datetime(2026, 7, 27, 12, 34, 56)

            first = create_run_output_dir(output_root, started_at)
            second = create_run_output_dir(output_root, started_at)

            self.assertEqual(first, output_root / "run_20260727_123456")
            self.assertEqual(second, output_root / "run_20260727_123456_01")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_default_pixel_budget_is_applied_to_native_attributes(self) -> None:
        processor = _FakeProcessor()
        configure_image_pixel_limits(
            processor, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
        )
        self.assertEqual(processor.image_processor.min_pixels, 100 * 28 * 28)
        self.assertEqual(processor.image_processor.max_pixels, 400 * 28 * 28)
        self.assertEqual(
            processor.image_processor.size,
            {
                "min_pixels": 100 * IMAGE_COMPRESSION_FACTOR**2,
                "max_pixels": 400 * IMAGE_COMPRESSION_FACTOR**2,
            },
        )

    def test_invalid_pixel_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            configure_image_pixel_limits(_FakeProcessor(), 200, 100)

    def test_exactly_128_language_attention_targets_are_selected(self) -> None:
        names = [
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(EXPECTED_DECODER_LAYERS)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        names.extend(
            [
                "vision_model.blocks.0.attn.q_proj",
                "abstractor.readout.0",
                "lm_head",
                "model.layers.0.mlp.up_proj",
            ]
        )
        targets = find_adapter_target_modules(names)
        self.assertEqual(len(targets), 128)
        self.assertFalse(any("vision_model" in target for target in targets))
        self.assertFalse(any("abstractor" in target for target in targets))
        self.assertFalse(any("lm_head" in target for target in targets))

    def test_missing_decoder_projection_raises(self) -> None:
        names = [
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(EXPECTED_DECODER_LAYERS)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        with self.assertRaises(RuntimeError):
            find_adapter_target_modules(names[:-1])

    def test_trainable_scope_verifier_accepts_only_llm_lora_parameters(self) -> None:
        class Parameter:
            def __init__(self, requires_grad):
                self.requires_grad = requires_grad

        class FrozenModule:
            @staticmethod
            def parameters():
                return [Parameter(False)]

        class FakeModel:
            vision_model = FrozenModule()
            abstractor = FrozenModule()

            @staticmethod
            def named_parameters():
                for layer in range(EXPECTED_DECODER_LAYERS):
                    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        stem = (
                            "language_model.base_model.model.model.layers."
                            f"{layer}.self_attn.{projection}"
                        )
                        yield f"{stem}.lora_A.default.weight", Parameter(True)
                        yield f"{stem}.lora_B.default.weight", Parameter(True)

        _verify_only_llm_adapters_are_trainable(FakeModel())

    def test_best_checkpoint_saves_only_the_language_adapter(self) -> None:
        class FakeLanguageModel:
            def save_pretrained(
                self, path, safe_serialization, save_embedding_layers
            ):
                self.path = Path(path)
                self.safe_serialization = safe_serialization
                self.save_embedding_layers = save_embedding_layers
                (self.path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (self.path / "adapter_model.safetensors").write_bytes(b"adapter")

        class FakeModel:
            def __init__(self):
                self.language_model = FakeLanguageModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "best_adapter"
            model = FakeModel()
            save_best_adapter(
                model,
                adapter_dir,
                epoch=2,
                best_score=0.75,
                metrics={"final_score": 0.75},
                args=argparse.Namespace(output_dir=Path(temp_dir)),
            )

            self.assertTrue(model.language_model.safe_serialization)
            self.assertFalse(model.language_model.save_embedding_layers)
            self.assertTrue((adapter_dir / "adapter_config.json").is_file())
            self.assertTrue((adapter_dir / "adapter_model.safetensors").is_file())
            metadata = json.loads(
                (adapter_dir / "training_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["adapter_scope"], "language_model_only")
            self.assertEqual(metadata["frozen_modules"], ["vision_model", "abstractor"])

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
