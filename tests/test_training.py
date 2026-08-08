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
    EXPECTED_ADAPTER_TARGET_COUNT,
    EXPECTED_DECODER_LAYERS,
    EXPECTED_VISION_LAYERS,
    IMAGE_COMPRESSION_FACTOR,
    MODEL_ID,
    _enable_llm_gradient_checkpointing,
    _verify_multimodal_adapters_are_trainable,
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

    def test_gradient_checkpointing_is_enabled_on_llm_after_peft_wrap(self) -> None:
        class FakeLanguageModel:
            _require_grads_hook = object()

            def gradient_checkpointing_enable(self, **kwargs):
                self.checkpointing_kwargs = kwargs

            def disable_input_require_grads(self):
                self.input_require_grads_disabled = True

        class FakePeftModel:
            language_model = FakeLanguageModel()

        model = FakePeftModel()
        _enable_llm_gradient_checkpointing(model, enabled=True)

        self.assertEqual(
            model.language_model.checkpointing_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        self.assertTrue(model.language_model.input_require_grads_disabled)

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

    def test_exactly_194_multimodal_targets_are_selected(self) -> None:
        names = [
            f"language_model.model.layers.{layer}.self_attn.{projection}"
            for layer in range(EXPECTED_DECODER_LAYERS)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        names.extend(
            f"vision_model.blocks.{layer}.attn.{projection}"
            for layer in range(EXPECTED_VISION_LAYERS)
            for projection in ("qkv", "proj")
        )
        names.extend(
            [
                "abstractor.readout.0",
                "abstractor.readout.2",
                "lm_head",
                "model.layers.0.mlp.up_proj",
            ]
        )
        targets = find_adapter_target_modules(names)
        self.assertEqual(len(targets), EXPECTED_ADAPTER_TARGET_COUNT)
        self.assertEqual(sum("vision_model" in target for target in targets), 64)
        self.assertEqual(sum("abstractor" in target for target in targets), 2)
        self.assertFalse(any("lm_head" in target for target in targets))

    def test_missing_multimodal_target_raises(self) -> None:
        names = [
            f"language_model.model.layers.{layer}.self_attn.{projection}"
            for layer in range(EXPECTED_DECODER_LAYERS)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        names.extend(
            f"vision_model.blocks.{layer}.attn.{projection}"
            for layer in range(EXPECTED_VISION_LAYERS)
            for projection in ("qkv", "proj")
        )
        names.extend(("abstractor.readout.0", "abstractor.readout.2"))
        with self.assertRaises(RuntimeError):
            find_adapter_target_modules(names[:-1])

    def test_trainable_scope_verifier_accepts_all_194_lora_targets(self) -> None:
        class Parameter:
            def __init__(self, requires_grad):
                self.requires_grad = requires_grad

        class FakeModel:
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
                for layer in range(EXPECTED_VISION_LAYERS):
                    for projection in ("qkv", "proj"):
                        stem = (
                            "base_model.model.vision_model.blocks."
                            f"{layer}.attn.{projection}"
                        )
                        yield f"{stem}.lora_A.default.weight", Parameter(True)
                        yield f"{stem}.lora_B.default.weight", Parameter(True)
                for index in (0, 2):
                    stem = f"base_model.model.abstractor.readout.{index}"
                    yield f"{stem}.lora_A.default.weight", Parameter(True)
                    yield f"{stem}.lora_B.default.weight", Parameter(True)

        _verify_multimodal_adapters_are_trainable(FakeModel())

    def test_best_checkpoint_saves_one_multimodal_adapter(self) -> None:
        class FakeModel:
            def save_pretrained(
                self, path, safe_serialization, save_embedding_layers
            ):
                self.path = Path(path)
                self.safe_serialization = safe_serialization
                self.save_embedding_layers = save_embedding_layers
                (self.path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (self.path / "adapter_model.safetensors").write_bytes(b"adapter")

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

            self.assertTrue(model.safe_serialization)
            self.assertFalse(model.save_embedding_layers)
            self.assertTrue((adapter_dir / "adapter_config.json").is_file())
            self.assertTrue((adapter_dir / "adapter_model.safetensors").is_file())
            metadata = json.loads(
                (adapter_dir / "training_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["adapter_scope"],
                "llm_and_vision_attention_with_abstractor_readout",
            )
            self.assertEqual(metadata["target_module_count"], 194)
            self.assertTrue(metadata["frozen_base_weights"])

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
