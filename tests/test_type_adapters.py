from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from type_adapters.data import build_type_subsets, restore_original_order
from type_adapters.modeling import (
    load_switchable_type_adapters,
    load_trainable_shared_adapter,
    validate_type_adapter_set,
)
from type_adapters.train import parse_args, selected_question_forms, selection_score


class _FakeDataset:
    def __init__(self) -> None:
        forms = ["SA", "MC", "LA", "MC", "SA"]
        self.records = [
            {
                "metadata": {
                    "question_id": str(index),
                    "question_form": question_form,
                }
            }
            for index, question_form in enumerate(forms)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "question_id": record["metadata"]["question_id"],
            "question_form": record["metadata"]["question_form"],
        }


def _write_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "base_model_name_or_path": "shared-base",
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


class TypeAdapterDataTest(unittest.TestCase):
    def test_subsets_filter_lazily_and_keep_source_indices(self) -> None:
        subsets = build_type_subsets(_FakeDataset())

        self.assertEqual(subsets["MC"].indices, [1, 3])
        self.assertEqual(subsets["SA"].indices, [0, 4])
        self.assertEqual(subsets["LA"].indices, [2])
        self.assertEqual(subsets["MC"][1]["source_index"], 3)

    def test_grouped_predictions_are_restored_to_source_order(self) -> None:
        subsets = build_type_subsets(_FakeDataset())
        predictions = restore_original_order(
            subsets,
            {
                "MC": ["mc-1", "mc-3"],
                "SA": ["sa-0", "sa-4"],
                "LA": ["la-2"],
            },
            source_length=5,
        )

        self.assertEqual(predictions, ["sa-0", "mc-1", "la-2", "mc-3", "sa-4"])

    def test_each_form_uses_its_own_selection_metric(self) -> None:
        metrics = {
            "mc_accuracy": 0.7,
            "sa_exact_match": 0.4,
            "descriptive_avg": 0.2,
        }
        self.assertEqual(selection_score("MC", metrics), ("mc_accuracy", 0.7))
        self.assertEqual(selection_score("SA", metrics), ("sa_exact_match", 0.4))
        self.assertEqual(selection_score("LA", metrics), ("descriptive_avg", 0.2))

    def test_training_can_select_all_or_one_question_form(self) -> None:
        self.assertEqual(selected_question_forms("ALL"), ("MC", "SA", "LA"))
        self.assertEqual(selected_question_forms("MC"), ("MC",))
        self.assertEqual(selected_question_forms("SA"), ("SA",))
        self.assertEqual(selected_question_forms("LA"), ("LA",))

        with patch(
            "sys.argv",
            ["train.py", "--shared-adapter-dir", "shared", "--question-form", "MC"],
        ):
            args = parse_args()
        self.assertEqual(args.question_form, "MC")


class TypeAdapterPeftTest(unittest.TestCase):
    def test_shared_adapter_is_loaded_trainable_without_new_lora(self) -> None:
        calls = []

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, llm, path, **kwargs):
                calls.append((llm, path, kwargs))
                return "continued-shared"

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "shared"
            _write_adapter(adapter_dir)
            with patch.dict(sys.modules, {"peft": types.SimpleNamespace(PeftModel=FakePeftModel)}):
                loaded = load_trainable_shared_adapter("base-llm", adapter_dir)

        self.assertEqual(loaded, "continued-shared")
        self.assertEqual(calls[0][0], "base-llm")
        self.assertEqual(calls[0][2], {"is_trainable": True})

    def test_three_adapters_are_registered_for_switching(self) -> None:
        calls = []

        class LoadedPeft:
            def load_adapter(self, path, **kwargs):
                calls.append(("load", Path(path).name, kwargs))

            def set_adapter(self, name):
                calls.append(("set", name))

        loaded_peft = LoadedPeft()

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, llm, path, **kwargs):
                calls.append(("from", llm, Path(path).name, kwargs))
                return loaded_peft

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dirs = {}
            for form in ("MC", "SA", "LA"):
                adapter_dirs[form] = root / f"{form.lower()}_adapter"
                _write_adapter(adapter_dirs[form])
            with patch.dict(sys.modules, {"peft": types.SimpleNamespace(PeftModel=FakePeftModel)}):
                result = load_switchable_type_adapters("base-llm", adapter_dirs)

        self.assertIs(result, loaded_peft)
        self.assertEqual(calls[0][3]["adapter_name"], "MC")
        self.assertFalse(calls[0][3]["is_trainable"])
        self.assertEqual([call[2]["adapter_name"] for call in calls[1:3]], ["SA", "LA"])
        self.assertEqual(calls[-1], ("set", "MC"))

    def test_type_adapter_directory_requires_matching_architectures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for form in ("MC", "SA", "LA"):
                _write_adapter(root / f"{form.lower()}_adapter")
            adapter_dirs = validate_type_adapter_set(root)
            self.assertEqual(set(adapter_dirs), {"MC", "SA", "LA"})


if __name__ == "__main__":
    unittest.main()
