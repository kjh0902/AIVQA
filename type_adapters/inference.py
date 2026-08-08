"""Batch test inference with MC/SA/LA adapter switching and order restoration."""

from __future__ import annotations

import argparse
from pathlib import Path

from aivqa.data import KananaVQADataset
from train_lora import (
    DATASET_DIR,
    DATASET_NAME,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    MODEL_ID,
    MODEL_MAX_PIXELS,
    create_run_output_dir,
    generate_predictions,
    save_test_predictions,
    set_seed,
)

from .data import QUESTION_FORMS, build_type_subsets, restore_original_order
from .modeling import (
    attach_type_adapters_for_inference,
    load_base_model_and_processor,
    release_cuda_memory,
    validate_type_adapter_set,
)


TYPE_PREDICTIONS_NAME = f"{DATASET_NAME}_test_predictions_type_adapters.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ordered AIVQA test predictions with MC/SA/LA adapters."
    )
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        required=True,
        help="Directory containing mc_adapter, sa_adapter, and la_adapter",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--test-json", type=Path, default=DATASET_DIR / f"{DATASET_NAME}_test.json"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kanana_1_5_v_3b_type_inference"),
    )
    parser.add_argument("--test-predictions-path", type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> dict[str, Path]:
    adapter_dirs = validate_type_adapter_set(args.adapters_dir)
    if not args.test_json.is_file():
        raise FileNotFoundError(f"Test JSON does not exist: {args.test_json}")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1")
    if args.max_length < 1 or args.max_new_tokens < 1:
        raise ValueError("--max-length and --max-new-tokens must be at least 1")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    if args.max_pixels > MODEL_MAX_PIXELS:
        raise ValueError(
            f"--max-pixels cannot exceed the model processor limit ({MODEL_MAX_PIXELS})"
        )
    return adapter_dirs


def main() -> int:
    args = parse_args()
    adapter_dirs = validate_args(args)
    set_seed(args.seed)

    run_output_dir = create_run_output_dir(args.output_dir)
    predictions_path = (
        args.test_predictions_path or run_output_dir / TYPE_PREDICTIONS_NAME
    )
    print(f"Run output directory: {run_output_dir}")

    dataset = KananaVQADataset(args.test_json, dataset_root=args.dataset_root)
    subsets = build_type_subsets(dataset)
    model = processor = None
    try:
        model, processor, dtype = load_base_model_and_processor(
            args,
            for_training=False,
        )
        model = attach_type_adapters_for_inference(model, adapter_dirs)

        grouped_predictions: dict[str, list[str]] = {}
        for question_form in QUESTION_FORMS:
            print(
                f"Generating {len(subsets[question_form])} {question_form} samples "
                f"with {adapter_dirs[question_form]}"
            )
            model.language_model.set_adapter(question_form)
            model.eval()
            grouped_predictions[question_form] = generate_predictions(
                model,
                processor,
                subsets[question_form],
                args.eval_batch_size,
                args.max_length,
                args.max_new_tokens,
                dtype,
            )

        predictions = restore_original_order(
            subsets,
            grouped_predictions,
            len(dataset),
        )
        save_test_predictions(args.test_json, predictions, predictions_path)
        print(f"Saved ordered type-adapter predictions: {predictions_path}")
        return 0
    finally:
        del processor, model
        release_cuda_memory()


if __name__ == "__main__":
    raise SystemExit(main())
