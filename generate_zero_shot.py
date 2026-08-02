"""Generate AIVQA test predictions with the pretrained Qwen3-VL model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aivqa.data import QwenVQADataset
from aivqa.text_crops import TextCropDataset, build_text_detector
from train_lora import (
    DATASET_DIR,
    DATASET_NAME,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    IMAGE_COMPRESSION_FACTOR,
    MODEL_ID,
    configure_image_pixel_limits,
    create_run_output_dir,
    generate_predictions,
    save_test_predictions,
    set_seed,
)


ZERO_SHOT_PREDICTIONS_NAME = f"{DATASET_NAME}_test_predictions_zero_shot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AIVQA test answers with pretrained Qwen3-VL."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--test-json", type=Path, default=DATASET_DIR / f"{DATASET_NAME}_test.json"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen3_vl_zero_shot"),
        help="Root directory where a unique run_YYYYMMDD_HHMMSS folder is created",
    )
    parser.add_argument(
        "--test-predictions-path",
        type=Path,
        help=f"Default: RUN_OUTPUT_DIR/{ZERO_SHOT_PREDICTIONS_NAME}",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Quantize pretrained weights to 4-bit for limited VRAM",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.test_json.is_file():
        raise FileNotFoundError(f"Test JSON does not exist: {args.test_json}")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    if args.max_pixels > DEFAULT_MAX_PIXELS:
        raise ValueError(
            f"--max-pixels cannot exceed DEFAULT_MAX_PIXELS ({DEFAULT_MAX_PIXELS})"
        )


def build_pretrained_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen3-VL zero-shot generation")

    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_id)
    configure_image_pixel_limits(processor, args.min_pixels, args.max_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    print(
        f"Image pixel budget: min={args.min_pixels:,}, max={args.max_pixels:,} "
        f"(~{args.min_pixels // IMAGE_COMPRESSION_FACTOR**2}-"
        f"{args.max_pixels // IMAGE_COMPRESSION_FACTOR**2} visual tokens)"
    )
    print(f"Loading pretrained processor and model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, **model_kwargs
    )
    model.requires_grad_(False)
    model.eval()
    return model, processor, dtype


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    run_output_dir = create_run_output_dir(args.output_dir)
    predictions_path = (
        args.test_predictions_path
        or run_output_dir / ZERO_SHOT_PREDICTIONS_NAME
    )
    print(f"Run output directory: {run_output_dir}")

    base_dataset = QwenVQADataset(args.test_json, dataset_root=args.dataset_root)
    print("Loading PP-OCRv5_server_det on CPU with MKLDNN disabled")
    text_detector = build_text_detector()
    debug_crop_dir = run_output_dir / "debug_text_crops"
    print(f"Saving detected text crops for debugging: {debug_crop_dir}")
    dataset = TextCropDataset(
        base_dataset,
        text_detector,
        args.max_pixels,
        debug_crop_dir=debug_crop_dir,
    )
    model, processor, dtype = build_pretrained_model_and_processor(args)
    predictions = generate_predictions(
        model,
        processor,
        dataset,
        args.eval_batch_size,
        args.max_new_tokens,
        dtype,
    )
    save_test_predictions(args.test_json, predictions, predictions_path)
    print(f"Saved zero-shot test predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
