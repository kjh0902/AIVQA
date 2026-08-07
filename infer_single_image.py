"""Run single-image visual question answering with Kanana-V."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train_lora import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    MODEL_ID,
    MODEL_MAX_PIXELS,
    configure_image_pixel_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Kanana-1.5-V-3B-Instruct a question about one local image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to an image file")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Hugging Face model ID (default: {MODEL_ID})",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Optional attention backend; the Transformers default is used when omitted",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise FileNotFoundError(f"Image file does not exist: {args.image}")
    if args.max_length < 1 or args.max_new_tokens < 1:
        raise ValueError("--max-length and --max-new-tokens must be at least 1")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    if args.max_pixels > MODEL_MAX_PIXELS:
        raise ValueError(
            f"--max-pixels cannot exceed the model processor limit ({MODEL_MAX_PIXELS})"
        )


def run_inference(args: argparse.Namespace) -> str:
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"[1/3] Loading processor: {args.model_id}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    configure_image_pixel_limits(processor, args.min_pixels, args.max_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_dtype: str | torch.dtype = (
        "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    )
    model_kwargs: dict[str, object] = {
        "device_map": "auto",
        "dtype": model_dtype,
        "trust_remote_code": True,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"[2/3] Loading model: {args.model_id}", file=sys.stderr)
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    with Image.open(args.image) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB").copy()

    sample = {
        "image": [image],
        "conv": [
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": args.question},
        ],
    }
    inputs = processor.batch_encode_collate(
        [sample],
        padding="longest",
        padding_side="left",
        max_length=args.max_length,
        add_generation_prompt=True,
    )
    inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    print("[3/3] Generating answer", file=sys.stderr)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    answers = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return answers[0].strip()


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        answer = run_inference(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
