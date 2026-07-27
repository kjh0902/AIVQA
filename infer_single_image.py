"""Run single-image visual question answering with Qwen3-VL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Qwen3-VL-8B-Instruct a question about one local image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to an image file")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of answer tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model weight dtype (default: auto)",
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
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")


def run_inference(args: argparse.Namespace) -> str:
    # Imports are intentionally delayed so `--help` works before dependencies are installed.
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[1/3] Loading processor: {args.model_id}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model_id)

    model_dtype: str | torch.dtype = (
        "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    )
    model_kwargs: dict[str, object] = {
        "device_map": "auto",
        "dtype": model_dtype,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"[2/3] Loading model: {args.model_id}", file=sys.stderr)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    model.eval()

    with Image.open(args.image) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    print("[3/3] Generating answer", file=sys.stderr)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    answer_ids = generated_ids[:, prompt_length:]
    answers = processor.batch_decode(
        answer_ids,
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
