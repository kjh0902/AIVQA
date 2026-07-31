"""Run ZwZ-8B inference on one local image."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_ID = "inclusionAI/ZwZ-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask inclusionAI/ZwZ-8B a question about one local image."
    )
    parser.add_argument("image", type=Path, help="입력 이미지 경로")
    parser.add_argument(
        "--question",
        default="Describe this image.",
        help="이미지에 대해 질문할 내용",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="생성할 최대 토큰 수 (기본값: 128)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"입력 이미지를 찾을 수 없습니다: {image_path}")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens는 1 이상이어야 합니다.")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
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

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
    )
    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
