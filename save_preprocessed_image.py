"""Save P22817.jpg at the resolution used by the Qwen3-VL image processor."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageOps

from train_lora import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    IMAGE_COMPRESSION_FACTOR,
)


INPUT_PATH = Path(
    "/home/junhyung/Documents/vscode/vqa/AIVQA/datasets/validation/P22817.jpg"
)
OUTPUT_PATH = INPUT_PATH.with_name(
    f"{INPUT_PATH.stem}_max_pixels_{DEFAULT_MAX_PIXELS}{INPUT_PATH.suffix}"
)


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Match Qwen3-VL's aspect-preserving, factor-aligned resize calculation."""
    factor = IMAGE_COMPRESSION_FACTOR
    if height < factor or width < factor:
        raise ValueError(f"Image dimensions must both be at least {factor} pixels")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must not exceed 200")

    resized_height = max(factor, round_by_factor(height, factor))
    resized_width = max(factor, round_by_factor(width, factor))
    resized_pixels = resized_height * resized_width

    if resized_pixels > DEFAULT_MAX_PIXELS:
        scale = math.sqrt(height * width / DEFAULT_MAX_PIXELS)
        resized_height = max(factor, floor_by_factor(height / scale, factor))
        resized_width = max(factor, floor_by_factor(width / scale, factor))
    elif resized_pixels < DEFAULT_MIN_PIXELS:
        scale = math.sqrt(DEFAULT_MIN_PIXELS / (height * width))
        resized_height = ceil_by_factor(height * scale, factor)
        resized_width = ceil_by_factor(width * scale, factor)

    return resized_height, resized_width


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input image does not exist: {INPUT_PATH}")

    with Image.open(INPUT_PATH) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")
        original_width, original_height = image.size
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
        )
        processed = image.resize(
            (resized_width, resized_height),
            Image.Resampling.BICUBIC,
        )
        processed.save(OUTPUT_PATH, quality=95, subsampling=0)

    print(f"Original: {INPUT_PATH} ({original_width}x{original_height})")
    print(f"Saved:    {OUTPUT_PATH} ({resized_width}x{resized_height})")
    print(f"Pixels:   {resized_width * resized_height:,} / {DEFAULT_MAX_PIXELS:,}")


if __name__ == "__main__":
    main()
