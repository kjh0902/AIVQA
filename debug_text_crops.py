"""Save OCR crops for one dataset question without running model inference."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageOps

from aivqa.data import QwenVQADataset
from aivqa.text_crops import build_text_detector, detect_text_crops
from train_lora import DATASET_DIR, DATASET_NAME, DEFAULT_MAX_PIXELS


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find one AIVQA question and save its OCR text crops."
    )
    parser.add_argument(
        "--question-id",
        required=True,
        help="metadata.question_id of the sample to inspect (for example: 0149)",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        help="Optional split filter; by default all three splits are searched",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/text_crop_debug")
    )
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    return parser.parse_args()


def find_question(
    question_id: str,
    dataset_dir: Path,
    dataset_root: Path,
    splits: Sequence[str] = SPLITS,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return the unique matching split, raw record, and runtime feature."""
    normalized_id = str(question_id).strip()
    matches: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    for split in splits:
        json_path = dataset_dir / f"{DATASET_NAME}_{split}.json"
        dataset = QwenVQADataset(json_path, dataset_root=dataset_root)
        for index, record in enumerate(dataset.records):
            metadata = record.get("metadata")
            record_question_id = (
                str(metadata.get("question_id"))
                if isinstance(metadata, dict)
                else None
            )
            if record_question_id == normalized_id:
                matches.append((split, record, dataset[index]))

    if not matches:
        searched = ", ".join(splits)
        raise ValueError(f"Question ID {normalized_id!r} was not found in: {searched}")
    if len(matches) > 1:
        found_splits = ", ".join(match[0] for match in matches)
        raise ValueError(
            f"Question ID {normalized_id!r} exists more than once ({found_splits}); "
            "specify --split"
        )
    return matches[0]


def create_output_dir(root: Path, split: str, question_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_question_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", question_id).strip("._")
    base = root / f"{split}_{safe_question_id or 'question'}_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def save_debug_bundle(
    output_dir: Path,
    split: str,
    record: dict[str, Any],
    feature: dict[str, Any],
    crops: Sequence[Image.Image],
) -> None:
    image_path = Path(feature["image_path"])
    with Image.open(image_path) as image_file:
        original = ImageOps.exif_transpose(image_file).convert("RGB")
        original.save(output_dir / "original.png", format="PNG")

    for index, crop in enumerate(crops, start=1):
        crop.save(output_dir / f"crop_{index}.png", format="PNG")

    model_input = record.get("model_input", {})
    metadata = {
        "question_id": feature["question_id"],
        "split": split,
        "question": model_input.get("question"),
        "options": model_input.get("options", []),
        "source_image": str(image_path),
        "crop_count": len(crops),
        "crop_files": [f"crop_{index}.png" for index in range(1, len(crops) + 1)],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    if args.max_pixels < 1:
        raise ValueError("--max-pixels must be positive")

    splits = (args.split,) if args.split else SPLITS
    split, record, feature = find_question(
        args.question_id,
        args.dataset_dir,
        args.dataset_root,
        splits,
    )

    model_input = record["model_input"]
    print(f"Question ID: {feature['question_id']} ({split})")
    print(f"Question: {model_input['question']}")
    print(f"Image: {feature['image_path']}")
    print("Loading PP-OCRv5_server_det on CPU with MKLDNN disabled")

    detector = build_text_detector()
    crops = detect_text_crops(
        detector,
        feature["image_path"],
        max_pixels=args.max_pixels,
    )
    output_dir = create_output_dir(args.output_dir, split, feature["question_id"])
    save_debug_bundle(output_dir, split, record, feature, crops)

    print(f"Detected text crops: {len(crops)}")
    print(f"Saved debug images: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
