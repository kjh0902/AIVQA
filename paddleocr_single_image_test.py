"""Run PaddleOCR detection and Korean recognition on one original image."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test PaddleOCR detection and Korean recognition on one image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paddleocr_single_image"),
        help="Root directory for a timestamped result directory",
    )
    parser.add_argument("--device", default="cpu", help="For example: cpu or gpu:0")
    parser.add_argument(
        "--font-path",
        type=Path,
        help="Optional Korean TrueType/OpenType font for visualization labels",
    )
    return parser.parse_args()


def build_ocr_pipeline(device: str) -> Any:
    """Build only the requested detection and Korean recognition modules."""
    os.environ["FLAGS_use_mkldnn"] = "0"
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=0.0,
        device=device,
        enable_mkldnn=False,
    )


def _result_payload(result: Any) -> Mapping[str, Any]:
    payload: Any = (
        result if isinstance(result, Mapping) else getattr(result, "json", None)
    )
    if callable(payload):
        payload = payload()
    if not isinstance(payload, Mapping):
        raise RuntimeError("PaddleOCR returned an unsupported OCR result")
    nested = payload.get("res")
    return nested if isinstance(nested, Mapping) else payload


def _polygon(polygon: Any) -> list[list[int | float]]:
    try:
        points = [[float(point[0]), float(point[1])] for point in polygon]
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError("PaddleOCR returned an invalid text polygon") from exc
    if len(points) < 3:
        raise RuntimeError("PaddleOCR text polygons must contain at least three points")
    return [
        [int(x) if x.is_integer() else x, int(y) if y.is_integer() else y]
        for x, y in points
    ]


def _polygon_key(
    polygon: Sequence[Sequence[int | float]],
) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in polygon)


def _axis_aligned_bbox(
    polygon: Sequence[Sequence[int | float]], image_size: tuple[int, int]
) -> list[int]:
    width, height = image_size
    x1 = max(0, min(width, math.floor(min(point[0] for point in polygon))))
    y1 = max(0, min(height, math.floor(min(point[1] for point in polygon))))
    x2 = max(0, min(width, math.ceil(max(point[0] for point in polygon))))
    y2 = max(0, min(height, math.ceil(max(point[1] for point in polygon))))
    return [x1, y1, x2, y2]


def extract_text_results(
    payload: Mapping[str, Any], image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    """Keep detector polygons separate and attach matching recognition output."""
    detected_polygons = list(payload.get("dt_polys", []))
    detection_scores = list(payload.get("dt_scores", []))
    if len(detected_polygons) != len(detection_scores):
        raise RuntimeError(
            "PaddleOCR returned mismatched detection polygons and scores"
        )

    entries: list[dict[str, Any]] = []
    detection_indices: defaultdict[
        tuple[tuple[float, float], ...], deque[int]
    ] = defaultdict(deque)
    for index, (raw_polygon, detection_score) in enumerate(
        zip(detected_polygons, detection_scores), start=1
    ):
        polygon = _polygon(raw_polygon)
        entries.append(
            {
                "index": index,
                "polygon": polygon,
                "bbox": _axis_aligned_bbox(polygon, image_size),
                "text": "",
                "detection_confidence": float(detection_score),
                "recognition_confidence": None,
            }
        )
        detection_indices[_polygon_key(polygon)].append(index - 1)

    recognized_polygons = list(payload.get("rec_polys", []))
    recognized_texts = list(payload.get("rec_texts", []))
    recognition_scores = list(payload.get("rec_scores", []))
    if not (
        len(recognized_polygons) == len(recognized_texts) == len(recognition_scores)
    ):
        raise RuntimeError("PaddleOCR returned mismatched recognition results")

    for raw_polygon, text, recognition_score in zip(
        recognized_polygons, recognized_texts, recognition_scores
    ):
        polygon = _polygon(raw_polygon)
        matching_indices = detection_indices.get(_polygon_key(polygon))
        if not matching_indices:
            raise RuntimeError("Recognition polygon did not match a detection polygon")
        entry = entries[matching_indices.popleft()]
        entry["text"] = str(text)
        entry["recognition_confidence"] = float(recognition_score)

    return entries


def create_output_dir(root: Path, image_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{image_stem}_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def load_original_rgb(image_path: Path) -> Image.Image:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    with Image.open(image_path) as image_file:
        return ImageOps.exif_transpose(image_file).convert("RGB")


def _load_font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    candidates = [
        font_path,
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]
    if font_path is not None and not font_path.is_file():
        raise FileNotFoundError(f"Font does not exist: {font_path}")
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def save_visualization(
    image: Image.Image,
    entries: Sequence[Mapping[str, Any]],
    output_path: Path,
    font_path: Path | None = None,
) -> None:
    visualized = image.copy()
    draw = ImageDraw.Draw(visualized)
    line_width = max(2, round(min(image.size) / 500))
    font_size = max(14, min(48, round(min(image.size) * 0.018)))
    font = _load_font(font_path, font_size)

    for entry in entries:
        points = [tuple(point) for point in entry["polygon"]]
        draw.line(
            [*points, points[0]],
            fill=(255, 0, 0),
            width=line_width,
            joint="curve",
        )
        text = str(entry["text"]) or "[unrecognized]"
        label = f"{entry['index']}: {text}"
        label_x = max(0, int(min(point[0] for point in points)))
        label_y = max(0, int(min(point[1] for point in points)) - font_size - 4)
        text_bbox = draw.textbbox((label_x, label_y), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 255, 0))
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    visualized.save(output_path, format="PNG")


def main() -> int:
    args = parse_args()
    image = load_original_rgb(args.image)
    output_dir = create_output_dir(args.output_dir, args.image.stem)
    normalized_image_path = output_dir / "original_rgb.png"
    image.save(normalized_image_path, format="PNG")

    print(f"Input image: {args.image}")
    print(f"Image size: {image.width}x{image.height}")
    print(f"Detection model: {DETECTION_MODEL}")
    print(f"Recognition model: {RECOGNITION_MODEL}")
    pipeline = build_ocr_pipeline(args.device)
    result = next(iter(pipeline.predict(str(normalized_image_path))), None)
    if result is None:
        raise RuntimeError("PaddleOCR returned no result")

    entries = extract_text_results(_result_payload(result), image.size)
    json_path = output_dir / "ocr_results.json"
    visualization_path = output_dir / "ocr_visualization.png"
    output = {
        "input_image": str(args.image.resolve()),
        "ocr_input_image": normalized_image_path.name,
        "image": {"width": image.width, "height": image.height, "mode": image.mode},
        "models": {
            "text_detection": DETECTION_MODEL,
            "text_recognition": RECOGNITION_MODEL,
        },
        "text_count": len(entries),
        "texts": entries,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    save_visualization(image, entries, visualization_path, args.font_path)

    print(f"Detected text regions: {len(entries)}")
    print(f"JSON: {json_path}")
    print(f"Visualization: {visualization_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
