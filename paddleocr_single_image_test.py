"""Run PaddleOCR detection and Korean recognition on one original image."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"
TEXT_DET_LIMIT_TYPE = "max"
TEXT_DET_LIMIT_SIDE_LEN = 1280


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


def build_ocr_models(device: str) -> tuple[Any, Any]:
    """Build separate models so the detector's confidence scores are retained."""
    os.environ["FLAGS_use_mkldnn"] = "0"
    from paddleocr import TextDetection, TextRecognition

    detector = TextDetection(
        model_name=DETECTION_MODEL,
        limit_type=TEXT_DET_LIMIT_TYPE,
        limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        device=device,
        enable_mkldnn=False,
    )
    recognizer = TextRecognition(
        model_name=RECOGNITION_MODEL,
        device=device,
        enable_mkldnn=False,
    )
    return detector, recognizer


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


def _axis_aligned_bbox(
    polygon: Sequence[Sequence[int | float]], image_size: tuple[int, int]
) -> list[int]:
    width, height = image_size
    x1 = max(0, min(width, math.floor(min(point[0] for point in polygon))))
    y1 = max(0, min(height, math.floor(min(point[1] for point in polygon))))
    x2 = max(0, min(width, math.ceil(max(point[0] for point in polygon))))
    y2 = max(0, min(height, math.ceil(max(point[1] for point in polygon))))
    return [x1, y1, x2, y2]


def extract_detection_results(
    payload: Mapping[str, Any], image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    """Keep every polygon and confidence returned by the detector independent."""
    detected_polygons = list(payload.get("dt_polys", []))
    detection_scores = list(payload.get("dt_scores", []))
    if len(detected_polygons) != len(detection_scores):
        raise RuntimeError(
            "PaddleOCR returned mismatched detection polygons and scores"
        )

    entries: list[dict[str, Any]] = []
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
    return entries


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _extract_text_line(
    image: Image.Image, polygon: Sequence[Sequence[int | float]]
) -> Image.Image:
    """Rectify one detector quadrilateral directly from the original RGB image."""
    if len(polygon) != 4:
        return image.crop(tuple(_axis_aligned_bbox(polygon, image.size)))

    points = [[float(point[0]), float(point[1])] for point in polygon]
    top_left = min(points, key=lambda point: point[0] + point[1])
    bottom_right = max(points, key=lambda point: point[0] + point[1])
    top_right = min(points, key=lambda point: point[1] - point[0])
    bottom_left = max(points, key=lambda point: point[1] - point[0])
    width = max(
        1,
        round(
            max(
                _distance(top_left, top_right),
                _distance(bottom_left, bottom_right),
            )
        ),
    )
    height = max(
        1,
        round(
            max(
                _distance(top_left, bottom_left),
                _distance(top_right, bottom_right),
            )
        ),
    )
    quad = (
        *top_left,
        *bottom_left,
        *bottom_right,
        *top_right,
    )
    text_line = image.transform(
        (width, height),
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC,
    )
    if text_line.height / max(1, text_line.width) >= 1.5:
        text_line = text_line.rotate(90, expand=True)
    return text_line


def recognize_text_results(
    recognizer: Any,
    image: Image.Image,
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recognize each independent polygon using temporary lossless line images."""
    recognized_entries = [dict(entry) for entry in entries]
    if not recognized_entries:
        return recognized_entries

    with tempfile.TemporaryDirectory(prefix="paddleocr_text_lines_") as temp_dir:
        temp_path = Path(temp_dir)
        text_line_paths = []
        for index, entry in enumerate(recognized_entries, start=1):
            text_line = _extract_text_line(image, entry["polygon"])
            text_line_path = temp_path / f"text_line_{index:05d}.png"
            text_line.save(text_line_path, format="PNG")
            text_line_paths.append(str(text_line_path))

        recognition_results = list(
            recognizer.predict(text_line_paths, batch_size=1)
        )

    if len(recognition_results) != len(recognized_entries):
        raise RuntimeError(
            "PaddleOCR returned a different number of recognition results"
        )
    for entry, result in zip(recognized_entries, recognition_results):
        payload = _result_payload(result)
        if "rec_text" not in payload or "rec_score" not in payload:
            raise RuntimeError(
                "PaddleOCR recognition result is missing required fields"
            )
        entry["text"] = str(payload["rec_text"])
        entry["recognition_confidence"] = float(payload["rec_score"])
    return recognized_entries


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
    print(
        f"Detection input limit: {TEXT_DET_LIMIT_TYPE} "
        f"{TEXT_DET_LIMIT_SIDE_LEN}px"
    )
    detector, recognizer = build_ocr_models(args.device)
    detection_result = next(
        iter(detector.predict(str(normalized_image_path), batch_size=1)), None
    )
    if detection_result is None:
        raise RuntimeError("PaddleOCR text detector returned no result")

    entries = extract_detection_results(
        _result_payload(detection_result), image.size
    )
    entries = recognize_text_results(recognizer, image, entries)
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
