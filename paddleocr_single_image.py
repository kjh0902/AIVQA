"""Run Korean PaddleOCR on one image and draw the detected bounding boxes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    from paddleocr import PaddleOCR
except ImportError as exc:
    raise SystemExit(
        "PaddleOCR가 설치되어 있지 않습니다. "
        "먼저 `pip install paddlepaddle paddleocr`를 실행하세요."
    ) from exc


@dataclass
class OCRItem:
    text: str
    confidence: float
    box: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="이미지 한 장에 한국어 PaddleOCR를 실행하고 바운딩 박스를 저장합니다."
    )
    parser.add_argument("image", type=Path, help="입력 이미지 경로")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="결과 이미지 경로 (기본값: 입력파일명_paddleocr_boxes.png)",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="PaddleOCR 추론 장치 (기본값: gpu:0, CPU 사용 시: cpu)",
    )
    return parser.parse_args()


def _as_result_dict(result: Any) -> dict[str, Any]:
    """Convert a PaddleOCR 3.x result object to its result dictionary."""
    if isinstance(result, dict):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()

    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError("PaddleOCR 결과를 사전 형태로 변환하지 못했습니다.")

    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _normalize_box(box: Any) -> list[tuple[float, float]]:
    points = np.asarray(box, dtype=float).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in points]


def _parse_v3_results(results: Iterable[Any]) -> list[OCRItem]:
    items: list[OCRItem] = []

    for result in results:
        data = _as_result_dict(result)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = data.get("rec_polys")
        if boxes is None:
            boxes = data.get("dt_polys", [])

        for text, score, box in zip(texts, scores, boxes):
            items.append(
                OCRItem(
                    text=str(text),
                    confidence=float(score),
                    box=_normalize_box(box),
                )
            )

    return items


def _looks_like_legacy_line(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 2
        and isinstance(value[1][0], str)
    )


def _parse_legacy_results(results: Any) -> list[OCRItem]:
    """Support the result structure used by PaddleOCR 2.x."""
    if not results:
        return []

    pages = [results] if _looks_like_legacy_line(results[0]) else results
    items: list[OCRItem] = []

    for page in pages:
        for line in page or []:
            if not _looks_like_legacy_line(line):
                continue
            box, (text, score, *_) = line
            items.append(
                OCRItem(
                    text=str(text),
                    confidence=float(score),
                    box=_normalize_box(box),
                )
            )

    return items


def validate_device(device: str) -> None:
    """Fail early with a useful message when the requested GPU is unavailable."""
    normalized = device.lower()
    if not normalized.startswith("gpu"):
        return

    import paddle

    if not paddle.is_compiled_with_cuda():
        raise SystemExit(
            "현재 환경의 PaddlePaddle이 CUDA를 지원하지 않습니다. "
            "CPU용 `paddlepaddle` 대신 CUDA 버전에 맞는 "
            "`paddlepaddle-gpu`를 설치하세요."
        )

    try:
        gpu_index = int(normalized.split(":", maxsplit=1)[1])
    except IndexError:
        gpu_index = 0
    except ValueError as exc:
        raise SystemExit(f"잘못된 GPU 장치 형식입니다: {device}") from exc

    gpu_count = paddle.device.cuda.device_count()
    if gpu_index < 0 or gpu_index >= gpu_count:
        raise SystemExit(
            f"요청한 장치 {device}를 사용할 수 없습니다. "
            f"PaddlePaddle이 감지한 GPU 수: {gpu_count}"
        )


def create_ocr(device: str) -> PaddleOCR:
    """Create a Korean OCR pipeline, with a fallback for PaddleOCR 2.x."""
    try:
        options: dict[str, Any] = {
            "device": device,
            "lang": "korean",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if device.lower() == "cpu":
            options["enable_mkldnn"] = False
        return PaddleOCR(**options)
    except TypeError:
        legacy_options: dict[str, Any] = {
            "lang": "korean",
            "use_angle_cls": True,
            "use_gpu": device.lower().startswith("gpu"),
        }
        if device.lower() == "cpu":
            legacy_options["enable_mkldnn"] = False
        return PaddleOCR(**legacy_options)


def run_ocr(ocr: PaddleOCR, rgb_image_array: np.ndarray) -> list[OCRItem]:
    # Pillow produces RGB arrays, while PaddleOCR/PaddleX expects OpenCV-style BGR.
    bgr_image_array = np.ascontiguousarray(rgb_image_array[:, :, ::-1])

    if hasattr(ocr, "predict"):
        return _parse_v3_results(ocr.predict(input=bgr_image_array))

    # PaddleOCR 2.x compatibility path
    return _parse_legacy_results(ocr.ocr(bgr_image_array, cls=True))


def print_results(items: list[OCRItem]) -> None:
    print(f"\n인식된 텍스트 수: {len(items)}")
    print("=" * 72)

    if not items:
        print("인식된 텍스트가 없습니다.")
        return

    for index, item in enumerate(items, start=1):
        formatted_box = [
            (round(x, 1), round(y, 1)) for x, y in item.box
        ]
        print(f"[{index:03d}]")
        print(f"  텍스트     : {item.text}")
        print(f"  신뢰도     : {item.confidence:.4f}")
        print(f"  바운딩 박스: {formatted_box}")
        print("-" * 72)


def draw_boxes(image: Image.Image, items: list[OCRItem]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    line_width = max(2, round(min(result.size) / 500))

    for item in items:
        if len(item.box) < 2:
            continue
        closed_box = item.box + [item.box[0]]
        draw.line(closed_box, fill=(255, 0, 0), width=line_width, joint="curve")

    return result


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"입력 이미지를 찾을 수 없습니다: {image_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else image_path.with_name(f"{image_path.stem}_paddleocr_boxes.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image).convert("RGB")

    validate_device(args.device)
    print(f"PaddleOCR 추론 장치: {args.device}")
    ocr = create_ocr(args.device)
    items = run_ocr(ocr, np.asarray(image))
    print_results(items)

    result_image = draw_boxes(image, items)
    result_image.save(output_path)
    print(f"\n결과 이미지 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
