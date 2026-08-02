"""PaddleOCR text-detection crops for Qwen3-VL inference."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps


MAX_TEXT_CROPS = 3
TARGET_READABLE_TEXT_HEIGHT = 24.0
MAX_SMALL_TEXT_BOOST = 6.0
MULTI_IMAGE_PROMPT_PREFIX = (
    "첫 번째 이미지는 전체 장면이고, 이후 이미지는 검출된 텍스트 영역을 확대한 "
    "이미지입니다. 모든 이미지를 함께 참고하여 답하시오."
)

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectedTextBox:
    bbox: BBox
    score: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


def build_text_detector() -> Any:
    """Build the requested CPU detector without enabling oneDNN/MKLDNN."""
    os.environ["FLAGS_use_mkldnn"] = "0"
    from paddleocr import TextDetection

    return TextDetection(
        model_name="PP-OCRv5_server_det",
        device="cpu",
        enable_mkldnn=False,
    )


def _result_payload(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
    if not isinstance(payload, Mapping):
        raise RuntimeError("PaddleOCR returned an unsupported detection result")
    nested = payload.get("res")
    if isinstance(nested, Mapping):
        payload = nested
    return payload


def _polygon_to_bbox(polygon: Any, image_size: tuple[int, int]) -> BBox | None:
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError("PaddleOCR returned an invalid text polygon") from exc
    if len(points) < 3:
        return None

    image_width, image_height = image_size
    x1 = max(0, min(image_width, math.floor(min(point[0] for point in points))))
    y1 = max(0, min(image_height, math.floor(min(point[1] for point in points))))
    x2 = max(0, min(image_width, math.ceil(max(point[0] for point in points))))
    y2 = max(0, min(image_height, math.ceil(max(point[1] for point in points))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def detect_text_boxes(
    detector: Any, image_path: str | Path, image_size: tuple[int, int]
) -> list[DetectedTextBox]:
    """Run detection only and convert polygons to clipped axis-aligned boxes."""
    results = iter(detector.predict(str(image_path), batch_size=1))
    result = next(results, None)
    if result is None:
        return []

    payload = _result_payload(result)
    polygons = payload.get("dt_polys", [])
    scores = payload.get("dt_scores")
    if scores is None:
        scores = [1.0] * len(polygons)
    if len(polygons) != len(scores):
        raise RuntimeError("PaddleOCR returned different polygon and score counts")

    boxes = []
    for polygon, score in zip(polygons, scores):
        bbox = _polygon_to_bbox(polygon, image_size)
        numeric_score = float(score)
        if bbox is not None and math.isfinite(numeric_score) and numeric_score > 0:
            boxes.append(DetectedTextBox(bbox=bbox, score=numeric_score))
    return boxes


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, max(start_a, start_b) - min(end_a, end_b))


def _should_merge(
    first: DetectedTextBox,
    second: DetectedTextBox,
    image_size: tuple[int, int],
) -> bool:
    x1a, y1a, x2a, y2a = first.bbox
    x1b, y1b, x2b, y2b = second.bbox
    horizontal_gap = _gap(x1a, x2a, x1b, x2b)
    vertical_gap = _gap(y1a, y2a, y1b, y2b)
    horizontal_overlap = _overlap(x1a, x2a, x1b, x2b)
    vertical_overlap = _overlap(y1a, y2a, y1b, y2b)

    min_width = max(1, min(first.width, second.width))
    min_height = max(1, min(first.height, second.height))
    text_height = max(first.height, second.height)
    height_similarity = min_height / text_height
    image_width, image_height = image_size

    same_line = (
        height_similarity >= 0.50
        and vertical_overlap / min_height >= 0.60
        and horizontal_gap <= min(1.5 * text_height, 0.04 * image_width)
    )
    first_center_x = (x1a + x2a) / 2
    second_center_x = (x1b + x2b) / 2
    horizontally_aligned = min(
        abs(x1a - x1b),
        abs(x2a - x2b),
        abs(first_center_x - second_center_x),
    ) <= 1.5 * text_height
    stacked_lines = (
        height_similarity >= 0.50
        and horizontal_overlap / min_width >= 0.50
        and vertical_gap <= min(1.25 * text_height, 0.03 * image_height)
        and horizontally_aligned
    )
    return same_line or stacked_lines


def _union_bbox(boxes: Sequence[DetectedTextBox]) -> BBox:
    return (
        min(box.bbox[0] for box in boxes),
        min(box.bbox[1] for box in boxes),
        max(box.bbox[2] for box in boxes),
        max(box.bbox[3] for box in boxes),
    )


def select_text_group_bboxes(
    boxes: Sequence[DetectedTextBox],
    image_size: tuple[int, int],
    max_groups: int = MAX_TEXT_CROPS,
    max_pixels: int | None = None,
) -> list[BBox]:
    """Merge related detections and prioritize text made small by full-image scaling."""
    if not boxes or max_groups < 1:
        return []
    if max_pixels is not None and max_pixels < 1:
        raise ValueError("max_pixels must be positive")

    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(boxes):
        for second_index in range(first_index + 1, len(boxes)):
            if _should_merge(first, boxes[second_index], image_size):
                union(first_index, second_index)

    components: dict[int, list[DetectedTextBox]] = {}
    for index, box in enumerate(boxes):
        components.setdefault(find(index), []).append(box)

    image_area = image_size[0] * image_size[1]
    full_image_scale = min(
        1.0,
        math.sqrt((max_pixels or image_area) / max(1, image_area)),
    )
    ranked_groups: list[tuple[float, BBox]] = []
    for members in components.values():
        bbox = _union_bbox(members)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        group_area = width * height
        if width < 8 or height < 8 or group_area < max(64, image_area * 1e-6):
            continue

        weighted_text_area = sum(box.area * box.score for box in members)
        density = weighted_text_area / max(1, group_area)
        nearly_full_image = (
            width >= 0.95 * image_size[0] and height >= 0.95 * image_size[1]
        )
        if nearly_full_image and density < 0.05:
            continue
        small_text_need = sum(
            box.score
            * min(
                MAX_SMALL_TEXT_BOOST,
                TARGET_READABLE_TEXT_HEIGHT
                / max(1.0, box.height * full_image_scale),
            )
            for box in members
        )
        compactness = 0.5 + 0.5 * min(1.0, density)
        priority = small_text_need * compactness
        ranked_groups.append((priority, bbox))

    ranked_groups.sort(key=lambda item: item[0], reverse=True)
    return [bbox for _, bbox in ranked_groups[: min(max_groups, MAX_TEXT_CROPS)]]


def expand_bbox_if_within_limit(
    bbox: BBox, image_size: tuple[int, int], max_pixels: int
) -> BBox:
    """Add a small context margin without resizing or exceeding max_pixels."""
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    x1, y1, x2, y2 = bbox
    margin_x = max(4, round((x2 - x1) * 0.08))
    margin_y = max(4, round((y2 - y1) * 0.08))
    expanded = (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_size[0], x2 + margin_x),
        min(image_size[1], y2 + margin_y),
    )
    expanded_area = (expanded[2] - expanded[0]) * (expanded[3] - expanded[1])
    return expanded if expanded_area <= max_pixels else bbox


def detect_text_crops(
    detector: Any,
    image_path: str | Path,
    max_pixels: int,
    max_crops: int = MAX_TEXT_CROPS,
) -> list[Image.Image]:
    """Detect, group, and crop text regions without image upscaling."""
    path = Path(image_path)
    with Image.open(path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")
        boxes = detect_text_boxes(detector, path, image.size)
        group_bboxes = select_text_group_bboxes(
            boxes,
            image.size,
            max_crops,
            max_pixels=max_pixels,
        )
        crops = []
        for bbox in group_bboxes:
            crop_bbox = expand_bbox_if_within_limit(bbox, image.size, max_pixels)
            crops.append(image.crop(crop_bbox).copy())
    return crops


def add_text_crops_to_feature(
    feature: dict[str, Any], crops: Sequence[Image.Image]
) -> dict[str, Any]:
    """Insert crops after the full image and prefix the existing user prompt."""
    if not crops:
        return feature

    messages = feature.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Feature messages must be a list")
    copied_messages = [dict(message) for message in messages]
    user_messages = [
        message for message in copied_messages if message.get("role") == "user"
    ]
    if len(user_messages) != 1:
        raise ValueError("Expected exactly one user message")

    user_message = user_messages[0]
    content = user_message.get("content")
    if not isinstance(content, list):
        raise ValueError("User message content must be a list")
    copied_content = [dict(item) for item in content]
    image_items = [item for item in copied_content if item.get("type") == "image"]
    text_items = [item for item in copied_content if item.get("type") == "text"]
    if len(image_items) != 1 or len(text_items) != 1:
        raise ValueError("Expected one full image and one text prompt")

    prompt = str(text_items[0].get("text", "")).strip()
    if not prompt:
        raise ValueError("User text prompt must not be empty")
    prefixed_text = dict(text_items[0])
    prefixed_text["text"] = f"{MULTI_IMAGE_PROMPT_PREFIX}\n\n{prompt}"
    user_message["content"] = [
        dict(image_items[0]),
        *({"type": "image", "image": crop} for crop in crops[:MAX_TEXT_CROPS]),
        prefixed_text,
    ]

    augmented = dict(feature)
    augmented["messages"] = copied_messages
    return augmented


class TextCropDataset:
    """Inference-only dataset wrapper that adds detected text crops lazily."""

    def __init__(self, dataset: Any, detector: Any, max_pixels: int) -> None:
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self.dataset = dataset
        self.detector = detector
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        feature = self.dataset[index]
        crops = detect_text_crops(
            self.detector,
            feature["image_path"],
            max_pixels=self.max_pixels,
        )
        return add_text_crops_to_feature(feature, crops)
