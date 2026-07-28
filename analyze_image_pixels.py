"""Analyze AIVQA train/validation/test image dimensions without loading a model."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from train_lora import (
    DATASET_DIR,
    DATASET_NAME,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    IMAGE_COMPRESSION_FACTOR,
)


SPLITS = ("train", "validation", "test")
EXIF_ORIENTATION_TAG = 274
SWAPPED_EXIF_ORIENTATIONS = {5, 6, 7, 8}
SUMMARY_COLUMNS = (
    "split",
    "image_count",
    "width_min",
    "width_mean",
    "width_median",
    "width_max",
    "height_min",
    "height_mean",
    "height_median",
    "height_max",
    "pixel_count_min",
    "pixel_count_p25",
    "pixel_count_p50",
    "pixel_count_p75",
    "pixel_count_p90",
    "pixel_count_p95",
    "pixel_count_p99",
    "pixel_count_mean",
    "pixel_count_max",
    "above_max_pixels_count",
    "above_max_pixels_percent",
    "within_max_pixels_count",
    "within_max_pixels_percent",
    "max_pixels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AIVQA image dimensions and Qwen max_pixels coverage."
    )
    parser.add_argument(
        "--train-json", type=Path, default=DATASET_DIR / f"{DATASET_NAME}_train.json"
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=DATASET_DIR / f"{DATASET_NAME}_validation.json",
    )
    parser.add_argument(
        "--test-json", type=Path, default=DATASET_DIR / f"{DATASET_NAME}_test.json"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/image_pixel_analysis"),
        help="Root directory where a unique run_YYYYMMDD_HHMMSS folder is created",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.train_json, args.validation_json, args.test_json):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset JSON does not exist: {path}")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")


def create_output_dir(output_root: Path, started_at: datetime | None = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"
    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        output_dir = output_root / name
        try:
            output_dir.mkdir(exist_ok=False)
            return output_dir
        except FileExistsError:
            suffix += 1


def resolve_image_path(dataset_root: Path, split: str, image_name: str) -> Path:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return image_path

    split_path = dataset_root / split / image_path
    direct_path = dataset_root / image_path
    if split_path.is_file() or not direct_path.is_file():
        return split_path
    return direct_path


def read_display_dimensions(image_path: Path) -> tuple[int, int, int]:
    """Return display-oriented width, height, and raw EXIF orientation."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                width, height = image.size
                orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG, 1))
    except (OSError, ValueError) as error:
        raise ValueError(f"Pillow could not read image: {image_path}") from error

    if orientation in SWAPPED_EXIF_ORIENTATIONS:
        width, height = height, width
    return int(width), int(height), orientation


def load_records(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Dataset JSON root must be a list: {json_path}")
    return records


def analyze_split(
    split: str,
    json_path: Path,
    dataset_root: Path,
    max_pixels: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(load_records(json_path)):
        metadata = record.get("metadata")
        model_input = record.get("model_input")
        if not isinstance(metadata, dict) or not isinstance(model_input, dict):
            raise ValueError(f"{json_path} sample {index}: invalid metadata/model_input")

        record_split = str(metadata.get("split", "")).strip()
        if record_split != split:
            raise ValueError(
                f"{json_path} sample {index}: expected split={split!r}, "
                f"found {record_split!r}"
            )
        image_name = model_input.get("image_name")
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"{json_path} sample {index}: missing model_input.image_name")

        image_path = resolve_image_path(dataset_root, split, image_name.strip())
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        width, height, orientation = read_display_dimensions(image_path)
        pixel_count = width * height
        rows.append(
            {
                "split": split,
                "question_id": str(metadata.get("question_id", index)),
                "image_name": image_name.strip(),
                "image_path": str(image_path),
                "exif_orientation": orientation,
                "width": width,
                "height": height,
                "pixel_count": pixel_count,
                "max_pixels": max_pixels,
                "above_max_pixels": pixel_count > max_pixels,
                "within_max_pixels": pixel_count <= max_pixels,
                "pixel_count_to_max_ratio": pixel_count / max_pixels,
            }
        )
    return pd.DataFrame(rows)


def _summary_row(split: str, frame: pd.DataFrame, max_pixels: int) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"Cannot summarize empty split: {split}")
    width = frame["width"]
    height = frame["height"]
    pixels = frame["pixel_count"]
    above_count = int(frame["above_max_pixels"].sum())
    within_count = int(frame["within_max_pixels"].sum())
    image_count = int(len(frame))
    return {
        "split": split,
        "image_count": image_count,
        "width_min": int(width.min()),
        "width_mean": float(width.mean()),
        "width_median": float(width.median()),
        "width_max": int(width.max()),
        "height_min": int(height.min()),
        "height_mean": float(height.mean()),
        "height_median": float(height.median()),
        "height_max": int(height.max()),
        "pixel_count_min": int(pixels.min()),
        "pixel_count_p25": float(pixels.quantile(0.25)),
        "pixel_count_p50": float(pixels.quantile(0.50)),
        "pixel_count_p75": float(pixels.quantile(0.75)),
        "pixel_count_p90": float(pixels.quantile(0.90)),
        "pixel_count_p95": float(pixels.quantile(0.95)),
        "pixel_count_p99": float(pixels.quantile(0.99)),
        "pixel_count_mean": float(pixels.mean()),
        "pixel_count_max": int(pixels.max()),
        "above_max_pixels_count": above_count,
        "above_max_pixels_percent": 100.0 * above_count / image_count,
        "within_max_pixels_count": within_count,
        "within_max_pixels_percent": 100.0 * within_count / image_count,
        "max_pixels": int(max_pixels),
    }


def summarize_images(details: pd.DataFrame, max_pixels: int) -> pd.DataFrame:
    rows = [
        _summary_row(split, details[details["split"] == split], max_pixels)
        for split in SPLITS
    ]
    rows.append(_summary_row("all", details, max_pixels))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_tables_and_reports(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    min_pixels: int,
    max_pixels: int,
) -> list[Path]:
    detail_csv = output_dir / "image_pixel_details.csv"
    summary_csv = output_dir / "image_pixel_summary.csv"
    summary_json = output_dir / "image_pixel_summary.json"
    report_txt = output_dir / "image_pixel_report.txt"

    details.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    payload = {
        "settings": {
            "image_compression_factor": IMAGE_COMPRESSION_FACTOR,
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "min_visual_tokens": int(min_pixels // IMAGE_COMPRESSION_FACTOR**2),
            "max_visual_tokens": int(max_pixels // IMAGE_COMPRESSION_FACTOR**2),
        },
        "summary": json.loads(summary.to_json(orient="records")),
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "AIVQA image pixel analysis",
        f"IMAGE_COMPRESSION_FACTOR: {IMAGE_COMPRESSION_FACTOR}",
        f"min_pixels: {min_pixels:,}",
        f"max_pixels: {max_pixels:,}",
        "",
        summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"),
        "",
    ]
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")
    return [detail_csv, summary_csv, summary_json, report_txt]


def write_plots(
    details: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, max_pixels: int
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"train": "#2563EB", "validation": "#F59E0B", "test": "#059669"}
    plot_paths: list[Path] = []

    distribution_path = output_dir / "pixel_count_distribution.png"
    positive_pixels = details.loc[details["pixel_count"] > 0, "pixel_count"]
    minimum_pixel_count = float(positive_pixels.min())
    maximum_pixel_count = float(positive_pixels.max())
    if minimum_pixel_count == maximum_pixel_count:
        bins = np.array([minimum_pixel_count * 0.9, maximum_pixel_count * 1.1])
    else:
        bins = np.geomspace(minimum_pixel_count, maximum_pixel_count, 50)
    figure, axis = plt.subplots(figsize=(10, 6))
    for split in SPLITS:
        values = details.loc[details["split"] == split, "pixel_count"]
        axis.hist(values, bins=bins, alpha=0.45, label=split, color=colors[split])
    axis.axvline(
        max_pixels,
        color="#DC2626",
        linestyle="--",
        linewidth=2,
        label=f"max_pixels={max_pixels:,}",
    )
    axis.set_xscale("log")
    axis.set_title("Original pixel-count distribution")
    axis.set_xlabel("Pixel count (log scale)")
    axis.set_ylabel("Image count")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(distribution_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(distribution_path)

    comparison_path = output_dir / "max_pixels_comparison.png"
    split_summary = summary[summary["split"].isin(SPLITS)].set_index("split")
    x_positions = np.arange(len(SPLITS))
    figure, axis = plt.subplots(figsize=(9, 6))
    within_bars = axis.bar(
        x_positions,
        split_summary.loc[list(SPLITS), "within_max_pixels_count"],
        label="At or below max_pixels",
        color="#0EA5E9",
    )
    above_bars = axis.bar(
        x_positions,
        split_summary.loc[list(SPLITS), "above_max_pixels_count"],
        bottom=split_summary.loc[list(SPLITS), "within_max_pixels_count"],
        label="Above max_pixels",
        color="#F97316",
    )
    axis.bar_label(within_bars, label_type="center", color="white", fontweight="bold")
    axis.bar_label(above_bars, label_type="center", color="white", fontweight="bold")
    axis.set_xticks(x_positions, SPLITS)
    axis.set_title("Images compared with max_pixels")
    axis.set_xlabel("Split")
    axis.set_ylabel("Image count")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(comparison_path)

    dimensions_path = output_dir / "image_dimensions_scatter.png"
    figure, axis = plt.subplots(figsize=(9, 7))
    for split in SPLITS:
        split_frame = details[details["split"] == split]
        axis.scatter(
            split_frame["width"],
            split_frame["height"],
            s=16,
            alpha=0.45,
            label=split,
            color=colors[split],
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Display-oriented image dimensions")
    axis.set_xlabel("Width (pixels, log scale)")
    axis.set_ylabel("Height (pixels, log scale)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(dimensions_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(dimensions_path)
    return plot_paths


def run_analysis(args: argparse.Namespace) -> tuple[Path, pd.DataFrame, list[Path]]:
    validate_args(args)
    output_dir = create_output_dir(args.output_dir)
    json_paths = {
        "train": args.train_json,
        "validation": args.validation_json,
        "test": args.test_json,
    }
    frames = [
        analyze_split(
            split,
            json_paths[split],
            args.dataset_root,
            args.max_pixels,
        )
        for split in SPLITS
    ]
    details = pd.concat(frames, ignore_index=True)
    summary = summarize_images(details, args.max_pixels)
    output_paths = write_tables_and_reports(
        details,
        summary,
        output_dir,
        args.min_pixels,
        args.max_pixels,
    )
    output_paths.extend(write_plots(details, summary, output_dir, args.max_pixels))
    return output_dir, summary, output_paths


def main() -> int:
    output_dir, summary, output_paths = run_analysis(parse_args())
    print(f"Saved image pixel analysis: {output_dir}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("Generated files:")
    for path in output_paths:
        print(f"- {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
