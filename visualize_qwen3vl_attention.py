"""Visualize per-layer Qwen3-VL decoder attention over one image.

This script performs one teacher-forced forward pass only. It does not generate
an answer, select a bounding box, crop the image, or feed any crop back to the
model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QUESTION = "이미지 오른쪽 위 안내판에 쓰인 내용을 읽어줘."
IMAGE_COMPRESSION_FACTOR = 32
MIN_PIXELS = 64 * IMAGE_COMPRESSION_FACTOR**2
MAX_PIXELS = 512 * IMAGE_COMPRESSION_FACTOR**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save the last prompt token's image-attention heatmap for every "
            "Qwen3-VL decoder layer."
        )
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help=f"Question paired with the image (default: {DEFAULT_QUESTION})",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: outputs/qwen3vl_attention/<image stem>)",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        default=12,
        help=(
            "GPU memory budget passed to device_map='auto'. The default leaves "
            "activation headroom on a 16 GB GPU (default: 12)."
        ),
    )
    parser.add_argument(
        "--preview-max-side",
        type=int,
        default=1600,
        help="Maximum side length of saved overlay images (default: 1600)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Heatmap opacity in overlay images (default: 0.5)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise FileNotFoundError(f"Image file does not exist: {args.image}")
    if not args.question.strip():
        raise ValueError("--question must not be empty")
    if args.gpu_memory_gib < 1:
        raise ValueError("--gpu-memory-gib must be at least 1")
    if args.preview_max_side < 1:
        raise ValueError("--preview-max-side must be at least 1")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")


def configure_image_pixel_limits(processor: Any) -> None:
    """Apply the same pixel-budget mechanism used by the training pipeline."""
    if not hasattr(processor, "image_processor"):
        raise TypeError("The loaded processor does not expose an image_processor")
    processor.image_processor.size = {
        "shortest_edge": MIN_PIXELS,
        "longest_edge": MAX_PIXELS,
    }


def prepare_inputs(processor: Any, image: Any, question: str) -> Any:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def locate_image_tokens(
    input_ids: Any, image_grid_thw: Any, tokenizer: Any, spatial_merge_size: int
) -> tuple[int, int, tuple[int, int], tuple[int, int, int]]:
    """Locate the single image-token span and its merged 2-D grid."""
    import torch

    if input_ids.shape[0] != 1 or image_grid_thw.shape[0] != 1:
        raise ValueError("This script supports exactly one image in one prompt")

    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    token_ids = input_ids[0]
    starts = torch.nonzero(token_ids == vision_start_id, as_tuple=False).flatten()
    ends = torch.nonzero(token_ids == vision_end_id, as_tuple=False).flatten()
    if starts.numel() != 1 or ends.numel() != 1:
        raise RuntimeError(
            "Expected one <|vision_start|> and one <|vision_end|> token, "
            f"found {starts.numel()} and {ends.numel()}"
        )

    image_start = int(starts.item()) + 1
    image_end = int(ends.item())
    if image_end <= image_start:
        raise RuntimeError("The image token span is empty")

    temporal, grid_height, grid_width = (
        int(value) for value in image_grid_thw[0].detach().cpu().tolist()
    )
    if temporal != 1:
        raise RuntimeError(
            f"Expected a still-image temporal grid of 1, received {temporal}"
        )
    if grid_height % spatial_merge_size or grid_width % spatial_merge_size:
        raise RuntimeError(
            "image_grid_thw is not divisible by the model's spatial_merge_size: "
            f"grid=({grid_height}, {grid_width}), merge={spatial_merge_size}"
        )

    heatmap_shape = (
        grid_height // spatial_merge_size,
        grid_width // spatial_merge_size,
    )
    expected_tokens = temporal * heatmap_shape[0] * heatmap_shape[1]
    actual_tokens = image_end - image_start
    if actual_tokens != expected_tokens:
        raise RuntimeError(
            "Image-token count does not match image_grid_thw after spatial merge: "
            f"tokens={actual_tokens}, expected={expected_tokens}, "
            f"grid=({temporal}, {grid_height}, {grid_width}), "
            f"merge={spatial_merge_size}"
        )
    return (
        image_start,
        image_end,
        heatmap_shape,
        (temporal, grid_height, grid_width),
    )


def find_decoder_layers(model: Any) -> Any:
    try:
        layers = model.model.language_model.layers
    except AttributeError as exc:
        raise RuntimeError(
            "Unexpected Qwen3-VL module layout: model.model.language_model.layers "
            "was not found. Check the installed Transformers version."
        ) from exc
    if not layers:
        raise RuntimeError("The model exposes no decoder layers")
    return layers


def collect_layer_attention(
    model: Any,
    inputs: Any,
    image_start: int,
    image_end: int,
    query_index: int,
) -> list[Any]:
    """Capture only one head-averaged image vector from each decoder layer.

    Qwen3-VL eager attention returns the full matrix from each self-attention
    module even though the decoder layer discards it. The hooks reduce that
    temporary matrix immediately and move only the small image vector to CPU.
    This avoids retaining every layer's full attention matrix on the GPU.
    """
    import torch

    layers = find_decoder_layers(model)
    vectors: list[Any | None] = [None] * len(layers)
    handles = []

    def make_hook(layer_index: int) -> Any:
        def capture(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise RuntimeError(
                    f"Layer {layer_index} did not return attention weights. "
                    "The model must be loaded with attn_implementation='eager'."
                )
            weights = output[1]
            if query_index >= weights.shape[-2] or image_end > weights.shape[-1]:
                raise RuntimeError(
                    f"Layer {layer_index} attention shape {tuple(weights.shape)} "
                    "does not cover the requested query/image token indices"
                )
            vector = weights[
                0, :, query_index, image_start:image_end
            ].float().mean(dim=0)
            vectors[layer_index] = vector.detach().cpu()

        return capture

    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(index)))

    try:
        # Calling the base multimodal model skips the unused language-model head.
        # No output_attentions=True is passed: hooks consume each eager attention
        # matrix before it can be accumulated in the model output.
        with torch.inference_mode():
            outputs = model.model(**inputs, use_cache=False, return_dict=True)
        del outputs
    finally:
        for handle in handles:
            handle.remove()

    missing = [index for index, vector in enumerate(vectors) if vector is None]
    if missing:
        raise RuntimeError(f"No attention vector was captured for layers: {missing}")
    return [vector for vector in vectors if vector is not None]


def normalize_heatmap(heatmap: Any) -> Any:
    import numpy as np

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if not np.isfinite(heatmap).all():
        raise ValueError("Attention heatmap contains NaN or infinite values")
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum == minimum:
        return np.zeros_like(heatmap)
    return (heatmap - minimum) / (maximum - minimum)


def make_preview(image: Any, max_side: int) -> Any:
    from PIL import Image

    preview = image.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return preview


def resize_heatmap(heatmap: Any, size: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image

    normalized = normalize_heatmap(heatmap)
    heatmap_image = Image.fromarray(normalized.astype(np.float32))
    return np.asarray(
        heatmap_image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32
    ).clip(0.0, 1.0)


def save_visualizations(
    image: Any, attention_maps: Any, output_dir: Path, max_side: int, alpha: float
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview = make_preview(image, max_side)
    preview.save(output_dir / "input_preview.png")

    for layer_index, heatmap in enumerate(attention_maps):
        resized = resize_heatmap(heatmap, preview.size)
        width, height = preview.size
        figure_width = 10.0
        figure_height = max(3.0, figure_width * height / width)
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        axis.imshow(preview)
        overlay = axis.imshow(
            resized, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0
        )
        axis.set_title(
            f"Decoder layer {layer_index:02d} | "
            f"raw attention {float(heatmap.min()):.3e}–{float(heatmap.max()):.3e}"
        )
        axis.axis("off")
        figure.colorbar(overlay, ax=axis, fraction=0.025, pad=0.02)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"layer_{layer_index:02d}_attention.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)

    columns = min(6, len(attention_maps))
    rows = math.ceil(len(attention_maps) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.2 * rows * preview.height / preview.width),
        squeeze=False,
    )
    contact_size = (min(preview.width, 480), min(preview.height, 480))
    contact_preview = preview.copy()
    contact_preview.thumbnail(contact_size)
    for layer_index, axis in enumerate(axes.flat):
        if layer_index >= len(attention_maps):
            axis.axis("off")
            continue
        resized = resize_heatmap(attention_maps[layer_index], contact_preview.size)
        axis.imshow(contact_preview)
        axis.imshow(resized, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0)
        axis.set_title(f"Layer {layer_index:02d}", fontsize=9)
        axis.axis("off")
    figure.suptitle("Qwen3-VL: last prompt token → image-token attention", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "all_layers_contact_sheet.png", dpi=150)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The active CUDA GPU does not support bfloat16")

    output_dir = args.output_dir or (
        Path("outputs") / "qwen3vl_attention" / args.image.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading processor: {args.model_id}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model_id)
    configure_image_pixel_limits(processor)

    print(
        f"[2/5] Loading BF16 model with a {args.gpu_memory_gib} GiB GPU budget",
        file=sys.stderr,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{args.gpu_memory_gib}GiB", "cpu": "64GiB"},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.requires_grad_(False)
    model.eval()

    with Image.open(args.image) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")

    print(
        f"[3/5] Preparing image with min_pixels={MIN_PIXELS:,}, "
        f"max_pixels={MAX_PIXELS:,}",
        file=sys.stderr,
    )
    inputs = prepare_inputs(processor, image, args.question)
    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
    image_start, image_end, heatmap_shape, image_grid = locate_image_tokens(
        inputs["input_ids"],
        inputs["image_grid_thw"],
        processor.tokenizer,
        spatial_merge_size,
    )
    attention_mask = inputs["attention_mask"][0]
    query_index = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    query_token_id = int(inputs["input_ids"][0, query_index].item())
    query_token = processor.tokenizer.convert_ids_to_tokens(query_token_id)

    inputs = inputs.to(model.device)
    print(
        f"[4/5] Capturing {len(find_decoder_layers(model))} decoder layers "
        f"(image tokens={image_end - image_start}, grid={heatmap_shape})",
        file=sys.stderr,
    )
    vectors = collect_layer_attention(
        model, inputs, image_start, image_end, query_index
    )
    attention_maps = np.stack(
        [vector.numpy().reshape(heatmap_shape) for vector in vectors]
    )
    np.save(output_dir / "attention_maps.npy", attention_maps)

    print(f"[5/5] Saving heatmaps to {output_dir}", file=sys.stderr)
    save_visualizations(
        image,
        attention_maps,
        output_dir,
        args.preview_max_side,
        args.overlay_alpha,
    )
    metadata = {
        "model_id": args.model_id,
        "image": str(args.image.resolve()),
        "question": args.question,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "image_grid_thw": image_grid,
        "spatial_merge_size": spatial_merge_size,
        "heatmap_shape": heatmap_shape,
        "image_token_span": [image_start, image_end],
        "query_token_index": query_index,
        "query_token_id": query_token_id,
        "query_token": query_token,
        "decoder_layers": len(vectors),
        "attention_definition": (
            "mean over heads of the last non-padding prompt token's causal "
            "attention to image tokens; visualizations are min-max normalized "
            "independently per layer"
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del inputs, model
    torch.cuda.empty_cache()
    return output_dir


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        output_dir = run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
