"""Build the AIVQA text + image Qdrant collection from unified_rag.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


LOGGER = logging.getLogger("aivqa.rag_db")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("rag_db/unified_rag.jsonl")
DEFAULT_QDRANT_PATH = Path("rag_db/qdrant_storage")
DEFAULT_MODEL_CACHE = Path("rag_db/huggingface_cache")

PAYLOAD_FIELDS = (
    "doc_id",
    "source",
    "title",
    "search_terms",
    "description",
    "image_path",
)


@dataclass
class BuildStats:
    lines_read: int = 0
    invalid_entities: int = 0
    indexed_entities: int = 0
    declared_images: int = 0
    embedded_images: int = 0
    failed_images: int = 0


def resolve_repository_path(path: str | Path) -> Path:
    """Resolve a CLI path against the repository root, independent of cwd."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def deterministic_point_id(doc_id: str) -> str:
    """Return a stable Qdrant-compatible UUID derived only from doc_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def make_search_text(search_terms: Sequence[Any]) -> str:
    """Join search terms into the single string embedded by KURE."""
    return " ".join(str(term).strip() for term in search_terms if str(term).strip())


def iter_entities(input_path: Path, stats: BuildStats) -> Iterator[dict[str, Any]]:
    """Stream valid entities without loading the large JSONL file into memory."""
    with input_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            stats.lines_read += 1
            if not line.strip():
                LOGGER.warning("Skipping blank JSONL line %d", line_number)
                stats.invalid_entities += 1
                continue

            try:
                entity = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.error("Skipping invalid JSON on line %d: %s", line_number, exc)
                stats.invalid_entities += 1
                continue

            if not isinstance(entity, dict):
                LOGGER.error("Skipping non-object entity on line %d", line_number)
                stats.invalid_entities += 1
                continue

            doc_id = entity.get("doc_id")
            search_terms = entity.get("search_terms")
            image_paths = entity.get("image_path")
            if not isinstance(doc_id, str) or not doc_id.strip():
                LOGGER.error("Skipping line %d: doc_id must be a non-empty string", line_number)
                stats.invalid_entities += 1
                continue
            if not isinstance(search_terms, list):
                LOGGER.error("Skipping %s: search_terms must be a list", doc_id)
                stats.invalid_entities += 1
                continue
            if image_paths is not None and not isinstance(image_paths, list):
                LOGGER.error("Skipping %s: image_path must be a list or null", doc_id)
                stats.invalid_entities += 1
                continue

            # Keep the payload shape consistent even when image_path is null/missing.
            entity.setdefault("image_path", [])
            yield entity


def batched(iterable: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def resolve_image_path(raw_path: str, input_path: Path) -> Path:
    """Resolve payload image paths, which are normally relative to the JSONL file."""
    image_path = Path(raw_path).expanduser()
    if image_path.is_absolute():
        return image_path

    jsonl_relative = input_path.parent / image_path
    if jsonl_relative.exists():
        return jsonl_relative.resolve()
    repository_relative = REPOSITORY_ROOT / image_path
    if repository_relative.exists():
        return repository_relative.resolve()
    # The common JSONL format is relative to the JSONL parent. Preserve that
    # interpretation in error logs even when the target is missing.
    return jsonl_relative.resolve()


def load_image(raw_path: str, input_path: Path) -> Image.Image:
    path = resolve_image_path(raw_path, input_path)
    with Image.open(path) as source_image:
        source_image.load()
        return ImageOps.exif_transpose(source_image).convert("RGB")


def _clip_features(
    images: Sequence[Image.Image],
    processor: Any,
    model: CLIPModel,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=list(images), return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    inputs = {
        name: tensor.to(
            device,
            dtype=model_dtype if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in inputs.items()
    }
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = F.normalize(features, p=2, dim=-1)
    return features.float().cpu().numpy()


def embed_entity_images(
    entities: Sequence[dict[str, Any]],
    input_path: Path,
    processor: Any,
    model: CLIPModel,
    device: torch.device,
    image_batch_size: int,
    stats: BuildStats,
) -> list[list[list[float]]]:
    """Embed every readable image and group the vectors back by entity."""
    vectors_by_entity: list[list[list[float]]] = [[] for _ in entities]
    image_buffer: list[Image.Image] = []
    owner_buffer: list[int] = []

    def flush() -> None:
        if not image_buffer:
            return
        try:
            features = _clip_features(image_buffer, processor, model, device)
            for owner, feature in zip(owner_buffer, features, strict=True):
                vectors_by_entity[owner].append(feature.tolist())
                stats.embedded_images += 1
        finally:
            for image in image_buffer:
                image.close()
            image_buffer.clear()
            owner_buffer.clear()

    for entity_index, entity in enumerate(entities):
        raw_paths = entity.get("image_path") or []
        stats.declared_images += len(raw_paths)
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                LOGGER.warning("Skipping invalid image path for %s: %r", entity["doc_id"], raw_path)
                stats.failed_images += 1
                continue
            try:
                image_buffer.append(load_image(raw_path, input_path))
                owner_buffer.append(entity_index)
            except Exception as exc:  # A bad image must not stop the collection build.
                LOGGER.warning(
                    "Skipping unreadable image for %s (%s): %s",
                    entity["doc_id"],
                    raw_path,
                    exc,
                )
                stats.failed_images += 1
                continue

            if len(image_buffer) >= image_batch_size:
                flush()

    flush()
    return vectors_by_entity


def create_client(qdrant_path: Path | None, qdrant_url: str | None) -> QdrantClient:
    if qdrant_url:
        return QdrantClient(
            url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=120,
        )
    assert qdrant_path is not None
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    text_dimension: int,
    image_dimension: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        LOGGER.warning("Deleting existing collection %s", collection_name)
        client.delete_collection(collection_name)
        exists = False

    if exists:
        LOGGER.info("Reusing existing collection %s", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text": models.VectorParams(
                size=text_dimension,
                distance=models.Distance.COSINE,
            ),
            "image": models.VectorParams(
                size=image_dimension,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
        },
    )
    LOGGER.info(
        "Created collection %s (text=%d, image=%d multivector/MAX_SIM)",
        collection_name,
        text_dimension,
        image_dimension,
    )


def load_models(
    text_model_name: str,
    image_model_name: str,
    cache_dir: Path,
    device: torch.device,
    use_fp16: bool,
) -> tuple[SentenceTransformer, Any, CLIPModel]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading text model %s", text_model_name)
    text_model = SentenceTransformer(
        text_model_name,
        device=str(device),
        cache_folder=str(cache_dir),
    )

    LOGGER.info("Loading image model %s", image_model_name)
    image_processor = AutoProcessor.from_pretrained(
        image_model_name,
        cache_dir=str(cache_dir),
    )
    image_model = CLIPModel.from_pretrained(
        image_model_name,
        cache_dir=str(cache_dir),
    ).to(device)

    if use_fp16:
        text_model.half()
        image_model.half()
        LOGGER.info("Using float16 inference on %s", device)

    text_model.eval()
    image_model.eval()
    return text_model, image_processor, image_model


def build_collection(args: argparse.Namespace) -> BuildStats:
    input_path = resolve_repository_path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"JSONL input does not exist: {input_path}")

    cache_dir = resolve_repository_path(args.model_cache)
    qdrant_path = None if args.qdrant_url else resolve_repository_path(args.qdrant_path)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else
        args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    use_fp16 = device.type == "cuda" and not args.fp32

    text_model, image_processor, image_model = load_models(
        args.text_model,
        args.image_model,
        cache_dir,
        device,
        use_fp16,
    )
    text_dimension = text_model.get_sentence_embedding_dimension()
    if text_dimension is None:
        raise RuntimeError("Could not determine the text embedding dimension")
    image_dimension = int(image_model.config.projection_dim)

    stats = BuildStats()
    client = create_client(qdrant_path, args.qdrant_url)
    progress = tqdm(desc="Indexed entities", unit="entity")
    try:
        ensure_collection(
            client,
            args.collection,
            int(text_dimension),
            image_dimension,
            args.recreate,
        )
        entity_stream: Iterable[dict[str, Any]] = iter_entities(input_path, stats)
        if args.limit is not None:
            from itertools import islice

            entity_stream = islice(entity_stream, args.limit)

        for entities in batched(entity_stream, args.batch_size):
            search_texts = [make_search_text(entity["search_terms"]) for entity in entities]
            text_vectors = text_model.encode(
                search_texts,
                batch_size=args.text_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
            image_vectors = embed_entity_images(
                entities,
                input_path,
                image_processor,
                image_model,
                device,
                args.image_batch_size,
                stats,
            )

            points: list[models.PointStruct] = []
            for entity, text_vector, entity_image_vectors in zip(
                entities, text_vectors, image_vectors, strict=True
            ):
                named_vectors: dict[str, Any] = {"text": text_vector.tolist()}
                if entity_image_vectors:
                    named_vectors["image"] = entity_image_vectors
                payload = {field: entity.get(field) for field in PAYLOAD_FIELDS}
                points.append(
                    models.PointStruct(
                        id=deterministic_point_id(entity["doc_id"]),
                        vector=named_vectors,
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=args.collection,
                points=points,
                wait=True,
            )
            stats.indexed_entities += len(points)
            progress.update(len(points))
    finally:
        progress.close()
        client.close()

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AIVQA's named text + image multivector Qdrant collection.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="JSONL path (repository-relative)")
    parser.add_argument("--collection", default="aivqa_unified_rag")
    qdrant_group = parser.add_mutually_exclusive_group()
    qdrant_group.add_argument(
        "--qdrant-path",
        default=str(DEFAULT_QDRANT_PATH),
        help="Embedded Qdrant storage path (repository-relative)",
    )
    qdrant_group.add_argument(
        "--qdrant-url",
        help="Qdrant server URL; QDRANT_API_KEY is read from the environment when set",
    )
    parser.add_argument("--model-cache", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--text-model", default="nlpai-lab/KURE-v1")
    parser.add_argument("--image-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--fp32", action="store_true", help="Disable automatic float16 on CUDA")
    parser.add_argument("--batch-size", type=int, default=64, help="Entities per Qdrant upsert")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Process only the first N valid entities")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before indexing",
    )
    args = parser.parse_args()

    for name in ("batch_size", "text_batch_size", "image_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    stats = build_collection(args)
    LOGGER.info(
        "Build complete: indexed=%d, lines=%d, invalid=%d, images=%d/%d, failed_images=%d",
        stats.indexed_entities,
        stats.lines_read,
        stats.invalid_entities,
        stats.embedded_images,
        stats.declared_images,
        stats.failed_images,
    )


if __name__ == "__main__":
    main()
