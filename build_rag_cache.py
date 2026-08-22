"""Precompute train/validation/test RAG retrieval caches for the AIVQA pipeline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivqa.data import KananaVQADataset
from rag_db.augmentation import (
    RAG_CACHE_DIR,
    rag_cache_paths,
    retrieve_dataset_candidates,
)
from rag_db.build_qdrant import (
    DEFAULT_COLLECTION,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MODEL_CACHE,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TEXT_MODEL,
)
from rag_db.infer_with_rag import (
    QdrantRetriever,
    RagEncoders,
    create_qdrant_client,
    resolve_repository_path,
    validate_collection_schema,
)
from train_lora import (
    DATASET_DIR,
    DATASET_NAME,
    DEFAULT_MIN_PIXELS,
    IMAGE_COMPRESSION_FACTOR,
    MODEL_ID,
    MODEL_MAX_PIXELS,
    set_seed,
)
from type_adapters.modeling import load_base_model_and_processor, release_cuda_memory


LOGGER = logging.getLogger("aivqa.rag_cache_builder")
DEFAULT_MAX_PIXELS = 400 * IMAGE_COMPRESSION_FACTOR**2


@dataclass
class RagResources:
    encoders: RagEncoders
    client: Any
    retriever: QdrantRetriever

    def close(self) -> None:
        self.client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed rag_cache/train.json, validation.json, and test.json "
            "before running the RAG training pipeline."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
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

    qdrant_group = parser.add_mutually_exclusive_group()
    qdrant_group.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    qdrant_group.add_argument("--qdrant-url")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--retrieval-page-size", type=int, default=100)
    parser.add_argument(
        "--rag-device",
        default="cpu",
        help="RAG encoder device; CPU preserves Kanana generation VRAM",
    )
    parser.add_argument("--rag-fp32", action="store_true")
    parser.add_argument("--search-max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("train JSON", args.train_json),
        ("validation JSON", args.validation_json),
        ("test JSON", args.test_json),
    ):
        resolved = resolve_repository_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not args.qdrant_url and not resolve_repository_path(args.qdrant_path).is_dir():
        qdrant_path = resolve_repository_path(args.qdrant_path)
        raise FileNotFoundError(
            f"Qdrant storage does not exist: {qdrant_path}"
        )
    if min(args.max_length, args.search_max_new_tokens, args.retrieval_page_size) < 1:
        raise ValueError(
            "--max-length, --search-max-new-tokens, and --retrieval-page-size "
            "must be at least 1"
        )
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    if args.max_pixels > MODEL_MAX_PIXELS:
        raise ValueError(f"--max-pixels cannot exceed {MODEL_MAX_PIXELS}")


def load_rag_resources(args: argparse.Namespace) -> RagResources:
    model_cache = resolve_repository_path(args.model_cache)
    qdrant_path = None if args.qdrant_url else resolve_repository_path(args.qdrant_path)
    encoders = RagEncoders(
        DEFAULT_TEXT_MODEL,
        DEFAULT_IMAGE_MODEL,
        model_cache,
        args.rag_device,
        args.rag_fp32,
    )
    client = create_qdrant_client(qdrant_path, args.qdrant_url)
    try:
        validate_collection_schema(
            client,
            args.collection,
            encoders.text_dimension,
            encoders.image_dimension,
        )
        retriever = QdrantRetriever(
            client,
            args.collection,
            encoders,
            args.score_threshold,
            args.retrieval_page_size,
            local_image_search=qdrant_path is not None,
        )
    except Exception:
        client.close()
        raise
    return RagResources(encoders, client, retriever)


def build_all_caches(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    dataset_root = resolve_repository_path(args.dataset_root)
    datasets = {
        "train": KananaVQADataset(
            resolve_repository_path(args.train_json), dataset_root=dataset_root
        ),
        "validation": KananaVQADataset(
            resolve_repository_path(args.validation_json), dataset_root=dataset_root
        ),
        "test": KananaVQADataset(
            resolve_repository_path(args.test_json), dataset_root=dataset_root
        ),
    }
    cache_paths = rag_cache_paths(resolve_repository_path(RAG_CACHE_DIR))

    rag = load_rag_resources(args)
    model = processor = None
    try:
        set_seed(args.seed)
        model, processor, dtype = load_base_model_and_processor(
            args, for_training=False
        )
        for split, dataset in datasets.items():
            cache_path = cache_paths[split]
            LOGGER.info(
                "Building %s RAG cache (%d samples): %s",
                split,
                len(dataset),
                cache_path,
            )
            retrieve_dataset_candidates(
                model,
                processor,
                dataset,
                rag.retriever,
                max_length=args.max_length,
                search_max_new_tokens=args.search_max_new_tokens,
                dtype=dtype,
                description=f"{split} RAG retrieval (Base Kanana)",
                cache_path=cache_path,
            )
        return cache_paths
    finally:
        del processor, model
        release_cuda_memory()
        rag.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    paths = build_all_caches(parse_args())
    for split, path in paths.items():
        LOGGER.info("Completed %s RAG cache: %s", split, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
