"""Run two-pass Kanana-V inference over the AIVQA test set with Qdrant RAG."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel

from aivqa.data import KananaVQADataset
from train_lora import (
    DATASET_DIR,
    DATASET_NAME,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    MODEL_ID,
    MODEL_MAX_PIXELS,
    _model_input_device,
    _move_batch_to_device,
    save_test_predictions,
    set_seed,
)
from type_adapters.modeling import (
    load_base_model_and_processor,
    validate_adapter_checkpoint,
)

from .build_qdrant import (
    DEFAULT_COLLECTION,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MODEL_CACHE,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TEXT_MODEL,
    REPOSITORY_ROOT,
    deterministic_point_id,
)
from .prompts import Candidate, build_answer_feature, build_search_feature


LOGGER = logging.getLogger("aivqa.rag_inference")
DEFAULT_ADAPTER_DIR = Path(
    "outputs/kanana_1_5_v_3b_lora/run_20260807_183229/best_adapter"
)
DEFAULT_OUTPUT = (
    Path("outputs/kanana_1_5_v_3b_rag")
    / f"{DATASET_NAME}_test_predictions.json"
)
TEXT_VECTOR_NAME = "text"
IMAGE_VECTOR_NAME = "image"
REQUIRED_PAYLOAD_FIELDS = {
    "doc_id",
    "source",
    "title",
    "search_terms",
    "description",
    "image_path",
}

def resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def normalize_exact_text(value: Any) -> str:
    """Apply only Unicode and whitespace normalization for title exact match."""
    normalized = unicodedata.normalize("NFC", str(value)).strip()
    return " ".join(normalized.split())


def _flatten_search_term_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        terms: list[str] = []
        for item in value:
            terms.extend(_flatten_search_term_value(item))
        return terms
    if isinstance(value, dict):
        preferred_keys = {"search_terms", "searchterms", "terms", "keywords", "검색어"}
        preferred_values = [
            item
            for key, item in value.items()
            if normalize_exact_text(key).casefold().replace(" ", "_") in preferred_keys
        ]
        values = preferred_values or list(value.values())
        terms = []
        for item in values:
            terms.extend(_flatten_search_term_value(item))
        return terms
    return []


def _parse_loose_search_terms(text: str) -> list[str]:
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        text = text[start + 1 : end]
    else:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            text = text[start + 1 : end]

    terms: list[str] = []
    for fragment in re.split(r"[,;|\n]+", text):
        fragment = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", fragment).strip()
        if ":" in fragment or "：" in fragment:
            fragment = re.split(r"[:：]", fragment, maxsplit=1)[1]
        fragment = fragment.strip().strip("\"'`[]{} ")
        if fragment:
            terms.append(fragment)
    return terms


def parse_search_terms(generated_text: str, maximum: int = 5) -> list[str]:
    """Recover search terms from JSON and common non-JSON Kanana responses."""
    if maximum < 1:
        return []
    text = generated_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    candidates = [text]
    for opening, closing in (("[", "]"), ("{", "}")):
        start = text.find(opening)
        end = text.rfind(closing)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])

    raw_terms: list[str] | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        raw_terms = _flatten_search_term_value(value)
        break

    if raw_terms is None:
        raw_terms = _parse_loose_search_terms(text)

    terms: list[str] = []
    seen: set[str] = set()
    for item in raw_terms:
        term = item.strip()
        key = normalize_exact_text(term)
        if not key or key in seen:
            continue
        terms.append(term)
        seen.add(key)
        if len(terms) == maximum:
            break
    if not terms and text not in {"", "[]", "{}"}:
        LOGGER.warning("Could not recover Kanana search terms from: %r", generated_text)
    return terms


def validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Qdrant result is missing its payload")
    missing = REQUIRED_PAYLOAD_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"Qdrant payload is missing fields: {sorted(missing)}")
    if not isinstance(payload["doc_id"], str) or not payload["doc_id"].strip():
        raise ValueError("Qdrant payload doc_id must be a non-empty string")
    return payload


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def validate_collection_schema(
    client: QdrantClient,
    collection_name: str,
    text_dimension: int,
    image_dimension: int,
) -> None:
    """Fail before inference when the completed DB differs from the build schema."""
    if not client.collection_exists(collection_name):
        raise ValueError(f"Qdrant collection does not exist: {collection_name}")
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict):
        raise ValueError("Qdrant collection does not use named vectors")
    if set(vectors) != {TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME}:
        raise ValueError(f"Unexpected named vectors: {sorted(vectors)}")

    text_config = vectors[TEXT_VECTOR_NAME]
    image_config = vectors[IMAGE_VECTOR_NAME]
    if int(text_config.size) != text_dimension:
        raise ValueError(
            f"Text dimension mismatch: collection={text_config.size}, model={text_dimension}"
        )
    if int(image_config.size) != image_dimension:
        raise ValueError(
            f"Image dimension mismatch: collection={image_config.size}, model={image_dimension}"
        )
    if _enum_value(text_config.distance) != "cosine":
        raise ValueError("The text vector distance must be cosine")
    if _enum_value(image_config.distance) != "cosine":
        raise ValueError("The image vector distance must be cosine")
    multivector = getattr(image_config, "multivector_config", None)
    if multivector is None or _enum_value(multivector.comparator) != "max_sim":
        raise ValueError("The image vector must be a MAX_SIM multivector")

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        raise ValueError(f"Qdrant collection is empty: {collection_name}")
    validate_payload(points[0].payload)


def create_qdrant_client(qdrant_path: Path | None, qdrant_url: str | None) -> QdrantClient:
    if qdrant_url:
        return QdrantClient(
            url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=120,
        )
    assert qdrant_path is not None
    if not qdrant_path.is_dir():
        raise FileNotFoundError(f"Qdrant storage does not exist: {qdrant_path}")
    return QdrantClient(path=str(qdrant_path))


def load_title_index(
    client: QdrantClient, collection_name: str, page_size: int = 1024
) -> dict[str, list[str]]:
    """Load only title/doc_id fields once, avoiding description corpus duplication."""
    title_index: dict[str, list[str]] = {}
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=["doc_id", "title"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload
            if not isinstance(payload, dict):
                raise ValueError("Qdrant title scroll result is missing its payload")
            doc_id = payload.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Qdrant title scroll result has an invalid doc_id")
            title_key = normalize_exact_text(payload.get("title", ""))
            if title_key:
                title_index.setdefault(title_key, []).append(doc_id)
        if offset is None:
            break
    return title_index


class RagEncoders:
    def __init__(
        self,
        text_model_name: str,
        image_model_name: str,
        cache_dir: Path,
        device_name: str,
        fp32: bool,
    ) -> None:
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for RAG encoders, but it is unavailable")
        self.use_fp16 = self.device.type == "cuda" and not fp32
        cache_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Loading RAG text encoder: %s", text_model_name)
        self.text_model = SentenceTransformer(
            text_model_name,
            device=str(self.device),
            cache_folder=str(cache_dir),
        )
        LOGGER.info("Loading RAG image encoder: %s", image_model_name)
        self.image_processor = AutoProcessor.from_pretrained(
            image_model_name, cache_dir=str(cache_dir)
        )
        self.image_model = CLIPModel.from_pretrained(
            image_model_name, cache_dir=str(cache_dir)
        ).to(self.device)
        if self.use_fp16:
            self.text_model.half()
            self.image_model.half()
        self.text_model.eval()
        self.image_model.eval()

    @property
    def text_dimension(self) -> int:
        dimension = self.text_model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine KURE embedding dimension")
        return int(dimension)

    @property
    def image_dimension(self) -> int:
        return int(self.image_model.config.projection_dim)

    def embed_text(self, text: str) -> list[float]:
        vector = self.text_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32).tolist()

    def embed_image(self, image: Image.Image) -> list[float]:
        inputs = self.image_processor(images=[image], return_tensors="pt")
        model_dtype = next(self.image_model.parameters()).dtype
        inputs = {
            key: value.to(
                self.device,
                dtype=model_dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output = self.image_model.get_image_features(**inputs)
            features = output if isinstance(output, torch.Tensor) else output.pooler_output
            features = F.normalize(features, p=2, dim=-1)
        return features[0].float().cpu().tolist()


@dataclass
class LocalImageIndex:
    """Compact fallback for Qdrant local mode's missing-multivector bug."""

    payloads: list[dict[str, Any]]
    vectors: np.ndarray
    owners: np.ndarray

    @classmethod
    def load(
        cls,
        client: QdrantClient,
        collection_name: str,
        page_size: int = 256,
    ) -> LocalImageIndex:
        payloads: list[dict[str, Any]] = []
        matrices: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        offset: Any = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[models.HasVectorCondition(has_vector=IMAGE_VECTOR_NAME)]
                ),
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=[IMAGE_VECTOR_NAME],
            )
            for point in points:
                payload = validate_payload(point.payload)
                named_vectors = point.vector
                if not isinstance(named_vectors, dict):
                    raise ValueError("Qdrant image point does not use named vectors")
                matrix = np.asarray(named_vectors.get(IMAGE_VECTOR_NAME), dtype=np.float32)
                if matrix.ndim != 2 or matrix.shape[0] == 0:
                    raise ValueError(
                        f"Qdrant point {payload['doc_id']} has an invalid image multivector"
                    )
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError(
                        f"Qdrant point {payload['doc_id']} has a zero image vector"
                    )
                matrix = matrix / norms
                owner = len(payloads)
                payloads.append(payload)
                matrices.append(matrix)
                owners.append(np.full(matrix.shape[0], owner, dtype=np.int32))
            if offset is None:
                break

        if not matrices:
            raise ValueError("Qdrant collection contains no image vectors")
        vectors = np.concatenate(matrices, axis=0)
        owner_ids = np.concatenate(owners)
        LOGGER.info(
            "Loaded local image search matrix: entities=%d, images=%d, dimension=%d",
            len(payloads),
            vectors.shape[0],
            vectors.shape[1],
        )
        return cls(payloads=payloads, vectors=vectors, owners=owner_ids)

    def search(
        self, query_vector: Sequence[float], score_threshold: float
    ) -> list[tuple[dict[str, Any], float]]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Image query shape {query.shape} does not match {self.vectors.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise ValueError("Image query embedding is a zero vector")
        similarities = self.vectors @ (query / norm)
        entity_scores = np.full(len(self.payloads), -np.inf, dtype=np.float32)
        # With one query vector, Qdrant MAX_SIM is the maximum cosine score
        # among all image vectors stored for the entity.
        np.maximum.at(entity_scores, self.owners, similarities)
        accepted = np.flatnonzero(entity_scores >= score_threshold)
        return [
            (self.payloads[index], float(entity_scores[index]))
            for index in accepted
        ]


class QdrantRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        encoders: RagEncoders,
        threshold: float,
        retrieval_page_size: int,
        local_image_search: bool = False,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.encoders = encoders
        self.threshold = threshold
        self.retrieval_page_size = retrieval_page_size
        self.title_index = load_title_index(client, collection_name)
        self.local_image_index = (
            LocalImageIndex.load(client, collection_name)
            if local_image_search
            else None
        )

    def _retrieve_payloads(self, doc_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[deterministic_point_id(doc_id) for doc_id in doc_ids],
            with_payload=True,
            with_vectors=False,
        )
        payloads = [validate_payload(point.payload) for point in points]
        returned_ids = {payload["doc_id"] for payload in payloads}
        missing = set(doc_ids) - returned_ids
        if missing:
            raise ValueError(f"Exact-title Qdrant points could not be retrieved: {missing}")
        return payloads

    def _query_all(
        self,
        query: list[Any],
        vector_name: str,
        *,
        require_vector: bool = False,
    ) -> list[Any]:
        """Page through every result accepted by the modality threshold."""
        points: list[Any] = []
        offset = 0
        while True:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query,
                using=vector_name,
                query_filter=(
                    models.Filter(
                        must=[models.HasVectorCondition(has_vector=vector_name)]
                    )
                    if require_vector
                    else None
                ),
                score_threshold=self.threshold,
                limit=self.retrieval_page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            page = list(response.points)
            points.extend(page)
            if len(page) < self.retrieval_page_size:
                break
            offset += len(page)
        return points

    @staticmethod
    def _merge(
        candidates: dict[str, Candidate],
        payload: dict[str, Any],
        *,
        text_score: float = 0.0,
        image_score: float = 0.0,
    ) -> None:
        doc_id = payload["doc_id"]
        candidate = candidates.setdefault(doc_id, Candidate(doc_id, payload))
        candidate.text_score = max(candidate.text_score, float(text_score))
        candidate.image_score = max(candidate.image_score, float(image_score))

    def retrieve(
        self, search_terms: Sequence[str], image: Image.Image
    ) -> list[Candidate]:
        candidates: dict[str, Candidate] = {}
        for term in search_terms:
            exact_doc_ids = self.title_index.get(normalize_exact_text(term), [])
            if exact_doc_ids:
                for payload in self._retrieve_payloads(exact_doc_ids):
                    self._merge(candidates, payload, text_score=2.0)
                continue

            for point in self._query_all(
                self.encoders.embed_text(term), TEXT_VECTOR_NAME
            ):
                self._merge(
                    candidates,
                    validate_payload(point.payload),
                    text_score=float(point.score),
                )

        image_vector = self.encoders.embed_image(image)
        if self.local_image_index is not None:
            for payload, score in self.local_image_index.search(
                image_vector, self.threshold
            ):
                self._merge(candidates, payload, image_score=score)
        else:
            for point in self._query_all(
                [image_vector], IMAGE_VECTOR_NAME, require_vector=True
            ):
                self._merge(
                    candidates,
                    validate_payload(point.payload),
                    image_score=float(point.score),
                )

        return sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.final_score, candidate.doc_id),
        )[:3]


def truncate_kanana_encoding(
    text_encoding: dict[str, Any],
    max_length: int,
    generation_suffix_length: int,
) -> bool:
    """Trim an encoded Kanana prompt without removing its image-token block."""
    input_ids = text_encoding.get("input_ids")
    attention_mask = text_encoding.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise TypeError("Kanana text encoding must contain one-dimensional input_ids")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise TypeError("Kanana text encoding must contain an aligned attention_mask")

    input_length = int(input_ids.numel())
    if input_length <= max_length:
        return False

    suffix_length = min(max(generation_suffix_length, 1), input_length, max_length)
    suffix_start = input_length - suffix_length
    media_positions = torch.nonzero(input_ids < 0, as_tuple=False).flatten()

    if media_positions.numel() == 0:
        keep_indices = list(range(max_length - suffix_length)) + list(
            range(suffix_start, input_length)
        )
    else:
        media_start = int(media_positions[0].item())
        media_end = int(media_positions[-1].item()) + 1
        if media_end > suffix_start:
            raise ValueError("Kanana image tokens overlap the generation prompt")

        media_length = media_end - media_start
        fixed_length = media_length + suffix_length
        if fixed_length > max_length:
            raise ValueError(
                "Image tokens alone exceed --max-length; reduce --max-pixels or "
                "increase --max-length"
            )

        text_budget = max_length - fixed_length
        before_length = media_start
        after_length = suffix_start - media_end
        if before_length + after_length <= text_budget:
            before_keep = before_length
            after_keep = after_length
        else:
            # Keep the end of the system prefix and the beginning of the user
            # prompt. The latter preserves the question before trailing RAG text.
            before_keep = min(before_length, text_budget // 2)
            after_keep = min(after_length, text_budget - before_keep)
            remaining = text_budget - before_keep - after_keep
            extra_before = min(before_length - before_keep, remaining)
            before_keep += extra_before
            remaining -= extra_before
            after_keep += min(after_length - after_keep, remaining)

        keep_indices = (
            list(range(media_start - before_keep, media_start))
            + list(range(media_start, media_end))
            + list(range(media_end, media_end + after_keep))
            + list(range(suffix_start, input_length))
        )

    index_tensor = torch.tensor(keep_indices, dtype=torch.long, device=input_ids.device)
    text_encoding["input_ids"] = input_ids.index_select(0, index_tensor)
    text_encoding["attention_mask"] = attention_mask.index_select(0, index_tensor)
    text_encoding["seq_length"] = len(keep_indices)
    return True


def collate_generation_feature(
    processor: Any,
    feature: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    """Encode once without Kanana's pre-truncation assertion, then trim safely."""
    image = feature.get("image")
    conversation = feature.get("conversation")
    if not isinstance(image, Image.Image):
        raise ValueError("Generation feature must contain one in-memory PIL image")
    if not isinstance(conversation, list):
        raise ValueError("Generation feature must contain a conversation list")

    encoded = processor.encode(
        {"image": [image], "conv": conversation},
        max_length=None,
        add_generation_prompt=True,
    )
    suffix_ids = processor.tokenizer("\nAI: ", add_special_tokens=False)["input_ids"]
    original_length = int(encoded["text"]["input_ids"].numel())
    was_truncated = truncate_kanana_encoding(
        encoded["text"], max_length, len(suffix_ids)
    )
    if was_truncated:
        LOGGER.warning(
            "Truncated Kanana input from %d to %d tokens (--max-length)",
            original_length,
            max_length,
        )
    return processor.collate(
        [encoded],
        padding="longest",
        padding_side="left",
        max_length=max_length,
    )


def generate_one(
    model: Any,
    processor: Any,
    feature: dict[str, Any],
    max_length: int,
    max_new_tokens: int,
    dtype: Any,
) -> str:
    model.eval()
    device = _model_input_device(model)
    batch = _move_batch_to_device(
        collate_generation_feature(processor, feature, max_length), device
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        generated_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.batch_decode(
        generated_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def load_kanana_with_adapter(
    args: argparse.Namespace, adapter_dir: str | Path | None = None
) -> tuple[Any, Any, Any]:
    model, processor, dtype = load_base_model_and_processor(args, for_training=False)
    model = attach_kanana_adapter(model, args, adapter_dir)
    return model, processor, dtype


def attach_kanana_adapter(
    model: Any,
    args: argparse.Namespace,
    adapter_dir: str | Path | None = None,
) -> Any:
    """Attach an answer adapter after Base Kanana has generated search queries."""
    from peft import PeftModel

    requested_adapter = args.adapter_dir if adapter_dir is None else adapter_dir
    resolved_adapter = validate_adapter_checkpoint(
        resolve_repository_path(requested_adapter)
    )
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        str(resolved_adapter),
        is_trainable=False,
    )
    model.requires_grad_(False)
    model.eval()
    LOGGER.info("Attached frozen LoRA adapter: %s", resolved_adapter)
    return model


def read_test_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Test JSON root must be a list of objects")
    return value


def run_inference(args: argparse.Namespace) -> list[str]:
    test_json = resolve_repository_path(args.test_json)
    dataset_root = resolve_repository_path(args.dataset_root)
    model_cache = resolve_repository_path(args.model_cache)
    qdrant_path = None if args.qdrant_url else resolve_repository_path(args.qdrant_path)

    records = read_test_records(test_json)
    dataset = KananaVQADataset(test_json, dataset_root=dataset_root)
    if len(records) != len(dataset):
        raise RuntimeError("Test record and dataset lengths differ")
    model, processor, dtype = load_base_model_and_processor(args, for_training=False)
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
        all_candidates: list[list[Candidate]] = []
        for index in tqdm(
            range(len(dataset)), desc="Base Kanana RAG retrieval", unit="sample"
        ):
            sample = dataset[index]
            model_input = records[index]["model_input"]
            question = str(model_input["question"])
            search_output = generate_one(
                model,
                processor,
                build_search_feature(sample, question),
                args.max_length,
                args.search_max_new_tokens,
                dtype,
            )
            search_terms = parse_search_terms(search_output)
            candidates = retriever.retrieve(search_terms, sample["image"])
            all_candidates.append(candidates)
            LOGGER.info(
                "sample=%s search_terms=%s candidates=%s",
                sample["question_id"],
                search_terms,
                [
                    {
                        "doc_id": item.doc_id,
                        "text": round(item.text_score, 4),
                        "image": round(item.image_score, 4),
                        "final": round(item.final_score, 4),
                    }
                    for item in candidates
                ],
            )

        model = attach_kanana_adapter(model, args)
        predictions: list[str] = []
        for index in tqdm(
            range(len(dataset)), desc="Adapter RAG answer generation", unit="sample"
        ):
            sample = dataset[index]
            model_input = records[index]["model_input"]
            question = str(model_input["question"])
            options = model_input.get("options", [])
            if not isinstance(options, list):
                raise ValueError(f"Sample {index}: model_input.options must be a list")
            predictions.append(
                generate_one(
                    model,
                    processor,
                    build_answer_feature(
                        sample,
                        question,
                        options,
                        all_candidates[index],
                    ),
                    args.max_length,
                    args.answer_max_new_tokens,
                    dtype,
                )
            )
    finally:
        client.close()
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shared-LoRA Kanana-V test inference with text/image Qdrant RAG."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--test-json", type=Path, default=DATASET_DIR / f"{DATASET_NAME}_test.json"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    qdrant_group = parser.add_mutually_exclusive_group()
    qdrant_group.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    qdrant_group.add_argument("--qdrant-url")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--retrieval-page-size", type=int, default=100)
    parser.add_argument("--rag-device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--rag-fp32", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--search-max-new-tokens", type=int, default=128)
    parser.add_argument("--answer-max-new-tokens", type=int, default=128)
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
    args = parser.parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be between 0 and 1")
    if args.retrieval_page_size < 1:
        parser.error("--retrieval-page-size must be at least 1")
    if min(args.max_length, args.search_max_new_tokens, args.answer_max_new_tokens) < 1:
        parser.error("generation lengths must be at least 1")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        parser.error("expected 0 < min-pixels <= max-pixels")
    if args.max_pixels > MODEL_MAX_PIXELS:
        parser.error(f"--max-pixels cannot exceed {MODEL_MAX_PIXELS}")
    return args


def validate_input_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("test JSON", args.test_json),
        ("adapter", args.adapter_dir),
    ):
        resolved = resolve_repository_path(path)
        expected = resolved.is_dir() if label == "adapter" else resolved.is_file()
        if not expected:
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
    output = resolve_repository_path(args.output_path)
    if output == resolve_repository_path(args.test_json):
        raise ValueError("Output path must not overwrite the source test JSON")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    validate_input_paths(args)
    set_seed(args.seed)
    predictions = run_inference(args)
    output_path = resolve_repository_path(args.output_path)
    save_test_predictions(
        resolve_repository_path(args.test_json), predictions, output_path
    )
    LOGGER.info("Saved %d predictions: %s", len(predictions), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
