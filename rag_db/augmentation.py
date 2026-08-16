"""Reusable RAG augmentation for training, validation, and generation datasets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .prompts import Candidate, build_answer_feature, build_search_feature


class CombinedVQADataset:
    """Concatenate VQA datasets while exposing records for type filtering."""

    def __init__(self, datasets: Sequence[Any]) -> None:
        if not datasets:
            raise ValueError("CombinedVQADataset requires at least one dataset")
        self.datasets = list(datasets)
        self.offsets: list[int] = []
        self.records: list[Any] = []
        total = 0
        for dataset in self.datasets:
            records = getattr(dataset, "records", None)
            if not isinstance(records, Sequence):
                raise TypeError("Every combined dataset must expose a records sequence")
            if len(records) != len(dataset):
                raise ValueError("Dataset records and sample counts must match")
            total += len(dataset)
            self.offsets.append(total)
            self.records.extend(records)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = 0
        for dataset, stop in zip(self.datasets, self.offsets):
            if index < stop:
                return dataset[index - start]
            start = stop
        raise IndexError(index)


class RagAugmentedDataset:
    """Replace each base prompt with the same sample plus retrieved context."""

    def __init__(
        self,
        dataset: Any,
        candidates: Sequence[Sequence[Candidate]],
        *,
        max_rag_chars: int | None = 2000,
    ) -> None:
        if len(dataset) != len(candidates):
            raise ValueError(
                f"RAG candidate count mismatch: {len(candidates)} for {len(dataset)} samples"
            )
        if max_rag_chars is not None and max_rag_chars < 1:
            raise ValueError("max_rag_chars must be positive when set")
        self.dataset = dataset
        self.candidates = [list(items) for items in candidates]
        self.max_rag_chars = max_rag_chars
        records = getattr(dataset, "records", None)
        if isinstance(records, Sequence):
            self.records = records

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        question = sample.get("question")
        options = sample.get("options", [])
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Sample {index}: RAG augmentation requires a question")
        if not isinstance(options, list):
            raise ValueError(f"Sample {index}: RAG augmentation requires list options")
        feature = build_answer_feature(
            sample,
            question,
            options,
            self.candidates[index],
            max_rag_chars=self.max_rag_chars,
        )
        sample["conversation"] = feature["conversation"]
        sample["rag_candidates"] = self.candidates[index]
        return sample


def retrieve_dataset_candidates(
    model: Any,
    processor: Any,
    dataset: Any,
    retriever: Any,
    *,
    max_length: int,
    search_max_new_tokens: int,
    dtype: Any,
    description: str,
    cache_path: Path | None = None,
) -> list[list[Candidate]]:
    """Run Kanana query generation and text/image retrieval once per sample."""
    from .infer_with_rag import generate_one, parse_search_terms
    from tqdm.auto import tqdm

    all_candidates: list[list[Candidate]] = []
    cache_rows: list[dict[str, Any]] = []
    for index in tqdm(range(len(dataset)), desc=description, unit="sample"):
        sample = dataset[index]
        question = sample.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Sample {index}: RAG retrieval requires a question")
        search_output = generate_one(
            model,
            processor,
            build_search_feature(sample, question),
            max_length,
            search_max_new_tokens,
            dtype,
        )
        search_terms = parse_search_terms(search_output)
        candidates = retriever.retrieve(search_terms, sample["image"])
        all_candidates.append(candidates)
        cache_rows.append(
            {
                "question_id": sample.get("question_id", str(index)),
                "search_terms": search_terms,
                "candidates": [
                    {
                        "doc_id": candidate.doc_id,
                        "text_score": candidate.text_score,
                        "image_score": candidate.image_score,
                        "final_score": candidate.final_score,
                        "payload": candidate.payload,
                    }
                    for candidate in candidates
                ],
            }
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(cache_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
    return all_candidates


def generate_rag_predictions(
    model: Any,
    processor: Any,
    dataset: Any,
    *,
    max_length: int,
    max_new_tokens: int,
    dtype: Any,
    description: str,
) -> list[str]:
    """Generate one answer at a time with image-safe prompt truncation."""
    from .infer_with_rag import generate_one
    from tqdm.auto import tqdm

    return [
        generate_one(
            model,
            processor,
            dataset[index],
            max_length,
            max_new_tokens,
            dtype,
        )
        for index in tqdm(range(len(dataset)), desc=description, unit="sample")
    ]
