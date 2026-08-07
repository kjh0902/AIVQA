"""Lazy MC/SA/LA views over the unchanged AIVQA JSON datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


QUESTION_FORMS = ("MC", "SA", "LA")


def normalize_question_form(question_form: str) -> str:
    normalized = str(question_form).strip().upper()
    if normalized not in QUESTION_FORMS:
        raise ValueError(
            f"Unsupported question form {question_form!r}; expected one of "
            f"{', '.join(QUESTION_FORMS)}"
        )
    return normalized


def _record_question_form(record: Any, index: int) -> str:
    if not isinstance(record, Mapping):
        raise ValueError(f"Sample {index}: record must be an object")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Sample {index}: metadata must be an object")
    question_form = metadata.get("question_form")
    if not isinstance(question_form, str) or not question_form.strip():
        raise ValueError(f"Sample {index}: metadata.question_form must be non-empty")
    return normalize_question_form(question_form)


class QuestionFormSubset:
    """Index-preserving lazy subset for exactly one question form."""

    def __init__(
        self,
        dataset: Any,
        question_form: str,
        *,
        require_non_empty: bool = True,
    ) -> None:
        records = getattr(dataset, "records", None)
        if not isinstance(records, Sequence):
            raise TypeError("The wrapped dataset must expose a records sequence")

        self.dataset = dataset
        self.question_form = normalize_question_form(question_form)
        self.indices = [
            index
            for index, record in enumerate(records)
            if _record_question_form(record, index) == self.question_form
        ]
        if require_non_empty and not self.indices:
            raise ValueError(f"Dataset contains no {self.question_form} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        sample = dict(self.dataset[source_index])
        sample["source_index"] = source_index
        return sample


def build_type_subsets(
    dataset: Any, *, require_non_empty: bool = True
) -> dict[str, QuestionFormSubset]:
    return {
        question_form: QuestionFormSubset(
            dataset,
            question_form,
            require_non_empty=require_non_empty,
        )
        for question_form in QUESTION_FORMS
    }


def restore_original_order(
    subsets: Mapping[str, QuestionFormSubset],
    grouped_predictions: Mapping[str, Sequence[str]],
    source_length: int,
) -> list[str]:
    """Restore type-batched predictions to the exact source JSON order."""
    if source_length < 0:
        raise ValueError("source_length cannot be negative")

    ordered: list[str | None] = [None] * source_length
    for question_form in QUESTION_FORMS:
        if question_form not in subsets or question_form not in grouped_predictions:
            raise ValueError(f"Missing subset or predictions for {question_form}")
        subset = subsets[question_form]
        predictions = grouped_predictions[question_form]
        if len(subset) != len(predictions):
            raise ValueError(
                f"{question_form} prediction count mismatch: "
                f"{len(predictions)} predictions for {len(subset)} samples"
            )
        for source_index, prediction in zip(subset.indices, predictions):
            if not 0 <= source_index < source_length:
                raise ValueError(f"Source index out of range: {source_index}")
            if ordered[source_index] is not None:
                raise ValueError(f"Duplicate prediction for source index {source_index}")
            ordered[source_index] = str(prediction)

    missing = [index for index, prediction in enumerate(ordered) if prediction is None]
    if missing:
        raise ValueError(f"Predictions are missing for source indices: {missing[:10]}")
    return [str(prediction) for prediction in ordered]
