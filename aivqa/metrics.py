"""Evaluation metrics for the MC, SA, and LA subsets."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Sequence


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return " ".join(normalized.split())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", _normalize_text(text), flags=re.UNICODE)


def _normalize_mc_answer(text: str) -> tuple[str, ...] | str:
    normalized = _normalize_text(text)
    choices = re.findall(r"(?<!\d)([1-5])(?!\d)", normalized)
    return tuple(sorted(set(choices))) if choices else normalized


def _rouge_1_f1(prediction: str, reference: str) -> float:
    """Compute sentence-level ROUGE-1 F1 from clipped unigram overlap."""
    predicted_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    overlap = sum(
        (Counter(predicted_tokens) & Counter(reference_tokens)).values()
    )
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _sentence_bleu_1(prediction: str, reference: str) -> float:
    """Compute sentence-level BLEU-1 with clipped unigram precision."""
    predicted_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    if not predicted_tokens:
        return float(not reference_tokens)
    if not reference_tokens:
        return 0.0

    overlap = sum(
        (Counter(predicted_tokens) & Counter(reference_tokens)).values()
    )
    unigram_precision = overlap / len(predicted_tokens)
    if unigram_precision == 0.0:
        return 0.0

    if len(predicted_tokens) >= len(reference_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(
            1.0 - len(reference_tokens) / len(predicted_tokens)
        )
    return brevity_penalty * unigram_precision


def compute_vqa_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    question_forms: Sequence[str],
) -> dict[str, float]:
    """Compute normalized [0, 1] task metrics and the aggregate score."""
    if not (len(predictions) == len(references) == len(question_forms)):
        raise ValueError("predictions, references, and question_forms must have equal lengths")

    grouped: dict[str, list[tuple[str, str]]] = {"MC": [], "SA": [], "LA": []}
    for prediction, reference, question_form in zip(
        predictions, references, question_forms
    ):
        normalized_form = str(question_form).strip().upper()
        if normalized_form not in grouped:
            raise ValueError(f"Unsupported question form: {question_form!r}")
        grouped[normalized_form].append((str(prediction), str(reference)))

    mc_pairs = grouped["MC"]
    mc_accuracy = (
        sum(_normalize_mc_answer(prediction) == _normalize_mc_answer(reference) for prediction, reference in mc_pairs)
        / len(mc_pairs)
        if mc_pairs
        else 0.0
    )

    sa_pairs = grouped["SA"]
    sa_exact_match = (
        sum(_normalize_text(prediction) == _normalize_text(reference) for prediction, reference in sa_pairs)
        / len(sa_pairs)
        if sa_pairs
        else 0.0
    )

    la_pairs = grouped["LA"]
    rouge = (
        sum(_rouge_1_f1(prediction, reference) for prediction, reference in la_pairs)
        / len(la_pairs)
        if la_pairs
        else 0.0
    )
    bleu = (
        sum(
            _sentence_bleu_1(prediction, reference)
            for prediction, reference in la_pairs
        )
        / len(la_pairs)
        if la_pairs
        else 0.0
    )

    descriptive_avg = (rouge + bleu) / 2.0
    final_score = (mc_accuracy + sa_exact_match + descriptive_avg) / 3.0
    return {
        "mc_accuracy": float(mc_accuracy),
        "sa_exact_match": float(sa_exact_match),
        "rouge": float(rouge),
        "bleu": float(bleu),
        "descriptive_avg": float(descriptive_avg),
        "final_score": float(final_score),
    }
