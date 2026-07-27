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


def _normalize_mc_answer(text: str) -> str:
    normalized = _normalize_text(text)
    match = re.search(r"(?<!\d)([1-5])(?!\d)", normalized)
    return match.group(1) if match else normalized


def _rouge_l_f1(prediction: str, reference: str) -> float:
    predicted_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    previous = [0] * (len(reference_tokens) + 1)
    for predicted_token in predicted_tokens:
        current = [0]
        for column, reference_token in enumerate(reference_tokens, start=1):
            if predicted_token == reference_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current

    lcs_length = previous[-1]
    precision = lcs_length / len(predicted_tokens)
    recall = lcs_length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute corpus BLEU-4 with effective order and add-one smoothing."""
    matches_by_order = [0, 0, 0, 0]
    possible_by_order = [0, 0, 0, 0]
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(predictions, references):
        predicted_tokens = _tokenize(prediction)
        reference_tokens = _tokenize(reference)
        prediction_length += len(predicted_tokens)
        reference_length += len(reference_tokens)
        for order in range(1, 5):
            predicted_ngrams = _ngrams(predicted_tokens, order)
            reference_ngrams = _ngrams(reference_tokens, order)
            matches_by_order[order - 1] += sum(
                min(count, reference_ngrams[ngram])
                for ngram, count in predicted_ngrams.items()
            )
            possible_by_order[order - 1] += sum(predicted_ngrams.values())

    if prediction_length == 0:
        return float(reference_length == 0)

    precisions = []
    for matches, possible in zip(matches_by_order, possible_by_order):
        if possible == 0:
            continue
        precisions.append((matches + 1.0) / (possible + 1.0))
    if not precisions:
        return 0.0

    geometric_mean = math.exp(sum(math.log(value) for value in precisions) / len(precisions))
    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * geometric_mean


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
        sum(_rouge_l_f1(prediction, reference) for prediction, reference in la_pairs)
        / len(la_pairs)
        if la_pairs
        else 0.0
    )
    bleu = (
        _corpus_bleu(
            [prediction for prediction, _ in la_pairs],
            [reference for _, reference in la_pairs],
        )
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
