"""Token-level length constraints for short-answer generation."""

from __future__ import annotations

import re
from typing import Any, TypeAlias

try:
    import torch
    from transformers import LogitsProcessor
except ImportError:  # Keep parsing and lightweight utility imports usable without ML deps.
    torch = None  # type: ignore[assignment]

    class LogitsProcessor:  # type: ignore[no-redef]
        pass


LengthSpec: TypeAlias = tuple[str, int]

_COUNT_RE = re.compile(r"(\d+)\s*(음절|어절)")
_TOKEN_STATS_CACHE: dict[int, tuple[Any, dict[str, torch.Tensor]]] = {}


def parse_sa_length_constraint(question_text: str) -> LengthSpec | None:
    """Parse one explicit ``N음절``/``N어절`` requirement from a question.

    Questions without a length requirement and questions containing multiple
    requirements are intentionally left unconstrained.  A multi-part answer cannot be
    represented safely by one total-length constraint.
    """
    question_text = question_text.split("\n\n[참고 정보", maxsplit=1)[0]
    matches = _COUNT_RE.findall(question_text)
    if len(matches) != 1:
        return None
    count, korean_unit = matches[0]
    unit = "syllable" if korean_unit == "음절" else "eojeol"
    return unit, int(count)


def get_sa_length_constraint(
    question_form: Any, question_text: str
) -> LengthSpec | None:
    """Return a length constraint only for an SA question."""
    if str(question_form).strip().upper() != "SA":
        return None
    return parse_sa_length_constraint(question_text)


def _is_hangul_syllable(character: str) -> bool:
    return "가" <= character <= "힣"


def build_token_stats(tokenizer: Any) -> dict[str, torch.Tensor]:
    """Build and cache vocabulary-wide length statistics for ``tokenizer``."""
    if torch is None:
        raise ImportError("torch and transformers are required for constrained decoding")
    cache_key = id(tokenizer)
    cached = _TOKEN_STATS_CACHE.get(cache_key)
    if cached is not None and cached[0] is tokenizer:
        return cached[1]

    vocab_size = len(tokenizer)
    syllable_count = torch.zeros(vocab_size, dtype=torch.long)
    chunk_count = torch.zeros(vocab_size, dtype=torch.long)
    starts_with_whitespace = torch.zeros(vocab_size, dtype=torch.bool)
    ends_with_whitespace = torch.zeros(vocab_size, dtype=torch.bool)
    whitespace_only = torch.zeros(vocab_size, dtype=torch.bool)

    for start in range(0, vocab_size, 4096):
        token_ids = list(range(start, min(start + 4096, vocab_size)))
        texts = tokenizer.batch_decode(
            [[token_id] for token_id in token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for offset, text in enumerate(texts):
            if not text:
                continue
            token_id = start + offset
            # A byte-level token that decodes to U+FFFD in isolation can complete at
            # most one Hangul syllable. Counting it as one prevents silent overshoot.
            syllable_count[token_id] = (
                1
                if "�" in text
                else sum(_is_hangul_syllable(char) for char in text)
            )
            chunk_count[token_id] = len(text.split())
            starts_with_whitespace[token_id] = text[0].isspace()
            ends_with_whitespace[token_id] = text[-1].isspace()
            whitespace_only[token_id] = not text.strip()

    stats = {
        "syllable_count": syllable_count,
        "chunk_count": chunk_count,
        "starts_with_whitespace": starts_with_whitespace,
        "ends_with_whitespace": ends_with_whitespace,
        "whitespace_only": whitespace_only,
    }
    _TOKEN_STATS_CACHE[cache_key] = (tokenizer, stats)
    return stats


class KoreanLengthLogitsProcessor(LogitsProcessor):
    """Mask next tokens that would violate per-row Korean length constraints.

    ``specs`` must align one-to-one with the generation batch. A ``None`` entry leaves
    that row untouched. This project's Kanana generation path uses ``inputs_embeds`` and
    therefore exposes only newly generated token IDs to logits processors.
    """

    EOJEOL_SAFETY_MAX_STEPS = 40

    def __init__(
        self,
        tokenizer: Any,
        specs: list[LengthSpec | None],
        eos_token_id: int,
    ) -> None:
        if torch is None:
            raise ImportError(
                "torch and transformers are required for constrained decoding"
            )
        self.tokenizer = tokenizer
        self.specs = specs
        self.eos_token_id = eos_token_id

        stats = build_token_stats(tokenizer)
        self.syllable_count = stats["syllable_count"]
        self.chunk_count = stats["chunk_count"]
        self.starts_with_whitespace = stats["starts_with_whitespace"]
        self.ends_with_whitespace = stats["ends_with_whitespace"]
        self.whitespace_only = stats["whitespace_only"]

        self._word_count = [0] * len(specs)
        self._at_boundary = [True] * len(specs)
        self._step_count = [0] * len(specs)
        self._expected_length = 0
        self._device_stats: dict[
            tuple[str, int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[0] != len(self.specs):
            raise RuntimeError(
                "KoreanLengthLogitsProcessor requires one length spec per generation "
                "row and does not support batch-expanding beam search"
            )

        current_length = input_ids.shape[1]
        if current_length != self._expected_length:
            raise RuntimeError(
                "KoreanLengthLogitsProcessor requires generated-only input_ids; "
                f"expected length {self._expected_length}, got {current_length}"
            )
        self._expected_length += 1

        device_key = (scores.device.type, scores.device.index)
        device_stats = self._device_stats.get(device_key)
        if device_stats is None:
            device_stats = (
                self.syllable_count.to(scores.device),
                self.chunk_count.to(scores.device),
                self.starts_with_whitespace.to(scores.device),
            )
            self._device_stats[device_key] = device_stats
        syllable_count, chunk_count, starts_with_whitespace = device_stats

        for row, spec in enumerate(self.specs):
            if spec is None:
                continue
            unit, target = spec

            if unit == "syllable":
                decoded = (
                    self.tokenizer.decode(
                        input_ids[row].tolist(), skip_special_tokens=True
                    )
                    if current_length
                    else ""
                )
                current = sum(_is_hangul_syllable(char) for char in decoded)
                remaining = target - current
                if remaining <= 0:
                    eos_score = scores[row, self.eos_token_id].clone()
                    scores[row, :] = float("-inf")
                    scores[row, self.eos_token_id] = eos_score
                else:
                    scores[row, self.eos_token_id] = float("-inf")
                    scores[row, syllable_count > remaining] = float("-inf")
                continue

            if unit != "eojeol":
                raise ValueError(f"Unsupported Korean length unit: {unit!r}")

            if current_length == 0:
                self._word_count[row] = 0
                self._at_boundary[row] = True
            else:
                last_token_id = int(input_ids[row, -1].item())
                chunks = int(self.chunk_count[last_token_id].item())
                if chunks == 0:
                    if bool(self.whitespace_only[last_token_id].item()):
                        self._at_boundary[row] = True
                else:
                    merges_with_previous = (
                        not self._at_boundary[row]
                        and self._word_count[row] > 0
                        and not bool(
                            self.starts_with_whitespace[last_token_id].item()
                        )
                    )
                    self._word_count[row] += chunks - int(merges_with_previous)
                    self._at_boundary[row] = bool(
                        self.ends_with_whitespace[last_token_id].item()
                    )

            self._step_count[row] += 1
            word_count = self._word_count[row]
            at_boundary = self._at_boundary[row]

            if (
                word_count < target
                and self._step_count[row] > self.EOJEOL_SAFETY_MAX_STEPS
            ):
                eos_score = scores[row, self.eos_token_id].clone()
                scores[row, :] = float("-inf")
                scores[row, self.eos_token_id] = eos_score
                continue

            if word_count < target:
                scores[row, self.eos_token_id] = float("-inf")
            else:
                if not at_boundary and word_count > 0:
                    added_chunks = torch.where(
                        starts_with_whitespace,
                        chunk_count,
                        chunk_count - 1,
                    )
                else:
                    added_chunks = chunk_count
                scores[row, added_chunks > 0] = float("-inf")

        return scores
