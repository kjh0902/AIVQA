"""Dataset conversion and collators for Kanana-V."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageOps


SYSTEM_PROMPT = (
    "이미지와 질문을 주의 깊게 확인한 뒤 정확하게 답하세요. "
    "질문에 지정된 답변 형식과 길이를 따르세요."
)

QUESTION_FORM_LABELS = {
    "MC": "객관식",
    "SA": "단답형",
    "LA": "서술형",
}

QUESTION_FORM_INSTRUCTIONS = {
    "MC": (
        "선택지 번호만 출력하세요. 복수 정답이면 번호를 오름차순으로 정렬하여 "
        "'/'로 구분하세요. 설명이나 다른 문장은 출력하지 마세요."
    ),
    "SA": (
        "질문에 명시된 출력 조건을 모두 지키고 정답만 출력하세요. "
        "설명이나 부가 문장은 출력하지 마세요."
    ),
    "LA": (
        "250자 이내의 한 문단으로 답하세요. 같은 내용을 반복하지 말고, 이미지와 "
        "질문에 필요한 내용만 구체적으로 서술하세요."
    ),
}

_KOREAN_COUNT_WORDS = "한|두|세|네|다섯|여섯|일곱|여덟|아홉|열"
SA_CONSTRAINT_PATTERN = re.compile(
    rf"(?<![\d.가-힣])(?P<count>\d+|{_KOREAN_COUNT_WORDS})"
    r"\s*(?P<unit>음절|어절|가지|개|답)"
)
SA_CONSTRAINT_LABELS = {
    "음절": "음절 수",
    "어절": "어절 수",
    "개": "답변 개수",
    "가지": "답변 개수",
    "답": "답 수",
}
_OUTPUT_DIRECTIVE_PATTERN = re.compile(
    r"(?:답하|작성하|제시하|나열하|열거하|말하|쓰|적|고르|찾)"
    r"(?:여|아|어|으)?(?:\s*주)?(?:시오|세요|라)"
)
_SENTENCE_BOUNDARIES = ".!?。！？\n"


def _sentence_span(text: str, position: int) -> tuple[int, int]:
    start = max(text.rfind(boundary, 0, position) for boundary in _SENTENCE_BOUNDARIES)
    ends = [text.find(boundary, position) for boundary in _SENTENCE_BOUNDARIES]
    valid_ends = [end for end in ends if end >= 0]
    return start + 1, min(valid_ends, default=len(text))


def _is_answer_count_constraint(question: str, match: re.Match[str]) -> bool:
    """Reject object counts unless they are grammatically tied to an answer command."""
    sentence_start, sentence_end = _sentence_span(question, match.start())
    sentence = question[sentence_start:sentence_end]
    relative_end = match.end() - sentence_start
    tail = sentence[relative_end:].lstrip()

    if match.group("unit") == "답" and re.match(r"^하(?:시오|세요|라)", tail):
        return True

    directive = _OUTPUT_DIRECTIVE_PATTERN.search(tail)
    if directive is None:
        return False

    before_directive = tail[: directive.start()].strip(" ,()")
    if not before_directive:
        return True
    if before_directive.startswith(("를", "을", "만", "씩", "로")):
        return True
    # Handles forms such as "3가지 이유를 답하시오" while excluding factual
    # descriptions such as "4개의 그릇은 ... 답하시오".
    return bool(re.fullmatch(r"[가-힣]{1,12}(?:을|를)", before_directive))


def extract_sa_constraints(question: str) -> list[tuple[str, str]]:
    """Extract explicit SA output constraints in their order of appearance."""
    constraints: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in SA_CONSTRAINT_PATTERN.finditer(question):
        unit = match.group("unit")
        if unit in {"개", "가지", "답"} and not _is_answer_count_constraint(
            question, match
        ):
            continue
        count = match.group("count")
        value = f"{count}{unit}" if count.isdigit() else f"{count} {unit}"
        constraint = (SA_CONSTRAINT_LABELS[unit], value)
        if constraint not in seen:
            constraints.append(constraint)
            seen.add(constraint)
    return constraints


def build_sa_instruction(question: str) -> str:
    """Build a per-sample SA instruction containing parsed output constraints."""
    constraints = extract_sa_constraints(question)
    lines = [
        "이 문제는 단답형입니다.",
        "",
        "반드시 질문에서 요구하는 출력 조건을 먼저 확인한 뒤 답하세요.",
    ]
    if constraints:
        lines.extend(f"- 요구 {label}: {value}" for label, value in constraints)
        lines.append("- 최종 답변이 위 조건을 만족하는지 확인한 뒤 출력하세요.")
    else:
        lines.append("- 질문에 별도의 출력 조건이 있다면 모두 준수하세요.")
    lines.extend(
        [
            "",
            "조건을 만족하는 정답만 출력하세요. 설명이나 부가 문장은 출력하지 마세요.",
        ]
    )
    return "\n".join(lines)


def build_question_form_instruction(question_form: str, question: str) -> str:
    """Return the form instruction, including per-question SA constraints."""
    normalized_form = str(question_form).strip().upper()
    if normalized_form not in QUESTION_FORM_INSTRUCTIONS:
        allowed = ", ".join(QUESTION_FORM_INSTRUCTIONS)
        raise ValueError(
            f"Unsupported question_form {question_form!r}; expected one of: {allowed}"
        )
    if normalized_form == "SA":
        return build_sa_instruction(question)
    return QUESTION_FORM_INSTRUCTIONS[normalized_form]


def format_question(question_form: str, question: str, options: Sequence[str]) -> str:
    """Combine the question form, question, and non-empty options."""
    normalized_form = str(question_form).strip().upper()
    if normalized_form not in QUESTION_FORM_LABELS:
        allowed = ", ".join(QUESTION_FORM_LABELS)
        raise ValueError(
            f"Unsupported question_form {question_form!r}; expected one of: {allowed}"
        )

    normalized_question = str(question).strip()
    if not normalized_question:
        raise ValueError("question must be a non-empty string")

    parts = [
        f"질문 유형: {QUESTION_FORM_LABELS[normalized_form]} ({normalized_form})",
        normalized_question,
    ]
    normalized_options = [str(option).strip() for option in options if str(option).strip()]
    if normalized_options:
        parts.append("선택지:\n" + "\n".join(normalized_options))
    return "\n\n".join(parts)


class KananaVQADataset:
    """Load source JSON and build one-image Kanana conversation samples lazily.

    The source file and images are never rewritten. ``dataset_root`` should be
    the directory containing the ``train``, ``validation``, and ``test`` image
    directories. By default it is inferred as the JSON file's grandparent.
    """

    def __init__(
        self,
        json_path: str | Path,
        dataset_root: str | Path | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        check_image_exists: bool = True,
    ) -> None:
        self.json_path = Path(json_path)
        if not self.json_path.is_file():
            raise FileNotFoundError(f"JSON file does not exist: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError("The dataset JSON root must be a list")

        self.records: list[dict[str, Any]] = records
        self.dataset_root = (
            Path(dataset_root) if dataset_root is not None else self.json_path.parent.parent
        )
        self.system_prompt = system_prompt
        self.check_image_exists = check_image_exists

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        metadata = self._require_mapping(record, "metadata", index)
        model_input = self._require_mapping(record, "model_input", index)

        question_form = self._require_text(metadata, "question_form", index).upper()
        if question_form not in QUESTION_FORM_LABELS:
            allowed = ", ".join(QUESTION_FORM_LABELS)
            raise ValueError(
                f"Sample {index}: unsupported question_form {question_form!r}; "
                f"expected one of: {allowed}"
            )
        question = self._require_text(model_input, "question", index)
        options = model_input.get("options", [])
        if not isinstance(options, list):
            raise ValueError(f"Sample {index}: model_input.options must be a list")

        split = self._require_text(metadata, "split", index)
        image_name = self._image_name(model_input, index)
        image_path = self._resolve_image_path(split, image_name)
        if self.check_image_exists and not image_path.is_file():
            raise FileNotFoundError(f"Sample {index}: image does not exist: {image_path}")
        image = self._load_rgb_image(image_path, index)

        formatted_question = format_question(question_form, question, options)
        form_instruction = build_question_form_instruction(question_form, question)
        instruction_prompt = f"{self.system_prompt}\n\n{form_instruction}"
        # This follows the model card's native input format: images are passed
        # separately and each image is represented by one <image> marker in conv.
        conversation = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": formatted_question},
        ]

        model_output = record.get("model_output")
        answer = model_output.get("answer") if isinstance(model_output, dict) else None
        if answer is not None:
            answer = str(answer).strip()

        return {
            "conversation": conversation,
            "image": image,
            "answer": answer,
            "question_id": str(metadata.get("question_id", index)),
            "question_form": question_form,
            "question": question,
            "options": [str(option) for option in options],
            "formatted_question": formatted_question,
            "image_path": str(image_path),
        }

    @staticmethod
    def _require_mapping(
        record: dict[str, Any], key: str, index: int
    ) -> dict[str, Any]:
        value = record.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Sample {index}: {key} must be an object")
        return value

    @staticmethod
    def _require_text(mapping: dict[str, Any], key: str, index: int) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Sample {index}: {key} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _image_name(model_input: dict[str, Any], index: int) -> str:
        for key in ("image_path", "image_name", "image"):
            value = model_input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError(
            f"Sample {index}: model_input must contain image_path, image_name, or image"
        )

    def _resolve_image_path(self, split: str, image_name: str) -> Path:
        image_path = Path(image_name)
        if image_path.is_absolute():
            return image_path

        split_path = self.dataset_root / split / image_path
        direct_path = self.dataset_root / image_path
        if split_path.is_file() or not direct_path.is_file():
            return split_path
        return direct_path

    @staticmethod
    def _load_rgb_image(image_path: Path, index: int) -> Image.Image:
        try:
            with Image.open(image_path) as image_file:
                return ImageOps.exif_transpose(image_file).convert("RGB").copy()
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Sample {index}: Pillow could not decode image: {image_path}"
            ) from error


def _conversation(feature: dict[str, Any]) -> list[dict[str, str]]:
    conversation = feature.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("Each feature must contain a non-empty conversation list")
    if any(message.get("role") == "assistant" for message in conversation):
        raise ValueError("Dataset conversations must contain only the prompt")

    copied: list[dict[str, str]] = []
    for message in conversation:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError("Conversation messages must have a valid role and text content")
        copied.append({"role": role, "content": content})
    return copied


def _processor_sample(
    feature: dict[str, Any], conversation: list[dict[str, str]]
) -> dict[str, Any]:
    image = feature.get("image")
    if not isinstance(image, Image.Image):
        raise ValueError("Each feature must contain one in-memory PIL image")
    return {"image": [image], "conv": conversation}


@dataclass
class TrainCollator:
    """Build a Kanana batch with assistant-only language-model labels."""

    processor: Any
    max_length: int = 2048
    ignore_index: int = -100

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("TrainCollator received an empty batch")

        full_samples: list[dict[str, Any]] = []
        prompt_samples: list[dict[str, Any]] = []
        for feature in features:
            prompt = _conversation(feature)
            answer = feature.get("answer")
            if answer is None or not str(answer).strip():
                question_id = feature.get("question_id", "<unknown>")
                raise ValueError(
                    f"Training sample {question_id} does not contain a non-empty answer"
                )
            prompt_samples.append(_processor_sample(feature, prompt))
            full_samples.append(
                _processor_sample(
                    feature,
                    prompt
                    + [{"role": "assistant", "content": str(answer).strip()}],
                )
            )

        batch = self.processor.batch_encode_collate(
            full_samples,
            padding="longest",
            padding_side="right",
            max_length=self.max_length,
            add_generation_prompt=False,
        )
        prompt_batch = self.processor.batch_encode_collate(
            prompt_samples,
            padding="longest",
            padding_side="right",
            max_length=self.max_length,
            add_generation_prompt=True,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_attention_mask = prompt_batch["attention_mask"]
        labels = input_ids.clone() if hasattr(input_ids, "clone") else input_ids.copy()
        labels[attention_mask == 0] = self.ignore_index
        labels[input_ids < 0] = self.ignore_index

        for row in range(len(features)):
            prompt_length = int(prompt_attention_mask[row].sum().item())
            full_length = int(attention_mask[row].sum().item())
            if prompt_length >= full_length:
                question_id = features[row].get("question_id", "<unknown>")
                raise ValueError(
                    f"Training sample {question_id} has no answer tokens after processing; "
                    "increase --max-length"
                )
            labels[row, :prompt_length] = self.ignore_index

        batch["labels"] = labels
        return batch


@dataclass
class GenerationCollator:
    """Build a left-padded Kanana generation batch without reference answers."""

    processor: Any
    max_length: int = 2048

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("GenerationCollator received an empty batch")
        samples = [
            _processor_sample(feature, _conversation(feature)) for feature in features
        ]
        return self.processor.batch_encode_collate(
            samples,
            padding="longest",
            padding_side="left",
            max_length=self.max_length,
            add_generation_prompt=True,
        )
