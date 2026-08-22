"""Parse, validate, and retry SA output-format requirements."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


Mode = Literal["none", "contains", "only"]
LengthUnit = Literal["syllable", "eojeol"]

_COUNT_WORDS = {
    "한": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}
_COUNT_TOKEN = r"\d+|" + "|".join(_COUNT_WORDS)
_LENGTH_RE = re.compile(
    rf"(?<![\d.가-힣])(?P<count>{_COUNT_TOKEN})\s*(?P<unit>음절|어절)"
)
_OUTPUT_VERB = r"(?:답하|작성하|쓰|적|기입하|제시하|나열하)"
_CJK_RANGES = (
    ("\u3400", "\u4dbf"),
    ("\u4e00", "\u9fff"),
    ("\uf900", "\ufaff"),
)


@dataclass(frozen=True)
class LengthRequirement:
    unit: LengthUnit
    count: int

    @property
    def korean_unit(self) -> str:
        return "음절" if self.unit == "syllable" else "어절"


@dataclass(frozen=True)
class SAFormatSpec:
    """Structured format constraints parsed from one SA question."""

    length_groups: tuple[tuple[LengthRequirement, ...], ...] = ()
    numeric_mode: Mode = "none"
    hanja_mode: Mode = "none"
    english_mode: Mode = "none"
    require_hangul: bool = False
    unit_mode: Literal["none", "include", "exclude"] = "none"
    unit_text: str | None = None
    answer_count: int | None = None

    @property
    def is_free(self) -> bool:
        return not any(
            (
                self.length_groups,
                self.numeric_mode != "none",
                self.hanja_mode != "none",
                self.english_mode != "none",
                self.require_hangul,
                self.unit_mode != "none",
                self.answer_count is not None,
            )
        )


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()


def _parse_count(value: str) -> int:
    return int(value) if value.isdigit() else _COUNT_WORDS[value]


def _parse_length_groups(question: str) -> tuple[tuple[LengthRequirement, ...], ...]:
    matches = list(_LENGTH_RE.finditer(question))
    if not matches:
        return ()
    requirements = [
        LengthRequirement(
            "syllable" if match.group("unit") == "음절" else "eojeol",
            _parse_count(match.group("count")),
        )
        for match in matches
    ]
    if len(requirements) == 1:
        return ((requirements[0],),)

    between = [
        question[first.end() : second.start()]
        for first, second in zip(matches, matches[1:])
    ]
    sequential = bool(re.search(r"각각|차례대로|순서대로", question)) or any(
        re.search(r"(?:과|와|,|그리고)", text) for text in between
    )
    if sequential:
        groups = [(requirement,) for requirement in requirements]
        before_first = question[: matches[0].start()]
        if (
            "각각" in before_first
            and len(groups) > 1
            and not re.match(r"\s*(?:과|와)", between[0])
        ):
            # "명칭과 재료는 각각 2음절, 지역은 3음절" assigns the first
            # requirement to two answers before moving to the next requirement.
            groups.insert(0, groups[0])
        return tuple(groups)
    return (tuple(requirements),)


def _parse_answer_count(question: str, length_group_count: int) -> int | None:
    patterns = (
        re.compile(
            rf"(?:정답|답|이름|명칭|이칭)\s*(?P<count>{_COUNT_TOKEN})\s*"
            r"(?:개|가지)"
        ),
        re.compile(
            rf"(?P<count>{_COUNT_TOKEN})\s*개의?\s*"
            r"(?:정답|답|이름|명칭|이칭)"
        ),
        re.compile(
            rf"(?P<count>{_COUNT_TOKEN})\s*(?:개|가지|답)(?:를|을)?"
            rf"[^.!?\n]{{0,40}}?{_OUTPUT_VERB}"
        ),
        re.compile(rf"(?P<count>{_COUNT_TOKEN})\s*답하"),
    )
    for pattern in patterns:
        match = pattern.search(question)
        if match:
            return _parse_count(match.group("count"))
    if length_group_count > 1:
        return length_group_count
    if length_group_count == 1 and re.search(r"각각|차례대로|순서대로", question):
        return 2
    return None


def _parse_script_mode(question: str, label_pattern: str) -> Mode:
    if re.search(
        rf"(?:{label_pattern})(?:\s*알파벳)?(?:으로만|만으로|만)\s*{_OUTPUT_VERB}",
        question,
    ):
        return "only"
    if re.search(
        rf"(?:{label_pattern})(?:\s*알파벳)?으로\s*{_OUTPUT_VERB}",
        question,
    ):
        return "only"
    if re.search(rf"(?:{label_pattern})(?:를|을)?\s*포함", question):
        return "contains"
    return "none"


def _parse_hanja_mode(question: str) -> Mode:
    if re.search(rf"한자(?:로만|만으로|만|로)\s*{_OUTPUT_VERB}", question):
        return "only"
    if re.search(r"한자(?:를|가)?\s*포함", question):
        return "contains"
    return "none"


def _parse_numeric_mode(question: str) -> Mode:
    if re.search(r"한글\s*(?:과|와)\s*숫자", question):
        return "contains"
    if re.search(r"숫자(?:를|가)?\s*포함", question):
        return "contains"
    if re.search(
        rf"숫자(?:로만|만으로|만|로|를)\s*{_OUTPUT_VERB}", question
    ):
        return "only"
    return "none"


def _parse_unit_requirement(
    question: str,
) -> tuple[Literal["none", "include", "exclude"], str | None]:
    if re.search(r"단위\s*(?:없이|제외)", question):
        return "exclude", None
    include = re.search(
        r"단위(?:를|까지)?\s*(?:포함|붙여|써서)|단위(?:와|랑)\s*함께",
        question,
    )
    if include is None:
        return "none", None
    unit_match = re.search(
        r"(?P<unit>km/h|km|kg|cm|mm|m|ml|L|%|원|만원|개월|년|월|일|"
        r"시간|분|초|명|개|회|번|층|호선)\s*단위",
        question,
        flags=re.IGNORECASE,
    )
    return "include", unit_match.group("unit") if unit_match else None


def parse_sa_format(question: str) -> SAFormatSpec:
    """Parse deterministic output-format requirements; no match means FREE."""
    length_groups = _parse_length_groups(question)
    numeric_mode = _parse_numeric_mode(question)
    hanja_mode = _parse_hanja_mode(question)
    english_mode = _parse_script_mode(question, r"영문|영어|알파벳")
    unit_mode, unit_text = _parse_unit_requirement(question)
    return SAFormatSpec(
        length_groups=length_groups,
        numeric_mode=numeric_mode,
        hanja_mode=hanja_mode,
        english_mode=english_mode,
        require_hangul=bool(re.search(r"한글\s*(?:과|와)\s*숫자", question)),
        unit_mode=unit_mode,
        unit_text=unit_text,
        answer_count=_parse_answer_count(question, len(length_groups)),
    )


def _is_hangul_syllable(character: str) -> bool:
    return "가" <= character <= "힣"


def _is_hanja(character: str) -> bool:
    return any(start <= character <= end for start, end in _CJK_RANGES)


def _split_answers(answer: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"\s*(?:/|;|\n)\s*|,\s+", answer)
        if part.strip()
    ]


def _validate_mode(
    answer: str,
    *,
    mode: Mode,
    label: str,
    contains: Callable[[str], bool],
    forbidden_for_only: Callable[[str], bool],
) -> list[str]:
    if mode == "none":
        return []
    if not any(contains(character) for character in answer):
        return [f"이전 답변에 질문이 요구한 {label}가 없습니다."]
    if mode == "only" and any(forbidden_for_only(character) for character in answer):
        return [f"이전 답변은 {label}만으로 작성되지 않았습니다."]
    return []


def _validate_length(
    answer: str,
    requirement: LengthRequirement,
    prefix: str = "이전 답변",
) -> str | None:
    if requirement.unit == "syllable":
        # This dataset uses "음절" as a visible answer-character count: Arabic
        # digits and Latin initials each occupy one requested slot (e.g. "6호선"
        # is labeled 3음절 and "LG" is labeled 2음절).
        actual = sum(character.isalnum() for character in answer)
    else:
        actual = len(answer.split())
    if actual == requirement.count:
        return None
    return (
        f"{prefix}은 {actual}{requirement.korean_unit}이지만 질문은 정확히 "
        f"{requirement.count}{requirement.korean_unit}을 요구합니다."
    )


def validate_sa_answer(answer: str, spec: SAFormatSpec) -> ValidationResult:
    """Validate one generated answer without editing or normalizing its content."""
    answer = str(answer).strip()
    reasons: list[str] = []
    if spec.is_free:
        return ValidationResult(True)
    if not answer:
        return ValidationResult(False, ("이전 답변이 비어 있습니다.",))

    if re.search(r"(?:^|\s)\d+\s*(?:음|음절|어절)\s*$", answer):
        reasons.append(
            "이전 답변 끝에 질문의 길이 조건을 옮겨 쓴 표현이 포함되어 있습니다."
        )

    reasons.extend(
        _validate_mode(
            answer,
            mode=spec.numeric_mode,
            label="숫자",
            contains=str.isdigit,
            forbidden_for_only=lambda character: character.isalpha(),
        )
    )
    reasons.extend(
        _validate_mode(
            answer,
            mode=spec.hanja_mode,
            label="한자",
            contains=_is_hanja,
            forbidden_for_only=lambda character: (
                character.isalnum() and not _is_hanja(character)
            ),
        )
    )
    reasons.extend(
        _validate_mode(
            answer,
            mode=spec.english_mode,
            label="영문 알파벳",
            contains=lambda character: character.isascii() and character.isalpha(),
            forbidden_for_only=lambda character: (
                (character.isalpha() and not character.isascii())
                or _is_hanja(character)
            ),
        )
    )
    if spec.require_hangul and not any(_is_hangul_syllable(char) for char in answer):
        reasons.append("이전 답변에 질문이 요구한 한글이 없습니다.")

    if spec.unit_mode == "exclude" and any(
        character.isalpha() or character in "%°" for character in answer
    ):
        reasons.append("이전 답변에 단위가 포함되어 있지만 질문은 단위 없는 답을 요구합니다.")
    elif spec.unit_mode == "include":
        if spec.unit_text is not None:
            if spec.unit_text.casefold() not in answer.casefold():
                reasons.append(
                    f"이전 답변에 질문이 요구한 단위 {spec.unit_text!r}가 없습니다."
                )
        elif not (
            any(character.isdigit() for character in answer)
            and any(character.isalpha() or character in "%°" for character in answer)
        ):
            reasons.append("이전 답변에 숫자와 단위가 함께 포함되어 있지 않습니다.")

    expected_parts = spec.answer_count
    if expected_parts is None and len(spec.length_groups) > 1:
        expected_parts = len(spec.length_groups)
    parts = (
        _split_answers(answer)
        if expected_parts and expected_parts > 1
        else [answer]
    )
    if expected_parts is not None and len(parts) != expected_parts:
        reasons.append(
            f"이전 답변은 {len(parts)}개의 답으로 구분되지만 질문은 정확히 "
            f"{expected_parts}개의 답을 요구합니다. 여러 답은 '/'로 구분해야 합니다."
        )

    if spec.length_groups:
        if len(spec.length_groups) == 1:
            targets = [(part, spec.length_groups[0]) for part in parts]
        else:
            targets = list(zip(parts, spec.length_groups))
        for index, (part, requirements) in enumerate(targets, start=1):
            prefix = "이전 답변" if len(targets) == 1 else f"이전 답변의 {index}번째 답"
            for requirement in requirements:
                reason = _validate_length(part, requirement, prefix)
                if reason is not None:
                    reasons.append(reason)

    return ValidationResult(not reasons, tuple(reasons))


def build_sa_retry_prompt(
    question: str,
    previous_answer: str,
    validation: ValidationResult,
) -> str:
    failures = "\n".join(f"- {reason}" for reason in validation.reasons)
    return (
        "원래 질문을 이미지와 함께 처음부터 다시 풀어야 합니다.\n\n"
        f"원래 질문:\n{question.strip()}\n\n"
        f"이전 답변:\n{previous_answer.strip()}\n\n"
        f"형식 검사 실패 이유:\n{failures}\n\n"
        "이전 답변을 자르거나 글자·숫자·조건 문구를 붙여 형식만 맞추지 마세요. "
        "원래 질문이 요구하는 의미상 올바른 정답 자체를 다시 찾으세요. "
        "모든 출력 조건을 만족하는 최종 정답만 출력하고 설명은 출력하지 마세요."
    )


def build_sa_retry_feature(
    feature: dict[str, Any],
    question: str,
    previous_answer: str,
    validation: ValidationResult,
) -> dict[str, Any]:
    """Fold the correction request into the existing single-turn user prompt."""
    conversation = feature.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("SA retry feature requires a non-empty conversation list")

    retry_conversation = copy.deepcopy(conversation)
    prompt_index = next(
        (
            index
            for index in range(len(retry_conversation) - 1, -1, -1)
            if retry_conversation[index].get("role") == "user"
            and retry_conversation[index].get("content") != "<image>"
        ),
        None,
    )
    if prompt_index is None:
        raise ValueError("SA retry feature requires a text user prompt")
    original_prompt = retry_conversation[prompt_index].get("content")
    if not isinstance(original_prompt, str) or not original_prompt.strip():
        raise ValueError("SA retry feature requires a non-empty text user prompt")

    retry_prompt = build_sa_retry_prompt(
        question, previous_answer, validation
    )
    retry_conversation[prompt_index]["content"] = (
        f"{original_prompt.rstrip()}\n\n[재시도 지시]\n{retry_prompt}"
    )
    retry_feature = dict(feature)
    retry_feature["conversation"] = retry_conversation
    return retry_feature


def generate_with_sa_retries(
    feature: dict[str, Any],
    initial_answer: str,
    generate_retry: Callable[[dict[str, Any]], str],
    *,
    max_retries: int = 2,
) -> str:
    """Retry only constrained SA answers, returning the final model output unchanged."""
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if str(feature.get("question_form", "")).strip().upper() != "SA":
        return initial_answer
    question = feature.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("SA retry requires the original question")

    spec = parse_sa_format(question)
    answer = initial_answer
    for _ in range(max_retries + 1):
        validation = validate_sa_answer(answer, spec)
        if validation.valid:
            return answer
        if _ == max_retries:
            break
        retry_feature = build_sa_retry_feature(
            feature, question, answer, validation
        )
        answer = generate_retry(retry_feature).strip()
    return answer
