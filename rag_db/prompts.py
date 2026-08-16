"""Lightweight RAG prompt construction shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from aivqa.data import (
    QUESTION_FORM_INSTRUCTIONS,
    SYSTEM_PROMPT,
    format_question,
)


SEARCH_QUERY_SYSTEM_PROMPT = (
    "당신은 한국 문화 지식 검색어 추출기입니다. 정답을 답하지 말고 요청된 JSON list만 출력하세요."
)
SEARCH_QUERY_PROMPT = """이미지와 질문에서 외부 지식 검색에 유용한 핵심 검색어를 추출하라.

- 인물명, 장소명, 기관명, 문화재명, 사건명, 음식명, 의복명, 전통문화 개념 등 검색 가치가 높은 명사구를 우선한다.
- "사진", "물건", "장소", "한국", "설명" 같은 일반어는 제외한다.
- 최대 5개까지만 선택한다.
- 검색할 만한 단어가 없다면 빈 리스트를 반환한다.
- 설명 없이 JSON list만 출력한다.
- 올바른 출력 예시: ["경복궁", "근정전"]

질문:
{question}"""

REFERENCE_CAUTION = (
    "검색된 참고정보는 외부 검색 결과이며 정답이 아닐 수 있습니다. "
    "사진 및 질문과 일치하지 않는 정보는 무시하고 답하십시오."
)


@dataclass
class Candidate:
    doc_id: str
    payload: dict[str, Any]
    text_score: float = 0.0
    image_score: float = 0.0

    @property
    def final_score(self) -> float:
        return self.text_score + self.image_score


def build_search_feature(sample: dict[str, Any], question: str) -> dict[str, Any]:
    prompt = SEARCH_QUERY_PROMPT.format(question=question.strip())
    return {
        "conversation": [
            {"role": "system", "content": SEARCH_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": prompt},
        ],
        "image": sample["image"],
    }


def build_answer_feature(
    sample: dict[str, Any],
    question: str,
    options: Sequence[str],
    candidates: Sequence[Candidate],
    max_rag_chars: int | None = None,
) -> dict[str, Any]:
    question_form = sample["question_form"]
    system_prompt = (
        f"{SYSTEM_PROMPT}\n\n{QUESTION_FORM_INSTRUCTIONS[question_form]}\n\n"
        f"{REFERENCE_CAUTION}"
    )
    parts = [format_question(question_form, question, options)]
    descriptions = [
        str(candidate.payload.get("description", "")).strip()
        for candidate in candidates
        if str(candidate.payload.get("description", "")).strip()
    ]
    if descriptions:
        rag_text = "\n\n---\n\n".join(descriptions)
        if max_rag_chars is not None:
            if max_rag_chars < 1:
                raise ValueError("max_rag_chars must be positive when set")
            rag_text = rag_text[:max_rag_chars]
        parts.append("RAG 참고정보:\n" + rag_text)
    return {
        "conversation": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        "image": sample["image"],
    }
