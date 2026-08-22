from __future__ import annotations

import unittest

from aivqa.data import _conversation
from aivqa.sa_validation import (
    build_sa_retry_prompt,
    generate_with_sa_retries,
    parse_sa_format,
    validate_sa_answer,
)


class SAFormatParserAndValidatorTest(unittest.TestCase):
    def assert_valid(self, question: str, answer: str) -> None:
        result = validate_sa_answer(answer, parse_sa_format(question))
        self.assertTrue(result.valid, result.reasons)

    def assert_invalid(self, question: str, answer: str) -> tuple[str, ...]:
        result = validate_sa_answer(answer, parse_sa_format(question))
        self.assertFalse(result.valid)
        return result.reasons

    def test_free_question_has_no_format_constraint(self) -> None:
        spec = parse_sa_format("사진 속 물건의 이름은 무엇인가?")
        self.assertTrue(spec.is_free)
        self.assert_valid("사진 속 물건의 이름은 무엇인가?", "긴 자유 답변")
        self.assert_valid("사진 속 물건의 이름은 무엇인가?", "")

    def test_single_syllable_and_eojeol_requirements(self) -> None:
        self.assert_valid("3음절로 답하시오.", "이마트")
        reasons = self.assert_invalid("4음절로 답하시오.", "이마트")
        self.assertIn("3음절", reasons[0])
        self.assert_valid("2어절로 답하시오.", "대한 민국")
        self.assert_invalid("2어절로 답하시오.", "대한민국")
        self.assert_valid("3음절로 답하시오.", "6호선")
        self.assert_valid("2음절로 답하시오.", "LG")

    def test_sequential_and_simultaneous_complex_lengths(self) -> None:
        sequential = parse_sa_format("차례대로 2음절과 3음절로 답하시오.")
        self.assertEqual(len(sequential.length_groups), 2)
        self.assert_valid(
            "차례대로 2음절과 3음절로 답하시오.", "단오/씨름판"
        )

        simultaneous = parse_sa_format("2어절 4음절로 답하시오.")
        self.assertEqual(len(simultaneous.length_groups), 1)
        self.assertEqual(len(simultaneous.length_groups[0]), 2)
        self.assert_valid("2어절 4음절로 답하시오.", "대한 민국")

        repeated = parse_sa_format(
            "명칭과 재료는 각각 2음절로 답하고, 지역은 3음절로 답하시오."
        )
        self.assertEqual(
            [group[0].count for group in repeated.length_groups], [2, 2, 3]
        )
        self.assert_valid(
            "명칭과 재료는 각각 2음절로 답하고, 지역은 3음절로 답하시오.",
            "명칭/재료/제주도",
        )

    def test_numeric_hanja_and_english_modes(self) -> None:
        self.assert_valid("단위 없이 숫자로 답하시오.", "3000")
        self.assert_invalid("단위 없이 숫자로 답하시오.", "3000원")
        self.assert_valid("한자로 답하시오.", "漢字")
        self.assert_invalid("한자로 답하시오.", "한자")
        self.assert_valid("영문 알파벳으로 답하시오.", "KISA")
        self.assert_invalid("영문 알파벳으로 답하시오.", "키사")

    def test_contains_and_unit_requirements(self) -> None:
        self.assert_valid("한글과 숫자로 답하시오.", "코로나19")
        self.assert_invalid("한글과 숫자로 답하시오.", "코로나")
        self.assert_valid("숫자와 단위를 포함하여 답하시오.", "10km")
        self.assert_invalid("숫자와 단위를 포함하여 답하시오.", "10")

    def test_multiple_answers_apply_shared_length_to_each_part(self) -> None:
        question = "2어절 길이의 이칭 3개를 차례대로 답하시오."
        spec = parse_sa_format(question)
        self.assertEqual(spec.answer_count, 3)
        self.assert_valid(question, "가 나/다 라/마 바")
        self.assert_invalid(question, "가 나/다 라")

    def test_echoed_length_text_is_rejected(self) -> None:
        reasons = self.assert_invalid("4음절로 답하시오.", "이마트 4음")
        self.assertTrue(any("길이 조건" in reason for reason in reasons))


class SARetryTest(unittest.TestCase):
    def test_retry_reasks_original_question_and_stops_on_valid_answer(self) -> None:
        feature = {
            "question_form": "SA",
            "question": "국내 대형마트 이름을 4음절로 답하시오.",
            "conversation": [{"role": "user", "content": "원래 프롬프트"}],
        }
        retry_features = []

        def generate(retry_feature):
            retry_features.append(retry_feature)
            return "롯데마트"

        result = generate_with_sa_retries(feature, "이마트 4음", generate)
        self.assertEqual(result, "롯데마트")
        self.assertEqual(len(retry_features), 1)
        retry_conversation = retry_features[0]["conversation"]
        self.assertEqual(len(retry_conversation), 1)
        self.assertEqual(retry_conversation[0]["role"], "user")
        retry_prompt = retry_conversation[0]["content"]
        self.assertIn("원래 프롬프트", retry_prompt)
        self.assertIn("이마트 4음", retry_prompt)
        self.assertIn(feature["question"], retry_prompt)
        self.assertIn("처음부터 다시", retry_prompt)
        self.assertIn("글자·숫자·조건 문구를 붙여", retry_prompt)
        self.assertEqual(_conversation(retry_features[0]), retry_conversation)

    def test_retry_preserves_native_image_prompt_without_assistant_turn(self) -> None:
        feature = {
            "question_form": "SA",
            "question": "4음절로 답하시오.",
            "conversation": [
                {"role": "system", "content": "시스템 지시"},
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": "원래 질문과 RAG 참고정보"},
            ],
        }
        retry_features = []

        generate_with_sa_retries(
            feature,
            "세글자",
            lambda retry_feature: retry_features.append(retry_feature) or "네글자답",
        )

        retry_conversation = retry_features[0]["conversation"]
        self.assertEqual(
            [message["role"] for message in retry_conversation],
            ["system", "user", "user"],
        )
        self.assertEqual(retry_conversation[1]["content"], "<image>")
        self.assertIn("원래 질문과 RAG 참고정보", retry_conversation[2]["content"])
        self.assertFalse(
            any(message["role"] == "assistant" for message in retry_conversation)
        )
        self.assertEqual(_conversation(retry_features[0]), retry_conversation)

    def test_retry_runs_at_most_twice_and_returns_last_output_unchanged(self) -> None:
        feature = {
            "question_form": "SA",
            "question": "4음절로 답하시오.",
            "conversation": [{"role": "user", "content": "원래 프롬프트"}],
        }
        outputs = iter(("두글자", "세글자"))
        calls = []

        def generate(retry_feature):
            calls.append(retry_feature)
            return next(outputs)

        result = generate_with_sa_retries(feature, "한글", generate)
        self.assertEqual(result, "세글자")
        self.assertEqual(len(calls), 2)

    def test_mc_la_and_free_sa_do_not_retry(self) -> None:
        def fail_if_called(feature):
            self.fail("retry callback must not be called")

        for question_form, question in (
            ("MC", "3음절로 답하시오."),
            ("LA", "3음절로 답하시오."),
            ("SA", "조건 없이 답하시오."),
        ):
            feature = {
                "question_form": question_form,
                "question": question,
                "conversation": [],
            }
            self.assertEqual(
                generate_with_sa_retries(feature, "기존 답변", fail_if_called),
                "기존 답변",
            )

    def test_retry_prompt_includes_each_failure_reason(self) -> None:
        validation = validate_sa_answer("한글", parse_sa_format("숫자로 답하시오."))
        prompt = build_sa_retry_prompt("숫자로 답하시오.", "한글", validation)
        for reason in validation.reasons:
            self.assertIn(reason, prompt)


if __name__ == "__main__":
    unittest.main()
