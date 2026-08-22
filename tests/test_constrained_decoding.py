from __future__ import annotations

import importlib.util
import math
import unittest

RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("torch", "transformers")
)
if RUNTIME_AVAILABLE:
    import torch

from aivqa.constrained_decoding import (
    KoreanLengthLogitsProcessor,
    get_sa_length_constraint,
    parse_sa_length_constraint,
)


class _FakeTokenizer:
    texts = (
        "",  # pad
        "",  # eos
        "가",
        "나다",
        "라마바사",
        "서울",
        " 대학",
        "원",
        " 부산",
        " ",
        "!",
    )

    def __len__(self) -> int:
        return len(self.texts)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    def decode(self, token_ids, **kwargs):
        return "".join(self.texts[int(token_id)] for token_id in token_ids)


class LengthConstraintParserTest(unittest.TestCase):
    def test_parses_syllable_constraint(self) -> None:
        self.assertEqual(
            parse_sa_length_constraint("정답을 3음절로 답하시오."),
            ("syllable", 3),
        )

    def test_parses_eojeol_constraint(self) -> None:
        self.assertEqual(
            parse_sa_length_constraint("두 단어를 2 어절로 작성하시오."),
            ("eojeol", 2),
        )

    def test_returns_none_without_one_unambiguous_constraint(self) -> None:
        self.assertIsNone(parse_sa_length_constraint("정답만 답하시오."))
        self.assertIsNone(
            parse_sa_length_constraint("2음절과 4음절로 각각 답하시오.")
        )

    def test_form_gate_constrains_only_sa(self) -> None:
        self.assertEqual(
            get_sa_length_constraint("SA", "3음절로 답하시오."),
            ("syllable", 3),
        )
        self.assertIsNone(get_sa_length_constraint("MC", "3음절로 답하시오."))
        self.assertIsNone(get_sa_length_constraint("LA", "2어절로 답하시오."))


@unittest.skipUnless(RUNTIME_AVAILABLE, "torch/transformers are not installed")
class KoreanLengthLogitsProcessorTest(unittest.TestCase):
    def test_syllable_masks_overshoot_and_forces_eos_at_target(self) -> None:
        processor = KoreanLengthLogitsProcessor(
            _FakeTokenizer(), [("syllable", 3)], eos_token_id=1
        )

        first_scores = processor(
            torch.empty((1, 0), dtype=torch.long), torch.zeros((1, 11))
        )
        self.assertTrue(math.isinf(float(first_scores[0, 1])))
        self.assertFalse(math.isinf(float(first_scores[0, 3])))
        self.assertTrue(math.isinf(float(first_scores[0, 4])))

        processor(torch.tensor([[2]], dtype=torch.long), torch.zeros((1, 11)))
        final_scores = processor(
            torch.tensor([[2, 3]], dtype=torch.long), torch.zeros((1, 11))
        )
        self.assertFalse(math.isinf(float(final_scores[0, 1])))
        self.assertTrue(math.isinf(float(final_scores[0, 2])))

    def test_eojeol_blocks_early_eos_and_a_third_word(self) -> None:
        processor = KoreanLengthLogitsProcessor(
            _FakeTokenizer(), [("eojeol", 2)], eos_token_id=1
        )

        first_scores = processor(
            torch.empty((1, 0), dtype=torch.long), torch.zeros((1, 11))
        )
        self.assertTrue(math.isinf(float(first_scores[0, 1])))

        second_scores = processor(
            torch.tensor([[5]], dtype=torch.long), torch.zeros((1, 11))
        )
        self.assertTrue(math.isinf(float(second_scores[0, 1])))
        self.assertFalse(math.isinf(float(second_scores[0, 6])))

        final_scores = processor(
            torch.tensor([[5, 6]], dtype=torch.long), torch.zeros((1, 11))
        )
        self.assertFalse(math.isinf(float(final_scores[0, 1])))
        self.assertFalse(math.isinf(float(final_scores[0, 7])))
        self.assertTrue(math.isinf(float(final_scores[0, 8])))

    def test_none_spec_leaves_row_unchanged(self) -> None:
        processor = KoreanLengthLogitsProcessor(
            _FakeTokenizer(), [None], eos_token_id=1
        )
        original = torch.arange(11, dtype=torch.float).unsqueeze(0)
        result = processor(torch.empty((1, 0), dtype=torch.long), original.clone())
        self.assertTrue(torch.equal(result, original))


if __name__ == "__main__":
    unittest.main()
