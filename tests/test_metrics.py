from __future__ import annotations

import unittest

from aivqa.metrics import compute_vqa_metrics


class MetricsTest(unittest.TestCase):
    def test_task_metrics_and_final_score(self) -> None:
        metrics = compute_vqa_metrics(
            predictions=["정답은 2번입니다.", "  경복궁 ", "한복은 한국의 전통 의복이다."],
            references=["2", "경복궁", "한복은 한국의 전통 의복이다."],
            question_forms=["MC", "SA", "LA"],
        )
        self.assertEqual(metrics["mc_accuracy"], 1.0)
        self.assertEqual(metrics["sa_exact_match"], 1.0)
        self.assertEqual(metrics["rouge"], 1.0)
        self.assertEqual(metrics["bleu"], 1.0)
        self.assertEqual(metrics["descriptive_avg"], 1.0)
        self.assertEqual(metrics["final_score"], 1.0)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_vqa_metrics(["1"], [], ["MC"])

    def test_final_score_uses_the_three_requested_components(self) -> None:
        metrics = compute_vqa_metrics(
            predictions=["1", "오답", "정답", "오답", "동일 문장"],
            references=["1", "2", "정답", "정답", "동일 문장"],
            question_forms=["MC", "MC", "SA", "SA", "LA"],
        )
        self.assertEqual(metrics["mc_accuracy"], 0.5)
        self.assertEqual(metrics["sa_exact_match"], 0.5)
        self.assertEqual(metrics["descriptive_avg"], 1.0)
        self.assertAlmostEqual(metrics["final_score"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
