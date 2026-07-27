from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_logger import HISTORY_FIELDS, TrainingLogger


def _record(epoch: int = 1) -> dict[str, int | float]:
    return {
        "epoch": epoch,
        "train_loss": 1.0,
        "val_loss": 0.9,
        "learning_rate": 0.0002,
        "mc_accuracy": 0.8,
        "sa_exact_match": 0.7,
        "rouge": 0.6,
        "bleu": 0.5,
        "descriptive_avg": 0.55,
        "final_score": 0.6833333333,
        "best_score": 0.6833333333,
        "elapsed_time": 12.5,
    }


class TrainingLoggerTest(unittest.TestCase):
    def test_numeric_json_csv_and_best_metrics_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(TrainingLogger, "_write_plots"):
                logger = TrainingLogger(temp_dir)
                logger.log_epoch(_record(), is_best=True)

            root = Path(temp_dir)
            history = json.loads((root / "training_history.json").read_text(encoding="utf-8"))
            best = json.loads((root / "best_metrics.json").read_text(encoding="utf-8"))
            with (root / "training_history.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(list(history[0]), HISTORY_FIELDS)
            self.assertIsInstance(history[0]["final_score"], float)
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(rows[0]["epoch"], "1")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_both_plot_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(temp_dir)
            logger.log_epoch(_record(), is_best=True)
            self.assertTrue((Path(temp_dir) / "loss_curve.png").is_file())
            self.assertTrue((Path(temp_dir) / "final_score_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
