"""Persistent numeric training history and plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "learning_rate",
    "mc_accuracy",
    "sa_exact_match",
    "rouge",
    "bleu",
    "descriptive_avg",
    "final_score",
    "best_score",
    "elapsed_time",
]


class TrainingLogger:
    """Write JSON/CSV histories, best metrics, and refreshed PNG curves."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, int | float]] = []

    def log_epoch(self, metrics: Mapping[str, int | float], is_best: bool) -> None:
        missing = [field for field in HISTORY_FIELDS if field not in metrics]
        if missing:
            raise ValueError(f"Missing training metrics: {', '.join(missing)}")

        record: dict[str, int | float] = {}
        for field in HISTORY_FIELDS:
            value = metrics[field]
            record[field] = int(value) if field == "epoch" else float(value)
        self.history.append(record)
        self._write_history_json()
        self._write_history_csv()
        if is_best:
            self._write_json(self.output_dir / "best_metrics.json", record)
        self._write_plots()

    def finalize(self) -> None:
        if not self.history:
            return
        self._write_history_json()
        self._write_history_csv()
        self._write_plots()

    def _write_history_json(self) -> None:
        self._write_json(self.output_dir / "training_history.json", self.history)

    def _write_history_csv(self) -> None:
        path = self.output_dir / "training_history.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self.history)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _write_plots(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [record["epoch"] for record in self.history]

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, [record["train_loss"] for record in self.history], marker="o", label="Train loss")
        axis.plot(epochs, [record["val_loss"] for record in self.history], marker="o", label="Validation loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("Training and validation loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "loss_curve.png", dpi=150)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, [record["final_score"] for record in self.history], marker="o", label="Final score")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("Validation final score")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "final_score_curve.png", dpi=150)
        plt.close(figure)
