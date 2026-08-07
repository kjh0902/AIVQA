"""Persistent numeric history for one question-form adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


HISTORY_FIELDS = (
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
    "selection_metric",
    "selection_score",
    "best_score",
    "elapsed_time",
)


class TypeTrainingHistory:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def log_epoch(self, metrics: Mapping[str, Any]) -> None:
        missing = [field for field in HISTORY_FIELDS if field not in metrics]
        if missing:
            raise ValueError(f"Missing type-adapter metrics: {', '.join(missing)}")
        record = {field: metrics[field] for field in HISTORY_FIELDS}
        self.history.append(record)
        self._write_json()
        self._write_csv()

    def _write_json(self) -> None:
        path = self.output_dir / "training_history.json"
        path.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_csv(self) -> None:
        path = self.output_dir / "training_history.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self.history)
