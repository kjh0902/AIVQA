"""Continue one Shared LoRA independently on MC, SA, and LA subsets."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from aivqa.data import KananaVQADataset, TrainCollator
from train_lora import (
    DATASET_DIR,
    DATASET_NAME,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    MODEL_ID,
    MODEL_MAX_PIXELS,
    create_run_output_dir,
    evaluate_generation,
    evaluate_loss,
    set_seed,
    train_one_epoch,
)

from .data import QUESTION_FORMS, QuestionFormSubset, build_type_subsets
from .history import TypeTrainingHistory
from .modeling import (
    attach_shared_adapter_for_training,
    load_base_model_and_processor,
    release_cuda_memory,
    validate_adapter_checkpoint,
)


SELECTION_METRICS = {
    "MC": "mc_accuracy",
    "SA": "sa_exact_match",
    "LA": "descriptive_avg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a completed Shared Kanana-V LoRA independently on MC, SA, and LA."
        )
    )
    parser.add_argument("--shared-adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--question-form",
        choices=("ALL", *QUESTION_FORMS),
        default="ALL",
        help="Train all adapters sequentially (default) or only one of MC/SA/LA",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--train-json", type=Path, default=DATASET_DIR / f"{DATASET_NAME}_train.json"
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=DATASET_DIR / f"{DATASET_NAME}_validation.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kanana_1_5_v_3b_type_adapters"),
        help="Root where one run directory containing mc/sa/la_adapter is created",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_adapter_checkpoint(args.shared_adapter_dir)
    for path in (args.train_json, args.validation_json):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset JSON does not exist: {path}")
    for field in (
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "max_length",
        "max_new_tokens",
    ):
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be at least 1")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("Optimizer hyperparameters must be non-negative and non-zero where required")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    if args.max_pixels > MODEL_MAX_PIXELS:
        raise ValueError(
            f"--max-pixels cannot exceed the model processor limit ({MODEL_MAX_PIXELS})"
        )


def selection_score(question_form: str, metrics: dict[str, float]) -> tuple[str, float]:
    metric_name = SELECTION_METRICS[question_form]
    return metric_name, float(metrics[metric_name])


def selected_question_forms(question_form: str) -> tuple[str, ...]:
    return QUESTION_FORMS if question_form == "ALL" else (question_form,)


def _serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_type_adapter(
    model: Any,
    adapter_dir: Path,
    question_form: str,
    epoch: int,
    selection_metric: str,
    best_score: float,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    metadata = {
        "question_form": question_form,
        "epoch": int(epoch),
        "selection_metric": selection_metric,
        "best_score": float(best_score),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "args": _serialized_args(args),
        "continued_from_shared_adapter": True,
        "shared_adapter_dir": str(args.shared_adapter_dir.resolve()),
        "adapter_scope": "language_model_only",
        "frozen_modules": ["vision_model", "abstractor"],
        "merge_and_reinitialize": False,
    }
    metadata_path = adapter_dir / "training_metadata.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(metadata_path)


def train_question_form(
    args: argparse.Namespace,
    question_form: str,
    train_dataset: QuestionFormSubset,
    validation_dataset: QuestionFormSubset,
    adapter_dir: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    model = processor = train_loader = validation_loader = None
    optimizer = scheduler = scaler = None
    try:
        set_seed(args.seed)
        model, processor, dtype = load_base_model_and_processor(args, for_training=True)
        model = attach_shared_adapter_for_training(model, args.shared_adapter_dir)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=TrainCollator(processor, max_length=args.max_length),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=TrainCollator(processor, max_length=args.max_length),
        )

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        updates_per_epoch = math.ceil(
            len(train_loader) / args.gradient_accumulation_steps
        )
        total_updates = updates_per_epoch * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_updates * args.warmup_ratio),
            num_training_steps=total_updates,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")

        history = TypeTrainingHistory(adapter_dir)
        best_score = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        started_at = time.monotonic()
        selection_metric = SELECTION_METRICS[question_form]

        for epoch in range(1, args.epochs + 1):
            print(f"\n[{question_form}] Epoch {epoch}/{args.epochs}")
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                dtype,
                args.gradient_accumulation_steps,
                args.max_grad_norm,
            )
            val_loss = evaluate_loss(model, validation_loader, dtype)
            _, validation_metrics = evaluate_generation(
                model,
                processor,
                validation_dataset,
                args.eval_batch_size,
                args.max_length,
                args.max_new_tokens,
                dtype,
            )
            _, score = selection_score(question_form, validation_metrics)
            is_best = score > best_score
            if is_best:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
                save_type_adapter(
                    model,
                    adapter_dir,
                    question_form,
                    epoch,
                    selection_metric,
                    best_score,
                    validation_metrics,
                    args,
                )
            else:
                epochs_without_improvement += 1

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **validation_metrics,
                "selection_metric": selection_metric,
                "selection_score": score,
                "best_score": best_score,
                "elapsed_time": time.monotonic() - started_at,
            }
            history.log_epoch(epoch_metrics)
            print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2))

            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"[{question_form}] Early stopping after "
                    f"{args.early_stopping_patience} non-improving epoch(s)."
                )
                break

        return {
            "question_form": question_form,
            "adapter_dir": str(adapter_dir),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "selection_metric": selection_metric,
            "best_score": best_score,
            "best_epoch": best_epoch,
        }
    finally:
        del scaler, scheduler, optimizer, validation_loader, train_loader, processor, model
        release_cuda_memory()


def _write_run_summary(
    run_output_dir: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "shared_adapter_dir": str(args.shared_adapter_dir.resolve()),
        "independent_shared_initialization": True,
        "question_form_order": list(selected_question_forms(args.question_form)),
        "args": _serialized_args(args),
        "results": results,
    }
    path = run_output_dir / "type_training_summary.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    run_output_dir = create_run_output_dir(args.output_dir)
    print(f"Run output directory: {run_output_dir}")

    train_base = KananaVQADataset(args.train_json, dataset_root=args.dataset_root)
    validation_base = KananaVQADataset(
        args.validation_json,
        dataset_root=args.dataset_root,
    )
    train_subsets = build_type_subsets(train_base)
    validation_subsets = build_type_subsets(validation_base)

    results: list[dict[str, Any]] = []
    for question_form in selected_question_forms(args.question_form):
        print(
            f"\nStarting independent {question_form} continuation from Shared Adapter: "
            f"{args.shared_adapter_dir}"
        )
        result = train_question_form(
            args,
            question_form,
            train_subsets[question_form],
            validation_subsets[question_form],
            run_output_dir / f"{question_form.lower()}_adapter",
        )
        results.append(result)
        _write_run_summary(run_output_dir, args, results)

    print(f"Saved all type adapters under: {run_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
