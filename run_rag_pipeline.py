"""Kanana-V Shared/MC/SA/LA LoRA training with precomputed RAG caches."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from aivqa.data import KananaVQADataset, TrainCollator
from rag_db.augmentation import (
    RAG_CACHE_DIR,
    RagAugmentedDataset,
    generate_rag_predictions,
    load_rag_cache,
    rag_cache_paths,
)
from train_lora import (
    DATASET_DIR,
    DATASET_NAME,
    DEFAULT_MIN_PIXELS,
    IMAGE_COMPRESSION_FACTOR,
    MODEL_ID,
    MODEL_MAX_PIXELS,
    build_model_and_processor,
    create_run_output_dir,
    evaluate_generation,
    evaluate_loss,
    save_test_predictions,
    set_seed,
    train_one_epoch,
)
from type_adapters.data import (
    QUESTION_FORMS,
    build_type_subsets,
    restore_original_order,
)
from type_adapters.modeling import (
    attach_type_adapters_for_inference,
    load_base_model_and_processor,
    release_cuda_memory,
)
from type_adapters.train import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TYPE_EPOCHS,
    train_question_form,
)


LOGGER = logging.getLogger("aivqa.rag_pipeline")
SHARED_EPOCHS = 5
SHARED_EARLY_STOPPING_PATIENCE = 2
TYPE_EPOCHS = DEFAULT_TYPE_EPOCHS
TYPE_EARLY_STOPPING_PATIENCE = DEFAULT_EARLY_STOPPING_PATIENCE
ANSWER_FILENAME = "answer.json"
PIPELINE_DEFAULT_MAX_PIXELS = 400 * IMAGE_COMPRESSION_FACTOR**2
REPOSITORY_ROOT = Path(__file__).resolve().parent


def resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a validation-selected RAG Shared LoRA, branch RAG MC/SA/LA LoRAs, "
            "and create answer.json with type-best adapters and RAG."
        )
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
    parser.add_argument(
        "--test-json", type=Path, default=DATASET_DIR / f"{DATASET_NAME}_test.json"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kanana_1_5_v_3b_rag_pipeline"),
    )

    parser.add_argument(
        "--max-rag-chars",
        type=int,
        default=1500,
        help="Maximum description characters included per retrieved candidate",
    )

    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--shared-learning-rate", type=float, default=3e-5)
    parser.add_argument("--type-learning-rate", type=float, default=2e-5)
    parser.add_argument("--shared-weight-decay", type=float, default=0.03)
    parser.add_argument("--type-weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument(
        "--max-pixels", type=int, default=PIPELINE_DEFAULT_MAX_PIXELS
    )

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
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
    for split, cache_path in rag_cache_paths(
        resolve_repository_path(RAG_CACHE_DIR)
    ).items():
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Required {split} RAG cache does not exist: {cache_path}. "
                "Run `python build_rag_cache.py` first."
            )
    for label, path in (
        ("train JSON", args.train_json),
        ("validation JSON", args.validation_json),
        ("test JSON", args.test_json),
    ):
        resolved = resolve_repository_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
    for field in (
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "max_new_tokens",
        "max_rag_chars",
        "lora_r",
        "lora_alpha",
    ):
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if min(args.shared_learning_rate, args.type_learning_rate, args.max_grad_norm) <= 0:
        raise ValueError("learning rates and --max-grad-norm must be positive")
    if min(args.shared_weight_decay, args.type_weight_decay) < 0:
        raise ValueError("weight decay cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    if args.max_pixels > MODEL_MAX_PIXELS:
        raise ValueError(f"--max-pixels cannot exceed {MODEL_MAX_PIXELS}")


def _serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _save_shared_adapter(
    model: Any,
    adapter_dir: Path,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    epoch: int,
    best_score: float,
    metrics: dict[str, float],
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    _write_json(
        adapter_dir / "training_metadata.json",
        {
            "stage": "shared",
            "epochs": args.epochs,
            "best_epoch": epoch,
            "best_score": best_score,
            "selection_metric": "final_score",
            "metrics": metrics,
            "training_data": "train",
            "question_forms": list(QUESTION_FORMS),
            "rag_enabled": True,
            "validation_data": "validation",
            "validation_performed": True,
            "best_model_selection": True,
            "early_stopping_patience": args.early_stopping_patience,
            "adapter_scope": "language_model_only",
            "frozen_modules": ["vision_model", "abstractor"],
            "args": _serialized_args(args),
            "history": history,
        },
    )


def train_shared_adapter(
    args: argparse.Namespace,
    train_dataset: RagAugmentedDataset,
    validation_dataset: RagAugmentedDataset,
    adapter_dir: Path,
) -> list[dict[str, Any]]:
    """Train on train only and select the best Shared adapter on validation."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    model = processor = train_loader = validation_loader = None
    optimizer = scheduler = scaler = None
    history: list[dict[str, Any]] = []
    try:
        set_seed(args.seed)
        model, processor, dtype = build_model_and_processor(args)
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
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
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
        best_score = float("-inf")
        epochs_without_improvement = 0
        started_at = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            LOGGER.info("[Shared] Epoch %d/%d", epoch, args.epochs)
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
            score = float(validation_metrics["final_score"])
            is_best = score > best_score
            if is_best:
                best_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **validation_metrics,
                "selection_metric": "final_score",
                "selection_score": score,
                "best_score": best_score,
                "is_best": is_best,
                "elapsed_time": time.monotonic() - started_at,
            }
            history.append(record)
            _write_json(adapter_dir.parent / "shared_training_history.json", history)
            LOGGER.info("[Shared] %s", json.dumps(record, ensure_ascii=False))
            if is_best:
                _save_shared_adapter(
                    model,
                    adapter_dir,
                    args,
                    history,
                    epoch,
                    best_score,
                    validation_metrics,
                )
            if epochs_without_improvement >= args.early_stopping_patience:
                LOGGER.info(
                    "[Shared] Early stopping after %d non-improving epoch(s).",
                    args.early_stopping_patience,
                )
                break
        return history
    finally:
        del scaler, scheduler, optimizer, validation_loader, train_loader
        del processor, model
        release_cuda_memory()


def run_test_inference(
    args: argparse.Namespace,
    test_dataset: KananaVQADataset,
    adapter_dirs: dict[str, Path],
    test_candidates: list[list[Any]],
) -> list[str]:
    model = processor = None
    try:
        model, processor, dtype = load_base_model_and_processor(
            args, for_training=False
        )
        subsets = build_type_subsets(test_dataset)
        model = attach_type_adapters_for_inference(model, adapter_dirs)
        grouped_predictions: dict[str, list[str]] = {}
        for question_form in QUESTION_FORMS:
            model.language_model.set_adapter(question_form)
            model.eval()
            rag_subset = RagAugmentedDataset(
                subsets[question_form],
                [test_candidates[index] for index in subsets[question_form].indices],
                max_rag_chars=args.max_rag_chars,
            )
            grouped_predictions[question_form] = generate_rag_predictions(
                model,
                processor,
                rag_subset,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                dtype=dtype,
                description=f"{question_form} test answers (best adapter + RAG)",
            )
        return restore_original_order(subsets, grouped_predictions, len(test_dataset))
    finally:
        del processor, model
        release_cuda_memory()


def run_pipeline(args: argparse.Namespace) -> Path:
    validate_args(args)
    run_dir = create_run_output_dir(resolve_repository_path(args.output_dir))
    LOGGER.info("Pipeline output: %s", run_dir)

    train_dataset = KananaVQADataset(
        resolve_repository_path(args.train_json),
        dataset_root=resolve_repository_path(args.dataset_root),
    )
    validation_dataset = KananaVQADataset(
        resolve_repository_path(args.validation_json),
        dataset_root=resolve_repository_path(args.dataset_root),
    )
    test_dataset = KananaVQADataset(
        resolve_repository_path(args.test_json),
        dataset_root=resolve_repository_path(args.dataset_root),
    )
    cache_paths = rag_cache_paths(resolve_repository_path(RAG_CACHE_DIR))
    LOGGER.info("Step 1/7: loading fixed train/validation/test RAG caches")
    train_candidates = load_rag_cache(cache_paths["train"], train_dataset)
    validation_candidates = load_rag_cache(
        cache_paths["validation"], validation_dataset
    )
    test_candidates = load_rag_cache(cache_paths["test"], test_dataset)

    set_seed(args.seed)
    LOGGER.info(
        "Step 2/7: Shared Adapter, train only, up to %d epochs with validation "
        "(early-stopping patience=%d)",
        SHARED_EPOCHS,
        SHARED_EARLY_STOPPING_PATIENCE,
    )
    shared_train_dataset = RagAugmentedDataset(
        train_dataset,
        train_candidates,
        max_rag_chars=args.max_rag_chars,
    )
    shared_validation_dataset = RagAugmentedDataset(
        validation_dataset,
        validation_candidates,
        max_rag_chars=args.max_rag_chars,
    )
    shared_args = copy.copy(args)
    shared_args.epochs = SHARED_EPOCHS
    shared_args.learning_rate = args.shared_learning_rate
    shared_args.weight_decay = args.shared_weight_decay
    shared_args.early_stopping_patience = SHARED_EARLY_STOPPING_PATIENCE
    shared_adapter_dir = run_dir / "shared_adapter"
    shared_history = train_shared_adapter(
        shared_args,
        shared_train_dataset,
        shared_validation_dataset,
        shared_adapter_dir,
    )
    shared_best_record = max(
        shared_history, key=lambda record: record["selection_score"]
    )

    rag_train = RagAugmentedDataset(
        train_dataset,
        train_candidates,
        max_rag_chars=args.max_rag_chars,
    )
    rag_validation = RagAugmentedDataset(
        validation_dataset,
        validation_candidates,
        max_rag_chars=args.max_rag_chars,
    )
    train_subsets = build_type_subsets(rag_train)
    validation_subsets = build_type_subsets(rag_validation)

    type_args = copy.copy(args)
    type_args.shared_adapter_dir = shared_adapter_dir
    type_args.epochs = TYPE_EPOCHS
    type_args.learning_rate = args.type_learning_rate
    type_args.weight_decay = args.type_weight_decay
    type_args.early_stopping_patience = TYPE_EARLY_STOPPING_PATIENCE
    adapter_dirs: dict[str, Path] = {}
    type_results: list[dict[str, Any]] = []
    for step, question_form in enumerate(QUESTION_FORMS, start=3):
        LOGGER.info(
            "Step %d/7: %s branch, up to %d epochs with RAG validation "
            "(early-stopping patience=%d)",
            step,
            question_form,
            TYPE_EPOCHS,
            TYPE_EARLY_STOPPING_PATIENCE,
        )
        adapter_dir = run_dir / f"{question_form.lower()}_adapter"
        result = train_question_form(
            type_args,
            question_form,
            train_subsets[question_form],
            validation_subsets[question_form],
            adapter_dir,
        )
        adapter_dirs[question_form] = adapter_dir
        type_results.append(result)
        _write_json(run_dir / "type_training_summary.json", type_results)

    LOGGER.info("Step 6/7: type-best Adapter + cached-RAG test inference")
    predictions = run_test_inference(
        args,
        test_dataset,
        adapter_dirs,
        test_candidates,
    )
    answer_path = run_dir / ANSWER_FILENAME
    LOGGER.info("Step 7/7: saving submission JSON to %s", answer_path)
    save_test_predictions(
        resolve_repository_path(args.test_json), predictions, answer_path
    )
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "rag_enabled": True,
            "rag_retrieval_performed": False,
            "rag_cache": {
                split: str(path) for split, path in cache_paths.items()
            },
            "search_query_model": "base_kanana",
            "shared_adapter": str(shared_adapter_dir),
            "shared_epochs": SHARED_EPOCHS,
            "shared_training_data": "train",
            "shared_validation_data": "validation",
            "shared_validation": True,
            "shared_selection_metric": "final_score",
            "shared_best_epoch": shared_best_record["epoch"],
            "shared_best_score": shared_best_record["selection_score"],
            "shared_trained_epochs": len(shared_history),
            "shared_early_stopping_patience": SHARED_EARLY_STOPPING_PATIENCE,
            "shared_history": shared_history,
            "type_epochs": TYPE_EPOCHS,
            "type_early_stopping_patience": TYPE_EARLY_STOPPING_PATIENCE,
            "type_adapters": {
                key: str(value) for key, value in adapter_dirs.items()
            },
            "type_results": type_results,
            "answer_json": str(answer_path),
            "args": _serialized_args(args),
        },
    )
    return answer_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    answer_path = run_pipeline(parse_args())
    LOGGER.info("Completed full RAG pipeline: %s", answer_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
