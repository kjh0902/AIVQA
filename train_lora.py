"""LoRA/DoRA fine-tuning, validation, and test generation for Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from aivqa.data import GenerationCollator, QwenVQADataset, TrainCollator
from aivqa.metrics import compute_vqa_metrics
from training_logger import TrainingLogger


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DATASET_NAME = "한국문화 멀티모달 질의응답"
DATASET_DIR = Path("datasets") / DATASET_NAME
TEST_PREDICTIONS_NAME = f"{DATASET_NAME}_test_predictions.json"
EXPECTED_DECODER_LAYERS = 36
PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
IMAGE_COMPRESSION_FACTOR = 32
DEFAULT_MIN_PIXELS = 64 * IMAGE_COMPRESSION_FACTOR * IMAGE_COMPRESSION_FACTOR
DEFAULT_MAX_PIXELS = 512 * IMAGE_COMPRESSION_FACTOR * IMAGE_COMPRESSION_FACTOR
TARGET_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL-8B-Instruct with LoRA or DoRA."
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
        default=Path("outputs/qwen3_vl_lora"),
        help="Root directory where a unique run_YYYYMMDD_HHMMSS folder is created",
    )
    parser.add_argument(
        "--test-predictions-path",
        type=Path,
        help="Default: RUN_OUTPUT_DIR/한국문화 멀티모달 질의응답_test_predictions.json",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help="Minimum pixels per image before Qwen3-VL processing",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Maximum pixels per image before Qwen3-VL processing",
    )

    # Adapter defaults are kept together here so they are easy to modify.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--use-dora",
        action="store_true",
        help="Use DoRA instead of the default LoRA adapter",
    )

    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Quantize frozen base weights to 4-bit (recommended for limited VRAM)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.train_json, args.validation_json, args.test_json):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset JSON does not exist: {path}")
    positive_integer_fields = (
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "max_new_tokens",
        "lora_r",
        "lora_alpha",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be at least 1")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning rate and max grad norm must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.min_pixels < 1 or args.max_pixels < 1:
        raise ValueError("--min-pixels and --max-pixels must be positive")
    if args.max_pixels < args.min_pixels:
        raise ValueError("--max-pixels must be greater than or equal to --min-pixels")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_output_dir(
    output_root: Path, started_at: datetime | None = None
) -> Path:
    """Create and return a collision-safe directory for one training run."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"

    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        run_output_dir = output_root / name
        try:
            run_output_dir.mkdir(exist_ok=False)
            return run_output_dir
        except FileExistsError:
            suffix += 1


def find_adapter_target_modules(module_names: Iterable[str]) -> list[str]:
    """Return and validate all 36 x 4 language self-attention projections."""
    targets: list[str] = []
    found: dict[int, set[str]] = {}
    for name in module_names:
        match = TARGET_PATTERN.fullmatch(name)
        if match is None:
            continue
        layer = int(match.group(1))
        projection = match.group(2)
        targets.append(name)
        found.setdefault(layer, set()).add(projection)

    expected_layers = set(range(EXPECTED_DECODER_LAYERS))
    if set(found) != expected_layers:
        missing = sorted(expected_layers - set(found))
        extra = sorted(set(found) - expected_layers)
        raise RuntimeError(
            f"Unexpected Qwen decoder layers; missing={missing}, extra={extra}"
        )
    expected_projections = set(PROJECTION_NAMES)
    incomplete = {
        layer: sorted(expected_projections - projections)
        for layer, projections in found.items()
        if projections != expected_projections
    }
    if incomplete or len(targets) != EXPECTED_DECODER_LAYERS * len(PROJECTION_NAMES):
        raise RuntimeError(f"Incomplete adapter projection set: {incomplete}")
    return sorted(targets)


def _torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, name)


def configure_image_pixel_limits(
    processor: Any, min_pixels: int, max_pixels: int
) -> None:
    """Set one shared image pixel budget on the official Qwen3-VL processor."""
    if not hasattr(processor, "image_processor"):
        raise TypeError("The loaded processor does not expose an image_processor")
    if min_pixels < 1 or max_pixels < min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    processor.image_processor.size = {
        "shortest_edge": int(min_pixels),
        "longest_edge": int(max_pixels),
    }


def build_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen3-VL-8B training")

    dtype = _torch_dtype(args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_id)
    configure_image_pixel_limits(processor, args.min_pixels, args.max_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    print(
        f"Image pixel budget: min={args.min_pixels:,}, max={args.max_pixels:,} "
        f"(~{args.min_pixels // IMAGE_COMPRESSION_FACTOR**2}-"
        f"{args.max_pixels // IMAGE_COMPRESSION_FACTOR**2} visual tokens)"
    )
    print(f"Loading processor and model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, **model_kwargs
    )
    model.requires_grad_(False)
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    target_modules = find_adapter_target_modules(
        name for name, _ in model.named_modules()
    )
    adapter_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_dora=args.use_dora,
    )
    model = get_peft_model(model, adapter_config)
    _verify_only_adapters_are_trainable(model)
    model.print_trainable_parameters()
    return model, processor, dtype


def _verify_only_adapters_are_trainable(model: Any) -> None:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable adapter parameters were created")

    target_path = re.compile(
        r"model\.language_model\.layers\.(?:[0-9]|[12][0-9]|3[0-5])\."
        r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\."
    )
    invalid = [
        name
        for name, _ in trainable
        if ".lora_" not in name or target_path.search(name) is None
    ]
    if invalid:
        raise RuntimeError(
            "Non-adapter or out-of-scope parameters are trainable: " + ", ".join(invalid[:10])
        )


def _move_batch_to_model(batch: Any, model: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(model.device)
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _autocast(dtype: Any) -> Any:
    import torch

    return torch.autocast(device_type="cuda", dtype=dtype)


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    dtype: Any,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    import torch
    from tqdm.auto import tqdm

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    token_count = 0
    progress = tqdm(loader, desc="train", leave=False)

    for step, batch in enumerate(progress):
        batch = _move_batch_to_model(batch, model)
        target_tokens = int((batch["labels"] != -100).sum().item())
        group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(gradient_accumulation_steps, len(loader) - group_start)

        with _autocast(dtype):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
        scaler.scale(loss / group_size).backward()
        loss_sum += float(loss.detach().item()) * target_tokens
        token_count += target_tokens

        should_step = (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        progress.set_postfix(loss=float(loss.detach().item()))

    if token_count == 0:
        raise RuntimeError("Training batch contained no assistant target tokens")
    return loss_sum / token_count


def evaluate_loss(model: Any, loader: Any, dtype: Any) -> float:
    import torch

    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch_to_model(batch, model)
            target_tokens = int((batch["labels"] != -100).sum().item())
            with _autocast(dtype):
                loss = model(**batch, use_cache=False).loss
            loss_sum += float(loss.item()) * target_tokens
            token_count += target_tokens
    if token_count == 0:
        raise RuntimeError("Validation batch contained no assistant target tokens")
    return loss_sum / token_count


def _feature_batches(dataset: QwenVQADataset, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(dataset), batch_size):
        yield [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]


def generate_predictions(
    model: Any,
    processor: Any,
    dataset: QwenVQADataset,
    batch_size: int,
    max_new_tokens: int,
    dtype: Any,
) -> list[str]:
    import torch
    from tqdm.auto import tqdm

    model.eval()
    collator = GenerationCollator(processor)
    predictions: list[str] = []
    previous_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    total_batches = math.ceil(len(dataset) / batch_size)
    try:
        with torch.inference_mode():
            for features in tqdm(
                _feature_batches(dataset, batch_size),
                total=total_batches,
                desc="generate",
                leave=False,
            ):
                batch = collator(features)
                prompt_width = batch["input_ids"].shape[1]
                batch = _move_batch_to_model(batch, model)
                with _autocast(dtype):
                    generated_ids = model.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                answer_ids = generated_ids[:, prompt_width:].detach().cpu()
                predictions.extend(
                    answer.strip()
                    for answer in processor.batch_decode(
                        answer_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
    finally:
        processor.tokenizer.padding_side = previous_padding_side
    return predictions


def evaluate_generation(
    model: Any,
    processor: Any,
    dataset: QwenVQADataset,
    batch_size: int,
    max_new_tokens: int,
    dtype: Any,
) -> tuple[list[str], dict[str, float]]:
    predictions = generate_predictions(
        model, processor, dataset, batch_size, max_new_tokens, dtype
    )
    references = []
    question_forms = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["answer"] is None:
            raise ValueError(f"Validation sample {sample['question_id']} has no answer")
        references.append(sample["answer"])
        question_forms.append(sample["question_form"])
    return predictions, compute_vqa_metrics(predictions, references, question_forms)


def save_best_checkpoint(
    model: Any,
    path: Path,
    epoch: int,
    best_score: float,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    import torch
    from peft import get_peft_model_state_dict

    adapter_state = {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(model).items()
    }
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "epoch": epoch,
        "best_score": float(best_score),
        "metrics": metrics,
        "args": serialized_args,
        "adapter_state_dict": adapter_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_best_checkpoint(model: Any, path: Path) -> dict[str, Any]:
    import torch
    from peft import set_peft_model_state_dict

    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    return checkpoint


def save_test_predictions(
    source_json: Path, predictions: Sequence[str], output_path: Path
) -> None:
    if source_json.resolve() == output_path.resolve():
        raise ValueError("Test prediction path must not overwrite the source JSON")
    with source_json.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Test JSON root must be a list")
    if len(records) != len(predictions):
        raise ValueError(
            f"Prediction count mismatch: {len(predictions)} predictions for {len(records)} samples"
        )

    for record, prediction in zip(records, predictions):
        model_output = record.get("model_output")
        if not isinstance(model_output, dict):
            model_output = {}
            record["model_output"] = model_output
        model_output["answer"] = str(prediction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    validate_args(args)

    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    set_seed(args.seed)
    args.output_dir = create_run_output_dir(args.output_dir)
    print(f"Run output directory: {args.output_dir}")
    predictions_path = args.test_predictions_path or args.output_dir / TEST_PREDICTIONS_NAME

    train_dataset = QwenVQADataset(args.train_json, dataset_root=args.dataset_root)
    validation_dataset = QwenVQADataset(
        args.validation_json, dataset_root=args.dataset_root
    )
    test_dataset = QwenVQADataset(args.test_json, dataset_root=args.dataset_root)
    # The same configured processor instance is shared by every train/validation/test
    # collator, so all image inputs use exactly the same pixel limits.
    model, processor, dtype = build_model_and_processor(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=TrainCollator(processor),
    )
    validation_loss_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=TrainCollator(processor),
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
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.dtype == "float16"
    )

    logger = TrainingLogger(args.output_dir)
    best_checkpoint_path = args.output_dir / "best_epoch.pth"
    best_score = float("-inf")
    epochs_without_improvement = 0
    started_at = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
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
        val_loss = evaluate_loss(model, validation_loss_loader, dtype)
        _, validation_metrics = evaluate_generation(
            model,
            processor,
            validation_dataset,
            args.eval_batch_size,
            args.max_new_tokens,
            dtype,
        )

        final_score = validation_metrics["final_score"]
        is_best = final_score > best_score
        if is_best:
            best_score = final_score
            epochs_without_improvement = 0
            save_best_checkpoint(
                model,
                best_checkpoint_path,
                epoch,
                best_score,
                validation_metrics,
                args,
            )
        else:
            epochs_without_improvement += 1

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **validation_metrics,
            "best_score": best_score,
            "elapsed_time": time.monotonic() - started_at,
        }
        logger.log_epoch(epoch_metrics, is_best=is_best)
        print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2))

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Early stopping: validation final_score did not improve for "
                f"{args.early_stopping_patience} epoch(s)."
            )
            break

    logger.finalize()
    checkpoint = load_best_checkpoint(model, best_checkpoint_path)
    print(
        f"Loaded best epoch {checkpoint['epoch']} "
        f"(final_score={checkpoint['best_score']:.6f})"
    )

    test_predictions = generate_predictions(
        model,
        processor,
        test_dataset,
        args.eval_batch_size,
        args.max_new_tokens,
        dtype,
    )
    save_test_predictions(args.test_json, test_predictions, predictions_path)
    print(f"Saved test predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
