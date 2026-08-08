"""Kanana-V LLM-only LoRA fine-tuning, validation, and test generation."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from aivqa.data import GenerationCollator, KananaVQADataset, TrainCollator
from aivqa.metrics import compute_vqa_metrics
from training_logger import TrainingLogger


MODEL_ID = "kakaocorp/kanana-1.5-v-3b-instruct"
DATASET_NAME = "한국문화 멀티모달 질의응답"
DATASET_DIR = Path("datasets") / DATASET_NAME
TEST_PREDICTIONS_NAME = f"{DATASET_NAME}_test_predictions.json"

EXPECTED_DECODER_LAYERS = 32
PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
IMAGE_COMPRESSION_FACTOR = 28
DEFAULT_MIN_PIXELS = 200 * IMAGE_COMPRESSION_FACTOR**2
DEFAULT_MAX_PIXELS = 1600 * IMAGE_COMPRESSION_FACTOR**2
MODEL_MAX_PIXELS = 1600 * IMAGE_COMPRESSION_FACTOR**2
TARGET_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Kanana-1.5-V-3B with LLM-only LoRA."
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
        default=Path("outputs/kanana_1_5_v_3b_lora"),
        help="Root directory where a unique run_YYYYMMDD_HHMMSS folder is created",
    )
    parser.add_argument(
        "--test-predictions-path",
        type=Path,
        help=f"Default: RUN_OUTPUT_DIR/{TEST_PREDICTIONS_NAME}",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help="Minimum image pixels after Kanana preprocessing",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Maximum image pixels after Kanana preprocessing",
    )

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit frozen base weights when BF16 does not fit in VRAM",
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
        "max_length",
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
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")
    if args.max_pixels > MODEL_MAX_PIXELS:
        raise ValueError(
            f"--max-pixels cannot exceed the model processor limit ({MODEL_MAX_PIXELS})"
        )


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
    """Return and validate every LLM self-attention projection."""
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
            f"Unexpected Kanana decoder layers; missing={missing}, extra={extra}"
        )
    expected_projections = set(PROJECTION_NAMES)
    incomplete = {
        layer: sorted(expected_projections - projections)
        for layer, projections in found.items()
        if projections != expected_projections
    }
    expected_count = EXPECTED_DECODER_LAYERS * len(PROJECTION_NAMES)
    if incomplete or len(targets) != expected_count:
        raise RuntimeError(f"Incomplete adapter projection set: {incomplete}")
    return sorted(targets)


def configure_image_pixel_limits(
    processor: Any, min_pixels: int, max_pixels: int
) -> None:
    """Configure the attributes used by the native Kanana image processor."""
    if not hasattr(processor, "image_processor"):
        raise TypeError("The loaded processor does not expose an image_processor")
    if min_pixels < 1 or max_pixels < min_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels")

    image_processor = processor.image_processor
    image_processor.min_pixels = int(min_pixels)
    image_processor.max_pixels = int(max_pixels)
    image_processor.size = {
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
    }


def _prepare_llm_for_training(
    model: Any, load_in_4bit: bool, gradient_checkpointing: bool
) -> Any:
    """Prepare only the language model, leaving both visual modules frozen."""
    llm = model.language_model
    if load_in_4bit:
        from peft import prepare_model_for_kbit_training

        # Quantization metadata is attached to the multimodal wrapper by
        # Transformers. PEFT needs the same flag on the LLM submodule.
        for attribute in ("is_loaded_in_4bit", "is_loaded_in_8bit"):
            if hasattr(model, attribute):
                setattr(llm, attribute, getattr(model, attribute))
        llm = prepare_model_for_kbit_training(
            llm,
            use_gradient_checkpointing=False,
        )
    if gradient_checkpointing:
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return llm


def build_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Kanana-V training")

    dtype = getattr(torch, args.dtype)
    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    configure_image_pixel_limits(processor, args.min_pixels, args.max_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": {"": torch.cuda.current_device()},
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
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
    print(f"Loading model: {args.model_id}")
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    for required_module in ("vision_model", "abstractor", "language_model"):
        if not hasattr(model, required_module):
            raise TypeError(f"Kanana model is missing required module: {required_module}")

    model.requires_grad_(False)
    model.config.use_cache = False
    model.language_model.config.use_cache = False

    llm = _prepare_llm_for_training(
        model,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    target_modules = find_adapter_target_modules(name for name, _ in llm.named_modules())
    adapter_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_llm = get_peft_model(llm, adapter_config)
    # PEFT normally makes embedding outputs require gradients when checkpointing
    # is enabled. Kanana replaces image placeholder embeddings in-place, which
    # is incompatible with a leaf embedding tensor. Non-reentrant checkpointing
    # does not need that hook to compute LoRA parameter gradients.
    if hasattr(llm, "_require_grads_hook"):
        llm.disable_input_require_grads()
    model.language_model = peft_llm
    _verify_only_llm_adapters_are_trainable(model)
    model.language_model.print_trainable_parameters()
    return model, processor, dtype


def _verify_only_llm_adapters_are_trainable(model: Any) -> None:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("No trainable LoRA parameters were created")

    adapted_targets: set[tuple[int, str]] = set()
    target_path = re.compile(
        r"\.model\.layers\.(\d+)\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)\."
    )
    invalid = []
    for name in trainable_names:
        match = target_path.search(name)
        if (
            not name.startswith("language_model.")
            or ".lora_" not in name
            or match is None
        ):
            invalid.append(name)
        else:
            adapted_targets.add((int(match.group(1)), match.group(2)))

    if invalid:
        raise RuntimeError(
            "Non-LoRA or out-of-scope parameters are trainable: "
            + ", ".join(invalid[:10])
        )
    expected_target_count = EXPECTED_DECODER_LAYERS * len(PROJECTION_NAMES)
    if len(adapted_targets) != expected_target_count:
        raise RuntimeError(
            f"Expected {expected_target_count} adapted LLM projections, "
            f"found {len(adapted_targets)}"
        )
    for module_name in ("vision_model", "abstractor"):
        module = getattr(model, module_name)
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{module_name} must remain frozen")


def _model_input_device(model: Any) -> Any:
    return next(model.vision_model.parameters()).device


def _move_batch_to_device(value: Any, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    if hasattr(value, "to"):
        return value.to(device, non_blocking=True)
    return value


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
    model.vision_model.eval()
    model.abstractor.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    token_count = 0
    progress = tqdm(loader, desc="train", leave=False)
    device = _model_input_device(model)

    for step, batch in enumerate(progress):
        batch = _move_batch_to_device(batch, device)
        target_tokens = int((batch["labels"] != -100).sum().item())
        group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(gradient_accumulation_steps, len(loader) - group_start)

        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = model(**batch).loss
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
    device = _model_input_device(model)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            target_tokens = int((batch["labels"] != -100).sum().item())
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = model(**batch).loss
            loss_sum += float(loss.item()) * target_tokens
            token_count += target_tokens
    if token_count == 0:
        raise RuntimeError("Validation batch contained no assistant target tokens")
    return loss_sum / token_count


def _feature_batches(
    dataset: KananaVQADataset, batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(dataset), batch_size):
        yield [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]


def generate_predictions(
    model: Any,
    processor: Any,
    dataset: KananaVQADataset,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    dtype: Any,
) -> list[str]:
    import torch
    from tqdm.auto import tqdm

    model.eval()
    collator = GenerationCollator(processor, max_length=max_length)
    predictions: list[str] = []
    total_batches = math.ceil(len(dataset) / batch_size)
    device = _model_input_device(model)
    with torch.no_grad():
        for features in tqdm(
            _feature_batches(dataset, batch_size),
            total=total_batches,
            desc="generate",
            leave=False,
        ):
            batch = _move_batch_to_device(collator(features), device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                generated_ids = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            # The native multimodal generate path passes inputs_embeds to the
            # LLM, so it returns answer token IDs without the prompt prefix.
            predictions.extend(
                answer.strip()
                for answer in processor.batch_decode(
                    generated_ids.detach().cpu(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    return predictions


def evaluate_generation(
    model: Any,
    processor: Any,
    dataset: KananaVQADataset,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    dtype: Any,
) -> tuple[list[str], dict[str, float]]:
    predictions = generate_predictions(
        model,
        processor,
        dataset,
        batch_size,
        max_length,
        max_new_tokens,
        dtype,
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


def save_best_adapter(
    model: Any,
    adapter_dir: Path,
    epoch: int,
    best_score: float,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    """Save a standard PEFT adapter directory, never the full VLM weights."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "args": serialized_args,
        "adapter_scope": "language_model_only",
        "frozen_modules": ["vision_model", "abstractor"],
    }
    metadata_path = adapter_dir / "training_metadata.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(metadata_path)


def load_best_adapter(model: Any, adapter_dir: Path) -> dict[str, Any]:
    from peft.utils.save_and_load import (
        load_peft_weights,
        set_peft_model_state_dict,
    )

    metadata_path = adapter_dir / "training_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Best adapter metadata does not exist: {metadata_path}")
    adapter_state = load_peft_weights(str(adapter_dir), device="cpu")
    set_peft_model_state_dict(model.language_model, adapter_state)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


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

    train_dataset = KananaVQADataset(args.train_json, dataset_root=args.dataset_root)
    validation_dataset = KananaVQADataset(
        args.validation_json, dataset_root=args.dataset_root
    )
    test_dataset = KananaVQADataset(args.test_json, dataset_root=args.dataset_root)
    model, processor, dtype = build_model_and_processor(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=TrainCollator(processor, max_length=args.max_length),
    )
    validation_loss_loader = DataLoader(
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
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")

    logger = TrainingLogger(args.output_dir)
    best_adapter_dir = args.output_dir / "best_adapter"
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
            args.max_length,
            args.max_new_tokens,
            dtype,
        )

        final_score = validation_metrics["final_score"]
        is_best = final_score > best_score
        if is_best:
            best_score = final_score
            epochs_without_improvement = 0
            save_best_adapter(
                model,
                best_adapter_dir,
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

    checkpoint = load_best_adapter(model, best_adapter_dir)
    print(
        f"Loaded best adapter from epoch {checkpoint['epoch']} "
        f"(final_score={checkpoint['best_score']:.6f})"
    )

    test_predictions = generate_predictions(
        model,
        processor,
        test_dataset,
        args.eval_batch_size,
        args.max_length,
        args.max_new_tokens,
        dtype,
    )
    save_test_predictions(args.test_json, test_predictions, predictions_path)
    print(f"Saved test predictions: {predictions_path}")
    print(f"Saved LoRA adapter: {best_adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
