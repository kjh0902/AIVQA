"""Kanana base loading and PEFT adapter continuation/switching utilities."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from train_lora import (
    IMAGE_COMPRESSION_FACTOR,
    _prepare_llm_for_training,
    _verify_multimodal_adapters_are_trainable,
    configure_image_pixel_limits,
)

from .data import QUESTION_FORMS


ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def validate_adapter_checkpoint(adapter_dir: str | Path) -> Path:
    path = Path(adapter_dir)
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Adapter config does not exist: {config_path}")
    if not any((path / name).is_file() for name in ADAPTER_WEIGHT_NAMES):
        expected = ", ".join(ADAPTER_WEIGHT_NAMES)
        raise FileNotFoundError(f"Adapter weights do not exist in {path}; expected {expected}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Adapter must be a LoRA checkpoint: {config_path}")
    return path


def _adapter_signature(adapter_dir: Path) -> tuple[Any, ...]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    targets = config.get("target_modules") or []
    return (
        str(config.get("peft_type", "")).upper(),
        config.get("r"),
        config.get("lora_alpha"),
        tuple(sorted(str(target) for target in targets)),
        config.get("base_model_name_or_path"),
    )


def validate_type_adapter_set(
    adapters_dir: str | Path,
) -> dict[str, Path]:
    root = Path(adapters_dir)
    adapter_dirs = {
        question_form: validate_adapter_checkpoint(
            root / f"{question_form.lower()}_adapter"
        )
        for question_form in QUESTION_FORMS
    }
    signatures = {_adapter_signature(path) for path in adapter_dirs.values()}
    if len(signatures) != 1:
        raise ValueError("MC, SA, and LA adapters do not share one LoRA architecture")
    return adapter_dirs


def load_trainable_shared_adapter(model: Any, shared_adapter_dir: str | Path) -> Any:
    """Load the Shared LoRA itself as trainable weights without merging it."""
    from peft import PeftModel

    path = validate_adapter_checkpoint(shared_adapter_dir)
    return PeftModel.from_pretrained(model, str(path), is_trainable=True)


def load_switchable_type_adapters(
    model: Any, adapter_dirs: Mapping[str, str | Path]
) -> Any:
    """Register all three multimodal inference adapters on one model."""
    from peft import PeftModel

    normalized_dirs = {
        question_form: validate_adapter_checkpoint(adapter_dirs[question_form])
        for question_form in QUESTION_FORMS
    }
    first_form = QUESTION_FORMS[0]
    peft_model = PeftModel.from_pretrained(
        model,
        str(normalized_dirs[first_form]),
        adapter_name=first_form,
        is_trainable=False,
    )
    for question_form in QUESTION_FORMS[1:]:
        peft_model.load_adapter(
            str(normalized_dirs[question_form]),
            adapter_name=question_form,
            is_trainable=False,
        )
    peft_model.set_adapter(first_form)
    return peft_model


def load_base_model_and_processor(
    args: Any, *, for_training: bool
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    operation = "training" if for_training else "generation"
    if not torch.cuda.is_available():
        raise RuntimeError(f"A CUDA GPU is required for Kanana-V {operation}")

    dtype = getattr(torch, args.dtype)
    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
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
    print(f"Loading base model: {args.model_id}")
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    for required_module in ("vision_model", "abstractor", "language_model"):
        if not hasattr(model, required_module):
            raise TypeError(f"Kanana model is missing required module: {required_module}")

    model.requires_grad_(False)
    if for_training:
        model.config.use_cache = False
        model.language_model.config.use_cache = False
        model.language_model = _prepare_llm_for_training(
            model,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        model.eval()
    return model, processor, dtype


def attach_shared_adapter_for_training(
    model: Any, shared_adapter_dir: str | Path
) -> Any:
    peft_model = load_trainable_shared_adapter(
        model,
        shared_adapter_dir,
    )
    llm = peft_model.language_model
    if hasattr(llm, "_require_grads_hook"):
        llm.disable_input_require_grads()
    _verify_multimodal_adapters_are_trainable(peft_model)
    peft_model.print_trainable_parameters()
    return peft_model


def attach_type_adapters_for_inference(
    model: Any, adapter_dirs: Mapping[str, str | Path]
) -> Any:
    model = load_switchable_type_adapters(
        model,
        adapter_dirs,
    )
    model.requires_grad_(False)
    model.eval()
    return model


def release_cuda_memory() -> None:
    """Release unreachable models before loading the next independent branch."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
