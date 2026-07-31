"""Run PaddleOCR 3.x on one image using only the official result API."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import paddle
import paddleocr
import paddlex
from paddleocr import PaddleOCR


DEFAULT_IMAGE = Path("datasets/validation/P12009.jpg")
DEFAULT_OUTPUT = Path("paddleocr_official_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PaddleOCR 3.x 공식 최소 방식으로 이미지 한 장을 추론합니다."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"입력 이미지 경로 (기본값: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"공식 결과 저장 디렉터리 (기본값: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def package_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def print_runtime_api() -> None:
    constructor_signature = inspect.signature(PaddleOCR)
    predict_signature = inspect.signature(PaddleOCR.predict)

    print("=== Runtime ===")
    print(f"PaddlePaddle: {package_version(paddle)}")
    print(f"PaddleOCR   : {package_version(paddleocr)}")
    print(f"PaddleX     : {package_version(paddlex)}")
    print(f"CUDA build  : {paddle.is_compiled_with_cuda()}")
    print(f"GPU count   : {paddle.device.cuda.device_count()}")
    print()
    print("=== Installed API ===")
    print(f"PaddleOCR source   : {inspect.getsourcefile(PaddleOCR)}")
    print(f"PaddleOCR signature: {constructor_signature}")
    print(f"predict signature  : {predict_signature}")

    parameters = constructor_signature.parameters.values()
    accepts_extra_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    requested_names = {
        "lang",
        "device",
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_textline_orientation",
    }
    unsupported = sorted(
        name
        for name in requested_names
        if name not in constructor_signature.parameters and not accepts_extra_keywords
    )
    if unsupported:
        raise SystemExit(
            "현재 PaddleOCR 생성자에서 지원하지 않는 인자: "
            + ", ".join(unsupported)
        )


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not image_path.is_file():
        raise SystemExit(f"입력 이미지를 찾을 수 없습니다: {image_path}")

    print_runtime_api()
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=== Official minimal inference ===")
    print(f"Input : {image_path}")
    print(f"Output: {output_dir}")

    ocr = PaddleOCR(
        lang="korean",
        device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    results = list(ocr.predict(str(image_path)))
    print(f"Result count: {len(results)}")

    for result in results:
        result.print()
        result.save_to_img(str(output_dir))
        result.save_to_json(str(output_dir))

    print(f"공식 결과 저장 완료: {output_dir}")


if __name__ == "__main__":
    main()
