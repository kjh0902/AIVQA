# Kanana-V 한국문화 멀티모달 질의응답 LoRA baseline

`kakaocorp/kanana-1.5-v-3b-instruct`로 한국문화 멀티모달 질의응답 데이터의
MC·SA·LA 전체 유형을 학습하는 LLM-only LoRA 구현입니다.

학습 범위는 다음과 같습니다.

```text
Image
  ↓
Kanana Vision Encoder       [Frozen]
  ↓
Kanana C-Abstractor         [Frozen]
  ↓
Kanana LLM                  [Frozen base weights + LoRA]
  ↓
Answer
```

공식 모델 구현의 `vision_model`, `abstractor`, `language_model` 경계를 그대로
사용합니다. LoRA는 32개 LLM decoder layer의 `q_proj`, `k_proj`, `v_proj`,
`o_proj` 128개에만 적용하며, 실행 시 trainable parameter 범위를 검사합니다.

## 전체 RAG 학습 및 추론 파이프라인

다음 명령 하나로 RAG 로드부터 `answer.json` 생성까지 순서대로 실행합니다.

```bash
python run_rag_pipeline.py
```

실행 순서는 고정되어 있습니다.

1. KURE/CLIP RAG encoder와 기존 Qdrant collection을 로드합니다.
2. Base Kanana가 train+validation 각 sample의 검색어를 생성하고 text/image RAG
   결과를 prompt에 추가합니다. 검색어 생성에는 어떤 LoRA도 사용하지 않습니다.
3. train+validation 전체로 Shared LoRA를 validation 없이 정확히 2 epoch 학습하고
   마지막 checkpoint를 `shared_adapter/`에 저장합니다.
4. 유형별 train/validation에도 2단계에서 Base Kanana가 만든 RAG context를 그대로
   사용합니다.
5. 동일한 Shared Adapter에서 MC, SA, LA를 각각 독립적으로 분기해 최대 10 epoch
   학습합니다. 매 epoch validation을 수행하고 유형별 지표가 2 epoch 연속 개선되지
   않으면 조기 종료하며, 유형별 best Adapter를 저장합니다.
6. Test 검색어는 Base Kanana로 생성하고, 문항 유형에 맞는 best Adapter는 RAG 최종
   답변 생성에만 사용합니다.
7. 원본 test 레코드 순서를 보존한 제출 파일을 `answer.json`으로 저장합니다.

기본 출력 구조는 다음과 같습니다.

```text
outputs/kanana_1_5_v_3b_rag_pipeline/run_YYYYMMDD_HHMMSS/
├── shared_adapter/
├── mc_adapter/
├── sa_adapter/
├── la_adapter/
├── rag_cache/
├── pipeline_summary.json
└── answer.json
```

학습 VRAM을 보존하기 위해 RAG encoder는 기본적으로 CPU에 둡니다. 별도 Qdrant
server나 RAG GPU를 사용하려면 다음처럼 지정할 수 있습니다.

```bash
python run_rag_pipeline.py \
  --qdrant-url http://localhost:6333 \
  --rag-device cuda
```

RAG context는 검색 점수 상위 3개 문서에서 만들며 긴 학습 prompt를 제한하기 위해
기본 2,000자로 자릅니다. `--max-rag-chars`로 바꿀 수 있습니다. 통합 파이프라인의
기본 이미지 상한은 약 400 visual token이며 `--max-pixels`로 바꿀 수 있습니다. 모든
단계의 입력은 image, question, options, 검색된 RAG context로만 구성됩니다.

## 환경 준비

Python 3.11 환경을 권장합니다. RTX 5070 Ti에는 CUDA 12.8 PyTorch wheel을 먼저
설치할 수 있습니다.

```bash
conda create -n aivqa python=3.11 -y
conda activate aivqa
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

모델은 Hugging Face의 custom code를 사용하므로 로딩 코드에
`trust_remote_code=True`가 필요합니다. 관련 런타임 의존성인 `timm`, `einops`,
`omegaconf`는 `requirements.txt`에 포함되어 있습니다.

## 데이터 형식

기본 경로는 다음과 같습니다.

```text
datasets/한국문화 멀티모달 질의응답/
├── 한국문화 멀티모달 질의응답_train.json
├── 한국문화 멀티모달 질의응답_validation.json
└── 한국문화 멀티모달 질의응답_test.json
```

각 레코드에서 아래 필드를 사용합니다.

- `metadata.question_id`, `metadata.split`, `metadata.question_form`
- `model_input.image_name` 또는 `image_path` 또는 `image`
- `model_input.question`, `model_input.options`
- train/validation의 `model_output.answer`

`KananaVQADataset`은 원본 JSON과 이미지 파일을 수정하지 않습니다. 이미지는
Pillow로 읽고 EXIF 방향을 반영한 RGB 메모리 객체로 변환합니다. Collator는 공식
processor가 요구하는 `{"image": [...], "conv": [...]}` 형식을 만들고
`batch_encode_collate`를 호출합니다.

문제 유형별 instruction은 모든 split에 동일하게 적용됩니다.

- MC: 선택지 번호만 출력, 복수 정답은 오름차순 `/` 구분
- SA: 질문에서 `N음절`, `N어절`, `N개`, `N가지`, `N답` 조건을 추출해 샘플별
  system prompt에 명시하고, 조건을 지킨 정답만 간결하게 출력
- LA: 250자 이내 한 문단, 같은 내용 반복 금지

학습 label은 assistant 답변 token에만 부여합니다. prompt, padding, 이미지
placeholder token은 모두 `-100`으로 마스킹합니다.

## LoRA 학습

RTX 5070 Ti 16GB용 기본값은 BF16, batch size 1, gradient checkpointing,
`max_length=2048`, 약 100~400 visual token입니다.

```bash
python train_lora.py \
  --epochs 5 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --max-length 2048 \
  --num-workers 2
```

기본 이미지 범위는 Kanana의 28×28 공간 압축 단위를 기준으로 합니다.

- `--min-pixels 78400`: 약 100 visual token
- `--max-pixels 313600`: 약 400 visual token
- 모델 processor 상한: `1254400`, 약 1600 visual token

첫 배치에서 메모리가 부족하면 `--max-pixels`를 더 낮추거나 Linux/WSL 환경에서
`--load-in-4bit`를 사용할 수 있습니다. 4-bit 옵션은 frozen base weight만
양자화하며 LoRA parameter는 계속 학습됩니다.

```bash
python train_lora.py --load-in-4bit --max-pixels 235200
```

기본 LoRA 설정은 `r=16`, `alpha=32`, `dropout=0.05`입니다.

```bash
python train_lora.py \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

각 epoch 후 validation loss와 생성 지표를 계산합니다.

- MC: 선택지 번호 Accuracy
- SA: 정규화 후 Exact Match
- LA: ROUGE-L F1과 BLEU-4
- `final_score = (MC Accuracy + SA Exact Match + LA 평균) / 3`

`final_score`가 개선되면 표준 PEFT adapter만 `best_adapter/`에 저장합니다. 전체
모델 가중치는 저장하지 않습니다. 기본 출력은 다음과 같습니다.

```text
outputs/kanana_1_5_v_3b_lora/run_YYYYMMDD_HHMMSS/
├── best_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_metadata.json
├── training_history.json
├── training_history.csv
├── best_metrics.json
├── loss_curve.png
├── final_score_curve.png
└── 한국문화 멀티모달 질의응답_test_predictions.json
```

저장된 adapter는 전체 VLM이 아니라 `language_model`에 다시 연결합니다.

```python
from peft import PeftModel
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained(
    "kakaocorp/kanana-1.5-v-3b-instruct",
    trust_remote_code=True,
    device_map="auto",
)
model.language_model = PeftModel.from_pretrained(
    model.language_model,
    "outputs/kanana_1_5_v_3b_lora/run_YYYYMMDD_HHMMSS/best_adapter",
)
```

## Zero-shot 및 단일 이미지 추론

학습 전 baseline 예측 JSON을 생성합니다.

```bash
python generate_zero_shot.py
```

단일 이미지에 질문할 수도 있습니다.

```bash
python infer_single_image.py \
  --image datasets/test/0001.jpg \
  --question "이 이미지에 무엇이 보이나요?"
```

두 경로 모두 공식 processor의 `batch_encode_collate`와 왼쪽 padding을 사용하며,
학습과 동일한 이미지 pixel 제한을 적용합니다.

## 테스트

단위 테스트는 모델 weight를 다운로드하지 않고 실행됩니다.

```bash
python -m unittest discover -s tests -v
```

공식 모델 및 사용법: [kakaocorp/kanana-1.5-v-3b-instruct](https://huggingface.co/kakaocorp/kanana-1.5-v-3b-instruct)
