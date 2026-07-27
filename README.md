# AIVQA

Hugging Face Transformers로 `Qwen/Qwen3-VL-8B-Instruct`를 불러와 단일 이미지에 질문하는 최소 추론 예제입니다. Qwen 공식 GitHub 저장소는 clone하지 않으며, 모델 가중치와 processor는 첫 실행 시 Hugging Face Hub에서 자동으로 내려받습니다.

## 환경 준비

GPU 서버의 터미널에서 저장소 디렉터리로 이동한 뒤 Conda 환경을 만듭니다.

```bash
conda create -n aivqa python=3.11 -y
conda activate aivqa
python -m pip install --upgrade pip
```

RTX 50 시리즈처럼 CUDA 12.8 빌드가 필요한 환경에서는 PyTorch CUDA wheel을 먼저 설치합니다.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

서버에 맞는 PyTorch가 이미 설치되어 있다면 첫 번째 명령은 생략하고 다음 명령만 실행해도 됩니다.

```bash
python -m pip install -r requirements.txt
```

새 SSH 세션에서는 환경을 다시 활성화합니다.

```bash
conda activate aivqa
cd /path/to/AIVQA
```

Conda 활성화가 되지 않는 최초 1회에는 `conda init bash`를 실행하고 셸을 다시 시작합니다.

## 단일 이미지 질의응답

저장소 루트에서 다음과 같이 실행합니다.

```bash
python infer_single_image.py \
  --image datasets/test/0001.jpg \
  --question "이 이미지에 무엇이 보이나요?"
```

스크립트는 processor 로드, 모델 로드, 답변 생성 단계를 표준 오류에 표시하고 최종 답변을 표준 출력에 출력합니다. 첫 실행에서는 모델 파일을 다운로드하므로 시간이 오래 걸릴 수 있습니다.

생성 길이나 데이터 타입을 지정할 수도 있습니다.

```bash
python infer_single_image.py \
  --image datasets/test/0001.jpg \
  --question "이미지의 핵심 내용을 한국어 한 문장으로 설명해 주세요." \
  --max-new-tokens 64 \
  --dtype bfloat16
```

기본값은 다음과 같습니다.

- 모델: `Qwen/Qwen3-VL-8B-Instruct`
- dtype: `auto`
- device map: `auto` (가능한 GPU를 자동 사용하고 필요하면 CPU로 일부 offload)
- 최대 생성 길이: 128 tokens

메모리가 부족하면 먼저 다른 GPU 프로세스를 종료했는지 확인하세요. `device_map=auto`가 CPU offload를 사용할 경우 실행 속도가 크게 느려질 수 있습니다.

## GPU 환경 확인

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

`cuda: True`와 GPU 이름이 출력되면 CUDA PyTorch 환경이 정상입니다.

## Dataset과 Collator

`aivqa.data.QwenVQADataset`은 원본 JSON을 메모리에 읽고, 각 샘플에 접근하는 시점에만 Qwen messages 형식으로 변환합니다. 원본 JSON과 이미지 파일을 수정하거나 변환 결과를 별도 파일로 저장하지 않습니다.

```python
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from aivqa.data import GenerationCollator, QwenVQADataset, TrainCollator

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
train_dataset = QwenVQADataset(
    "datasets/한국문화 멀티모달 질의응답/한국문화 멀티모달 질의응답_train.json"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=TrainCollator(processor),
)

generation_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=GenerationCollator(processor),
)
```

- `TrainCollator`: assistant 정답을 messages에 추가하고, prompt와 padding 위치를 `-100`으로 마스킹한 `labels`를 만듭니다.
- `GenerationCollator`: assistant 정답을 제외하고 generation prompt까지 추가합니다.
- MC 선택지는 원본 순서를 유지하여 질문 아래에 붙이며, SA/LA의 빈 선택지 배열은 출력하지 않습니다.

테스트는 다음 명령으로 실행합니다.

```bash
python -m unittest discover -s tests -v
```

## LoRA/DoRA 학습·검증·테스트

`train_lora.py`는 36개 language decoder layer의 `q_proj`, `k_proj`, `v_proj`, `o_proj`에만 adapter를 적용합니다. Vision encoder, visual merger, LM head와 decoder의 원본 가중치는 모두 frozen 상태로 유지하며, 실행 시 대상 projection 144개와 trainable parameter 범위를 검사합니다.

RTX 5070처럼 VRAM이 제한된 GPU에서는 frozen base model을 4-bit로 로드하는 옵션을 권장합니다.

```bash
conda activate aivqa
python train_lora.py \
  --load-in-4bit \
  --epochs 10 \
  --early-stopping-patience 3 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8
```

모든 train/validation/test 이미지에는 동일한 processor pixel 제한이 적용됩니다. Qwen3-VL의 32×32 spatial compression을 기준으로 기본 범위는 약 64~512 visual tokens입니다.

- `--min-pixels 65536` (`64 × 32 × 32`)
- `--max-pixels 524288` (`512 × 32 × 32`)

첫 배치에서 여전히 CUDA OOM이 발생하면 최대치를 약 256 visual tokens로 더 낮출 수 있습니다.

```bash
python train_lora.py \
  --load-in-4bit \
  --min-pixels 65536 \
  --max-pixels 262144
```

기본 adapter는 LoRA이며 `r=16`, `alpha=32`, `dropout=0.05`입니다. DoRA는 `--use-dora`를 추가합니다.

```bash
python train_lora.py \
  --load-in-4bit \
  --use-dora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

각 epoch가 끝나면 validation 생성 평가를 수행합니다.

- MC: 선택지 번호 Accuracy
- SA: 유니코드·공백 정규화 후 Exact Match
- LA: 한국어 토큰 기반 ROUGE-L F1과 corpus BLEU-4
- `descriptive_avg = (ROUGE + BLEU) / 2`
- `final_score = (MC Accuracy + SA Exact Match + descriptive_avg) / 3`

`final_score`가 기존 최고값을 초과할 때 adapter 가중치를 `best_epoch.pth`에 저장합니다. 개선이 없는 epoch 수가 `--early-stopping-patience`에 도달하면 학습을 종료합니다.

기본 출력 디렉터리 `outputs/qwen3_vl_lora/`에는 다음 파일이 생성됩니다.

- `best_epoch.pth`
- `training_history.json`
- `training_history.csv`
- `best_metrics.json`
- `loss_curve.png`
- `final_score_curve.png`
- `한국문화 멀티모달 질의응답_test_predictions.json`

학습 종료 후 `best_epoch.pth`를 다시 로드하고 test split에 greedy generation을 수행합니다. 예측 JSON은 원본 순서와 기존 필드를 유지하면서 `model_output.answer`만 추가하며, 원본 test JSON은 수정하지 않습니다.

## 데이터 및 산출물

`datasets/`는 로컬 및 GPU 서버에 별도로 존재하는 데이터이므로 Git에 포함하지 않습니다. 모델 캐시, 체크포인트, 실행 결과 폴더도 `.gitignore`에서 제외합니다.
