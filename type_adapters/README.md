# Type-specific LoRA adapters

이 패키지는 기존 Shared Adapter 및 기존 학습 코드를 수정하지 않고 다음 세 분기를
순차적으로 학습합니다.

```text
Shared Adapter -> MC subset -> mc_adapter/
Shared Adapter -> SA subset -> sa_adapter/
Shared Adapter -> LA subset -> la_adapter/
```

각 분기는 Kanana Base를 새로 로드한 뒤 Shared LoRA checkpoint를
`is_trainable=True`로 직접 불러옵니다. Shared LoRA를 Base에 merge하거나 새 LoRA를
초기화하지 않습니다. 각 분기가 끝나면 모델 참조를 제거하고 CUDA cache를 정리한 뒤
다음 분기를 다시 Shared Adapter에서 시작합니다.

## Training

프로젝트 루트에서 실행합니다.

```bash
python -m type_adapters.train \
  --shared-adapter-dir outputs/kanana_1_5_v_3b_lora/run_YYYYMMDD_HHMMSS/best_adapter
```

기본 출력 구조는 다음과 같습니다.

```text
outputs/kanana_1_5_v_3b_type_adapters/run_YYYYMMDD_HHMMSS/
├── mc_adapter/
├── sa_adapter/
├── la_adapter/
└── type_training_summary.json
```

유형별 best checkpoint 선택 기준은 MC=`mc_accuracy`, SA=`sa_exact_match`,
LA=`descriptive_avg`입니다. LoRA rank, alpha, dropout과 target module은 Shared Adapter의
`adapter_config.json`에서 그대로 이어받습니다.

## Test inference

학습 run directory를 전달하면 Base 모델을 한 번만 로드하고 세 Adapter를 등록합니다.
test sample을 MC/SA/LA 순으로 batch 처리한 뒤 원래 JSON index 순서로 복원합니다.

```bash
python -m type_adapters.inference \
  --adapters-dir outputs/kanana_1_5_v_3b_type_adapters/run_YYYYMMDD_HHMMSS \
  --eval-batch-size 1
```

출력 JSON은 원본 test record와 기존 필드를 보존하고 각 `model_output.answer`에 생성
결과를 기록합니다.
