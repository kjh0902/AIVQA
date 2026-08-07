# Type-specific LoRA adapters

완료된 Shared LoRA를 초기값으로 사용해 MC, SA, LA Adapter를 추가 학습하고 test
문항 유형에 맞춰 추론합니다. Shared LoRA를 Base에 merge하거나 새 LoRA를 만들지 않으며,
rank, alpha, dropout과 target module은 Shared Adapter 설정을 그대로 이어받습니다.

모든 명령은 프로젝트 루트에서 실행합니다.

## 준비

- `--shared-adapter-dir`: 기존 Shared 학습 결과의 `best_adapter/`
- 기본 데이터: `datasets/한국문화 멀티모달 질의응답/`의 train/validation/test JSON
- 기본 학습값: 3 epochs, learning rate `2e-5`, effective batch size 8
- VRAM이 부족하면 `--load-in-4bit`를 추가합니다.

## 세 Adapter 일괄 학습

한 번의 실행으로 동일한 Shared Adapter에서 각각 독립적으로 시작해 MC → SA → LA
순서로 학습합니다.

```bash
python -m type_adapters.train --shared-adapter-dir outputs/kanana_1_5_v_3b_lora/run_YYYYMMDD_HHMMSS/best_adapter
```

출력:

```text
outputs/kanana_1_5_v_3b_type_adapters/run_YYYYMMDD_HHMMSS/
├── mc_adapter/
├── sa_adapter/
├── la_adapter/
└── type_training_summary.json
```

## 개별 Adapter 학습

`--question-form`으로 한 유형만 학습할 수 있습니다. 각 명령은 별도의 run directory를
만듭니다.

```bash
# MC Adapter
python -m type_adapters.train --shared-adapter-dir PATH/TO/best_adapter --question-form MC

# SA Adapter
python -m type_adapters.train --shared-adapter-dir PATH/TO/best_adapter --question-form SA

# LA Adapter
python -m type_adapters.train --shared-adapter-dir PATH/TO/best_adapter --question-form LA
```

validation은 매 epoch 자동 실행됩니다. best checkpoint 선택 기준은 다음과 같습니다.

- MC: `mc_accuracy`
- SA: `sa_exact_match`
- LA: `(ROUGE-L + BLEU-4) / 2`인 `descriptive_avg`

유형별 `training_history.json`, `training_history.csv`, `training_metadata.json`에서 결과를
확인할 수 있습니다.

## 전체 test 평가 및 제출 JSON 생성

세 Adapter가 들어 있는 일괄 학습 run directory를 지정합니다. test를 유형별로 batch
추론한 뒤 원래 index 순서로 복원하고 기존 제출 형식의 `model_output.answer`에 결과를
저장합니다.

```bash
python -m type_adapters.inference --adapters-dir outputs/kanana_1_5_v_3b_type_adapters/run_YYYYMMDD_HHMMSS --eval-batch-size 1
```

기본 출력은 다음 경로에 생성됩니다.

```text
outputs/kanana_1_5_v_3b_type_inference/run_YYYYMMDD_HHMMSS/
└── 한국문화 멀티모달 질의응답_test_predictions_type_adapters.json
```

저장 위치를 직접 지정하려면 `--test-predictions-path PATH/TO/predictions.json`을
사용합니다.
