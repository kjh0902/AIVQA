# AIVQA Text + Image Qdrant 구축

`build_qdrant.py`는 `unified_rag.jsonl`을 스트리밍으로 읽어 문화 entity 하나를
Qdrant point 하나로 저장한다.

- `text`: `search_terms`를 공백으로 합친 뒤 `nlpai-lab/KURE-v1`로 임베딩한 named vector
- `image`: 각 이미지를 `openai/clip-vit-base-patch32`로 따로 임베딩한 named multivector
- image multivector comparator: `MAX_SIM`
- point ID: `doc_id`를 UUID v5로 변환한 deterministic UUID
- payload: 입력의 `doc_id`, `source`, `title`, `search_terms`, `description`, `image_path`

이미지가 없는 entity는 `image` vector만 생략한다. 개별 이미지가 없거나 깨졌으면
경고 로그를 남기고 그 이미지만 건너뛴다. JSONL 전체와 이미지 전체를 메모리에 올리지
않고 entity/image batch 단위로 처리한다.

## 설치

저장소 루트에서 기존 방식대로 의존성을 설치한다.

```bash
python -m pip install -r requirements.txt
```

## 구축

어느 작업 디렉터리에서 실행해도 기본 경로는 repository root를 기준으로 계산된다.
최초 구축 시 다음 명령을 실행한다.

```bash
python rag_db/build_qdrant.py --recreate
```

기본 출력은 `rag_db/qdrant_storage/`, collection 이름은
`aivqa_unified_rag`이다. CUDA가 있으면 CUDA와 FP16을 자동으로 사용한다. GPU에서
FP32가 필요하면 `--fp32`를 지정한다.

실행이 중단된 경우 `--recreate` 없이 다시 실행하면 deterministic ID로 upsert하므로
기존 collection에 이어서 안전하게 재구축할 수 있다.

전체 구축 전 소량으로 동작을 확인하려면 별도 collection을 권장한다.

```bash
python rag_db/build_qdrant.py \
  --collection aivqa_unified_rag_smoke \
  --limit 100 \
  --recreate
```

이미 실행 중인 Qdrant server에 저장할 수도 있다. 이 경우 server는 multivector를
지원하는 Qdrant 1.10 이상이어야 한다. API key가 필요하면 `QDRANT_API_KEY` 환경
변수로 전달한다.

```bash
python rag_db/build_qdrant.py \
  --qdrant-url http://localhost:6333 \
  --recreate
```

다른 batch 크기나 모델을 사용하려면 `--help`에서 전체 옵션을 확인한다. 모델 cache는
기본적으로 `rag_db/huggingface_cache/`에 저장되며 Git에서 제외된다.

## Image query 형식

`image`는 multivector이므로 단일 query image embedding도 한 행짜리 matrix로
전달한다.

```python
hits = client.query_points(
    collection_name="aivqa_unified_rag",
    query=[query_image_embedding],
    using="image",
    limit=10,
)
```

## Kanana + Text/Image RAG test inference

이미 구축된 `aivqa_unified_rag` collection과
`paddleocr_image_corpus.jsonl`을 사용해 전체 test dataset을 두 단계로 추론한다.
Qdrant collection을 새로 만들거나 수정하지 않는다.

1. 동일한 Kanana + shared LoRA가 image/question/OCR에서 JSON 검색어를 생성한다.
2. title exact match 또는 KURE text 검색과 CLIP image 검색을 수행한다.
3. `doc_id`로 결과를 합치고 `text_score + image_score` 상위 3개 description을 사용한다.
4. 같은 Kanana + LoRA 인스턴스가 기존 MC/SA/LA 출력 규칙으로 최종 답을 생성한다.

기본 adapter 경로는 요청된 checkpoint의 repository-relative 경로다.

```bash
python -m rag_db.infer_with_rag
```

기본 입출력은 다음과 같다.

```text
adapter: outputs/kanana_1_5_v_3b_lora/run_20260807_183229/best_adapter/
test:    datasets/한국문화 멀티모달 질의응답/한국문화 멀티모달 질의응답_test.json
OCR:     rag_db/paddleocr_image_corpus.jsonl
Qdrant:  rag_db/qdrant_storage/ (collection: aivqa_unified_rag)
output:  outputs/kanana_1_5_v_3b_rag/한국문화 멀티모달 질의응답_test_predictions.json
```

Qdrant server를 사용하거나 GPU 메모리를 절약하기 위해 RAG encoder를 CPU에 둘 수도
있다.

```bash
python -m rag_db.infer_with_rag \
  --qdrant-url http://localhost:6333 \
  --rag-device cpu
```

API key가 필요한 server는 `QDRANT_API_KEY` 환경 변수를 읽는다. `--qdrant-url`을
지정하지 않으면 기존 embedded storage를 읽는다. 검색 threshold는 text/image 모두
기본 `0.9`이며 `--score-threshold`로 조정할 수 있다. threshold를 통과한 결과는
`--retrieval-page-size` 단위로 모두 조회한 다음 fusion한다.

이미지가 없는 entity는 DB에 그대로 존재한다. Qdrant server 연결에서는 `has_vector`
filter와 native multivector 검색을 사용한다. Embedded local mode의 multivector 구현은
누락된 vector를 거리 계산 전에 제외하지 못하므로, 시작할 때 image vector를 Qdrant에서
한 번 읽어 compact NumPy 행렬을 만들고 동일한 cosine + entity별 MAX_SIM을 계산한다.
