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
