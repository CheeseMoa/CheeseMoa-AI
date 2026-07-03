# 프로젝트 구조

CheeseMoa-AI 저장소의 폴더 구조와 각 모듈의 책임을 정의한다.
상위 방향은 [architecture/pipeline-overview.md](../architecture/pipeline-overview.md)를 따른다.

## 저장소 최상위

```text
CheeseMoa-AI/
  app/                     # Python 워커 애플리케이션 (SQS consumer)
  docs/                    # 설계·결정·컨벤션 문서
  .vscode/                 # 공유 에디터 설정 (format on save 등)
  .pre-commit-config.yaml  # 커밋 전 ruff 자동 실행
  .gitignore
  requirements.txt
  README.md
```

## 앱 내부 (`app/`)

```text
app/
  worker.py            # 워커 진입점. SQS 폴링 루프 + 오류 정책(재시도/DLQ/포이즌), 처리는 handlers에 위임
                       # `--smoke` 플래그로 AWS 없이 페이크 전체 배선 자가 검증
  handlers.py          # 인바운드 3종(classify/feedback/delete) 처리 로직. image_id↔face_id↔행 인덱스 번역,
                       # 보정→제약 변환 + later-wins 충돌 해소, 결과(ClassifyResult) 조립
  core/                # 설정, 의존성 주입
    config.py          # 환경변수 기반 설정 (pydantic-settings, `Settings` 하나) — 큐 URL·버킷명 플레이스홀더
    deps.py            # 프로덕션 조립 전용: boto3 클라이언트·AI 모델을 부품에 연결 (페이크 분기 없음)
    model_source.py    # 모델 파일 획득 추상화 (로컬 오버라이드/HuggingFace)
  messaging/           # SQS 연동 (Protocol + SQS 구현 + 인메모리 페이크 병치)
    consumer.py        # 단일 FIFO 인바운드 큐 수신 (MaxNumberOfMessages=1)
    publisher.py       # 결과 큐 발행 (classify-result)
  storage/             # 저장 계층 (Protocol + S3 구현 + 인메모리 페이크 병치)
    event_embeddings.py  # event .npz 도메인 타입 + 직렬화 코덱 + 불변식 (ADR 007)
    embedding_store.py   # event .npz 로드/저장 인터페이스
    image_source.py      # 원본 이미지 획득 (s3_key → BGR ndarray)
  pipeline/            # AI 파이프라인 로직
    detect.py          # YuNet 얼굴 감지
    align.py           # face_align 직접 구현 (Umeyama)
    embed.py           # AuraFace 임베딩
    cluster.py         # 전체 재군집 + cluster_id 재조정 (순수 로직)
    hdbscan_standalone.py  # HDBSCAN numpy 전용 이식본 (PoC 검증, cluster.py가 사용)
  schemas/             # Pydantic 메시지 스키마
    messages.py        # ClassifyRequest·ClusterFeedback(merge/split/reassign)·DeleteRequest·ClassifyResult
                       # + 인바운드 판별 유니온(parse_inbound_message, body `type` 필드)
```

## 레이어 의존 방향

```text
worker ──▶ handlers ──▶ storage ──▶ (외부: S3 — 원본 이미지 + event 단위 .npz)
   │           │    ──▶ pipeline.cluster (순수 numpy)
   │           └──▶ schemas
   ├──▶ messaging ──▶ (외부: SQS)
   └──▶ schemas, core
core/deps ──▶ pipeline(detect/embed) + boto3   # 무거운 체인은 조립 지점에서만
```

- `worker`는 `messaging`·`handlers`·`schemas`에 의존한다. 폴링/오류 정책만 알고 처리 내용은 모른다.
- `handlers`는 `storage`·`schemas`·`pipeline.cluster`(순수 numpy)에 의존하되 **detect/embed는 모른다** —
  얼굴 추출은 `FaceExtractor` 콜러블로 주입받는다(합성은 `core/deps`). 덕분에 handlers 스모크는
  모델 다운로드 없이 돈다.
- `messaging`은 `schemas`(메시지 형식)에 의존하고 `pipeline`은 모른다.
- `storage`·`messaging`은 Protocol 인터페이스와 인메모리 페이크를 구현과 병치한다 — 테스트 디렉터리가
  없는 현 단계에서 각 모듈 `__main__` 스모크와 `worker --smoke`가 페이크를 직접 조립한다.
- `pipeline` 모듈들은 데이터가 `detect → align → embed → cluster` 순서로 흐르지만, **모듈 간 직접
  import는 최소화한다**: 순수 수학 모듈(`align`, `cluster`)이 무거운 임포트 체인
  (`model_source`→huggingface_hub, onnxruntime)을 끌어오지 않도록 필요한 상수·헬퍼는 로컬로
  중복 선언하고(`align._ensure_bgr`, `cluster.EMBED_DIM`), `cluster`는 내부 알고리즘인
  `hdbscan_standalone`만 import한다.
- `core`는 어느 레이어에서도 참조 가능하다.
- `schemas`는 `pipeline`에 의존하지 않는다.

## 모듈별 책임

| 모듈 | 책임 |
|------|------|
| `worker.py` | SQS 폴링 루프 + 메시지 수준 오류 정책(포이즌 삭제 / 재시도·DLQ / 마지막 시도 failed 발행). 처리 순서 「저장 → 발행 → 삭제」 강제 |
| `handlers.py` | 메시지 3종 → (event .npz 갱신 + ClassifyResult). ADR 007 재군집 흐름의 구현 지점 |
| `messaging/` | SQS 큐 수신·발행. 메시지 직렬화/역직렬화 |
| `storage/` | event .npz 코덱·불변식·S3 입출력, 원본 이미지 획득 |
| `core/config.py` | 환경변수 파싱. `Settings` 클래스 하나만 존재. 미정 주소(큐 URL·버킷명)는 기본값 없는 필수 필드 |
| `core/deps.py` | 프로덕션 의존성 조립 + 레디니스 검사 (모델 적재 → SQS/S3 연결 확인 → 폴링 시작) |
| `pipeline/detect.py` | YuNet 모델 로딩 및 얼굴 감지 실행 |
| `pipeline/align.py` | Umeyama 변환 계산 및 112×112 정렬 이미지 생성 |
| `pipeline/embed.py` | AuraFace 모델 로딩 및 512-dim 벡터 생성 |
| `pipeline/cluster.py` | 전체 임베딩 HDBSCAN 재군집 + 보정 제약 강제 + 기존 `cluster_id` 재조정·대표벡터 계산 (저장소를 모르는 순수 로직) |
| `pipeline/hdbscan_standalone.py` | HDBSCAN 알고리즘의 numpy 전용 이식본 ([ADR 005](../decisions/005-hdbscan-standalone-port.md)) |
| `schemas/` | SQS 메시지 입출력 데이터 형식 정의 (Pydantic v2) |

## 명명

폴더·파일·클래스 명명 규칙은 [code-style.md](./code-style.md)를 따른다.
메시징 파일은 역할 이름(`consumer.py`, `publisher.py`),
파이프라인 파일은 동작 이름(`detect.py`, `align.py`, `embed.py`, `cluster.py`)을 쓴다.
예외: 검증된 외부 알고리즘의 이식본은 원본과의 대조(diff) 가능성을 위해 원본 파일명을 유지한다
(`hdbscan_standalone.py`, [ADR 005](../decisions/005-hdbscan-standalone-port.md)).
