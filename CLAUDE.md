# CheeseMoa-AI

## 프로젝트 개요

**치즈모아** (CheeseMoa) — "사진이 주인을 찾아가는 AI 공유 앨범"

여러 사람이 함께 촬영한 사진을 AI 얼굴 인식으로 자동 분류·공유하는 서비스. 이 레포는 **Python 워커 기반 AI 추론 서버**로, Spring 백엔드로부터 AWS SQS를 통해 작업을 받아 얼굴 감지 → 정렬 → 임베딩 → 클러스터링 파이프라인을 실행한다.

- GitHub 조직: [CheeseMoa](https://github.com/CheeseMoa)
- Jira 프로젝트: CHMO (티켓 번호 접두사 `CHMO-XX`)

---

## 전체 시스템 아키텍처

```
Flutter App
  │
  ▼
Spring API Server ──────────── PostgreSQL (Metadata)
  │                                  ▲
  │ Presigned URL 발급                │ 분류 결과 저장
  ▼                                  │
AWS S3 (원본 이미지 저장)              │
  │                                  │
  ▼                                  │
Message Queue (AWS SQS) ─────────────┘
  │
  ▼
[이 서버] Python AI Worker
  │
  ├── Face Detection (YuNet)
  ├── Face Alignment (직접 구현 - Umeyama 유사변환)
  ├── Face Embedding (AuraFace)
  └── Clustering (HDBSCAN)
       │
       ▼
  S3 (임베딩 저장소 — event 단위 .npz)
```

Spring 백엔드가 SQS 요청 큐에 분류 작업을 발행하면, 이 서버의 워커가 consumer로 소비해 파이프라인을 실행하고 결과를 SQS 결과 큐로 발행한다. HTTP는 사용하지 않는다.

---

## AI 파이프라인

### 파이프라인 흐름

```
S3에서 이미지 읽기
  │
  ▼
YuNet (얼굴 감지 + 5점 랜드마크 추출)
  │
  ▼
face_align (직접 구현 — Umeyama 유사변환, 112×112 ArcFace 기준점으로 정렬)
  │
  ▼
AuraFace (512-dim 임베딩 벡터 생성)
  │
  ▼
HDBSCAN (PoC numpy 이식본, cluster_selection_epsilon=0.15, cosine — event 전체 임베딩 재군집)
  │
  ▼
기존 cluster_id 재조정 (overlap 매칭으로 번호 승계, 사용자 보정은 제약)
  │
  ▼
클러스터 결과 → SQS 결과 큐로 발행 / event .npz 갱신
```

### 핵심 설계 결정사항

**face_align — 직접 구현 유지**
- `insightface.utils.face_align` 대신 `_umeyama()` 함수와 `_ARCFACE_DST` 상수를 코드 내 직접 구현
- 외부 의존성 제거 (insightface, skimage 불필요 → OpenCV + numpy만 사용)
- image_size=112 고정이므로 분기 로직 불필요, 30줄 닫힌형식 수식으로 완결
- 변환행렬 `np.allclose=True`, 픽셀 차이 0으로 기존과 동등성 검증 완료

**HDBSCAN — PoC numpy 전용 이식본 사용** ([ADR 005](docs/decisions/005-hdbscan-standalone-port.md))
- 알고리즘은 HDBSCAN 유지: 단순 직접 구현(UnionFind 코사인 임계값) 대비 ARI 0.601 vs 0.219로 정확도 약 2.7배 우수 (ADR 002)
- 구현체는 sklearn 라이브러리 대신 PoC의 `hdbscan_standalone.py`(sklearn 알고리즘을 numpy로 그대로 이식) —
  scikit-learn 1.9.0과 라벨 완전 일치 검증, 의존성 제거(face_align과 같은 패턴), PoC 최종 검증 레시피와 일치
- 파라미터 (PoC 검증값): `min_cluster_size=2, min_samples=2, metric='cosine', cluster_selection_epsilon=0.15`
- 재군집 뒤 결정적 후처리로 정확도 보강: 연결 성분 부분 승격([ADR 008](docs/decisions/008-blob-promotion-connected-components.md),
  클러스터 0개 퇴화 교정 — 쌍 유사도 ≥0.45 간선 성분 중 내부 완전 연결 ≥0.4인 성분만 승격) →
  보정 제약 강제 → 파편 병합(완전 연결) → 노이즈 구제(전역 유사도 내림차순) →
  저신뢰 `ambiguous` 분리(leave-one-out 유사도·마진, 제약 당사자 보호). 임계값은 전부 `ClusterConfig` 설정값
- 클러스터링은 전체 파이프라인 비용 0.1% 미만 (cosine은 sklearn도 brute 경로라 성능 특성 동급)

**전체 재군집 + ID 재조정 (정확도 최우선)**
- 재군집 격리 단위는 **event**(인물 앨범은 모임 안의 이벤트 단위로 생성, [ADR 007](docs/decisions/007-embedding-storage-s3.md))
- 군집의 진실은 항상 event 전체 임베딩(기존+신규)에 대한 HDBSCAN 재군집. 개별 임베딩을 event 단위 S3 `.npz`에 전부 보관해 매 트리거마다 전체를 다시 군집화한다
- 재군집 파티션을 기존 클러스터와 overlap 최대 매칭으로 연결해 `cluster_id`를 승계(연속성 유지). 대응 없는 군집만 신규 인물
- 사용자 보정(병합/분리/이동)은 must-link/cannot-link 제약으로 반영해 재군집이 사람 결정을 뒤집지 않게 함
- 대표벡터(L2 정규화 평균)는 조회·표시용 파생 캐시일 뿐, 군집 판단의 원천이 아님
- 클러스터링 연산은 파이프라인 비용 0.1% 미만이라 전체 재군집 비용 부담 없음
- 상세: [docs/spec/feature-spec.md](docs/spec/feature-spec.md) §4

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 실행 런타임 | Python 워커 프로세스 |
| 얼굴 감지 | YuNet (OpenCV DNN) |
| 얼굴 임베딩 | AuraFace (onnxruntime CPU) |
| 얼굴 정렬 | 직접 구현 (OpenCV + numpy, Umeyama) |
| 클러스터링 | HDBSCAN (PoC numpy 이식본) 전체 재군집 + cluster_id 재조정 (event 단위) |
| 임베딩 저장 | S3 (event 단위 `.npz`) — [ADR 007](docs/decisions/007-embedding-storage-s3.md) |
| 메시지 큐 연동 | AWS SQS |
| 데이터 검증 | Pydantic v2 |
| 코드 포맷터 | Ruff |
| 환경 변수 | pydantic-settings + python-dotenv (`app/core/config.py`의 `Settings` 단일 클래스) |

---

## 프로젝트 구조

```
CheeseMoa-AI/
├── app/
│   ├── worker.py            # SQS consumer 워커 엔트리포인트 (폴링 루프 + 오류 정책, --smoke 자가 검증)
│   ├── handlers.py          # 인바운드 3종(classify/feedback/delete) 처리 로직 (ADR-007 재군집 흐름)
│   ├── core/                # 설정(config.py)·프로덕션 조립(deps.py)·모델 소싱(model_source.py)
│   ├── messaging/           # SQS 수신(consumer.py)·발행(publisher.py) + 인메모리 페이크
│   ├── storage/             # event .npz 코덱(event_embeddings.py)·저장소(embedding_store.py)·
│   │                        # 원본 이미지 소스(image_source.py) + 인메모리 페이크
│   ├── pipeline/            # AI 파이프라인 로직
│   │   ├── detect.py        # YuNet 얼굴 감지
│   │   ├── align.py         # face_align 직접 구현
│   │   ├── embed.py         # AuraFace 임베딩
│   │   ├── cluster.py       # 전체 재군집 + cluster_id 재조정 (순수 로직)
│   │   └── hdbscan_standalone.py  # HDBSCAN numpy 이식본 (PoC 검증)
│   └── schemas/             # Pydantic 스키마 (SQS 메시지)
├── .env.example             # 환경변수 예시 — SQS 큐 URL·S3 버킷명은 미정(placeholder)
├── requirements.txt
├── .pre-commit-config.yaml  # ruff linter + formatter
└── .vscode/settings.json    # formatOnSave (ruff)
```

---

## 개발 환경 세팅

```bash
# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 환경변수 준비 — .env.example을 .env로 복사해 실값 주입 (큐 URL·버킷명 확정 전까지는 placeholder)
copy .env.example .env

# 워커 실행 (SQS consumer — 모델 적재 + SQS/S3 레디니스 통과 후 폴링 시작)
python -m app.worker

# AWS·모델 없이 전체 배선 자가 검증 (인메모리 페이크 e2e)
python -m app.worker --smoke

# pre-commit 훅 설치
pip install pre-commit
pre-commit install
```

---

## 코드 컨벤션

### 들여쓰기 / 포맷
- **스페이스 2칸** (AI/Flutter/Web 진영 공통)
- 저장 시 ruff 자동 포맷 (VSCode `formatOnSave: true`)
- 최대 줄 길이: 120자
- pre-commit hook: `ruff --fix` + `ruff-format` 자동 실행

### 네이밍
- 클래스: `PascalCase`
- 함수/변수: `snake_case`
- 상수: `UPPER_SNAKE_CASE`

### 주석
- WHY가 불명확한 경우에만 작성 (숨겨진 제약, 수학적 불변식 등)
- 할 일: `# TODO:` 표시
- `_umeyama()` 같은 수학 로직은 예외적으로 설명 주석 허용

---

## Git 컨벤션

### 브랜치 전략 (Git Flow)
- `main`: 배포 브랜치
- `develop`: 개발 통합 브랜치
- `feature/CHMO-XX-설명`: 기능 개발 브랜치

```bash
# 예시
git checkout -b feature/CHMO-54-face-detection-api
```

### 커밋 메시지
```
[CHMO-XX] type: 메시지
```

| type | 설명 |
|------|------|
| feat | 새로운 기능 |
| fix | 버그 수정 |
| docs | 문서 수정 |
| style | 포맷팅, 공백 등 |
| refactor | 리팩토링 |
| test | 테스트 코드 |
| chore | 빌드, 설정 등 |

```bash
# 예시
git commit -m "[CHMO-54] feat: SQS consumer 워커 골격 및 얼굴 감지 구현"
```

---

## 현재 상태

- 구현 완료 (파이프라인): `app/pipeline/detect.py`(YuNet), `app/pipeline/align.py`(Umeyama),
  `app/pipeline/embed.py`(AuraFace, onnxruntime CPU), `app/core/model_source.py`(모델 획득 추상화),
  `app/pipeline/cluster.py`(전체 재군집 + 보정 제약 + cluster_id 재조정, 순수 로직),
  `app/pipeline/hdbscan_standalone.py`(HDBSCAN numpy 이식본),
  `app/schemas/messages.py`(SQS 메시지 스키마 + 인바운드 판별 유니온 — 예시:
  [docs/spec/message-examples.md](docs/spec/message-examples.md))
- 구현 완료 (워커 계층, CHMO-165): `app/worker.py`(폴링 루프 + 오류 정책 + `--smoke`),
  `app/handlers.py`(3종 핸들러 + 보정 제약 later-wins 조정), `app/core/config.py`(Settings),
  `app/core/deps.py`(프로덕션 조립 + 레디니스), `app/messaging/`(SQS 수신·발행 + 페이크),
  `app/storage/`(event .npz 코덱·저장소·이미지 소스 + 페이크) — [ADR 007](docs/decisions/007-embedding-storage-s3.md)
- 미정: SQS 큐 URL·S3 버킷명 (feature-spec §10 #7) — `.env` 필수 항목이며 값 확정 시 주입만 하면 됨
- `app/main.py`: 비어있음 (엔트리포인트는 `app/worker.py`)
- `healthcare_api.py`: FastAPI 학습용 샘플 코드 (실제 프로젝트 코드 아님)

### 다음 구현 목표
1. 눈감음/흔들림 품질 판정 — `ClassifyOptions` 토글은 파싱만 되고 무시됨 (`eyes_closed`/`blurry` 항상 빈 리스트)
2. 확정된 큐 URL·버킷명 주입 + 실 AWS 환경 통합 검증 (visibility timeout·redrive policy를 `.env.example` 메모대로 설정)
3. pytest 도입 — 각 모듈 `__main__` 스모크를 tests/로 승격 (`# TODO(CHMO-165)` 표시 지점)

완료된 목표: 소규모 단일 인물 이벤트 앨범 미생성 개선 (2026-07-04,
[ADR 008](docs/decisions/008-blob-promotion-connected-components.md)) — 클러스터 0개 갈래는
`_promote_single_blob` 연결 성분 부분 승격 재설계로, 파편화 갈래는 파편 병합 임계
`merge_centroid_similarity` 0.7→0.6 완화(실측: 동일 인물 파편 centroid ~0.68, 타인 최대 0.167)로
교정. face-test 실사진 검증: 단일 인물 8건 전부 앨범 1개, 20인물 180얼굴 오병합 0건.
잔여 과제(임계 실데이터 추가 튜닝)는 feature-spec §10 #4로 흡수.
