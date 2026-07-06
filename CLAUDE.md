# CheeseMoa-AI

## 프로젝트 개요

**치즈모아** (CheeseMoa) — "사진이 주인을 찾아가는 AI 공유 앨범"

여러 사람이 함께 촬영한 사진을 AI 얼굴 인식으로 자동 분류·공유하는 서비스. 이 레포는 **Python 워커 기반 AI 추론 서버**로, Spring 백엔드로부터 AWS SQS를 통해 작업을 받아 얼굴 감지 → 정렬 → 임베딩 → 클러스터링 파이프라인을 실행한다.

- GitHub 조직: [CheeseMoa](https://github.com/CheeseMoa)
- Jira 프로젝트: CHMO (티켓 번호 접두사 `CHMO-XX`)

---

## 전체 시스템 아키텍처

Flutter App → Spring API Server(+ PostgreSQL 메타데이터, Presigned URL 발급) → AWS S3(원본 이미지) →
AWS SQS(분류 작업 큐) → **[이 서버] Python AI Worker**(YuNet 감지 → Umeyama 정렬 → AuraFace 임베딩 →
HDBSCAN 클러스터링) → S3(event 단위 `.npz` 임베딩 저장) + SQS 결과 큐 → Spring이 결과를 PostgreSQL에 저장.

Spring 백엔드가 SQS 요청 큐에 분류 작업을 발행하면, 이 서버의 워커가 consumer로 소비해 파이프라인을 실행하고 결과를 SQS 결과 큐로 발행한다. HTTP는 사용하지 않는다.

---

## AI 파이프라인

### 파이프라인 흐름

S3 이미지 읽기 → YuNet(얼굴 감지 + 5점 랜드마크) → face_align(직접 구현 Umeyama, 112×112 ArcFace 기준점) →
AuraFace(512-dim 임베딩) → 품질 게이트(눈감음 CNN + 흔들림 Laplacian, 토글 ON시 eyes_closed/blurry로 분리·
재군집 제외) → HDBSCAN(PoC numpy 이식본, cosine, epsilon=0.15, event 전체 임베딩 재군집) → cluster_id 재조정
(overlap 매칭으로 번호 승계, 사용자 보정은 제약) → SQS 결과 큐 발행 / event `.npz` 갱신.

### 핵심 설계 결정사항

**face_align — 직접 구현 유지**: `insightface.utils.face_align` 대신 `_umeyama()`/`_ARCFACE_DST`를
코드 내 직접 구현해 insightface·skimage 의존성 제거(OpenCV+numpy만). 변환행렬 동등성 검증 완료.

**HDBSCAN — PoC numpy 전용 이식본 사용** ([ADR 005](docs/decisions/005-hdbscan-standalone-port.md)):
알고리즘은 HDBSCAN 유지(단순 UnionFind 대비 ARI 2.7배 우수, ADR 002)하되 구현체는 sklearn이 아닌
PoC 이식본(라벨 완전 일치 검증, 의존성 제거). 파라미터(ARI 스윕 재확인, [ADR 009](docs/decisions/009-clustering-parameter-tuning.md)):
`min_cluster_size=2, min_samples=2, metric='cosine', cluster_selection_epsilon=0.15`. 재군집 후 결정적
후처리로 정확도 보강: 연결 성분 부분 승격([ADR 008](docs/decisions/008-blob-promotion-connected-components.md))
→ 보정 제약 강제 → 파편 병합(완전 연결) → 노이즈 구제(전역 유사도 내림차순) → 저신뢰 `ambiguous` 분리
(leave-one-out, 제약 당사자 보호). 임계값은 전부 `ClusterConfig` 설정값. 클러스터링은 전체 비용 0.1% 미만.

**전체 재군집 + ID 재조정 (정확도 최우선)**: 재군집 격리 단위는 **event**([ADR 007](docs/decisions/007-embedding-storage-s3.md)).
군집의 진실은 항상 event 전체 임베딩(기존+신규)에 대한 HDBSCAN 재군집 — 개별 임베딩을 S3 `.npz`에 전부
보관해 매 트리거마다 전체를 다시 군집화한다. 재군집 파티션은 기존 클러스터와 overlap 최대 매칭으로
`cluster_id`를 승계(대응 없는 군집만 신규 인물). 사용자 보정(merge/split/reassign/confirm_distinct)은
must-link/cannot-link 제약으로 반영해 재군집이 사람 결정을 뒤집지 않게 함:
- `uncertain`("분류가 어려워요") 사진은 실 `cluster_id`가 없어(.npz엔 None) 일반 reassign 대상이 못 되므로,
  예약 앨범 id `"__uncertain__"`을 `uncertain[].album_id`로 보내고 reassign의 `from_cluster_id`가 그 값이면
  미매칭 얼굴을 must-link로 인물 앨범 편입 (계약 확장, feature-spec §6.2·§6.3)
- must-link는 "같이 있어야 한다"만 강제할 뿐 "떨어져 있어야 한다"는 강제 못해, 확정된 두 인물 앨범 사이로
  유사도가 애매한 신규 사진(다리 사진)이 들어오면 오병합 위험이 있음 → `confirm_distinct` 액션(계약 확장,
  feature-spec §6.3)으로 `cluster_ids`(2개 이상)의 대표 얼굴 전 쌍에 cannot-link — merge의 반대 방향 선언

대표벡터(L2 정규화 평균)는 조회·표시용 파생 캐시일 뿐 군집 판단의 원천이 아님. 상세: [docs/spec/feature-spec.md](docs/spec/feature-spec.md) §4.

**품질 게이트 — 눈감음/흔들림 (CHMO-172)** (`app/pipeline/quality.py`, feature-spec §7 註): 눈감음은
YuNet 5점 정렬 크롭의 눈 좌표를 경량 CNN(`open-closed-eye-0001`, OpenVINO)으로 분류, 흔들림은 얼굴 crop
Laplacian variance(얼굴 미검출 시 전체 이미지로 fallback). 이미지 단위 "얼굴 1개라도 해당", 토글 ON시
`eyes_closed`/`blurry`로 분리·재군집 제외(request-scoped). 임계는 `QualityConfig`. 한계: 웃음 캔디드 오탐
(표정 CNN 후속 필요), 부분 모션블러+선명 배경 사각지대. 모델 소싱: `model_source.py`의 `UrlModelSource`.

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 실행 런타임 | Python 워커 프로세스 |
| 얼굴 감지 | YuNet (OpenCV DNN) |
| 얼굴 임베딩 | AuraFace (onnxruntime CPU) |
| 얼굴 정렬 | 직접 구현 (OpenCV + numpy, Umeyama) |
| 클러스터링 | HDBSCAN (PoC numpy 이식본) 전체 재군집 + cluster_id 재조정 (event 단위) |
| 품질 판정 | 눈감음 CNN (`open-closed-eye-0001`, onnxruntime CPU) + 흔들림 (Laplacian variance, OpenCV) |
| 모델 소싱 | `app/core/model_source.py` — YuNet·AuraFace는 HF Hub, 눈감음 CNN은 OpenVINO URL(`UrlModelSource`) |
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
│   │   ├── quality.py       # 품질 게이트 — 눈감음 CNN(EyeStateClassifier) + 흔들림 Laplacian (순수 로직)
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

로컬 Docker + 실 AWS(SQS/S3) end-to-end 테스트(AWS SSO 프로필 설정·세션 만료 대응 포함)는
[docs/guides/local-docker-e2e-testing.md](docs/guides/local-docker-e2e-testing.md) 참고.

---

## 코드 컨벤션

- **들여쓰기/포맷**: 스페이스 2칸(AI/Flutter/Web 공통), 저장 시 ruff 자동 포맷, 최대 줄 길이 120자,
  pre-commit hook에서 `ruff --fix` + `ruff-format` 자동 실행
- **네이밍**: 클래스 `PascalCase` · 함수/변수 `snake_case` · 상수 `UPPER_SNAKE_CASE`
- **주석**: WHY가 불명확한 경우에만(숨겨진 제약·수학적 불변식 등), 할 일은 `# TODO:`. `_umeyama()` 같은
  수학 로직은 예외적으로 설명 주석 허용

---

## Git 컨벤션

- **브랜치 (Git Flow)**: `main`(배포) · `develop`(개발 통합) · `feature/CHMO-XX-설명`(기능 개발)
  — 예: `git checkout -b feature/CHMO-54-face-detection-api`
- **커밋 메시지**: `[CHMO-XX] type: 메시지` (type: `feat`·`fix`·`docs`·`style`·`refactor`·`test`·`chore`)
  — 예: `git commit -m "[CHMO-54] feat: SQS consumer 워커 골격 및 얼굴 감지 구현"`

---

## 현재 상태

- 구현 완료 (파이프라인): `app/pipeline/detect.py`(YuNet), `app/pipeline/align.py`(Umeyama),
  `app/pipeline/embed.py`(AuraFace, onnxruntime CPU), `app/core/model_source.py`(모델 획득 추상화 + `UrlModelSource`),
  `app/pipeline/cluster.py`(전체 재군집 + 보정 제약 + cluster_id 재조정, 순수 로직),
  `app/pipeline/quality.py`(눈감음 CNN + 흔들림 Laplacian 품질 게이트, CHMO-172),
  `app/pipeline/hdbscan_standalone.py`(HDBSCAN numpy 이식본),
  `app/schemas/messages.py`(SQS 메시지 스키마 + 인바운드 판별 유니온 — 예시:
  [docs/spec/message-examples.md](docs/spec/message-examples.md))
- 구현 완료 (워커 계층, CHMO-165): `app/worker.py`(폴링 루프 + 오류 정책 + `--smoke`),
  `app/handlers.py`(4종 액션 핸들러 + 보정 제약 later-wins 조정), `app/core/config.py`(Settings),
  `app/core/deps.py`(프로덕션 조립 + 레디니스), `app/messaging/`(SQS 수신·발행 + 페이크),
  `app/storage/`(event .npz 코덱·저장소·이미지 소스 + 페이크) — [ADR 007](docs/decisions/007-embedding-storage-s3.md)
- 미정: SQS 큐 URL·S3 버킷명 (feature-spec §10 #7) — `.env` 필수 항목이며 값 확정 시 주입만 하면 됨
- `app/main.py`: 비어있음 (엔트리포인트는 `app/worker.py`)
- `healthcare_api.py`: FastAPI 학습용 샘플 코드 (실제 프로젝트 코드 아님)

### 다음 구현 목표
1. 확정된 큐 URL·버킷명 주입 + 실 AWS 환경 통합 검증 (visibility timeout·redrive policy를 `.env.example` 메모대로 설정).
   배포 미완 항목: Dockerfile·모델 프리베이크(콜드스타트)·CloudWatch·IAM 자격증명·Spring 실계약 통합검증
2. pytest 도입 — 각 모듈 `__main__` 스모크를 tests/로 승격 (`# TODO(CHMO-165)` 표시 지점)
3. (후속) 품질 게이트 개선 — 웃음 예외용 표정 CNN, 눈/흔들림 임계 라벨셋 튜닝(현재 라벨 부재), 부분 블러 대응
4. Spring과 `confirm_distinct` 트리거 정책 합의 — 즉시 발행(안전) vs 공유 시점 일괄 발행(발행 전 새
   업로드가 끼면 그 사이 재군집은 보호 공백)

### 완료된 목표
- **confirm_distinct — 확정 앨범 간 오병합 방지 (계약 확장)** (2026-07-06) — must-link는 응집만 강제하고
  이격은 못해 다리 사진이 확정된 두 앨범을 오병합할 위험을 `cluster_feedback`의 4번째 action
  `confirm_distinct`(`cluster_ids` 대표 얼굴 전 쌍 cannot-link)로 방지. 실측 오병합 기하로 `handlers.py`
  자가검증(⑬)에 회귀 고정 (feature-spec §6.3).
- **uncertain 사진의 인물 앨범 편입 (계약 확장)** (2026-07-04) — 실 `cluster_id`가 없어 일반 reassign
  대상이 못 되던 uncertain 얼굴을, 예약 앨범 id `"__uncertain__"`(`uncertain[].album_id`)로 해결. 실 AWS
  end-to-end 검증 완료.
- **눈감음/흔들림 품질 게이트** (2026-07-04, CHMO-172) — `quality.py` 신설, CNN+Laplacian 판정으로
  `eyes_closed`/`blurry` 라우팅. `eye_closed_confidence=0.85` face-test 보정.
- **클러스터링 파라미터 ARI 스윕** (2026-07-04, [ADR 009](docs/decisions/009-clustering-parameter-tuning.md)) —
  현행 `ClusterConfig` 값이 최적 근방·안전임을 확정(개선 후보 전부 회귀/과적합으로 기각).
- **소규모 단일 인물 이벤트 앨범 미생성 개선** (2026-07-04,
  [ADR 008](docs/decisions/008-blob-promotion-connected-components.md)) — 연결 성분 부분 승격 + peel로
  재설계, 병합 임계 0.7 유지.
