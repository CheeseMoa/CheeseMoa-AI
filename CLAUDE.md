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

S3 이미지 읽기 → YuNet(얼굴 감지 + 5점 랜드마크, 배경 인물 필터 — bbox 폭이 이미지 긴 변의 2.5%
미만인 얼굴 제거, [ADR 013](docs/decisions/013-background-face-size-filter.md)) → face_align(직접 구현 Umeyama, 112×112 ArcFace 기준점) →
AuraFace(512-dim 임베딩) → 품질 게이트(눈감음 CNN + 흔들림 Laplacian, 토글 ON시 eyes_closed/blurry로 분리·
재군집 제외) → HDBSCAN(PoC numpy 이식본, cosine, epsilon=0.15, event 전체 임베딩 재군집) → cluster_id 재조정
(overlap 매칭으로 번호 승계, 사용자 보정은 제약) → SQS 결과 큐 발행 / event `.npz` 갱신.

### 핵심 설계 결정사항

**face_align — 직접 구현 유지**: `insightface.utils.face_align` 대신 `_umeyama()`/`_ARCFACE_DST`를
코드 내 직접 구현해 insightface·skimage 의존성 제거(OpenCV+numpy만). 변환행렬 동등성 검증 완료.

**입력 품질이 임베딩 모델 선택보다 먼저다 (2026-07-14 실측 원칙,
[review](docs/reviews/2026-07-14-input-quality-alignment-landmark.md))**: 파이프라인이 스스로 주입하는
노이즈가 신원 신호보다 크다 — 같은 사진·같은 얼굴인데 `max_side`만 1600↔2400으로 바꿔도 임베딩
유사도가 **최저 0.43**까지 흔들린다(파편화 문제의 파편 간 거리 0.587보다 큰 변동). 원인은 ①
`align.py`의 `warpAffine` 기본 보간(INTER_LINEAR)이 최대 12.7배 축소에서 저역통과 없이 서브샘플링하는
에일리어싱, ② YuNet이 WIDER FACE 학습이라 초근접 대형 얼굴의 랜드마크가 불안정(얼굴폭 대비 최대 15.8%
이동). **따라서 정확도 개선은 반드시 정렬·랜드마크 → 그 다음 모델 순서로 접근한다.** 노이즈 바닥이
0.43인 파이프라인에 더 좋은 임베딩을 얹는 것은 밑 빠진 독에 물 붓기다. `max_side`를 올리는 방향은
오검출 폭증(28→85개)으로 해법이 아니다. **두 원인 모두 2026-07-15 교정 완료**(정렬 AA 프리블러 +
대형 얼굴 정규 스케일 재검출, 노이즈 바닥 0.33→0.69 — 완료된 목표 참조). 남은 선행 과제는 세션 간
거리 분포 측정(다음 구현 목표 0.5).

**임베딩 모델 교체 — 라이선스로 봉쇄됨 (2026-07-14 조사 확정)**: 무료 + 상용 가능 + AuraFace보다
판별력 우수한 모델은 **존재하지 않는다**. 라이선스는 코드가 아니라 **가중치·학습 데이터**에서 막힌다 —
InsightFace 모델 주는 "ALL models ... non-commercial research purposes only"(코드만 MIT), Glint360K는
"데이터셋 및 그 데이터로 학습된 모델"까지, WebFace260M은 "그 서브셋"까지 상용 금지. AdaFace(IR-101/
WebFace12M)는 동일 crop에서 파편 간 centroid 0.587→**0.809**로 압도적이지만 **상용 불가**이므로
정확도 기준선(yardstick)으로만 쓴다. LVFace의 HF `mit` 태그는 **함정**(본문은 비상용). AuraFace가
사실상 유일한 합법 선택지다. 유일한 합법적 성능 향상 경로는 **InsightFace 상용 라이선스 구매**이며,
결제 전 ⓐ 동아시아 코호트 수치(74.96 vs 백인 94.70) 자체 데이터 A/B 검증, ⓑ 유료 라이선스가 학습
데이터 출처까지 면책하는지 서면 확인이 **선행 조건**이다. TTA(좌우 반전 평균)·자체 학습·합성 데이터는
전부 실측/조사로 기각.

**HDBSCAN — PoC numpy 전용 이식본 사용** ([ADR 005](docs/decisions/005-hdbscan-standalone-port.md)):
알고리즘은 HDBSCAN 유지(단순 UnionFind 대비 ARI 2.7배 우수, ADR 002)하되 구현체는 sklearn이 아닌
PoC 이식본(라벨 완전 일치 검증, 의존성 제거). 파라미터(ARI 스윕 재확인, [ADR 009](docs/decisions/009-clustering-parameter-tuning.md)):
`min_cluster_size=2, min_samples=2, metric='cosine', cluster_selection_epsilon=0.15`. 재군집 후 결정적
후처리로 정확도 보강: 연결 성분 부분 승격([ADR 008](docs/decisions/008-blob-promotion-connected-components.md))
→ 제약 강제(보정 must/cannot-link + 같은 사진 자동 cannot-link — 같은 사진의 두 얼굴은 타인 확정,
[ADR 011](docs/decisions/011-same-photo-cannot-link.md)) → 파편 병합(완전 연결, 임계 0.55 —
분포 측정 기반 재보정, [ADR 012](docs/decisions/012-merge-threshold-recalibration.md)) → 노이즈
구제(전역 유사도 내림차순) → 저신뢰 `ambiguous` 분리(leave-one-out, 사람 제약 당사자만 보호) →
2차 파편 병합(구제·축출로 바뀐 최종 멤버십에 같은 임계 재적용,
[ADR 010](docs/decisions/010-post-rescue-second-merge.md)). 임계값은 전부 `ClusterConfig` 설정값.
클러스터링은 전체 비용 0.1% 미만.

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
Laplacian variance — 단 **주 인물 얼굴만**(최대 얼굴 폭의 50% 미만은 배경 인물로 보고 제외, 배경
아웃포커스는 흔들림 증거가 아님 — event 30 오탐 실측). 얼굴 미검출 시 전체 이미지 fallback: variance
폭락 OR 방향성 블러(그라디언트 방향 쏠림 ≥0.40 AND 정규화 variance <60 — variance는 텍스처 양을 재는
지표라 놓치는 사진을 손떨림의 방향 쏠림으로 잡는다, [ADR 014](docs/decisions/014-directional-blur-fallback.md)).
이미지 단위 "얼굴 1개라도 해당", 토글 ON시 `eyes_closed`/`blurry`로 분리·재군집 제외(request-scoped).
임계는 `QualityConfig`. 한계: 웃음 캔디드 오탐(표정 CNN 후속 필요), 부분 모션블러+선명 배경 사각지대,
회전 손떨림(전역 쏠림 낮음). 모델 소싱: `model_source.py`의 `UrlModelSource`.

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

macOS / Linux:
```bash
# 가상환경 생성 & 활성화
python3 -m venv .venv
source .venv/bin/activate

# 환경변수 준비 — .env.example을 .env로 복사해 실값 주입 (큐 URL·버킷명 확정 전까지는 placeholder)
cp .env.example .env
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
copy .env.example .env
```

이후는 플랫폼 공통:
```sh
pip install -r requirements.txt

# 워커 실행 (SQS consumer — 모델 적재 + SQS/S3 레디니스 통과 후 폴링 시작)
python -m app.worker

# AWS·모델 없이 전체 배선 자가 검증 (인메모리 페이크 e2e)
python -m app.worker --smoke

# pre-commit 훅 설치
pip install pre-commit
pre-commit install
```

`python -m app.worker`는 실 AWS에 붙으므로 자격증명이 필요하다. 로컬은 AWS SSO 프로필을 쓴다
(`.env`에 액세스 키를 넣지 않는다):
```sh
aws sso login --profile cheesemoa   # 세션 만료 시에도 이 한 줄
export AWS_PROFILE=cheesemoa        # PowerShell: $env:AWS_PROFILE = "cheesemoa"
```

AWS CLI v2 설치(`brew install awscli` / `winget install -e --id Amazon.AWSCLI`), SSO 프로필 최초
등록, 로컬 Docker + 실 AWS(SQS/S3) end-to-end 테스트 절차는
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
- 배포 완료 (2026-07-11): EC2에 Docker 컨테이너로 상시 실행 중 — 큐 URL·버킷명 확정 주입, 모델
  프리베이크(콜드스타트 해소), ECR `cheesemoa-ai`. 상세: [docs/guides/ec2-deployment.md](docs/guides/ec2-deployment.md)
- 자동 배포 (2026-07-11): main 푸시(=PR 머지) 시 GitHub Actions가 arm64 빌드 → 오프라인 스모크 →
  ECR 푸시 → SSM으로 EC2 컨테이너 교체 → 기동 로그 검증까지 수행 (`.github/workflows/deploy.yml`,
  OIDC 롤 `cheesemoa-github-actions-ai` — [ec2-deployment.md §5](docs/guides/ec2-deployment.md))
- `app/main.py`: 비어있음 (엔트리포인트는 `app/worker.py`)
- `healthcare_api.py`: FastAPI 학습용 샘플 코드 (실제 프로젝트 코드 아님)

### 다음 구현 목표
0. **[P0] 실 데이터 오염 대응** ([docs/backlog/2026-07-11-followups.md](docs/backlog/2026-07-11-followups.md)) —
   동일 사진 재업로드가 만든 중복 임베딩이 앨범을 쌍 단위로 쪼갬 + `delete_request` 미도달 유령 행
   (원인·재현: [reviews/2026-07-11-duplicate-embedding-split.md](docs/reviews/2026-07-11-duplicate-embedding-split.md))
0.5 **[P1] ADR-012 리스크 해소 — 아동 교차연령 재검증**: 병합 임계 0.55의 유일한 미검증 리스크는
   ADR-008의 "타인 centroid 최대 0.635"(face-test child 8명 교차연령, **교정 전** 임베딩)다.
   face-test 데이터 확보 시 교정 후 파이프라인으로 재측정해 ADR-012를 확정할 것. 아동 다수 실
   이벤트의 오병합 리포트가 트리거. 별건: YuNet이 화장품 팔레트 그림을 얼굴로 오검출
   ([분포 측정 리뷰](docs/reviews/2026-07-15-distance-distribution-verdict.md) §별건) —
   배경 인물 크기 필터는 [ADR 013](docs/decisions/013-background-face-size-filter.md)으로 구현됐으나
   팔레트는 크게 찍힌 오검출이라 못 거른다. score_threshold 또는 종횡비 필터가 여전히 후보.
1. 배포 후속 — 남은 항목: CloudWatch 지표 연동(로그는 완료 — 2026-07-14, awslogs 드라이버로
   `/cheesemoa/ai-worker` 직송, [cloudwatch-logging.md](docs/guides/cloudwatch-logging.md)), Spring 실계약 통합검증, 큐의 visibility timeout·
   redrive policy를 `.env.example` 메모대로 설정. **인스턴스 분리 검토**: 현재 워커가 Spring과 t4g.small
   (2코어·RAM 1846MB)을 공유하는데, t4g는 버스터블이라 실트래픽으로 추론이 지속되면 CPU 크레딧 소진 →
   Spring API까지 함께 스로틀된다 ([ec2-deployment.md](docs/guides/ec2-deployment.md) §리스크)
2. pytest 도입 — 각 모듈 `__main__` 스모크를 tests/로 승격 (`# TODO(CHMO-165)` 표시 지점)
3. (후속) 품질 게이트 개선 — 웃음 예외용 표정 CNN, 눈/흔들림 임계 라벨셋 튜닝(현재 라벨 부재), 부분 블러 대응
4. Spring과 `confirm_distinct` 트리거 정책 합의 — 즉시 발행(안전) vs 공유 시점 일괄 발행(발행 전 새
   업로드가 끼면 그 사이 재군집은 보호 공백). 단, 상태 기반 계약 개편
   ([docs/backlog/state-based-feedback-contract.md](docs/backlog/state-based-feedback-contract.md))
   채택 시 자동 해소되는 항목

### 완료된 목표
- **배경 인물(행인) 앨범 방지 — 얼굴 크기 필터** (2026-07-15,
  [ADR 013](docs/decisions/013-background-face-size-filter.md)) — event 27에서 배경에 멀리 찍힌
  행인(이미지 긴 변의 0.8%)이 사진 2장에 반복 등장해 앨범이 생긴 문제. 실사진 16개 이벤트 210
  얼굴 분포 측정에서 행인 최대 0.82% vs 앨범 얼굴 최소 3.29%의 빈 구간을 확인, 검출 단계에
  rel_w(bbox 폭/긴 변) 2.5% 하한을 추가(`DETECT_MIN_FACE_REL_WIDTH`, 0=비활성). 앨범 손실 0 +
  노이즈 얼굴 52% 제거 실측. 절대 px 기준은 저해상도 업로드를 잘라 기각. 앨범 최소값(3.29%)
  쪽 마진이 1.3배로 얇으니 앨범 사진 누락 리포트 시 이 값부터 하향 검토.
- **병합 임계 재보정 0.68 → 0.55 — 파편화 주 원인 제거** (2026-07-15,
  [ADR 012](docs/decisions/012-merge-threshold-recalibration.md),
  [분포 측정 리뷰](docs/reviews/2026-07-15-distance-distribution-verdict.md)) — 교정 후 라벨 코퍼스
  (5인, 동일인 103쌍·타인 781쌍)에서 타인 최고 0.4584 vs 동일인 쌍 77%가 0.68 미달을 실측. ARI 스윕
  0.45~0.60 고원(5인 완벽 분리) 중앙 0.55 채택, test4가 기본값만으로 앨범 정확히 2개. 유료 라이선스
  검토는 근거 상실로 보류. 미검증 리스크(아동 교차연령)는 다음 구현 목표 0.5.
- **입력 품질 교정 — 정렬 AA + 랜드마크 2단계 정제** (2026-07-15,
  [review §구현 결과](docs/reviews/2026-07-14-input-quality-alignment-landmark.md)) — `align.py`에 ROI 한정
  가우시안 프리블러(σ=(1/s)/2, 확대 경로는 픽셀 동일 유지), `detect.py`에 대형 얼굴 정규 스케일 재검출
  (실패 시 원 랜드마크 폴백, 파라미터는 스윕으로 **224/0.75** 확정 — 리뷰 초기값 160/0.5보다 우수).
  같은 얼굴 임베딩 유사도(max_side 스윕) 평균 0.9133→**0.9596**, 최저 0.3254→**0.6881**, 랜드마크 지터
  최대 26.5%→11.0%. 토글 5종(`ALIGN_ANTIALIAS`·`DETECT_REFINE_*` 등)으로 .env 롤백 가능.
- **EC2 배포 + ORT 스레드 정합** (2026-07-11) — Docker 이미지(arm64, 모델 프리베이크)를 ECR `cheesemoa-ai`로
  올려 EC2에서 상시 실행. 배포 직후 임베딩이 로컬 대비 45배 느렸는데, 원인은 CPU 크레딧 스로틀링이 아니라
  ORT 스레드 오버서브스크립션(2코어 호스트에 기본값 8스레드)이었다 — 코어 수 정합으로 **6배 개선**
  (2860ms/장 → 476ms/장). 실측: [worker-scaling-and-performance.md §7](docs/guides/worker-scaling-and-performance.md)
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
