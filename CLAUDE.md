# CheeseMoa-AI

## 프로젝트 개요

**치즈모아** (CheeseMoa) — "사진이 주인을 찾아가는 AI 공유 앨범"

여러 사람이 함께 촬영한 사진을 AI 얼굴 인식으로 자동 분류·공유하는 서비스. 이 레포는 **Python 워커 기반
AI 추론 서버**로, Spring 백엔드로부터 AWS SQS를 통해 작업을 받아 얼굴 감지 → 정렬 → 임베딩 → 클러스터링
파이프라인을 실행한다.

- GitHub 조직: [CheeseMoa](https://github.com/CheeseMoa) · Jira 프로젝트: CHMO (티켓 접두사 `CHMO-XX`)

---

## 전체 시스템 아키텍처

Flutter App → Spring API Server(+ PostgreSQL 메타데이터, Presigned URL 발급) → AWS S3(원본 이미지) →
AWS SQS(분류 작업 큐) → **[이 서버] Python AI Worker**(YuNet 감지 → Umeyama 정렬 → AuraFace 임베딩 →
HDBSCAN 클러스터링) → S3(event 단위 `.npz` 임베딩 저장) + SQS 결과 큐 → Spring이 결과를 PostgreSQL에 저장.

Spring이 SQS 요청 큐에 작업을 발행하면 워커가 consumer로 소비해 파이프라인을 실행하고 결과를 SQS 결과
큐로 발행한다. HTTP는 사용하지 않는다. classify 중에는 처리 장수를 별도 progress 큐로도 발행해 백엔드가
진행바를 그린다(CHMO-274, 큐 미설정 시 비활성 — [message-examples §⑤](docs/spec/message-examples.md)).

---

## AI 파이프라인

### 흐름

S3 이미지 읽기 → YuNet 감지(+아래 검출 방어층) → face_align(직접 구현 Umeyama, 112×112 ArcFace 기준점) →
AuraFace(512-dim 임베딩) → 품질 게이트(눈감음·흔들림, 토글 ON시 `eyes_closed`/`blurry` 분리·재군집 제외) →
HDBSCAN(event 전체 임베딩 재군집) → cluster_id 재조정(overlap 승계, 사용자 보정은 제약) →
SQS 결과 발행 / event `.npz` 갱신.

### 검출 방어층 (각 규칙의 근거·실측·한계·재보정 이력은 해당 ADR 참조)

- 배경 인물 필터: bbox 폭 < 이미지 긴 변의 2.5% 제거 — [ADR 013](docs/decisions/013-background-face-size-filter.md)
- 대형 오검출 결합 필터: score<0.78 AND 종횡비(w/h)<0.70 제거 —
  [ADR 015](docs/decisions/015-detection-false-positive-combined-filter.md)
- 대형 근접 얼굴 재검출 회복: rel_w≥0.20 저score(<0.6) 후보를 정규 스케일 재검출, score≥0.80이면 되살림
  (YuNet이 초근접 대형 얼굴에 저score를 주는 문제) — [ADR 017](docs/decisions/017-size-aware-detection-score-threshold.md)
- 재검출 랜드마크 신뢰: 재검출 score≥0.80이면 이동량 가드를 무시하고 재검출 랜드마크 채택(파편 bbox 기준
  가드가 올바른 교정을 오판하던 문제) — [ADR 023](docs/decisions/023-refine-trust-redetect-landmarks.md)
- confident 파편 디둡: 정제 랜드마크 중심거리 <0.1×얼굴폭인 대형(>224px) 쌍은 score 최상 박스만 유지
  (1인 셀피가 2명 단체로 오인) — [ADR 027](docs/decisions/027-duplicate-face-fragment-dedup.md)
- 크기 인지형 confident 게이트: 대형(rel_w≥0.20)은 게이트 0.70, [0.6,0.70) 구간은 회복 재검출로 재판정
  (손·종이 오검출 유령 앨범 차단, 소형은 0.6 유지) — [ADR 028](docs/decisions/028-size-aware-confident-score-gate.md)

### 핵심 설계 결정

**face_align 직접 구현**: `_umeyama()`/`_ARCFACE_DST`를 코드 내 구현해 insightface·skimage 의존성 제거
(OpenCV+numpy만). 변환행렬 동등성 검증 완료.

**입력 품질이 임베딩 모델보다 먼저다**
([2026-07-14 review](docs/reviews/2026-07-14-input-quality-alignment-landmark.md)): 파이프라인 자체 주입
노이즈(정렬 에일리어싱 + 초근접 얼굴 랜드마크 불안정)가 신원 신호보다 컸다(`max_side`만 바꿔도 동일 얼굴
유사도 최저 0.43). **정확도 개선은 반드시 정렬·랜드마크 → 그 다음 모델 순서로 접근한다.** 두 원인 모두
2026-07-15 교정 완료(정렬 AA 프리블러 + 정규 스케일 재검출, 노이즈 바닥 0.33→0.69).

**임베딩 모델 교체는 라이선스로 봉쇄** (2026-07-14 조사 확정): 무료 + 상용 가능 + AuraFace보다 우수한
모델은 **존재하지 않는다** — 라이선스는 코드가 아니라 가중치·학습 데이터에서 막힌다(InsightFace 모델 주
비상용, Glint360K·WebFace260M은 파생 모델까지 금지, LVFace의 HF `mit` 태그는 함정 — 본문은 비상용).
AdaFace는 판별력 압도적(파편 간 centroid 0.587→0.809)이지만 상용 불가라 정확도 기준선(yardstick) 전용.
자체 호스팅의 유일한 합법 성능 향상 경로는 InsightFace 상용 라이선스 구매이며, 결제 전 ⓐ 동아시아
코호트(74.96 vs 백인 94.70) 자체 A/B ⓑ 학습 데이터 출처 면책 서면 확인이 선행 조건. TTA·자체 학습·
합성 데이터는 전부 기각. **관리형 API는 이 봉쇄 밖이다** — AWS Rekognition(서울 리전, 기존 수탁자)이
AuraFace가 원리적으로 못 가르는 하드케이스 대역을 실측으로 갈랐고, uncertain 재판정 보조 신호로 **구현
완료·기본 활성**(`app/pipeline/rejudge.py`, CHMO-420 — `REJUDGE_ENABLED=false` 롤백,
[ADR 030](docs/decisions/030-rekognition-uncertain-rejudge.md) ·
[실측 리뷰](docs/reviews/2026-07-23-rekognition-uncertain-ab.md)). **동일 인물 앨범 쪼개짐 회수는
같은 API의 별건**이다 — 얼굴 단위 재판정은 미배정 얼굴만 보므로 앨범 A·B가 둘 다 형성되면 훅이 아예
실행되지 않는다. 앨범 쌍 재판정은 미구현·설계 확정 상태
([ADR 031](docs/decisions/031-rekognition-cluster-pair-merge.md), §다음 구현 목표 0.2).

**HDBSCAN — PoC numpy 이식본** ([ADR 005](docs/decisions/005-hdbscan-standalone-port.md), 파라미터 스윕
[ADR 009](docs/decisions/009-clustering-parameter-tuning.md)): `min_cluster_size=2, min_samples=2,
metric='cosine', cluster_selection_epsilon=0.15`. 재군집 후 결정적 후처리로 정확도 보강(순서 고정):
연결 성분 부분 승격([ADR 008](docs/decisions/008-blob-promotion-connected-components.md)) → 제약 강제
(보정 must/cannot-link + 같은 사진 자동 cannot-link, [ADR 011](docs/decisions/011-same-photo-cannot-link.md))
→ 파편 병합(centroid 0.55 [ADR 012](docs/decisions/012-merge-threshold-recalibration.md) AND face-pair
평균 0.475 [ADR 016](docs/decisions/016-merge-facepair-cohesion-gate.md), 승인은 컴포넌트 '현재 전체 멤버'
재평가 [ADR 024](docs/decisions/024-merge-component-linkage.md)) → 노이즈 구제(전역 유사도 내림차순) →
저신뢰 ambiguous 분리(leave-one-out + 회색지대 face-pair 축출,
[ADR 020](docs/decisions/020-evict-facepair-gray-gate.md)) → margin 구제(top1≥0.40 AND 2위 군집 대비
1.7배 여유 — 절대 임계가 못 붙이는 [0.40, 0.60) 대역 재심, 실 이벤트 적대 검증 후 활성화
[2026-07-23 리뷰](docs/reviews/2026-07-23-margin-gate-real-event-validation.md)) → 2차 파편 병합
([ADR 010](docs/decisions/010-post-rescue-second-merge.md)). 재군집 입력은 근중복 행(≥0.985 — 재업로드·
유령 행 복제)을 대표 1행으로 접고 결과에서 펼친다([ADR 029](docs/decisions/029-duplicate-embedding-collapse.md),
쌍 앨범 와해 방어). 임계는 전부 `ClusterConfig` 설정값. 클러스터링은 전체 비용 0.1% 미만.

**전체 재군집 + ID 재조정 (정확도 최우선)** ([ADR 007](docs/decisions/007-embedding-storage-s3.md)):
재군집 격리 단위는 **event**. 군집의 진실은 항상 event 전체 임베딩(S3 `.npz` 보관) 재군집이며, 파티션은
기존 클러스터와 overlap 최대 매칭으로 `cluster_id` 승계(대응 없는 군집만 신규 인물). 사용자 보정
(merge/split/reassign/confirm_distinct)은 must/cannot-link 제약으로 반영해 재군집이 사람 결정을 뒤집지
않게 한다. uncertain("분류가 어려워요") 사진은 예약 앨범 id `"__uncertain__"`로 reassign 편입을 받고,
항목마다 주 인물 얼굴 bbox 배열 `face_bboxes`(CHMO-407)와 분류 어려움 이유 `causes`(CHMO-404)를 동봉 —
계약 상세·결정 배경은 [feature-spec §6.2·§6.3](docs/spec/feature-spec.md)·
[message-examples §④](docs/spec/message-examples.md). uncertain 확정 직전에는 (CHMO-420 기본 활성)
Rekognition CompareFaces 재판정 훅이 하드케이스를 자동 편입(≥90, must-link 기록 후 2차 재군집)·제안
(85~94, `suggestions` 동봉)한다 — 점수는 (face, 대표) 쌍으로 S3 캐싱해 재군집 재과금 없음
([ADR 030](docs/decisions/030-rekognition-uncertain-rejudge.md)). 대표벡터(L2 정규화 평균)는 조회·표시용 파생
캐시일 뿐 군집 판단의 원천이 아님(feature-spec §4).

**품질 게이트 — 눈감음/흔들림 (CHMO-172)** (`app/pipeline/quality.py`, feature-spec §7 註): 눈감음은
Face Landmarker blendshape litert 이식본(`app/pipeline/blink.py`, min(eyeBlinkL/R) ≥ 0.40 —
[ADR 021](docs/decisions/021-blink-blendshape-litert.md))로 판정하되, 주 인물(최대 얼굴 폭의 50% 이상,
[ADR 022](docs/decisions/022-eye-main-face-ratio.md)) AND 이미지 긴 변의 8% 이상
([ADR 026](docs/decisions/026-eye-closed-relative-size-gate.md)) 얼굴만. `QUALITY_BLINK_THRESHOLD=0`
롤백 시에만 종전 눈 CNN + 판정 자격 게이트([ADR 019](docs/decisions/019-eye-judgment-eligibility-gate.md)).
흔들림은 얼굴 crop Laplacian variance(주 인물만), 얼굴 미검출 시 전체 이미지 fallback — variance 폭락 OR
방향성 블러([ADR 014](docs/decisions/014-directional-blur-fallback.md)). variance 기반 blurry 판정은
최종적으로 흔들림 재확인 게이트(방향 쏠림 바닥 0.35, whole_var·face_var 붕괴 면제 3종 —
[ADR 018](docs/decisions/018-shake-coherence-floor.md) §보강 1~3)를 거친다. 임계는 `QualityConfig`,
알려진 한계·사각지대 목록은 ADR 018·021·022·026 각 §한계 참조.

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 얼굴 감지 | YuNet (OpenCV DNN) |
| 얼굴 정렬 | 직접 구현 (OpenCV + numpy, Umeyama) |
| 얼굴 임베딩 | AuraFace (onnxruntime CPU) |
| 클러스터링 | HDBSCAN (PoC numpy 이식본) — event 단위 전체 재군집 + cluster_id 재조정 |
| 품질 판정 | 눈감음 blendshape (Face Landmarker tflite ×2, ai-edge-litert) + 흔들림 (Laplacian variance) |
| 모델 소싱 | `app/core/model_source.py` — YuNet·AuraFace는 HF Hub, 그 외는 URL(`UrlModelSource`) |
| 임베딩 저장 | S3 (event 단위 `.npz`) — [ADR 007](docs/decisions/007-embedding-storage-s3.md) |
| 메시지 큐 | AWS SQS · 데이터 검증: Pydantic v2 |
| 환경 변수 | pydantic-settings + python-dotenv (`app/core/config.py`의 `Settings` 단일 클래스) |

---

## 프로젝트 구조

```
CheeseMoa-AI/
├── app/
│   ├── worker.py            # SQS consumer 엔트리포인트 (폴링 루프 + 오류 정책, --smoke 자가 검증)
│   ├── handlers.py          # 인바운드 3종(classify/feedback/delete) 처리 (ADR-007 재군집 흐름)
│   ├── core/                # 설정(config.py)·프로덕션 조립(deps.py)·모델 소싱(model_source.py)
│   ├── messaging/           # SQS 수신·발행 + 인메모리 페이크
│   ├── storage/             # event .npz 코덱·저장소·이미지 소스·썸네일 저장소 + 인메모리 페이크
│   ├── pipeline/            # detect(YuNet)·align(Umeyama)·embed(AuraFace)·cluster(재군집)·
│   │                        # quality(품질 게이트)·blink(눈감음)·thumbnail(대표 얼굴)·hdbscan_standalone
│   └── schemas/             # Pydantic 스키마 (SQS 메시지)
├── .env.example             # 환경변수 예시
└── .pre-commit-config.yaml  # ruff linter + formatter (저장 시 포맷은 .vscode/settings.json)
```

`app/main.py`는 비어 있고(엔트리포인트는 `app/worker.py`), `healthcare_api.py`는 학습용 샘플이다.

---

## 개발 환경 세팅

```sh
python3 -m venv .venv && source .venv/bin/activate  # Windows: python -m venv .venv && .venv\Scripts\activate
cp .env.example .env                                # Windows: copy — 실값 주입 전까지는 placeholder
pip install -r requirements.txt
pip install pre-commit && pre-commit install
python -m app.worker --smoke                        # AWS·모델 없이 전체 배선 자가 검증 (인메모리 페이크 e2e)
python -m app.worker                                # 실 워커 (모델 적재 + SQS/S3 레디니스 후 폴링)
```

실 AWS는 SSO 프로필을 쓴다(`.env`에 액세스 키를 넣지 않는다): `aws sso login --profile cheesemoa` 후
`export AWS_PROFILE=cheesemoa`(PowerShell: `$env:AWS_PROFILE`). AWS CLI 설치·SSO 최초 등록·로컬 Docker +
실 AWS e2e 절차: [local-docker-e2e-testing.md](docs/guides/local-docker-e2e-testing.md).

## 코드 컨벤션

- **들여쓰기/포맷**: 스페이스 2칸(AI/Flutter/Web 공통), ruff 자동 포맷(pre-commit `ruff --fix` +
  `ruff-format`), 최대 줄 길이 120자
- **네이밍**: 클래스 `PascalCase` · 함수/변수 `snake_case` · 상수 `UPPER_SNAKE_CASE`
- **주석**: WHY가 불명확한 경우에만(숨겨진 제약·수학적 불변식 등), 할 일은 `# TODO:`

## Git 컨벤션

- **브랜치 (Git Flow)**: `main`(배포) · `feature/CHMO-XX-설명` — 예: `feature/CHMO-54-face-detection-api`
- **커밋**: `[CHMO-XX] type: 메시지` (type: `feat`·`fix`·`docs`·`style`·`refactor`·`test`·`chore`)

## 현재 상태

- 파이프라인·워커 계층 전부 구현 완료. 검증은 각 모듈 `__main__` 자가검증 + `python -m app.worker --smoke`.
- 배포 완료(2026-07-11): EC2 Docker 상시 실행(arm64, 모델 프리베이크, ECR `cheesemoa-ai`), main 푸시 시
  GitHub Actions 자동 배포(빌드→오프라인 스모크→ECR→SSM 컨테이너 교체) —
  [ec2-deployment.md](docs/guides/ec2-deployment.md), 로그는 CloudWatch `/cheesemoa/ai-worker` 직송.

### 다음 구현 목표

0. **[P0] Rekognition 재판정 — 운영 잔여분** — 워커 구현·기본 활성화(재판정 훅·점수 캐싱·
   `suggestions` 계약, [ADR 030](docs/decisions/030-rekognition-uncertain-rejudge.md))는 2026-07-24
   완료(CHMO-420, `REJUDGE_ENABLED=false` 롤백). 잔여: ⓐ **운영 전제 조건 확인** — AWS AI 학습
   opt-out(샌드박스 계정은 설정 불가 — 운영자 요청/정식 계정 이전 필요)·처리방침 위탁 문구·민감정보
   동의 점검(리뷰 §선행 조건, 미충족 환경은 false 배포) ⓑ 워커 IAM에 `rekognition:CompareFaces`
   권한(없으면 best-effort 폴백으로 종전 동작 + 경고 로그) ⓒ Spring `suggestions` 소비·제안 UX
   합의(미구현이어도 빈/미소비 필드로 무해) ⓓ 파편 병합 힌트 wire 계약(현재 로그만 — 목표 0.2가 흡수).
0.1 **[P0] 실 데이터 오염 대응 — 잔여분** — 워커 방어(근중복 행 붕괴, ADR 029)는 2026-07-23 완료.
    잔여: Spring ETag 재업로드 검사 + `delete_request` 발행 검증(PIPA, 유령 행 자체는 여전히 존재) +
    S3 버킷 버저닝 ([backlog](docs/backlog/2026-07-11-followups.md) ·
    [원인·재현](docs/reviews/2026-07-11-duplicate-embedding-split.md))
0.2 **[P0] 앨범 쪼개짐 회수 — Rekognition 앨범 쌍 재판정 (미구현, ADR 초안 확정)** — 오병합은 거의
    없는데 같은 인물이 두 앨범으로 갈라지는 문제. 실측 34개 이벤트 중 12개(35%)에 회수 가능한 쪼개짐이
    있고, 회수 대상 13쌍 중 12쌍이 파편병합 face-pair 바닥(ADR 016)에 막힌 쌍이다 — 그 바닥은 아동
    체인 융합 때문에 낮출 수 없다(기각 리뷰). 설계: 회색지대 앨범 쌍(centroid ≥0.35, 같은 사진 공존은
    호출 전 탈락) × 대표 2장씩 K×K → **전원 ≥90 AND 산포 ≤5**일 때만 `soft_merge`로 병합(must-link는
    event 134 회귀 재현이라 금지). 3개 이상이 합쳐질 땐 컴포넌트 완전 연결 요구 — 다리 타는 체인 융합
    차단(ADR 024와 같은 계열). 대표 1쌍 argmax는 실측 기각(확실한 타인 쌍이 `[99.99, 33.7, 2.0, 1.6]`).
    비용 이벤트당 최초 $0.19 + 월 $1.48(아이 20명)
    ([ADR 031](docs/decisions/031-rekognition-cluster-pair-merge.md) ·
    [실측](docs/reviews/2026-07-24-rekognition-cluster-pair-survey.md))
0.5 **[P1] 화장품 팔레트 그림 오검출** — 크게 찍힌 얼굴 그림은 크기·score·종횡비 필터 전부 정상값으로
    통과, 임베딩 단계 신호가 필요한 별도 문제
    ([분포 조사](docs/reviews/2026-07-15-detect-score-aspect-survey.md) §한계)
1. 배포 후속 — CloudWatch 지표 연동, Spring 실계약 통합검증, 큐 visibility timeout·redrive 설정,
   **인스턴스 분리 검토**(Spring과 t4g.small 공유 — 버스터블 크레딧 소진 시 API까지 스로틀,
   [ec2-deployment.md](docs/guides/ec2-deployment.md) §리스크)
2. pytest 도입 — 각 모듈 `__main__` 스모크를 tests/로 승격(`# TODO(CHMO-165)` 표시 지점)
3. 품질 게이트 후속 — 눈/흔들림 임계 라벨셋 튜닝(라벨 부재), 부분 블러, 잔존 사각지대는 ADR 018·019·
   021·026 §한계 참조(웃음 표정 CNN·원본 해상도 눈 crop은 실측 기각 —
   [완료 이력](docs/completed-goals.md) 참조)
4. `confirm_distinct` 트리거 정책 Spring 합의 — 즉시 발행 vs 공유 시점 일괄 발행. 단
   [상태 기반 계약 개편](docs/backlog/state-based-feedback-contract.md) 채택 시 자동 해소

### 완료된 목표

이력 전문(30건 — 문제·원인·해법·실측 검증·롤백 스위치 기록)은
[docs/completed-goals.md](docs/completed-goals.md)로 이동했다. **새 목표를 완료하면 CLAUDE.md가 아니라
그 파일 맨 위에 같은 형식으로 추가할 것.** 최근 3건: Rekognition uncertain 재판정(ADR 030, 2026-07-24) ·
근중복 행 붕괴(CHMO-419/ADR 029, 2026-07-23) · `face_bboxes` 배열 계약 교체(CHMO-407, 2026-07-22).
