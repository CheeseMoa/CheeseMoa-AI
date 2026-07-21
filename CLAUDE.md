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
미만인 얼굴 제거, [ADR 013](docs/decisions/013-background-face-size-filter.md) + 대형 오검출 결합 필터 —
score<0.78 AND 종횡비(w/h)<0.70 제거, [ADR 015](docs/decisions/015-detection-false-positive-combined-filter.md)
+ 대형 근접 얼굴 재검출 회복 — rel_w≥0.20 저score(<0.6) 후보를 정규 스케일 재검출해 재검출 score≥0.80이면
되살림, YuNet이 초근접 대형 얼굴에 저score를 줘 공통첩으로 빠지던 문제, rel_w 하한 0.30→0.20 재보정 —
화면을 덮는 얼굴은 bbox 파편이 rel_w 0.28대로 나와 게이트 미달(event 60 미검출),
[ADR 017](docs/decisions/017-size-aware-detection-score-threshold.md) §재보정
+ 재검출 랜드마크 신뢰 — 재검출 score≥0.80이면 랜드마크 이동량 가드를 무시하고 재검출 랜드마크 채택,
가드 기준이 파편 bbox 폭이라 초대형 얼굴의 올바른 교정을 오매칭으로 오판해 깨진 랜드마크가 유지되고
그 쓰레기 임베딩이 공통첩 유출·유령 앨범을 만들던 문제(event 73),
[ADR 023](docs/decisions/023-refine-trust-redetect-landmarks.md)
+ confident 파편 디둡 — 초대형 얼굴의 파편 박스 2개가 둘 다 score 게이트를 통과하면(YuNet NMS도
IoU 0.297<0.3으로 미발동) 한 사람이 두 명으로 검출됨, 정제 랜드마크 중심거리 < 0.1×얼굴폭인 대형
(>224px) 쌍은 score 최상 박스만 남김 — 1인 셀피가 "주 인물 2명 단체"로 공용 앨범에 노출되던 문제
(event 105), [ADR 027](docs/decisions/027-duplicate-face-fragment-dedup.md)) →
face_align(직접 구현 Umeyama, 112×112 ArcFace 기준점) →
AuraFace(512-dim 임베딩) → 품질 게이트(눈감음 CNN + 흔들림 Laplacian, 토글 ON시 eyes_closed/blurry로 분리·
재군집 제외) → HDBSCAN(PoC numpy 이식본, cosine, epsilon=0.15, event 전체 임베딩 재군집) → cluster_id 재조정
(overlap 매칭으로 번호 승계, 사용자 보정은 제약) → SQS 결과 큐 발행 / event `.npz` 갱신.
classify 처리 중에는 이미지 루프 도중 처리 장수를 별도 progress 큐로도 발행해 백엔드가 분류
진행바를 그린다(CHMO-274, 큐 미설정 시 비활성 — [message-examples §⑤](docs/spec/message-examples.md)).

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
[ADR 011](docs/decisions/011-same-photo-cannot-link.md)) → 파편 병합(centroid 임계 0.55
[ADR 012](docs/decisions/012-merge-threshold-recalibration.md) AND 파편 간 face-pair 평균 바닥 0.475 —
아동 교차연령 오병합 차단, [ADR 016](docs/decisions/016-merge-facepair-cohesion-gate.md); 병합 승인은
컴포넌트 '현재 전체 멤버' 재평가로 판정 — 구 스냅샷 완전 연결이 2얼굴 파편의 노이즈 centroid 쌍에
걸려 같은 인물 앨범을 쪼개던 문제(event 90) 해소, 다리(bridge) 융합은 남남 쌍이 전체 face-pair
평균을 끌어내려 여전히 차단, [ADR 024](docs/decisions/024-merge-component-linkage.md)) → 노이즈
구제(전역 유사도 내림차순) → 저신뢰 `ambiguous` 분리(leave-one-out, 사람 제약 당사자만 보호
+ 회색지대(LOO<0.46) 멤버는 클러스터 내 최강 face-pair<0.45면 남남 부착으로 축출 —
동일인/남남 LOO가 [0.40,0.46)에서 섞여 전역 바닥 상향은 불가, 판별 신호는 쌍에 남는다,
[ADR 020](docs/decisions/020-evict-facepair-gray-gate.md)) →
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
  미매칭 얼굴을 must-link로 인물 앨범 편입 (계약 확장, feature-spec §6.2·§6.3); 항목마다 주 얼굴 bbox
  `face_bbox`(원본 px, null 가능)도 동봉 — 앱 상세 화면 얼굴 crop용 (계약 확장 CHMO-388, 완료된 목표 참조)
- must-link는 "같이 있어야 한다"만 강제할 뿐 "떨어져 있어야 한다"는 강제 못해, 확정된 두 인물 앨범 사이로
  유사도가 애매한 신규 사진(다리 사진)이 들어오면 오병합 위험이 있음 → `confirm_distinct` 액션(계약 확장,
  feature-spec §6.3)으로 `cluster_ids`(2개 이상)의 대표 얼굴 전 쌍에 cannot-link — merge의 반대 방향 선언

대표벡터(L2 정규화 평균)는 조회·표시용 파생 캐시일 뿐 군집 판단의 원천이 아님. 상세: [docs/spec/feature-spec.md](docs/spec/feature-spec.md) §4.

**품질 게이트 — 눈감음/흔들림 (CHMO-172)** (`app/pipeline/quality.py`, feature-spec §7 註): 눈감음은
Face Landmarker blendshape(`app/pipeline/blink.py` — mediapipe pip이 linux aarch64 휠이 없어
face_landmarker.task 내부 tflite 2개를 ai-edge-litert로 직접 실행하는 이식본, 파리티 |Δ| med 0.008·
판정 뒤집힘 0)로 판정한다 — YuNet 5점 RoI → 478 랜드마크 → min(eyeBlinkL/R) ≥ 0.40, presence <0.5는
미판정(CNN 폴백은 정탐 기여 0으로 제거), [ADR 021](docs/decisions/021-blink-blendshape-litert.md).
눈감음도 흔들림처럼 **주 인물 얼굴만** 판정한다(최대 얼굴 폭의 50% 미만은 배경 인물로 보고 미판정,
blink·CNN 경로 공통 — 프레임 끝 행인의 감은 눈이 주인물 2명 뜬 사진을 eyes_closed로 보내던
event 69 실측, [ADR 022](docs/decisions/022-eye-main-face-ratio.md)); 추가로 bbox 폭이 이미지 긴 변의
8% 미만인 원거리 얼굴도 미판정 — blink가 멀리 찍힌 얼굴의 "아래 쳐다봄"을 감은 눈으로 오탐하던 event 99,
[ADR 026](docs/decisions/026-eye-closed-relative-size-gate.md).
`QUALITY_BLINK_THRESHOLD=0` 롤백 시에만 종전 눈 CNN(`open-closed-eye-0001`) + 판정 자격 게이트
(bbox 짧은 변 ≥64px AND 눈/볼 밝기 비 ≤1.4 — 그림·가림 오탐 제외,
[ADR 019](docs/decisions/019-eye-judgment-eligibility-gate.md)) 경로를 쓴다. 흔들림은 얼굴 crop
Laplacian variance — 단 **주 인물 얼굴만**(최대 얼굴 폭의 50% 미만은 배경 인물로 보고 제외, 배경
아웃포커스는 흔들림 증거가 아님 — event 30 오탐 실측). 얼굴 미검출 시 전체 이미지 fallback: variance
폭락 OR 방향성 블러(그라디언트 방향 쏠림 ≥0.35 AND 정규화 variance <60 — 0.40→0.35 재보정, ADR 014
§재보정 — variance는 텍스처 양을 재는
지표라 놓치는 사진을 손떨림의 방향 쏠림으로 잡는다, [ADR 014](docs/decisions/014-directional-blur-fallback.md)).
이미지 단위 "얼굴 1개라도 해당", 토글 ON시 `eyes_closed`/`blurry`로 분리·재군집 제외(request-scoped).
variance 기반 blurry 판정(얼굴·fallback 공통)은 최종적으로 흔들림 재확인 게이트를 거친다 — 전체
이미지 방향 쏠림이 바닥(0.35) 미만이면 손떨림이 아니라 소프트 원판(옛날 인화 재촬영·앱 스무딩)으로
보고 해제, variance는 잔결의 양만 재서 둘을 구분 못한다([ADR 018](docs/decisions/018-shake-coherence-floor.md)).
단 whole_var가 붕괴 수준(<40)이면 게이트 면제·흔들림 확정 — 고스팅형 손떨림은 쏠림이
낮게 나온다(event 55 실측, ADR 018 보강). 면제는 fallback + **대형 주 인물 얼굴 경로**(blurry 얼굴
rel_w ≥ 0.22 — 얼굴이 화면 대부분이면 whole_var가 얼굴 자체를 재므로 fallback 논리가 이식되고,
옛날 인화 오탐은 얼굴이 작거나 whole_var 미붕괴로 걸러진다. 고스팅 셀피가 대형 얼굴 회복 검출로
얼굴 경로에 빠져 uncertain으로 새던 event 64 해소, ADR 018 §보강 2) + **소형 얼굴 face_var 붕괴**
(blurry 얼굴 최저 face_var<7 AND whole_var<40이면 얼굴·이미지 잔결이 둘 다 붕괴 = 소프트 원판을
넘어선 흔들림 확정 — 소형 얼굴 하나가 얼굴 경로로 새서 fallback 붕괴 면제를 못 받던 test9 흔들린
단체샷 미탐 해소, whole_var 결합은 선명 사진의 어둡거나 배경인 얼굴 오탐 방어(실 이벤트 event 51),
`QUALITY_FACE_VAR_COLLAPSE_FLOOR`, ADR 018 §보강 3)에 적용된다.
임계는 `QualityConfig`. 한계: 부분 모션블러+선명 배경 사각지대, 소형 얼굴(rel_w<0.22)의 고스팅·
회전 손떨림·등방 블러 중 face_var 10~25 구간(전역 쏠림 낮음 — ADR 018 게이트가 해제하는 방향,
옛날 인화 face_var와 겹쳐 §보강 3 붕괴 면제로도 못 가름), 아웃포커스 주 인물(등방이라 동일), 눈감음 presence 미달
~9% 미판정(실측 감음 손실 0 — 코퍼스 한정, ADR 021 §한계), blendshape는 눈 감김을 문자 그대로
재서 웃으며 감은 캔디드도 잡힘(의미론 — 임계로 조정. 단 아래 쳐다봄은 threshold로 못 가르나
원거리는 rel_w 게이트로 제외, 프레임에서 큰 내려뜸은 잔존 — ADR 026 §한계), 최대 얼굴의 절반 미만
또는 이미지 긴 변의 8% 미만 크기 얼굴은 눈감음 미판정이라 그런 구도의 일행 감음은 놓침(ADR 022·026 트레이드오프 — blur와 동일). CNN 시절 한계였던 유아 오탐·보정 이미지
미탐·수면 미탐·웃음 오탐 우려는 ADR 021로 해소.
모델 소싱: `model_source.py`의 `UrlModelSource`.

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 실행 런타임 | Python 워커 프로세스 |
| 얼굴 감지 | YuNet (OpenCV DNN) |
| 얼굴 임베딩 | AuraFace (onnxruntime CPU) |
| 얼굴 정렬 | 직접 구현 (OpenCV + numpy, Umeyama) |
| 클러스터링 | HDBSCAN (PoC numpy 이식본) 전체 재군집 + cluster_id 재조정 (event 단위) |
| 품질 판정 | 눈감음 blendshape (Face Landmarker tflite ×2, ai-edge-litert — ADR 021, 롤백 시 눈 CNN) + 흔들림 (Laplacian variance, OpenCV) |
| 모델 소싱 | `app/core/model_source.py` — YuNet·AuraFace는 HF Hub, 눈감음 CNN·face_landmarker.task는 URL(`UrlModelSource`) |
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
│   │                        # 원본 이미지 소스(image_source.py)·썸네일 저장소(thumbnail_store.py) + 인메모리 페이크
│   ├── pipeline/            # AI 파이프라인 로직
│   │   ├── detect.py        # YuNet 얼굴 감지
│   │   ├── align.py         # face_align 직접 구현
│   │   ├── embed.py         # AuraFace 임베딩
│   │   ├── cluster.py       # 전체 재군집 + cluster_id 재조정 (순수 로직)
│   │   ├── quality.py       # 품질 게이트 — 눈감음 CNN(EyeStateClassifier) + 흔들림 Laplacian (순수 로직)
│   │   ├── thumbnail.py     # 대표 얼굴 썸네일 렌더 — bbox crop→다운스케일→JPEG (순수 함수, CHMO-335)
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
0.1 **[P0] 실 데이터 오염 대응** ([docs/backlog/2026-07-11-followups.md](docs/backlog/2026-07-11-followups.md)) —
   동일 사진 재업로드가 만든 중복 임베딩이 앨범을 쌍 단위로 쪼갬 + `delete_request` 미도달 유령 행
   (원인·재현: [reviews/2026-07-11-duplicate-embedding-split.md](docs/reviews/2026-07-11-duplicate-embedding-split.md))
0.5 **[P1] YuNet 화장품 팔레트 그림 오검출** (ADR-012 아동 교차연령 리스크는 ADR-016으로 해소 —
   완료된 목표 참조): YuNet이 화장품 팔레트 그림을 얼굴로 오검출
   ([분포 측정 리뷰](docs/reviews/2026-07-15-distance-distribution-verdict.md) §별건) —
   배경 인물 크기 필터는 [ADR 013](docs/decisions/013-background-face-size-filter.md)으로 구현됐으나
   팔레트는 크게 찍힌 오검출이라 못 거른다. score·종횡비 결합 필터(ADR 015)로도 못 잡는다
   — 얼굴처럼 그려진 그림은 score도 종횡비도 정상([분포 조사](docs/reviews/2026-07-15-detect-score-aspect-survey.md)
   §한계). 임베딩 단계 신호가 필요한 별도 문제로 남는다.
1. 배포 후속 — 남은 항목: CloudWatch 지표 연동(로그는 완료 — 2026-07-14, awslogs 드라이버로
   `/cheesemoa/ai-worker` 직송, [cloudwatch-logging.md](docs/guides/cloudwatch-logging.md)), Spring 실계약 통합검증, 큐의 visibility timeout·
   redrive policy를 `.env.example` 메모대로 설정. **인스턴스 분리 검토**: 현재 워커가 Spring과 t4g.small
   (2코어·RAM 1846MB)을 공유하는데, t4g는 버스터블이라 실트래픽으로 추론이 지속되면 CPU 크레딧 소진 →
   Spring API까지 함께 스로틀된다 ([ec2-deployment.md](docs/guides/ec2-deployment.md) §리스크)
2. pytest 도입 — 각 모듈 `__main__` 스모크를 tests/로 승격 (`# TODO(CHMO-165)` 표시 지점)
3. (후속) 품질 게이트 개선 — 눈/흔들림 임계 라벨셋 튜닝(현재 라벨 부재), 부분 블러 대응, 실물 대형
   선글라스·유아 눈 오탐(ADR 019 §한계 — 리포트 축적 시), 소형 얼굴(rel_w<0.22) 등방성 블러의
   face_var 10~25 잔존 구간(face_var<7 붕괴분은 ADR 018 §보강 3으로 회복 — 옛날 인화와 겹치는
   구간만 남음, 고스팅 정탐 표본도 유니크 1장이라 미탐 리포트 축적 시 재실측, 도구
   `scripts/survey_face_collapse.py`·`scripts/sim_facevar_floor.py`). 웃음 예외용 표정 CNN은 2026-07-17 실측에서
   오탐 0건으로 착수 근거 상실([survey](docs/reviews/2026-07-17-smile-eyes-geometry-survey.md)).
   원본 해상도 눈 crop도 실측 기각 — 이중 리샘플링의 소프트한 입력이 오히려 CNN 학습 분포(운전자
   모니터링 저해상도)에 정합해, 원본 직접 crop은 기존 정탐을 붕괴시키고 웃음 실눈 오탐을 새로
   유발한다([리뷰](docs/reviews/2026-07-17-native-eyecrop-verdict.md)). 대체 모델(Face Landmarker
   blendshape)은 A/B 실측 후 ADR 021로 **채택 완료** — 눈감음 임계 재보정 대상은 이제
   `QUALITY_BLINK_THRESHOLD`(0.40)와 presence 바닥이다
4. Spring과 `confirm_distinct` 트리거 정책 합의 — 즉시 발행(안전) vs 공유 시점 일괄 발행(발행 전 새
   업로드가 끼면 그 사이 재군집은 보호 공백). 단, 상태 기반 계약 개편
   ([docs/backlog/state-based-feedback-contract.md](docs/backlog/state-based-feedback-contract.md))
   채택 시 자동 해소되는 항목

### 완료된 목표
- **매칭 사진의 미매칭 주 인물 얼굴 uncertain 동시 노출 — 미등록 인물 수동 구제 진입점** (2026-07-21,
  feature-spec §6.2 결정) — 2명이 인식된 사진에서 한 명만 매칭되면 사진이 인물 앨범(+공용)에만 실려,
  미매칭 인물을 "분류가 어려워요 → 인물 앨범 편입"(`__uncertain__` reassign)으로 수동 구제할 진입점이
  없던 문제. 백엔드 합의로 계약 확장: `_assemble_result` 라우팅의 "매칭 얼굴 있으면 uncertain 제외"를
  주 인물 미매칭 얼굴에 한해 해제(인물·공용·uncertain 중복 노출 허용). 행인(크기 게이트)·오검출
  (ADR 025)·파편(ADR 027)은 종전대로 숨김 — 주 인물 자격(counted+크기)이 그대로 노이즈 방어를 겸한다.
  face_bbox는 미매칭 주 얼굴(CHMO-388 crop_face_of 재사용), 편입 reassign은 원래 `cluster_id=None`
  얼굴만 must-link라 수정 없이 그대로 동작. `CLUSTER_UNMATCHED_MAIN_TO_UNCERTAIN`(false=구 정책 롤백),
  자가검증 51건(신규 ⑳ 3건 — 동시 노출·행인 숨김·토글 OFF 재현).
- **uncertain 주 얼굴 face_bbox 계약 확장 — 상세 화면 얼굴 crop** (2026-07-21, CHMO-388) —
  "분류가 어려워요" 사진 상세 화면에서 어느 얼굴이 분류가 어려웠는지 보여주기 위해
  `uncertain[].face_bbox`(`FaceBox` — 원본 픽셀 x·y 좌상단, w·h 폭·높이, 정수)를 결과 계약에 추가.
  워커 crop→S3 업로드(인물 앨범 썸네일 방식)는 uncertain 목록이 매 재군집 event 전체 스냅샷이라
  원본 재fetch·디코드 반복 + 고아 썸네일 정리가 필요해 기각 — 상세 화면엔 원본이 이미 있어 앱이
  bbox로 직접 오린다(.npz v3 bboxes가 원본 px라 좌표계 일치). crop 대상은 그 사진 uncertain 얼굴 중
  머릿수 자격(ADR 025·027 통과) 우선 → 최대 폭 순(행인·파편·오검출 배제, `_uncertain_face_box`).
  bbox 미상(v2 이하 .npz 행)은 null — crop 없이 사진만 표시. 자가검증 48건(신규 2)·스키마 34건
  (신규 1)·스모크 통과. Spring은 값 저장·전달만(표시 측 경계 클램프 필요), 상세는 feature-spec §6.2·
  message-examples §④·노션 "SQS 메시지 스키마" 갱신 완료.
- **초대형 얼굴 파편 이중 검출 디둡 — 1인 사진 공용 앨범 오노출 해소** (2026-07-21,
  [ADR 027](docs/decisions/027-duplicate-face-fragment-dedup.md)) — group 37 / event 105에서 혼자
  찍힌 셀피가 공용 앨범에도 노출되던 문제. YuNet이 초대형 얼굴에 그린 파편 박스 2개가 둘 다 score
  게이트를 통과(0.769/0.638)했고, YuNet NMS(쌍 IoU 0.297<0.3)·ADR-017 디둡(회복 경로 전용)·이중
  검출 안전판 0.95(cannot-link 면제 전용이라 인물 앨범은 무사)를 전부 비껴가 머릿수만 2명으로
  세어졌다. 두 층 수정: ① 검출 — confident 얼굴도 정제 후 디둡(양쪽 폭>224px AND 랜드마크 중심거리
  <0.1×얼굴폭이면 score 최상만, 원본 1,081장 실측 파편 쌍 0.019·0.024 vs 실제 타인 겹침 최저 0.436,
  최초 후보 0.5는 실제 타인 쌍을 삼켜 기각), ② 라우팅 — 머릿수에 이중 검출 붕괴 이식(같은사진 근중복
  ≥0.95 그룹은 폭 최대 행만 카운트, 같은사진 쌍 758개 실측 이중 검출 0.978~0.979 vs 타인 최고 0.756
  — ①만으로는 .npz에 저장된 파편 행이 안 고쳐진다) + ADR-025 최근접에서 같은사진 근중복 제외(파편
  쌍끼리 바닥을 뚫는 구멍, 전면 제외는 그 사진에만 있는 낯선 단체를 오판해 기각). 검증: 검출 diff
  1,081장 중 파편 15장(유니크 2)만 정리, 라우팅 diff 24개 이벤트 중 이중 검출 6장만 공용→인물앨범,
  자가검증 46건(신규 ⑲)·스모크 통과. 의미 변화: 미배정 근중복만 있는 사진은 공용이 아니라 uncertain.
  `DETECT_CONFIDENT_DEDUP_LANDMARK_RATIO`·`CLUSTER_COMMON_DUPLICATE_FACE_SIMILARITY`(각 0=비활성),
  도구 `scripts/survey_confident_dup.py`·`scripts/verify_dup_dedup.py`. 잔여: .npz 파편 행 자체는
  남음(P0 중복 정리와 동류), 저장 이벤트 치유는 다음 재군집 트리거 때(즉시 치유는 85·90·91·104·105
  재트리거).
- **눈감음 상대 크기 게이트 — 원거리 "아래 쳐다봄" 오탐 제외** (2026-07-21,
  [ADR 026](docs/decisions/026-eye-closed-relative-size-gate.md)) — group 35 / event 99에서 아래를
  쳐다보는 얼굴이 눈감음으로 오탐되던 문제. blink blendshape는 "윗눈꺼풀 내려옴"을 재서 내려뜬 뜬
  눈과 감은 눈을 못 가른다 — threshold 불가(정탐 0.516~0.637 한가운데 오탐 박힘, event 61 정탐과
  event 99 오탐이 같은 0.625), 절대 px 불가(정탐 함성 182px가 오탐 58·188px 사이 중첩). 판별축은
  **이미지 대비 상대 크기**: 고해상도 사진의 원거리 얼굴은 절대 188~210px여도 프레임에선 4.7~5.2%
  작은 피사체, 진짜 감음은 14~17% 주 피사체. 전 이벤트 961장 실측에서 내려뜸 오탐 전부 ≤5.2% vs
  진짜 감음 전부 ≥11.3%, 빈 구간에서 0.08 채택. `eye_main_face_ratio`(최대 얼굴 대비)와 달리
  denominator가 이미지라 솔로 사진의 원거리 얼굴을 잡는다(ADR 013·022 "주 피사체 vs 배경" 원리).
  검증: 전 이벤트 959장 before/after diff에서 37장 해제(전부 오탐), 진짜 감음 미탐 0, event 99 오탐
  4장 해제·정탐 2장 유지, 스모크 통과. `QUALITY_EYE_MIN_REL_WIDTH`(0=비활성), 도구
  `scripts/survey_eye_rel_width.py`. 한계: 프레임에서 큰 내려뜸(event 103 rel 32.8%)은 못 거름,
  가림 오탐(선글라스·안경)은 별개 문제, 원거리 진짜 감음은 미판정(ADR 013 원리상 허용).
- **공통(단체) 판정 머릿수 실인물 자격 게이트 — 오검출 얼굴 제외** (2026-07-20,
  [ADR 025](docs/decisions/025-common-headcount-facesim-gate.md)) — event 93에서 퍼 후드 털 뭉치
  오검출(score 0.652·종횡비 0.96·rel_w 16.8%로 기존 검출 필터 전부 통과)이 주 인물로 세어져 1인
  셀피가 "주 인물 2명 단체"로 오판, 공용 앨범에 노출되던 문제. 판별 신호는 임베딩에 남는다 —
  오검출은 event 내 어떤 얼굴과도 안 닮는다(max-sim 0.183 vs 실인물 미배정 최저 0.191, 배정 최저
  0.407). 미배정 얼굴이 event 내 최근접 유사도 < 0.185(빈 구간 하단, 실인물 보호 쪽)면 머릿수에서
  제외(라우팅 전용, 군집 불변). 검증: 실 이벤트 13개 diff에서 클러스터 전 이벤트 불변, 변화는 공용
  3건 전부 의도된 수정(93 퍼 후드 + 84·87 w=0 레거시 행이 미러 이벤트의 크기 게이트와 정렬), 진짜
  2인 사진(0.191·0.239) 공용 유지. `CLUSTER_COMMON_FACE_MIN_SIMILARITY`(0=비활성). 한계: 보정 표본
  오검출 1건·빈 구간 0.008로 얇음 — 리포트 축적 시 검출 score 병행 재설계 (ADR 025 §한계).
- **인물 앨범 대표 얼굴 썸네일 — 워커 crop·S3 업로드 + 계약 확장** (2026-07-20, CHMO-335) — 앱의
  인물 앨범 목록용 썸네일. 대표 선정 신호(LOO centroid 유사도)·디코딩 원본·bbox가 전부 워커 안에
  있으므로 워커가 재군집 직후 클러스터마다 대표 얼굴을 crop(bbox 1.4배 여백)→다운스케일(긴 변 256px)
  →JPEG→S3 업로드하고 결과에 키만 싣는다(`ResultCluster.thumbnail_s3_key`, null 가능 — Spring은
  presigned URL 매 조회 발급으로 서빙만). Lambda·백엔드 크롭은 원본 재디코딩 중복으로 기각.
  `.npz` **스키마 v3**(bboxes·s3_keys 열 — v2 이하는 미상 폴백으로 해당 행만 대표 후보 제외),
  `pipeline/thumbnail.py`(순수 렌더)·`storage/thumbnail_store.py`(Protocol+S3+페이크) 신설,
  키는 `thumbnails/{event_id}/{cluster_id}.jpg` 고정·덮어쓰기, 은퇴 클러스터 썸네일은 best-effort
  삭제. 대표 원본은 클러스터당 1장만 재fetch(공유 t4g RAM 제약 — 요청 전체 이미지 캐시 금지),
  썸네일 실패는 경고 로그 + 해당 키 null로 격리(job 정상 진행). `THUMBNAIL_MAX_SIDE=0`이 롤백
  스위치(기능 전체 비활성). 잔여: 워커 IAM에 embeddings 버킷 `s3:DeleteObject` 권한 확인,
  Spring과 presigned URL 캐시 정책(매 조회 발급) 합의.
- **파편병합 승인을 컴포넌트 전체 재평가로 — 같은 인물 앨범 분리 해소** (2026-07-20,
  [ADR 024](docs/decisions/024-merge-component-linkage.md)) — event 90(group 35)에서 주 인물의
  2얼굴 파편 앨범이 병합 게이트(0.632/0.479)를 통과하고도 별도 앨범으로 남던 문제. 구 완전 연결
  검사가 "병합 전 파편 스냅샷 쌍" 전부에 게이트를 요구해, 먼저 합류한 2얼굴 파편과의 노이즈 낀
  스냅샷 쌍(0.508/0.458)이 합류를 차단 — 15얼굴 컴포넌트의 안정된 증거(0.641/0.476)는 반영되지
  않는 구조. 임계 조정은 2D 그리드 스윕으로 기각(치유하는 어느 조합도 ADR-016이 잡은 아동
  오병합을 이벤트 7~15개에서 재점화). 대신 승인 검사를 컴포넌트 '현재 전체 멤버' 재평가(재계산
  centroid + 전체 얼굴 교차 face-pair 평균, 같은 임계 재사용)로 교체 — 다리(bridge) 융합은 남남
  쌍이 전체 평균을 끌어내려 여전히 차단(자가검증 (m) 합성 기하). 검증: 이벤트 52개 중 event 90만
  치유([15,9,8,2,2]→[17,9,8,2]), 코퍼스·나머지 51개·자가검증 전부 무변화.
  `CLUSTER_MERGE_COMPONENT_LINKAGE`(false=구 동작). 잔여 한계: face-pair 0.420짜리 파편은 동일인
  증거 부재로 의도적 미병합(사용자 merge 보정 영역), 거대 컴포넌트의 이론상 다리 리스크는 아동
  코퍼스·이벤트 무변화로 실측상 안전(리포트 축적 시 구성 쌍 veto 재보정 — ADR 024 §한계).
- **재검출 랜드마크 신뢰 임계 — 초대형 얼굴 파편 bbox의 가드 오판 해소** (2026-07-17,
  [ADR 023](docs/decisions/023-refine-trust-redetect-landmarks.md)) — event 73(group 27)에서 얼굴이
  크게 나온 1인 사진이 공통 사진첩으로 빠지던 문제. YuNet이 초대형 얼굴에 파편 bbox(score 0.640,
  게이트 0.6을 살짝 통과해 ADR-017 회복 경로 미적용)를 주고, 랜드마크 정제의 재검출이 올바른
  랜드마크(score 0.860)를 찾고도 이동량 가드(0.5 × 파편 bbox 폭)에 걸려 폐기 → 깨진 랜드마크의
  쓰레기 임베딩(동일인과도 0.24)이 노이즈로 유출. 같은 뿌리로 recover 경로의 "가드 걸리면 원
  랜드마크 유지" 폴백이 offset 파편 박스 3개를 쓰레기 임베딩으로 살려 유령 인물 앨범까지 생성.
  실측(34개 이벤트 520장, 가드 발동 33건): 좋은 교정은 재검출 score 전부 ≥0.86(NN 0.32→0.98 등),
  무익한 후보는 전부 ≤0.39 — 빈 구간에서 회복 임계와 같은 0.80을 신뢰 임계로 채택, 이상이면
  가드 무시하고 재검출 랜드마크 채택(refine·recover 공통). 교정된 중심이 본 얼굴을 가리켜 파편
  박스는 디둡으로 제거 → 유령 앨범 뿌리 차단. 검증: 31/34 이벤트 불변, 변경 3개 전부 의도된
  수정(리포트 사진 앨범 편입 + 유령 앨범 소멸), 라벨 코퍼스 child ARI 0.573→0.794(순개선)·나머지
  불변. `DETECT_REFINE_TRUST_REDETECT_SCORE`(0=비활성), 도구 `scripts/survey_refine_shift.py`.
- **눈감음 판정 blendshape 교체 — Face Landmarker litert 이식** (2026-07-17,
  [ADR 021](docs/decisions/021-blink-blendshape-litert.md)) — 눈 패치 CNN의 도메인 실패(유아 오탐·
  보정 스톡 미탐·수면 미탐, event 61)를 A/B 실측(871 얼굴, [리뷰](docs/reviews/2026-07-17-mediapipe-blink-ab.md))
  으로 검증된 eyeBlink blendshape로 교체. mediapipe pip은 linux aarch64 휠이 없어 EC2(t4g) 배포
  불가 → face_landmarker.task(Apache 2.0) 내부 tflite 2개만 ai-edge-litert(aarch64 휠 있음)로
  실행하는 이식본 `blink.py` 신설(HDBSCAN·face_align 이식 패턴). YuNet 5점 RoI(배율 3.0/시프트
  −0.05 — 참조 파리티 스윕 확정: |Δ| med 0.008, 판정 뒤집힘 0, 감음 6/6)라 mediapipe 자체 검출
  대비 판정 가능률 69%→91%(누운 수면 옆얼굴 회복). presence<0.5는 미판정 — CNN 폴백은 실측
  정탐 기여 0에 유아 오탐만 재생산해 제거, CNN+ADR-019는 `QUALITY_BLINK_THRESHOLD=0` 롤백
  경로로만 유지. 비용 +35MB RSS·3ms/얼굴, aarch64 컨테이너 검증 완료, Dockerfile 프리베이크
  4모델. event 61: 감음 5장 전부 eyes_closed(미탐 2 회복), 오탐 0.
- **저신뢰 분리 회색지대 face-pair 재확인 — 남남 부착 축출** (2026-07-17,
  [ADR 020](docs/decisions/020-evict-facepair-gray-gate.md)) — event 61에서 남남 얼굴이 LOO centroid
  0.425로 저신뢰 바닥(0.4)을 통과해 인물 앨범에 남던 문제. 전역 바닥 상향은 스윕으로 기각(0.42에서
  이미 단일인물 코퍼스 회귀 + blob 불변식 커플링, 동일인/남남 LOO가 [0.40,0.46)에서 원리적으로 겹침).
  판별 신호는 ADR-016처럼 개별 쌍에 남는다: 남남 부착 top쌍 ≤0.440 vs 진짜 멤버 ≥0.469의 빈 구간에서
  facepair floor 0.45(blob 승격 간선과 동일값 — 승격 즉시 해체 churn 차단 불변식), 회색지대 ceiling
  0.46(남남 관측 최고 0.456 직상, 코퍼스 진짜 멤버 LOO 최저 0.502 아래). 재검측: 코퍼스 회귀 0,
  실 이벤트 45개 중 13개에서 20얼굴 강등(전부 타인 정합, 동일인 증거 보유 0, 의도 밖 변화 0),
  나머지 32개 불변, event 61 해소. 도구: `scripts/tune_membership_floor.py`.
- **눈감음 판정 자격 게이트 — 가림·초소형 얼굴 오탐 해소** (2026-07-17,
  [ADR 019](docs/decisions/019-eye-judgment-eligibility-gate.md)) — 웃음 캔디드 오탐을 실측하려던
  조사([survey](docs/reviews/2026-07-17-smile-eyes-geometry-survey.md), 859 얼굴)에서 웃음 오탐은
  0건이고 실제 오탐 축은 다른 것으로 판명: eyes_closed flagged 11건 중 정탐 1건, 오탐은 초소형
  그림·옆얼굴(32~52px) 4건 + 고글·마스크(눈 뜸) 2건 + 눈 뜬 아기 1건. 가림 검출용 신호 가설
  (눈 어두움·렌즈 매끈함)은 실측 기각 — 선글라스는 반사로 오히려 고텍스처. 대신 판정 자격 게이트
  2개가 오탐을 가른다: **bbox 짧은 변 ≥64px**(min_blur_face_px와 같은 근거) AND **눈/볼 밝기 비
  ≤1.4**(감은 눈꺼풀은 피부 — 초과면 고글 반사·마스크 가림, 빈 구간 [1.21, 1.73]). 코퍼스
  eyes_closed 이미지 11→5장(해제 전부 오탐, 정탐 유지, 신규 0). `quality.py`
  `_eye_judgment_eligible`+`eye_cheek_ratio`, `QualityConfig`/`.env` 설정 2개(0=비활성). 남은 한계:
  대형 실물 선글라스(무표본)·유아 눈(CNN 도메인). 도구: `scripts/survey_smile_eyes.py`·
  `scripts/survey_eye_occlusion.py`(신호 실측 + 크롭 육안 분류).
- **흔들림 재확인 게이트 — 옛날 사진 blurry 오탐 해소** (2026-07-17,
  [ADR 018](docs/decisions/018-shake-coherence-floor.md)) — event 50(앨범 205)에서 옛날 인화 사진
  재촬영본 8장이 전부 blurry로 오분류되던 문제. variance는 잔결의 양만 재서 "원판이 소프트한 사진"과
  "흔들린 사진"을 구분 못한다(임계 조정 불가 — 옛날 사진 분포가 연속). 판별축은 전체 이미지 방향
  쏠림(손떨림은 이방성, 소프트 원판은 등방): 오탐 최고 0.268 vs 흔들림 최저 0.444(얼굴 경로)의 빈
  구간에서 0.35를 바닥으로 채택, variance 판정 말미에 게이트로 적용(`shake_confirmed`,
  `QUALITY_SHAKE_COHERENCE_FLOOR`, 0=비활성). 얼굴 crop 쏠림은 판별력 없음(겹침 실측). event 50
  오탐 8장 전부 해제 + 나머지 51장 무변경, test2 라벨셋 무회귀. **보강(당일)**: event 55에서
  고스팅형 손떨림(겹침 번짐 — 쏠림 0.306으로 낮음)이 게이트에 오해제되어 공통첩으로 유출 →
  fallback 한정 variance 붕괴 면제(whole_var<40이면 쏠림 무관 흔들림 확정, 흔들림 13.5 vs 무얼굴
  옛날 사진 98.9) 추가. **보강 2(당일, §보강 2)**: 같은 고스팅 셀피(재업로드)가 event 64에서 대형
  얼굴 회복 재보정(rel_w 0.20, 98a093c)으로 이번엔 얼굴 검출되어 얼굴 경로로 빠짐 → 게이트 오해제 →
  쓰레기 임베딩이 노이즈로 uncertain 앨범 유출. 실측(코퍼스 499장)에서 단일 축은 전부 겹치나
  (whole_var·face_var·rel_w·쏠림·자기상관 피크·타일 쏠림 각각 기각) **결합 규칙이 성립**: 붕괴 면제를
  대형 blurry 얼굴(rel_w ≥ 0.22, 빈 구간 [0.172, 0.280] 기하 중앙)에 한해 얼굴 경로로 확장 —
  옛날 인화 오탐은 얼굴이 작거나(붕괴 6장 rel_w≤0.172) whole_var 미붕괴(113.1)로 걸러진다.
  검증: 고스팅 blurry 복원, event 50 재점화 0, 라벨셋·event 64 나머지 무회귀, 스윕 발동 고스팅뿐.
  `quality.face_collapse_exempt` + `QUALITY_COLLAPSE_FACE_REL_WIDTH`(0=비활성),
  도구 `scripts/survey_face_collapse.py`. **보강 3(2026-07-21, CHMO-380, §보강 3)**: 소형 얼굴
  등방성 손떨림(test9 dcb66942 단체샷)이 소형 얼굴 검출 하나로 fallback 붕괴 면제를 비켜가 미탐 →
  blurry 얼굴 최저 face_var<7(빈 구간 [5.4, 10.0]) AND whole_var<40이면 게이트 면제·흔들림 확정.
  로컬 142장 중 대상 1장만 전환·회귀 0, 실 이벤트 58개 diff에서 정탐 8장(dcb 재업로드)만 전환.
  whole_var 결합은 실 이벤트 검증에서 추가 — face_var 단독은 선명 사진의 배경 얼굴을 오탐(event 51
  배경 사진기자 face_var 4.1·whole_var 1133). `quality.face_var_collapse_exempt` +
  `QUALITY_FACE_VAR_COLLAPSE_FLOOR`(0=비활성), 도구 `scripts/sim_facevar_floor.py`. 남은 한계: 소형
  얼굴 face_var 10~25 구간 등방성 블러(옛날 인화와 겹침), 어두운 선명 사진(event 16 암실 전시 —
  whole_var도 어둠으로 붕괴, 밝기 게이트는 저조도 흔들림 놓쳐 미도입·수용), 정탐 표본 유니크 1장.
- **대형 근접 얼굴 재검출 회복 — 초근접 얼굴 미검출 해소** (2026-07-16,
  [ADR 017](docs/decisions/017-size-aware-detection-score-threshold.md)) — event 36에서 얼굴이 크게 나온
  아이(0010=`6acd1055`)가 공통 사진첩으로 빠지던 문제. 원인은 YuNet(WIDER FACE 학습)이 초근접 대형
  얼굴에 저score(0.55)를 줘 score 게이트 0.6에 탈락 → 검출 0. **단순 크기 인지형 score 임계는 오검출
  0 불가**(실측: 대형 저score 실얼굴 37 vs 오검출 41이 score·선명도 모두 겹침). 대신 **정규 스케일
  재검출**이 깨끗한 판별축 — 실얼굴은 "너무 커서" 저score였을 뿐이라 정규 크기 재검출 시 score가
  오르고(재검출≥0.80에서 실얼굴 회복·오검출 0/41), 진짜 FP(블러 블롭)는 어느 스케일에서도 낮다.
  `detect.py`에 재검출 회복(`_recover_large_face`, refine와 코어 공유) + 랜드마크 중심 디둡(같은사진
  cannot-link 분열 방지), `DetectorConfig` 설정 2개(0=비활성). 실측: child·event35·36 각 +7 실얼굴
  회복(0010 포함), 성인 event13·27 무회귀. 도구: `scripts/survey_bigface.py`(임베딩 매칭 조사) +
  vision 크롭 분류 워크플로우. **rel_w 하한 재보정 0.30→0.20** (2026-07-17, ADR-017 §재보정) —
  event 60에서 화면을 다 덮는 얼굴이 검출 0으로 공통첩에 빠짐. YuNet은 초대형 얼굴일수록 bbox를
  파편으로 작게 그려(실제 폭 ~1,100px에 박스 500~600px) rel_w 0.280·0.295로 게이트 미달 — 크기
  게이트가 가장 극단적인 얼굴에서 무너지는 구조. 게이트 스윕 실측(전 이벤트 783장 + child, 0.30/
  0.25/0.20 최종 출력 diff)에서 0.20이 기존 검출 손실 0·오검출 통과 0·실얼굴 2장(유일 사진) 회복.
  FP 방어는 rel_w가 아니라 재검출 score(≥0.80)가 담당. 도구: `scripts/sweep_bigface_gate.py`.
- **분류 진행률 SQS 발행 — job 내부 진행바** (2026-07-16, CHMO-274) — classify가 결과 1건만 끝에
  발행해 백엔드·앱이 처리 중 진행도를 알 수 없던 문제. `_handle_classify`의 이미지 루프(job 비용의
  사실상 전부)에서 처리 장수를 별도 progress 큐로 흘려보낸다: 루프 진입 시 `0/total` 1회 + 이후
  3장마다(`_PROGRESS_REPORT_EVERY`) + 마지막 `total/total`. `processed`가 단조 증가해 백엔드가 순서·중복·재전달을
  방어(마지막 본 값 이하 버림). best-effort(발행 실패가 job을 안 죽임), progress 큐 URL 미설정 시
  비활성. `ProgressUpdate`(messages.py)·`SqsProgressPublisher`(publisher.py)·`report_progress` 콜백
  주입(handlers·deps). Spring의 큐 소비·메모리 보관·FE 폴링은 이 레포 밖(백엔드 담당).
- **파편병합 face-level 응집 게이트 — 아동 교차연령 오병합 해소** (2026-07-16, CHMO-269,
  [ADR 016](docs/decisions/016-merge-facepair-cohesion-gate.md)) — event 35에서 서로 다른 아이 30장이
  한 앨범으로 뭉치던 문제. 원인은 검출·해상도가 아니라 파편병합이 **centroid**로 판정하는데 아동
  얼굴은 평균이 "아기 얼굴" 영역으로 수렴해 타인도 0.55~0.63으로 붙는 것(나이대 효과). 판별 신호는
  개별 얼굴 쌍에 남아(같은인물 face평균 0.65 vs 다른아이 ≤0.50), 병합 조건에 **파편 간 face-pair 평균
  바닥**(`merge_facepair_floor`)을 centroid와 AND로 추가. 라벨 코퍼스 child 8인 ARI 0.245→0.788
  (성인·단일인물 무회귀), 실 S3 이벤트 분해는 적대적 검증 전부 GOOD(다른 인물 분리). event 35는
  30장 blob→7명 분리, 최대 클러스터 내부 median 0.403→0.592. ADR-012가 남긴 아동 미검증 리스크 해소.
  `cluster.py`에 게이트 + `ClusterConfig`/`.env` 설정(0=비활성). 도구: `scripts/tune_merge_facepair.py`
  (floor 스윕)·`scripts/verify_split.py`(분해 fix/regression 판별)·`scripts/diagnose_event.py`(층별 진단).
  **재보정 0.45→0.475** (당일, ADR-016 §재보정) — event 43·38(아동)에서 0.45로도 최대 앨범 오병합
  잔존(내부 타인쌍 37%). 재스윕에서 0.475가 코퍼스 사실상 무회귀(meanARI −0.002)로 실 이벤트를
  0.50과 동일하게 완전 분해(43 타인쌍 6%, 35는 0%), 갈라진 조각 전 쌍 cross 0.40~0.47(타인)로 검증.
  교훈: 라벨 코퍼스는 교차연령이라 조임에 민감하지만 실 이벤트는 단일 세션이라 더 조여도 안전 —
  재보정 시 실 이벤트 오병합 지표(최대 앨범 내부 타인쌍 비율)를 함께 볼 것.
- **대형 오검출 결합 필터 — score<0.78 AND 종횡비<0.70** (2026-07-16,
  [ADR 015](docs/decisions/015-detection-false-positive-combined-filter.md)) — 팔짱 낀 팔·조형물 등 진짜
  얼굴 크기의 오검출이 ADR-013 크기 필터·품질 게이트를 통과해 사진을 blurry 오분류하던 문제. 단일
  필터는 불가(album 얼굴도 score·종횡비 각각 최저까지 내려감), 두 축이 동시에 낮은 것은 오검출뿐이라
  결합 규칙이 album 손실 0/114로 오검출 6종 전부 제거. `detect.py`에 `DetectorConfig` 설정 2개
  (+.env, 둘 중 0이면 비활성)로 구현.
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
