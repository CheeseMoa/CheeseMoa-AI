# AI 파이프라인 개요

CheeseMoa-AI의 핵심은 얼굴 인식 파이프라인이다. S3에서 이미지를 읽어 인물별로
클러스터링한 결과를 SQS 결과 큐로 발행한다.

## 파이프라인 흐름

```text
S3에서 이미지 읽기
  │
  ▼
YuNet (얼굴 감지 + 5점 랜드마크 추출)
  │
  ▼
face_align (Umeyama 유사변환 — 112×112 ArcFace 기준점으로 정렬)
  │
  ▼
AuraFace (512-dim 임베딩 벡터 생성)
  │
  ▼
HDBSCAN (인물별 클러스터링, cosine distance)
  │
  ▼
클러스터 결과 → SQS 결과 큐 발행 / event .npz 저장(S3)
```

## 각 단계

### 1. 얼굴 감지 — YuNet

OpenCV DNN 기반 경량 얼굴 감지 모델이다. 얼굴 바운딩 박스와 5개 랜드마크
(양쪽 눈, 코, 양쪽 입꼬리)를 추출한다.

구현 위치: `app/pipeline/detect.py`

### 2. 얼굴 정렬 — face_align

5점 랜드마크를 ArcFace 기준 좌표(`_ARCFACE_DST`)에 맞춰 Umeyama 유사변환으로
정렬한다. 출력은 항상 112×112 BGR 이미지다. RGB 변환은 embed 전처리에서 수행한다.

외부 의존성(insightface, skimage) 없이 OpenCV + NumPy만으로 직접 구현했다.
결정 근거: [decisions/001-face-align-custom.md](../decisions/001-face-align-custom.md)

구현 위치: `app/pipeline/align.py`

### 3. 임베딩 생성 — AuraFace

정렬된 112×112 BGR 얼굴 크롭을 L2 정규화된 512차원 float32 벡터로 변환한다.
같은 인물의 얼굴은 코사인 공간에서 가깝게, 다른 인물은 멀게 위치한다.

- 모델: `fal/AuraFace-v1`의 `glintr100.onnx`를 onnxruntime(CPUExecutionProvider)으로 실행
- 전처리: BGR→RGB → NCHW float32 → `(x - 127.5) / 127.5` — PoC 검증 레시피와 수치 동일 (최대 절대 오차 0 확인)
- 후처리: 항상 L2 정규화 — 하류(HDBSCAN cosine, 대표벡터 평균)가 전부 단위벡터를 전제
- 배치 처리: 모델이 동적 배치를 지원해 한 이미지의 여러 얼굴을 1회 추론으로 처리
- 퇴화 출력(비유한값·영벡터)은 해당 얼굴만 `None`으로 스킵 — align의 `None` 정책과 일관

모델 로딩(HuggingFace 다운로드 포함)은 `FaceEmbedder` 생성 시 1회만 일어난다.
워커는 부트스트랩에서 임베더를 생성해 모델을 메모리에 적재한 뒤 SQS 폴링을 시작한다.
추론 런타임 결정 근거: [decisions/004-embedding-onnxruntime.md](../decisions/004-embedding-onnxruntime.md)

구현 위치: `app/pipeline/embed.py`

### 4. 클러스터링 — HDBSCAN 전체 재군집 + cluster_id 재조정

512-dim 임베딩 벡터들을 코사인 거리 기반으로 군집화해 인물별 클러스터를 만든다.
HDBSCAN 구현은 PoC 검증 이식본 `hdbscan_standalone.py`(sklearn 알고리즘의 numpy 전용
재구현 — scikit-learn 1.9.0과 라벨 완전 일치 검증)를 사용한다.
파라미터는 PoC 검증 레시피: `min_cluster_size=2, min_samples=2, metric='cosine',
cluster_selection_epsilon=0.15`.

증분 매칭이 아니라 **매 트리거마다 event 전체 임베딩(기존+신규)을 재군집**하고, 그 결과를
기존 `cluster_id`에 재조정해 인물 번호의 연속성을 유지한다(정확도 최우선). 재군집 격리 단위는
event다([ADR 007](../decisions/007-embedding-storage-s3.md)). 전략 상세:
[decisions/003-full-reclustering.md](../decisions/003-full-reclustering.md) 및 [spec/feature-spec.md](../spec/feature-spec.md) §4.

`recluster()`는 저장소·SQS를 모르는 **순수 함수**다 — 임베딩 로드/저장(S3 `.npz`)은 워커(호출자)의 책임.

- 입력: event 전체 임베딩 행렬 `(N, 512)` + 직전 클러스터 배정 + 사용자 보정 제약
- 사용자 보정(must-link/cannot-link)은 재군집 후 **결정적 후처리로 강제**한다 — must-link
  컴포넌트는 비노이즈 다수결 라벨로 병합(전원 노이즈면 클러스터로 승격), cannot-link 위반은
  greedy 그래프 컬러링으로 최소 분리하고 비제약 멤버는 코사인 최근접 앵커를 따라간다
  (split 처리 방식 TBD #5의 기본 정책)
- 재군집 결과에 **정확도 보강 후처리**를 순서대로 적용한다 (임계는 전부 `ClusterConfig` 설정값):
  1. **연결 성분 부분 승격** (보정 강제 전, [ADR 008](../decisions/008-blob-promotion-connected-components.md)) —
     HDBSCAN이 클러스터를 하나도 못 만든 경우(소규모 단일 인물 이벤트 등 분할 지점이 없는 event),
     쌍 유사도 ≥ 간선 임계(기본 0.45)로 연결 성분을 만들어 내부 완전 연결 ≥ floor(기본 0.4)인
     성분만 각각 클러스터로 승격 — 행인·타 성분은 노이즈 유지, 체이닝 오병합은 floor가 차단
  2. **파편 병합** — centroid 코사인 유사도가 동일 인물 수준(기본 0.7, PoC값 유지 — 교차연령
     오병합 회피, [ADR 008](../decisions/008-blob-promotion-connected-components.md)) 이상인 클러스터끼리 병합.
     **완전 연결(complete linkage)**: 병합될 모든 구성 쌍이 임계 이상이어야 하며(전이 체인으로 타인이
     융합되는 것 차단), cannot-link로 연결된 클러스터 쌍은 병합하지 않는다(사용자 분리 결정 보존)
  3. **노이즈 구제** — 최근접 centroid 유사도가 기본 0.6 이상인 노이즈 얼굴을 그 클러스터에 편입.
     (얼굴, 클러스터) 후보를 **전역 유사도 내림차순**으로 처리해 cannot-link 경합 시 더 나은 매치가 우선
  4. **저신뢰 분리** (TBD #3 기본 정책) — leave-one-out centroid 유사도가 바닥(기본 0.4) 미만이면
     노이즈로 강등, 2위 클러스터와의 마진이 기본 0.05 미만이면 `ambiguous`로 분리. 사용자 제약에 직접
     걸린 얼굴(must-link 컴포넌트·cannot-link 당사자)은 보호하고, cannot-link로 연결된 클러스터 쌍은
     마진 비교에서 서로 제외한다
- `cluster_id` 승계는 신·구 클러스터 간 **Jaccard 내림차순 greedy 매칭** (최소 Jaccard 임계는
  설정값, TBD #4) — 대응 없는 신규 군집은 새 id(`is_new`), 매칭 실패한 기존 id는 은퇴
- 출력: 클러스터(멤버 + L2 정규화 평균 대표벡터) + 노이즈 + 저신뢰 `ambiguous` + 은퇴 id 목록
  — 노이즈(어디에도 못 붙음)와 `ambiguous`(두 인물 사이 저신뢰)는 둘 다 `uncertain` 후보지만 사유가 다르다

> 단일 인물 event 엣지(ADR 005)는 두 갈래로 깨지며 각각 교정한다: **파편화**(1인물 20장 →
> `[14, 2]+노이즈 4`)는 파편 병합이, **클러스터 0개 퇴화**(분할 지점이 없는 소규모 단일 인물
> 이벤트 → 전원 노이즈)는 연결 성분 부분 승격(ADR 008)이 처리한다 — 합성 데이터로 검증, 승격
> 조건은 face-test 실사진 소견(동일 인물 쌍 0.46~0.70)으로 보정. `allow_single_cluster=True`는
> `cluster_selection_epsilon=0.15`와의 상호작용으로 전원 노이즈가 되어 해법이 아님도 실험으로 확인.

결정 근거: [decisions/002-hdbscan-sklearn.md](../decisions/002-hdbscan-sklearn.md) (알고리즘),
[decisions/005-hdbscan-standalone-port.md](../decisions/005-hdbscan-standalone-port.md) (구현체)

구현 위치: `app/pipeline/cluster.py`, `app/pipeline/hdbscan_standalone.py`

## 관련 문서

- 전체 시스템 구조: [system-overview.md](./system-overview.md)
- 폴더 구조: [conventions/project-structure.md](../conventions/project-structure.md)
