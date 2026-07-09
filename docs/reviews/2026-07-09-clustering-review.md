# 코드 리뷰 — 재군집·클러스터링 경로 (유지보수성 · 운영 관점)

- 일자: 2026-07-09
- 대상: `app/pipeline/cluster.py`(전체 재군집 + 후처리 + ID 재조정),
  `app/pipeline/hdbscan_standalone.py`(HDBSCAN numpy 이식본), 그리고 재군집 소비 지점
  `app/handlers.py`의 `_recluster_and_save`와 `app/worker.py` 오류 정책과의 상호작용
- 관점: 유지보수성, 운영 환경(실 AWS 배포)에서 발생 가능한 문제
- 상태: **발견 기록 — 전 항목 미수정** (수정 시 각 항목에 반영 커밋을 기입할 것)
- 관련: [감지 리뷰](./2026-07-09-face-detection-review.md) ·
  [정렬 리뷰](./2026-07-09-face-alignment-review.md) ·
  [임베딩 리뷰](./2026-07-09-face-embedding-review.md) — 임베딩 단계 발견(ORT 스레드 하드코딩,
  모델 provenance, `log_severity_level`)은 임베딩 리뷰를 따르고, 여기서는 **클러스터링 단계에
  고유한 발견**만 다룬다.

## 요약

이 경로의 코드 품질은 리뷰한 네 경로 중 가장 높다 — `ClusterConfig.__post_init__`의 임계값 간
불변식(구제 즉시 재강등 churn 차단), 모든 후처리 단계의 결정성 설계(동률 규칙까지 문서화),
`recluster` 진입 검증(비유한값·비정규 벡터 거부)은 과거 리뷰에서 잡힌 버그의 재발 방지가 코드에
새겨진 모범 사례다. 고유 리스크는 하나가 구조적이다: **O(N²) 전체 재군집에 이벤트 규모 가드가
없다** — brute HDBSCAN이 N×N float64 행렬 2개를 피크에 동시 보유하고(N=1만에서 ~1.6GB), 매
classify가 event 전체를 재군집하므로 대형 이벤트는 메시지당 비용이 제곱 증가하다 OOM 또는
visibility timeout 초과(동시 처리 → `.npz` lost update)에 도달한다.

## 운영 환경 리스크

### 1. O(N²) 전체 재군집에 이벤트 규모 가드가 없음 — 메모리·지연·동시성 모두

brute 경로 HDBSCAN은 `_pairwise_distance`가 N×N float64 거리 행렬을 만들고,
`_mutual_reachability_graph`가 **또 하나의 N×N 사본**을 반환한다 — 거리 행렬이 `_run_hdbscan`
스코프에 살아있는 채로. 즉 피크에 N² 행렬 2개가 공존한다: N=1만이면 ~1.6GB, N=2만이면 ~6.4GB.
재군집 단위가 event이고(ADR-007) 매 classify가 event 전체를 재군집하므로(ADR-003), 대형
이벤트(결혼식·행사에서 사진 수천 장 × 다인 얼굴)는 시간이 갈수록 매 메시지 처리 비용이 제곱으로
증가하다가 컨테이너 OOM에 도달한다. OOM은 handlers의 이미지 단위 격리를 무력화하는 프로세스
사망이라(감지 리뷰 5번과 동일 계열) poison message 반복 사망으로 이어진다.

파생 위험 — **visibility timeout 초과 시 동시 처리**: 처리 시간이 timeout을 넘으면 같은 event를
두 워커가 동시에 재군집·저장해 `.npz` lost update가 난다. `_recluster_and_save`의 멱등성 주석은
"크래시 후 재전달"(순차 재처리)은 방어하지만 "동시 처리"는 방어하지 못한다.

- 권장: ① event 얼굴 수 상한을 정해 계약으로 명시 — 초과 시 failed로 명확히 실패(OOM보다 낫다),
  ② 실측으로 "N별 피크 메모리·소요 시간" 표를 만들어 visibility timeout(`.env.example` 메모
  항목)을 최악 케이스에 맞춤, ③ 여유 시 MRD 계산을 in-place로 바꿔 N² 사본 하나 제거 — 단
  이식본의 기존 in-place 수정 2건과 같은 "결과 비트 동일 검증" 절차 필수 (ADR-005)

### 2. 재군집 결과에 요약 로그가 없음 — 군집 품질 회귀를 운영에서 감지 불가

`recluster`가 순수 모듈이라 로그가 없는 것은 옳은 설계인데, 이를 보상해야 할
`_recluster_and_save`(handlers.py)에도 재군집 요약 로그가 없다(보정 관련 warning만 존재).
파라미터 튜닝·모델 교체·데이터 변화로 클러스터가 파편화되거나 전원 noise가 되는 회귀가 생겨도,
현재는 사용자 불만 외에 감지 경로가 없다.

- 권장: 재군집마다 `event_id, N, 클러스터 수, noise/ambiguous 수, 신규/은퇴 id 수, 소요 ms`
  한 줄 — 1번의 실측 데이터 수집과 CloudWatch 지표(배포 로드맵)의 기반이 되는 같은 로그다.
  감지 리뷰 4번(파이프라인 카운트 로그)과 합치면 사진 유입→앨범 반영 전 구간이 추적된다

### 3. 결정적 계약 위반(ValueError)도 재시도를 소진함

`recluster`의 모순 제약·비정규 임베딩 ValueError는 몇 번을 재시도해도 같은 결과인데, worker의
오류 정책은 일반 Exception으로 취급해 maxReceiveCount만큼 재시도 후 DLQ로 보낸다.
ValidationError는 이미 poison 즉시 격리 경로가 있으므로, 파이프라인 계약 위반 ValueError도 같은
fail-fast 분류가 가능한지 검토할 가치가 있다. 정확성 문제는 아니고 재시도 낭비·DLQ 도달 지연이다
— 우선순위 낮음.

## 유지보수성

### 4. 944줄, 7단계 in-place 라벨 변이 파이프라인 — 확장 시 함정이 있는 구조

`recluster`의 후처리는 각 단계가 `labels`를 제자리 수정하고, 정확성이 실행 순서 불변식
(승격 → must → cannot → 병합 → 구제 → 축출)과 라벨 카운터 규율에 걸려 있다.
`_enforce_cannot_link` 독스트링이 직접 경고하듯 **라벨을 발급하는 단계를 잘못 삽입하면 조용히
라벨이 충돌**한다. 현재는 문서화가 훌륭해 결함이 아니지만, 이 방어가 전부 주석과 `__main__`
자가검증 (a)~(h)에 있으므로 pytest 승격(로드맵 2번)이 **이 모듈에서 특히 시급**하다 —
자가검증이 회귀 방지선의 전부다.

### 5. `EMBED_DIM` 중복 주석의 근거 절반이 무효 (`cluster.py` 상단)

"embed를 import하면 onnxruntime·model_source(**huggingface_hub**) 임포트 체인" — onnxruntime은
사실(embed.py 모듈 레벨 import라 중복 자체는 정당)이지만 huggingface_hub는 `resolve()` 내부
지연 import라 무효다. 정렬 리뷰 2번(`_ensure_bgr`)과 같은 오류 패턴의 반복이며, 임베딩 리뷰가
"의도된 중복"으로 판정한 것과 모순되지 않는다 — 중복은 유지하되 주석에서 huggingface_hub만
지우면 된다. 이 패턴이 반복되는 근본 원인(경량 공유 상수·헬퍼를 둘 곳이 없음)은 의존성 없는
공용 모듈 하나로 해소 가능하다 — 정렬 리뷰 2번의 권장과 같은 해법.

### 6. 소소한 것들

- `hdbscan_standalone`의 `fit_predict`는 labels만 쓰는데 `_get_probabilities`가 항상 계산된다.
  비용은 미미하니 제거보다는, probabilities가 향후 uncertain 신뢰도 신호로 활용 가능한 값이라는
  점만 기록해 둔다.
- `HDBSCAN.__init__`의 `**kwargs`가 미지원 인자를 조용히 무시한다 — sklearn 호환 의도지만 오타
  파라미터(`min_cluster_szie=`)가 기본값으로 조용히 동작하는 통로다. 프로젝트 내 호출자는
  `recluster` 하나뿐이므로 무시 목록을 명시 인자로 제한해도 호환 부담이 없다.
- `embed_batch`의 입력 블롭이 작다는 판정(임베딩 리뷰 "배치 메모리 상한 불요")에 한 가지 뉘앙스:
  ResNet100 **중간 활성화** 메모리는 배치에 비례하므로, 1번의 N별 실측 때 얼굴 수백 개 이미지의
  임베딩 피크도 함께 재면 상한 불요 판정이 확정된다.

## 확인해서 문제 없었던 것 (오탐 방지 기록)

- **퇴화 임베딩 이중 방어**: embed가 비유한·영벡터를 None으로 거르고, `recluster` 진입에서
  비유한·비정규(norm≠1) 입력을 ValueError로 재검증 — 계층 간 계약이 양쪽에서 강제된다.
- **`ClusterConfig` 불변식 설계**: rescue ≥ membership floor, blob floor ≥ membership floor 등
  "구제 즉시 재강등" churn을 생성 시점에 차단 — 임계값 상호작용이 코드로 문서화된 드문 사례.
- **재귀 안전성**: 이식 시 재귀→반복 전환 2건(ADR-005)을 확인했고, 남아있는
  `_TreeUnionFind.find`의 재귀는 union-by-rank로 깊이 O(log N)이라 N=수만에서도 안전하다.
- **결정성**: 모든 후처리 단계의 동률 규칙 고정, `new_id_factory` 주입, `_cluster_groups`의
  최소 멤버 인덱스 정렬 — 멱등 재처리(저장 후 크래시 재전달) 안전성의 실질 근거.
- **in-place 최적화 2건**(cosine 거리, MRD)은 결과 비트 동일 검증을 거쳤음이 주석·ADR에 기록.
- **n=0·n=1 경계**: 빈 event는 HDBSCAN 사전 분기(전원 노이즈)로 안전하고, 이식본의 n<2 사전
  검증이 불투명한 numpy 예외를 명확한 ValueError로 바꿔놓았다(주석에 재현 기록).
- **`_enforce_cannot_link`의 next_label 미반환**: 결함처럼 보이지만 독스트링이 이유(오래된
  카운터 재사용으로 인한 조용한 라벨 충돌 방지)와 후속 단계 작성 규칙(`labels.max() + 1`)을
  명시한 의도적 설계다.

## 우선순위

| 순위 | 항목 | 비용 |
|------|------|------|
| 실 AWS 통합 검증 전 | 1 (이벤트 규모 가드 + N별 실측 + visibility timeout 정합), 2 (재군집 요약 로그) | 실측 필요 / 로그는 소규모 |
| pytest 도입 시 | 4 (자가검증 승격 — 네 경로 중 이 모듈 최우선) | 로드맵 편승 |
| 여유 시 | 3 (ValueError fail-fast), 5 (주석 정정), 6 | 소규모 |

감지·정렬·임베딩 경로 공통 항목의 우선순위는 각 리뷰 문서의 우선순위 표를 따른다.
