# ADR 005: HDBSCAN은 PoC의 numpy 전용 이식본을 사용한다

## Status

Accepted — [ADR 002](./002-hdbscan-sklearn.md)의 구현체 선택(scikit-learn 라이브러리)을 대체한다

## Context

[ADR 002](./002-hdbscan-sklearn.md)는 HDBSCAN 알고리즘 채택과 함께 구현체로 scikit-learn
라이브러리를 선택했다. 이후 PoC(face-detection-PoC)가 sklearn HDBSCAN의 알고리즘 핵심
(`_reachability.pyx` / `_linkage.pyx` / `_tree.pyx` / brute 경로)을 **numpy만으로** 옮긴
`hdbscan_standalone.py`를 작성했고, PoC 최종 파이프라인 검증(`measure_8vcpu_time.py`)은
sklearn이 아닌 이 이식본으로 수행됐다. 프로덕션 이식 시 어느 구현체를 쓸지 재결정이 필요했다.

주의: ADR 002가 비교 끝에 기각한 "직접 구현"(UnionFind 코사인 임계값, ARI 0.219)과 이
이식본은 다르다 — 이식본은 sklearn 알고리즘을 그대로 옮겨 **결과 라벨이 동일**하다.

## Decision

`app/pipeline/hdbscan_standalone.py`(PoC 이식본, numpy 전용)를 사용한다.
scikit-learn은 의존성에 추가하지 않는다.

## Rationale

- **PoC 최종 레시피와 일치**: 배포 타깃 검증(8vCPU 시뮬레이션)에 실제로 쓰인 구현체를
  그대로 이식한다. "검증된 레시피의 수치 동일 이식"이라는 파이프라인 원칙(embed·align과 동일)과 맞다.
- **의존성 제거**: face_align([ADR 001](./001-face-align-custom.md))과 같은 패턴 —
  sklearn/scipy 패키지 버전과 무관하게 항상 동일한 라벨을 보장한다(알고리즘이 코드로 고정됨).
- **정확도 동일**: scikit-learn 1.9.0의 `sklearn.cluster.HDBSCAN`과 라벨 완전 일치를 검증했다
  (합성 단위벡터 30세트 — 군집 구조 20 + 균등 랜덤 10, 파라미터 `min_cluster_size=2,
  min_samples=2, metric='cosine', cluster_selection_epsilon=0.15`). ADR 002의 알고리즘 채택
  근거(UnionFind 임계값 대비 ARI 2.7배)는 그대로 유효하다.
- **성능 특성 동급**: `metric='cosine'`은 sklearn에서도 KD/Ball-Tree를 못 쓰고 brute 경로
  (전체 쌍거리 행렬)로 떨어지므로, 이식본의 dense O(N²) 특성은 sklearn 대비 손해가 아니다.
  클러스터링은 전체 파이프라인 비용 0.1% 미만이라 문제되지 않는다(ADR 002).

## Consequences

- `requirements.txt`에 scikit-learn을 추가하지 않는다 (numpy만 사용).
- `app/pipeline/cluster.py`는 `app.pipeline.hdbscan_standalone.HDBSCAN`을 사용한다.
- sklearn 업스트림 개선을 자동으로 따라가지 않는다 — 알고리즘은 원본 커밋(cc50648cc) 기준으로
  고정이며, 갱신이 필요하면 이식본을 직접 수정한다.
- **PoC 원본과의 편차는 안전 수정 3가지 + 포맷이 전부다** (모두 결과 라벨 비트 동일을 검증):
  ① 재귀 → 반복 전환(`_traverse_upwards`·`_recurse_leaf_dfs`) — 순수 파이썬 재귀는 깊은 체인형
  cluster tree에서 RecursionError로 죽는 것이 프로덕션 경로(eom + eps)에서 재현됐다(Cython
  원본에는 없던 제약). ② 표본 2개 미만 사전 검증 — sklearn과 동일하게 명확한 ValueError로
  거부(없으면 n=1·min_samples=1이 불투명한 numpy 예외로 죽음). ③ in-place 연산 2건
  (`1.0 - S`, 두 번째 `np.maximum`) — N×N float64 임시 행렬 2개 제거.
- dense O(N²) 메모리·시간 — ③ 적용 후 피크는 약 2×N² float64. **재군집 격리 단위가 event
  (이벤트당 수백 벡터)라 사실상 무시 가능**하다(참고: N=1만 ≈ 1.6GB, N=3만 ≈ 14GB). 극단적으로
  한 이벤트가 수만+로 커지는 경우에만 feature-spec §4의 규모 탈출구(fast-path + 주기 재군집)로 대응한다.
- `allow_single_cluster=False`(sklearn 기본값이자 PoC 레시피)에서는 **event 전체가 사실상
  단일 군집(전원 같은 인물)이면 루트를 선택할 수 없어** 두 형태로 깨진다: 분산이 있으면 파편화
  (1인물 20장 → `[14, 2]+노이즈 4`), 분할 지점이 없으면 클러스터 0개(소규모 단일 인물 이벤트 →
  전원 노이즈). 전자는 `cluster.py`의 **파편 병합**(완전 연결 기준 centroid 유사도 병합)이,
  후자는 **연결 성분 부분 승격**(쌍 유사도 간선으로 성분을 만들어 내부 완전 연결인 성분만 승격,
  [ADR 008](./008-blob-promotion-connected-components.md))이 교정한다.
  `allow_single_cluster=True`는 해법이 아니다: 단일 클러스터 모드의 루트 소속 판정이
  `cluster_selection_epsilon=0.15`(유사도 0.85 이내)를 요구해 실제 동일 인물 분산(유사도 ~0.7)이
  전원 노이즈로 떨어짐을 실험으로 확인했다.
