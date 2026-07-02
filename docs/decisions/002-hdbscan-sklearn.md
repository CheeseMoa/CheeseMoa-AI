# ADR 002: HDBSCAN은 sklearn 라이브러리를 사용한다

## Status

Superseded by [ADR 005](./005-hdbscan-standalone-port.md) — 구현체가 PoC의 numpy 전용
이식본(`hdbscan_standalone.py`)으로 바뀌었다. HDBSCAN **알고리즘** 채택 근거(아래 Rationale의
UnionFind 임계값 방식 대비 ARI 2.7배)는 여전히 유효하다.

## Context

인물별 클러스터링을 위해 HDBSCAN 알고리즘이 필요하다. 이를 sklearn 라이브러리로
처리하는 방법과 UnionFind 기반 코사인 임계값 방식으로 직접 구현하는 방법을
비교했다.

직접 구현은 외부 의존성을 제거하지만, 정확도와 확장성 측면의 트레이드오프가 있다.

## Decision

sklearn의 HDBSCAN을 사용한다 (`eps=0.15`, `metric='cosine'`).

## Rationale

- **정확도**: 직접 구현(UnionFind + 코사인 임계값) 대비 ARI 0.601 vs 0.219로
  정확도가 약 2.7배 우수하다. 인물 분류 오류는 사용자 경험에 직접 영향을 주므로
  정확도가 최우선이다.
- **확장성**: sklearn HDBSCAN은 O(n²) 회피 및 가변 밀도 군집 자동 처리를
  지원한다. 사진 수가 늘어도 성능이 유지된다.
- **비용 대비 효과**: 클러스터링은 전체 파이프라인 연산 비용의 0.1% 미만이다.
  의존성 추가 부담보다 정확도 이득이 훨씬 크다.

## Consequences

- `requirements.txt`에 scikit-learn이 포함된다.
- `app/pipeline/cluster.py`에서 `sklearn.cluster.HDBSCAN`을 직접 사용한다.
- `eps`와 `metric` 파라미터는 데이터 특성 변화 시 재조정이 필요할 수 있다.
- 노이즈 레이블(`-1`)로 분류된 얼굴은 어느 인물에도 속하지 않는 것으로 처리한다.
