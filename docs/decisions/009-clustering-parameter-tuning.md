# ADR 009: 클러스터링 파라미터 ARI 스윕 — 현행 값 확정, min_samples=3 기각

## Status

Accepted (2026-07-04). [ADR 005](./005-hdbscan-standalone-port.md)·[ADR 008](./008-blob-promotion-connected-components.md)의
`ClusterConfig` 값을 재검증한다. 결론적으로 **현행 값을 유지**한다.

## Context

`ClusterConfig`의 임계값 10종은 클러스터링 정확도를 좌우하는데, 기존 값은 PoC에서 **ad-hoc 순도
지표 + raw HDBSCAN(euclidean)** 으로 HDBSCAN 3종만 훑어 정했다(ADR-008). 후처리 임계(병합·구제·
저신뢰·blob 승격)는 체계적으로 스윕된 적이 없고 지표도 ARI가 아니었다. 배포 전 **프로덕션과 동일한
경로(cosine + `recluster()` 전체 후처리)로 ARI 기반 재현 가능한 스윕**을 돌려 최적값을 확인·반영하려 했다.

## Decision

**모든 `ClusterConfig` 기본값을 현행대로 유지한다** (`min_samples=2` 포함). 스윕이 시사한 유일한 개선
후보 `min_samples 2→3`은 소규모 이벤트 회귀로 **기각**했다.

## 방법 (하네스)

- `scripts/tune_cluster.py` (로컬 dev 도구). 프로덕션 레시피(detect→align→embed)로 임베딩을 뽑아
  폴더별 캐시하고, 후보 `ClusterConfig`마다 `recluster()`(cosine·전체 후처리)를 돌려 파티션을
  인물 라벨과 비교한다. ARI는 numpy 자체 구현(sklearn 미도입, ADR-005).
- **지표**: 다인 세트 = **ARI**. 단일 인물 세트 = **최대 클러스터 비율**(ARI는 단일 클래스에 퇴화하므로).
  단일 인물 비율은 "앨범 형성 + 오분리 없음" 가드로만 쓰고 목적함수(다인 평균 ARI)엔 넣지 않는다.
- **데이터** (face-test): `different2`(10명)·`child`(8명 교차연령)·`different`(2명) = ARI 대상;
  `similar`·`similar2`·`similar3`(각 1명) = 가드.
- 스윕은 불변식(`__post_init__`) 위반 조합을 skip. 범위는 ADR-008 실측 앵커(동일 인물 0.46~0.70, 타인 ≲0.3) 기반.

## 결과

| | 현행(baseline) | min_samples=3 | 스윕 best(임계 하향) |
|---|---|---|---|
| 다인 평균 ARI | 0.886 | 0.906 (+0.020) | 0.914 (+0.028) |
| child(교차연령) | 0.681 | 0.740 | 0.753 |
| different2(10명) | 0.977 | 0.977 | 0.989 |
| 단일 인물 세트 | 정상 | 정상 | 정상 |

## Rationale (왜 현행 유지인가)

1. **`min_samples 2→3` 기각 — 소규모 이벤트 회귀.** face-test ARI는 올랐으나(교차연령 +0.06),
   `cluster.py` 합성 자가검증 **(e) "동일 인물 2장 → 승격"** 이 실패했다. 원인: 소규모 단일 인물 앨범을
   구제하는 `_promote_single_blob`(ADR-008)이 `n >= max(min_cluster_size, min_samples)` 게이트 **안**에서만
   실행되므로, `min_samples=3`이면 `n=2` 이벤트가 게이트에 걸려 승격 없이 전원 노이즈가 된다 → **2장 인물
   앨범 미형성**. face-test 라벨셋엔 n=2 단일 인물 케이스가 없어 스윕이 이 회귀를 못 봤고, 자가검증이 잡았다.
2. **후처리 임계 하향 기각 — ADR-008 가드 충돌 + 동률.** best의 추가 이득(+0.008)은 `merge_centroid_similarity`
   0.7→0.65 등 임계를 하한으로 내려서 얻는데, 상위 12개가 전부 동률(진짜 최적 아닌 tie)이고, 0.65는
   ADR-008이 명시적으로 기각한 값이다(교차연령 다른 사람 centroid가 0.635까지 → 오병합). face-test 과적합.
3. **부수 성과 — 현행 값의 정량 확인.** 스윕은 "현재 후처리 임계가 이미 최적 근방이고 더 내리면 위험"함을
   ARI로 재확인했다. ADR-008의 보수적 선택(오병합보다 미병합)이 옳았음을 뒷받침한다.

## Consequences

- `ClusterConfig`·`Settings.cluster_*`·`.env.example` 전부 현행 값 유지 (이번 커밋은 문서·하네스만).
- **하네스(`scripts/tune_cluster.py`)는 재사용 가능**하게 남는다 — 데이터 분포가 바뀌면(실사용 데이터
  확보 시) 재실행해 재튜닝한다. `scripts/`는 gitignore라 방법론·결과는 이 ADR로 보존한다.
- **후속 과제**: `min_samples=3`을 안전하게 채택하려면 blob 승격을 `min_samples` 게이트에서 분리하는
  코드 수정이 필요하다(그러면 교차연령 이득 + n=2 앨범 형성 양립 가능). 별도 티켓.
- **한계**: face-test(VGGFace2 등)는 한국·아동·교차연령 프로덕션 분포와 다르다. 여기서의 "최적"이
  프로덕션 최적을 보장하지 않으므로, 실사용 라벨 데이터 확보 시 재검증한다.
