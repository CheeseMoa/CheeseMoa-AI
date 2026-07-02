"""순수 파이프라인 단계로서의 인물 클러스터링 (전체 재군집 + cluster_id 재조정).

군집의 진실은 group 전체 임베딩(기존+신규)에 대한 HDBSCAN 재군집이다 (ADR-003).
이 모듈은 저장소(pgvector)·SQS를 모르는 순수 로직으로, 임베딩 행렬과 직전 배정을 받아

  ① HDBSCAN 전체 재군집 (PoC 검증 이식본, cosine) — 클러스터 0개 균질 blob은 단일 클러스터로 승격
  ② 사용자 보정(must-link/cannot-link) 후처리 강제 — 재군집이 사람 결정을 뒤집지 않게
  ③ 파편 병합 — centroid 유사도가 동일 인물 수준인 클러스터 병합 (완전 연결, 단일 인물 파편화 교정, ADR 005)
  ④ 노이즈 구제 — 최근접 centroid 유사도가 충분한 노이즈 얼굴을 클러스터에 편입
  ⑤ 저신뢰 분리 — 절대 유사도·2위 마진 임계 미달 멤버를 ambiguous로 분리 (TBD #3 기본 정책)
  ⑥ 기존 클러스터와의 overlap(Jaccard) 매칭으로 cluster_id 승계 / 신규 발급 / 은퇴
  ⑦ 클러스터별 대표벡터(L2 정규화 평균, 파생 캐시) 계산

을 수행한다 (feature-spec §4). 임베딩 로드/저장과 보정 메시지(merge/split/reassign)의
제약 변환은 호출자(워커)의 책임이다.
"""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.hdbscan_standalone import HDBSCAN

# embed.EMBED_DIM과 같은 값 — embed를 import하면 onnxruntime·model_source(huggingface_hub)
# 임포트 체인이 순수 수학 모듈에 유입되므로 로컬 상수로 중복 선언한다 (align._ensure_bgr와 같은 이유).
EMBED_DIM = 512
_NOISE = -1


@dataclass(frozen=True)
class ClusterConfig:
  """`recluster`의 튜닝 파라미터. 기본값은 PoC(face-detection-PoC)가 검증한 레시피다.

  거리 metric은 cosine 고정으로 노출하지 않는다 — 임베딩이 L2 정규화 단위벡터라는
  파이프라인 전제(embed 후처리·대표벡터 정의)가 cosine에 결합되어 있다.
  """

  min_cluster_size: int = 2
  min_samples: int = 2
  cluster_selection_epsilon: float = 0.15
  # 기존 cluster_id 승계에 필요한 최소 Jaccard (TBD feature-spec §10 #4). 대량 업로드 시
  # 신규 멤버가 많을수록 Jaccard가 자연히 낮아지므로(기존 10 + 신규 100이면 최대 0.09)
  # 기본값은 0.0 — 겹침이 하나라도 있으면 승계 후보가 되고, 최강 겹침부터 배정된다.
  min_match_jaccard: float = 0.0
  # 파편 병합 임계 — centroid 코사인 유사도가 이 이상이면 같은 인물의 파편으로 보고 병합한다.
  # AuraFace에서 동일 인물 ≳0.7 / 타인 ≲0.3으로 간격이 넓어 0.7이 안전한 초기값 (TBD #4에서 실데이터 재조정).
  merge_centroid_similarity: float = 0.7
  # 노이즈 구제 임계 — 최근접 centroid 유사도가 이 이상인 노이즈 얼굴을 그 클러스터에 편입한다.
  # 동일 인물 하한(≈0.6) 수준. 1.0에 가깝게 올리면 사실상 비활성.
  rescue_similarity: float = 0.6
  # 저신뢰 분리 임계 (TBD feature-spec §10 #3의 초기값) — 아래 둘 중 하나라도 걸리면 ambiguous로 뺀다:
  # 자기 centroid 절대 유사도 바닥, 그리고 2위 클러스터와의 유사도 마진.
  min_membership_similarity: float = 0.4
  min_membership_margin: float = 0.05

  def __post_init__(self) -> None:
    # 이식한 HDBSCAN이 min_cluster_size < 2에서 raise하므로 생성 시점에 같은 계약을 강제한다
    if self.min_cluster_size < 2:
      raise ValueError(f"min_cluster_size는 2 이상이어야 합니다. 받은 값: {self.min_cluster_size}")
    if self.min_samples < 1:
      raise ValueError(f"min_samples는 1 이상이어야 합니다. 받은 값: {self.min_samples}")
    if self.cluster_selection_epsilon < 0.0:
      raise ValueError(f"cluster_selection_epsilon은 0 이상이어야 합니다. 받은 값: {self.cluster_selection_epsilon}")
    for name in (
      "min_match_jaccard",
      "merge_centroid_similarity",
      "rescue_similarity",
      "min_membership_similarity",
      "min_membership_margin",
    ):
      value = getattr(self, name)
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}은(는) [0, 1] 범위여야 합니다. 받은 값: {value}")
    if self.cluster_selection_epsilon > 2.0:
      # cosine 거리 범위는 [0, 2] — 밖의 값은 기하학적으로 무의미한데도 조용히 동작해 군집 선택을 왜곡한다
      raise ValueError(
        f"cluster_selection_epsilon은 cosine 거리 범위 [0, 2] 안이어야 합니다. 받은 값: {self.cluster_selection_epsilon}"
      )
    if self.rescue_similarity < self.min_membership_similarity:
      # 이 순서가 깨지면 [rescue, floor) 대역에서 구제된 얼굴이 같은 실행의 저신뢰 축출에서 곧바로
      # 노이즈로 재강등된다 — 구제 시점과 축출 시점의 자기 제외 유사도가 정확히 같기 때문 (리뷰 재현).
      raise ValueError(
        "rescue_similarity는 min_membership_similarity 이상이어야 합니다. "
        f"받은 값: rescue={self.rescue_similarity}, floor={self.min_membership_similarity}"
      )


@dataclass(frozen=True)
class Constraints:
  """사용자 보정(병합/분리/이동)을 임베딩 행 인덱스 쌍으로 표현한 제약.

  보정 메시지(cluster-feedback의 merge/split/reassign) → 인덱스 쌍 변환은 호출자의 책임이다.
  must-link로 (전이적으로) 연결된 두 얼굴 사이의 cannot-link는 모순이라 `recluster`가
  ValueError로 거부한다 — 보정 간 충돌의 시간순 해소(나중 결정 우선)는 보정 이력을 아는
  워커 계층에서 끝내고, 이 모듈에는 일관된 제약 셋만 전달해야 한다.
  """

  must_link: tuple[tuple[int, int], ...] = ()
  cannot_link: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class PersonCluster:
  """재군집 결과의 인물 클러스터 1개."""

  cluster_id: str
  is_new: bool  # 이번 재조정에서 기존 cluster_id 승계에 실패해 새로 발급된 인물인지
  member_indices: tuple[int, ...]  # 입력 embeddings의 행 인덱스 (오름차순)
  # 멤버별 자기 클러스터 신뢰도 — leave-one-out centroid 코사인 유사도, 저신뢰 축출과 동일 정의
  # (member_indices와 자리 대응, 단독 멤버는 1.0). 워커가 uncertain reason·저신뢰 표시
  # (feature-spec §6.2·§7·TBD #2)에 쓸 값을 재계산 없이 노출한다 — 외부 재계산은 LOO 보정·
  # cannot-link 마진 제외가 빠져 축출 결정과 어긋난 신뢰도를 만든다 (리뷰 지적).
  membership_similarities: tuple[float, ...]
  # 조회·표시용 파생 캐시(멤버 임베딩의 L2 정규화 평균) — 군집 판단의 원천이 아니다 (ADR-003).
  # ndarray 필드의 자동 __eq__는 진리값 모호성으로 예외를 던지므로 비교 대상에서 제외한다 (DetectedFace와 동일).
  centroid: np.ndarray = field(compare=False)  # shape (EMBED_DIM,), float32


@dataclass(frozen=True)
class ReclusterResult:
  """`recluster` 1회 실행의 결과 — 결과 메시지(classify-result)와 pgvector 갱신의 원천."""

  clusters: tuple[PersonCluster, ...]  # 최소 멤버 인덱스 오름차순
  # 어느 인물에도 배정되지 않은 얼굴 (uncertain 후보) — 밀도 노이즈(구제 실패)뿐 아니라
  # 저신뢰 절대 바닥 미달로 클러스터에서 강등된 얼굴도 포함한다
  noise_indices: tuple[int, ...]
  ambiguous_indices: tuple[
    int, ...
  ]  # 인물 배정이 저신뢰(절대 유사도·마진 미달)라 분리된 얼굴 (uncertain 'ambiguous' 후보)
  retired_cluster_ids: tuple[str, ...]  # 이번 재군집에서 승계되지 못한 기존 cluster_id


class _UnionFind:
  """must-link 컴포넌트 계산용 union-find (경로 압축 + 크기 기준 합치기)."""

  def __init__(self, n: int) -> None:
    self._parent = list(range(n))
    self._size = [1] * n

  def find(self, x: int) -> int:
    root = x
    while self._parent[root] != root:
      root = self._parent[root]
    while self._parent[x] != root:  # 경로 압축
      self._parent[x], x = root, self._parent[x]
    return root

  def union(self, a: int, b: int) -> None:
    root_a, root_b = self.find(a), self.find(b)
    if root_a == root_b:
      return
    if self._size[root_a] < self._size[root_b]:
      root_a, root_b = root_b, root_a
    self._parent[root_b] = root_a
    self._size[root_a] += self._size[root_b]


def _validate_pairs(pairs: tuple[tuple[int, int], ...], n: int, kind: str) -> None:
  for i, j in pairs:
    if not (0 <= i < n and 0 <= j < n):
      raise ValueError(f"{kind} 제약 인덱스가 범위를 벗어났습니다. 받은 쌍: ({i}, {j}), 임베딩 수: {n}")


def _must_link_components(n: int, constraints: Constraints) -> tuple[list[int], dict[int, list[int]]]:
  """must-link 폐포(전이적 연결)를 계산하고 cannot-link와의 모순을 거부한다.

  반환: (각 인덱스의 컴포넌트 루트, 루트 → 멤버 오름차순 목록). 단독 얼굴도 자기 컴포넌트를 가진다.
  """
  uf = _UnionFind(n)
  for i, j in constraints.must_link:
    uf.union(i, j)
  comp_of = [uf.find(i) for i in range(n)]
  components: dict[int, list[int]] = {}
  for idx in range(n):
    components.setdefault(comp_of[idx], []).append(idx)
  for i, j in constraints.cannot_link:
    if i == j:
      # 일반 모순 검사(comp_of[i] == comp_of[j])에 맡기면 존재하지 않는 must-link를 탓하는
      # 오도성 메시지가 나간다 (리뷰 재현) — 보정 메시지 번역 버그는 원인 그대로 알려준다
      raise ValueError(f"cannot-link 쌍은 서로 다른 얼굴이어야 합니다. 받은 쌍: ({i}, {j})")
    if comp_of[i] == comp_of[j]:
      raise ValueError(f"모순된 제약입니다: must-link로 연결된 얼굴 쌍 ({i}, {j})에 cannot-link가 지정되었습니다.")
  return comp_of, components


def _enforce_must_link(labels: np.ndarray, components: dict[int, list[int]], next_label: int) -> int:
  """must-link 컴포넌트 전원을 같은 라벨로 강제한다 (labels 제자리 수정).

  대상 라벨은 컴포넌트 내 비노이즈 다수결(동률 시 작은 라벨). 전원 노이즈면 새 라벨을 발급한다
  — 사용자가 같은 인물이라고 확정한 그룹은 밀도와 무관하게 클러스터로 승격한다.
  반환: 다음 합성 라벨 번호.
  """
  for root in sorted(components):  # 루트 순서 고정 — 새 라벨 발급 순서의 결정성
    members = components[root]
    if len(members) < 2:
      continue
    member_labels = labels[members]
    non_noise = member_labels[member_labels != _NOISE]
    if non_noise.size:
      values, counts = np.unique(non_noise, return_counts=True)  # unique는 오름차순 → argmax 동률 시 작은 라벨
      target = int(values[np.argmax(counts)])
    else:
      target = next_label
      next_label += 1
    labels[members] = target
  return next_label


def _normalized_mean(embeddings: np.ndarray, members: Sequence[int]) -> np.ndarray:
  """멤버 임베딩의 L2 정규화 평균 — 대표벡터 정의 (feature-spec §4 ⑤).

  단위벡터들의 평균은 정확히 대척(antipodal)일 때만 0이 되어 사실상 발생하지 않지만,
  0 나눗셈 대신 첫 멤버 임베딩으로 결정적 폴백한다.
  """
  mean = embeddings[list(members)].mean(axis=0)
  norm = float(np.linalg.norm(mean))
  if norm == 0.0:
    return embeddings[members[0]].astype(np.float32)
  return (mean / norm).astype(np.float32)


def _loo_similarities(embeddings: np.ndarray, members: Sequence[int]) -> np.ndarray:
  """멤버별 자기 클러스터 leave-one-out centroid 코사인 유사도 — 신뢰도의 공통 정의.

  자기가 포함된 centroid는 유사도를 부풀려 경계 얼굴이 신뢰도 검사를 통과해 버리므로 자기를 뺀
  평균과 비교한다. 저신뢰 축출 판정과 결과(PersonCluster.membership_similarities) 노출이 같은
  함수를 쓰게 해 두 값이 어긋나지 않게 한다. 단독 멤버와 퇴화(LOO 합이 영벡터)는 판단 불능이라
  1.0(유지)으로 둔다.
  """
  block = embeddings[list(members)]
  if len(members) < 2:
    return np.ones(len(members), dtype=np.float64)
  loo = block.astype(np.float64).sum(axis=0) - block
  norms = np.linalg.norm(loo, axis=1)
  sims = np.einsum("ij,ij->i", block.astype(np.float64), loo)
  safe_norms = np.where(norms == 0.0, 1.0, norms)
  return np.where(norms == 0.0, 1.0, sims / safe_norms)


def _enforce_cannot_link(
  labels: np.ndarray,
  comp_of: list[int],
  components: dict[int, list[int]],
  cannot_link: tuple[tuple[int, int], ...],
  embeddings: np.ndarray,
  next_label: int,
) -> None:
  """같은 클러스터에 남은 cannot-link 쌍을 분리한다 (labels 제자리 수정, TBD #5의 기본 정책).

  이동 단위는 must-link 컴포넌트(또는 단독 얼굴)라 컴포넌트가 쪼개지지 않는다. 위반 쌍에 관여한
  컴포넌트(앵커)들을 greedy 그래프 컬러링으로 최소한만 갈라(제약 없는 앵커끼리는 과분리하지 않음),
  가장 큰 앵커 무리가 원 라벨을 유지하고 나머지 색은 새 라벨을 받는다. 제약에 안 걸린 나머지 멤버는
  컴포넌트 단위로 코사인 최근접 앵커 대표벡터를 따라간다. 앵커 처리 순서(크기 내림차순 → 최소 인덱스)와
  라벨 오름차순 순회로 결과는 결정적이다.

  next_label은 여기서 발급할 합성 라벨의 시작값일 뿐, 진행된 카운터를 반환하지 않는다 — 반환하면
  버리는 호출부가 생기고, 그 오래된 카운터로 라벨을 발급하는 후속 단계가 여기서 만든 라벨과
  조용히 충돌한다 (리뷰 지적). 라벨을 발급하는 후속 단계를 추가하려면 카운터가 아니라
  `labels.max() + 1`에서 다시 시작할 것.
  """
  if not cannot_link:
    return
  # 처리 중 이동은 처리 대상 라벨 안에서 새 라벨로만 일어나므로(기존 라벨로 유입 없음),
  # 위반 라벨 집합을 처음 한 번만 계산해도 안전하다. 노이즈(-1)는 클러스터가 아니라 위반이 아니다.
  violated_labels = sorted({int(labels[i]) for i, j in cannot_link if labels[i] == labels[j] and labels[i] != _NOISE})
  for current in violated_labels:
    member_idx = [int(i) for i in np.flatnonzero(labels == current)]
    pairs = [(i, j) for i, j in cannot_link if labels[i] == current and labels[j] == current]

    # 앵커 = 위반 쌍에 관여한 컴포넌트 전체 (must-link 강제 이후 컴포넌트는 라벨 균일)
    anchors: dict[int, list[int]] = {}
    adjacency: dict[int, set[int]] = {}
    for i, j in pairs:
      root_i, root_j = comp_of[i], comp_of[j]
      anchors.setdefault(root_i, components[root_i])
      anchors.setdefault(root_j, components[root_j])
      adjacency.setdefault(root_i, set()).add(root_j)
      adjacency.setdefault(root_j, set()).add(root_i)

    # greedy 컬러링: cannot-link로 인접한 앵커만 다른 색 — 색 0(가장 큰 앵커 우선)이 원 라벨 유지
    ordered_roots = sorted(anchors, key=lambda root: (-len(anchors[root]), anchors[root][0]))
    color: dict[int, int] = {}
    for root in ordered_roots:
      used = {color[neighbor] for neighbor in adjacency[root] if neighbor in color}
      chosen = 0
      while chosen in used:
        chosen += 1
      color[root] = chosen
    color_label = {0: current}
    for extra in range(1, max(color.values()) + 1):
      color_label[extra] = next_label
      next_label += 1
    for root in ordered_roots:
      labels[anchors[root]] = color_label[color[root]]

    # 제약 없는 나머지 멤버는 컴포넌트 단위로 코사인 최근접 앵커를 따라간다 — split된 인물 양쪽이
    # 이후 업로드에서도 각자 사진을 이어받을 수 있게 하기 위함이다 (원 라벨 고정 시 한쪽만 성장).
    anchor_members = {idx for members in anchors.values() for idx in members}
    units: dict[int, list[int]] = {}
    for idx in member_idx:
      if idx not in anchor_members:
        units.setdefault(comp_of[idx], []).append(idx)
    if not units:
      continue
    anchor_centroids = np.stack([_normalized_mean(embeddings, anchors[root]) for root in ordered_roots])
    anchor_labels = [color_label[color[root]] for root in ordered_roots]
    for members in units.values():
      similarities = anchor_centroids @ embeddings[members].mean(axis=0)
      labels[members] = anchor_labels[int(np.argmax(similarities))]  # argmax 동률 시 앞선(큰) 앵커


def _cannot_link_partners(cannot_link: tuple[tuple[int, int], ...]) -> dict[int, list[int]]:
  """얼굴 → cannot-link 상대 목록. 대다수인 비제약 얼굴의 차단 검사가 O(전체 쌍) 스캔 대신 O(1)이 된다."""
  partners: dict[int, list[int]] = {}
  for a, b in cannot_link:
    partners.setdefault(a, []).append(b)
    partners.setdefault(b, []).append(a)
  return partners


def _sets_blocked(set_a: set[int], set_b: set[int], cannot_link: tuple[tuple[int, int], ...]) -> bool:
  """두 멤버 집합 사이에 cannot-link 쌍이 걸쳐 있는지 (병합 차단 판정)."""
  for a, b in cannot_link:
    if (a in set_a and b in set_b) or (a in set_b and b in set_a):
      return True
  return False


def _cluster_groups(labels: np.ndarray) -> list[tuple[int, list[int]]]:
  """비노이즈 라벨별 멤버 목록을 최소 멤버 인덱스 순으로 반환한다 (합성 라벨 번호 무관 결정성)."""
  groups: dict[int, list[int]] = {}
  for idx, label in enumerate(labels):
    if label != _NOISE:
      groups.setdefault(int(label), []).append(idx)
  return sorted(groups.items(), key=lambda item: item[1][0])


def _promote_single_blob(labels: np.ndarray, embeddings: np.ndarray, threshold: float) -> None:
  """HDBSCAN이 클러스터를 하나도 못 만들었을 때, 전체가 동일 인물 수준의 밀집이면 단일 클러스터로 승격한다.

  allow_single_cluster=False에서 group 전체가 사실상 단일 군집이면 두 갈래로 깨진다: 파편화되거나
  (파편 병합이 교정), 분할 지점이 아예 없으면 클러스터 0개(전원 노이즈)가 된다 — 후자는 병합·구제가
  손댈 클러스터가 없어 인물 앨범이 아예 생기지 않는 것이 리뷰에서 재현됐다(동일 사진 버스트 등).
  모든 쌍별 유사도가 threshold 이상일 때만(완전 연결 기준) 전체를 라벨 0 하나로 승격한다 — 낯선
  두 얼굴(유사도 ~0)은 승격되지 않고, 이후 must/cannot-link 강제는 이 라벨 위에서 정상 동작한다.
  """
  if labels.size == 0 or (labels != _NOISE).any():
    return
  # 전원 노이즈일 때만 계산 — 실사용에서 이 경로는 소규모 blob이고, N² 행렬은 HDBSCAN이 이미 만든 규모다
  gram = embeddings @ embeddings.T
  if float(gram.min()) >= threshold:
    labels[:] = 0


def _merge_fragments(
  labels: np.ndarray,
  embeddings: np.ndarray,
  cannot_link: tuple[tuple[int, int], ...],
  threshold: float,
) -> None:
  """centroid 코사인 유사도가 동일 인물 수준(threshold)인 클러스터끼리 병합한다 (labels 제자리 수정).

  allow_single_cluster=False 특성상 한 인물 위주의 밀집이 파편화되는 케이스(ADR 005)와 일반적인
  과분할을 함께 교정한다. cannot-link로 연결된 클러스터 쌍은 병합하지 않는다(사용자 분리 결정 보존).
  병합 조건은 완전 연결(complete linkage): 두 컴포넌트의 모든 구성 클러스터 쌍이 임계 이상이어야
  한다 — 쌍별 검사만 하면 전이 체인(A~B, B~C)이 서로 타인인 A와 C(유사도 ~0.1)를 한 앨범으로
  융합하는 것이 리뷰에서 재현됐다. 유사도 내림차순 greedy에 병합 컴포넌트의 대표 라벨을 최소 멤버
  인덱스 클러스터로 고정해 결과가 결정적이다. 유사도는 병합 전 centroid 스냅샷 기준이다.
  """
  ordered = _cluster_groups(labels)
  if len(ordered) < 2:
    return
  centroids = np.stack([_normalized_mean(embeddings, members) for _, members in ordered])
  similarities = centroids @ centroids.T
  candidates = [
    (-float(similarities[i, j]), i, j)
    for i in range(len(ordered))
    for j in range(i + 1, len(ordered))
    if similarities[i, j] >= threshold
  ]
  if not candidates:
    return

  parent = list(range(len(ordered)))

  def find(x: int) -> int:
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  merged_members = {pos: set(members) for pos, (_, members) in enumerate(ordered)}
  merged_positions = {pos: {pos} for pos in range(len(ordered))}  # 완전 연결 검사용 구성 클러스터 위치
  for _, i, j in sorted(candidates):
    root_i, root_j = find(i), find(j)
    if root_i == root_j:
      continue
    if _sets_blocked(merged_members[root_i], merged_members[root_j], cannot_link):
      continue
    if not all(similarities[p, q] >= threshold for p in merged_positions[root_i] for q in merged_positions[root_j]):
      continue  # 완전 연결 위반 — 다리(bridge) 클러스터를 통한 타인 융합 차단
    if root_j < root_i:  # 작은 위치가 루트 — 컴포넌트 라벨이 최소 멤버 인덱스 클러스터로 수렴
      root_i, root_j = root_j, root_i
    parent[root_j] = root_i
    merged_members[root_i] |= merged_members.pop(root_j)
    merged_positions[root_i] |= merged_positions.pop(root_j)

  for pos, (_, members) in enumerate(ordered):
    root = find(pos)
    if root != pos:
      labels[members] = ordered[root][0]


def _rescue_noise(
  labels: np.ndarray,
  embeddings: np.ndarray,
  cannot_link: tuple[tuple[int, int], ...],
  threshold: float,
) -> None:
  """최근접 centroid 유사도가 threshold 이상인 노이즈 얼굴을 그 클러스터에 편입한다 (labels 제자리 수정).

  파편 병합 뒤에 실행해 병합된 centroid를 기준으로 삼는다 (centroid는 시작 시점 스냅샷).
  (얼굴, 클러스터) 후보를 전역 유사도 내림차순으로 처리한다 — 얼굴별 인덱스 순 처리는 cannot-link
  경합(서로 배타인 두 노이즈 얼굴이 같은 클러스터를 원할 때) 시 유사도가 낮은 쪽이 자리를 선점하는
  역전이 리뷰에서 재현됐다. 전역 내림차순에서는 더 나은 매치가 항상 먼저 배정되고, 결과는 결정적이다
  (동률은 얼굴 인덱스 → 클러스터 순). cannot-link 상대가 있는 클러스터는 건너뛰고 다음 후보를 본다.
  must-link로 묶인 전원-노이즈 컴포넌트는 이미 클러스터로 승격됐으므로 여기 도달하는 노이즈는
  전부 제약상 단독 얼굴이다.
  """
  noise_idx = [int(i) for i in np.flatnonzero(labels == _NOISE)]
  ordered = _cluster_groups(labels)
  if not noise_idx or not ordered:
    return
  centroids = np.stack([_normalized_mean(embeddings, members) for _, members in ordered])
  similarities = embeddings[noise_idx] @ centroids.T  # (노이즈 수, 클러스터 수) — 얼굴별 GEMV 대신 1회 GEMM
  candidates = sorted(
    (-float(similarities[row, pos]), idx, pos)
    for row, idx in enumerate(noise_idx)
    for pos in range(len(ordered))
    if similarities[row, pos] >= threshold
  )
  partners = _cannot_link_partners(cannot_link)
  rescued: set[int] = set()
  for _, idx, pos in candidates:
    if idx in rescued:
      continue
    target = ordered[pos][0]
    # 라이브 labels 검사 — 먼저(더 높은 유사도로) 구제된 상대가 있으면 그 클러스터는 차단된다
    if any(labels[partner] == target for partner in partners.get(idx, ())):
      continue
    labels[idx] = target
    rescued.add(idx)


def _evict_ambiguous(
  labels: np.ndarray,
  embeddings: np.ndarray,
  cannot_link: tuple[tuple[int, int], ...],
  protected: set[int],
  config: ClusterConfig,
) -> tuple[int, ...]:
  """저신뢰 멤버를 클러스터에서 분리한다 (labels는 노이즈로 수정, ambiguous 인덱스만 반환).

  자신 없는 배정을 인물 앨범에 넣지 않는다 (feature-spec §7, TBD #3의 기본 정책):
  - 자기 클러스터 유사도가 바닥(min_membership_similarity) 미만 → 어디에도 속하지 않는 얼굴이므로
    노이즈로 강등한다 (반환 목록에는 없음 — noise_indices로 집계된다).
  - 2위 클러스터와의 유사도 마진이 min_membership_margin 미만 → 두 인물 사이의 애매한 얼굴이므로
    ambiguous로 분리해 반환한다.
  자기 클러스터 유사도는 leave-one-out centroid(자기를 뺀 평균) 기준이다 — 자기가 포함된
  centroid는 유사도를 부풀려 경계 얼굴이 마진 검사를 통과해 버린다. 두 가지 예외:
  - 사용자 제약에 직접 걸린 얼굴(must-link 컴포넌트, cannot-link 당사자 — 호출자가 protected로 전달)과
    단독 멤버 클러스터(자기가 곧 centroid)는 빼지 않는다.
  - cannot-link로 연결된 클러스터 쌍은 마진 비교에서 서로 제외한다 — 사용자가 갈라둔 동일 인물
    양쪽에 가까운 것은 당연하므로, 분리 유지가 애매함으로 오판되면 split된 앨범이 전부 비게 된다.
  평가는 시작 시점 멤버십 스냅샷으로 일괄 수행해 축출 순서에 결과가 의존하지 않는다.
  """
  ordered = _cluster_groups(labels)
  if not ordered:
    return ()
  centroids = np.stack([_normalized_mean(embeddings, members) for _, members in ordered])
  position_of = {label: pos for pos, (label, _) in enumerate(ordered)}
  count = len(ordered)
  linked = np.zeros((count, count), dtype=bool)
  for a, b in cannot_link:
    label_a, label_b = int(labels[a]), int(labels[b])
    if label_a != _NOISE and label_b != _NOISE and label_a != label_b:
      pos_a, pos_b = position_of[label_a], position_of[label_b]
      linked[pos_a, pos_b] = linked[pos_b, pos_a] = True

  all_sims = embeddings @ centroids.T  # (N, 클러스터 수) — 멤버별 matmul 대신 1회 BLAS 호출 (N=8천에서 ~3배 차이)
  demoted_noise: list[int] = []
  ambiguous: list[int] = []
  for pos, (_, members) in enumerate(ordered):
    if len(members) < 2:
      continue
    member_arr = np.asarray(members)
    loo_sims = _loo_similarities(embeddings, members)
    unlinked = [q for q in range(count) if q != pos and not linked[pos, q]]
    others_max = all_sims[member_arr][:, unlinked].max(axis=1) if unlinked else None
    for row, idx in enumerate(members):
      if idx in protected:
        continue
      sim_own = float(loo_sims[row])
      if sim_own < config.min_membership_similarity:
        demoted_noise.append(idx)
      elif others_max is not None and sim_own - float(others_max[row]) < config.min_membership_margin:
        ambiguous.append(idx)
  for idx in demoted_noise + ambiguous:
    labels[idx] = _NOISE
  return tuple(sorted(ambiguous))


def _match_cluster_ids(
  new_clusters: list[tuple[int, list[int]]],
  previous_cluster_ids: Sequence[str | None],
  min_match_jaccard: float,
) -> tuple[dict[int, str], tuple[str, ...]]:
  """신규 파티션 ↔ 기존 클러스터를 Jaccard 내림차순 greedy 1:1 매칭한다 (feature-spec §4 ④).

  스펙의 'overlap 최대 매칭(Jaccard / 헝가리안)' 중 greedy Jaccard를 채택 — 가장 강한 겹침이
  그 번호를 가져가는 규칙이 결정적·설명 가능하고, numpy 전용 원칙(scipy 헝가리안 배제)과 맞다.
  동률은 (교집합 크기 내림차순 → 기존 id 등장 순 → 신규 클러스터 순)으로 고정한다.
  반환: (신규 클러스터 위치 → 승계한 cluster_id, 승계되지 못해 은퇴하는 기존 id들).
  """
  previous_members: dict[str, set[int]] = {}
  previous_order: list[str] = []  # 첫 등장 순서 — 은퇴 목록과 동률 처리의 결정성
  for idx, previous_id in enumerate(previous_cluster_ids):
    if previous_id is None:
      continue
    if previous_id not in previous_members:
      previous_members[previous_id] = set()
      previous_order.append(previous_id)
    previous_members[previous_id].add(idx)

  candidates: list[tuple[float, int, int, int]] = []  # (-jaccard, -교집합, 기존 순번, 신규 순번)
  for new_pos, (_, members) in enumerate(new_clusters):
    member_set = set(members)
    for prev_pos, previous_id in enumerate(previous_order):
      intersection = len(member_set & previous_members[previous_id])
      if intersection == 0:
        continue
      union = len(member_set) + len(previous_members[previous_id]) - intersection
      jaccard = intersection / union
      if jaccard < min_match_jaccard:
        continue
      candidates.append((-jaccard, -intersection, prev_pos, new_pos))

  matched: dict[int, str] = {}
  used_previous: set[int] = set()
  for _, _, prev_pos, new_pos in sorted(candidates):
    if new_pos in matched or prev_pos in used_previous:
      continue
    matched[new_pos] = previous_order[prev_pos]
    used_previous.add(prev_pos)

  retired = tuple(pid for pos, pid in enumerate(previous_order) if pos not in used_previous)
  return matched, retired


def recluster(
  embeddings: np.ndarray,
  previous_cluster_ids: Sequence[str | None],
  constraints: Constraints | None = None,
  config: ClusterConfig | None = None,
  new_id_factory: Callable[[], str] | None = None,
) -> ReclusterResult:
  """group 전체 임베딩을 재군집하고 기존 cluster_id를 재조정한다 (feature-spec §4 ③④⑤).

  재군집 뒤 결정적 후처리를 순서대로 적용한다: 보정 강제(must→cannot-link) → 파편 병합 →
  노이즈 구제 → 저신뢰 ambiguous 분리 → ID 재조정 → 대표벡터 (모듈 독스트링 ①~⑦).

  Args:
    embeddings: shape (N, EMBED_DIM) — group 전체(기존+신규) 임베딩. L2 정규화 단위벡터 전제.
    previous_cluster_ids: 길이 N — 각 행의 직전 클러스터 배정 (신규·직전 노이즈는 None).
    constraints: 사용자 보정 제약. 모순 셋은 ValueError.
    config: HDBSCAN·후처리·매칭 파라미터 (기본: PoC 검증 레시피 + 보수적 후처리 임계).
    new_id_factory: 신규 cluster_id 발급자 (기본 uuid4) — 테스트에서 결정적 주입용.

  같은 입력(과 같은 factory)에 대해 결과는 항상 동일하다(결정적).
  """
  resolved_config = config if config is not None else ClusterConfig()
  resolved_constraints = constraints if constraints is not None else Constraints()
  factory = new_id_factory if new_id_factory is not None else (lambda: str(uuid.uuid4()))

  emb = np.asarray(embeddings)
  if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
    raise ValueError(f"embeddings는 shape (N, {EMBED_DIM})이어야 합니다. 받은 shape: {emb.shape}")
  if emb.size and not np.isfinite(emb).all():
    # 비유한 벡터는 cosine 거리가 정의되지 않아 군집 전체를 오염시킨다 — embed 단계가 None으로
    # 걸러 보냈어야 하는 값이므로 프로그래밍 오류로 거부한다 (embed._preprocess와 동일 철학).
    raise ValueError("embeddings에 비유한값(NaN/inf)이 있습니다. embed 단계는 퇴화 임베딩을 걸러야 합니다.")
  if emb.size:
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
      # HDBSCAN cosine 경로는 내부 정규화하지만 병합·구제·저신뢰 후처리는 단위벡터 전제의 생 내적을
      # 코사인으로 쓴다 — 비정규 입력은 모든 유사도 임계를 조용히 우회하는 것이 리뷰에서 재현됐으므로
      # 거부한다 (embed는 항상 L2 정규화 출력, 저장소 왕복 오차는 atol로 흡수).
      worst = float(norms[int(np.argmax(np.abs(norms - 1.0)))])
      raise ValueError(f"embeddings는 L2 정규화 단위벡터여야 합니다. 받은 norm 예: {worst:.4f}")
  n = emb.shape[0]
  if len(previous_cluster_ids) != n:
    raise ValueError(
      f"previous_cluster_ids 길이는 임베딩 수와 같아야 합니다. 받은 길이: {len(previous_cluster_ids)}, 임베딩 수: {n}"
    )
  _validate_pairs(resolved_constraints.must_link, n, "must-link")
  _validate_pairs(resolved_constraints.cannot_link, n, "cannot-link")
  comp_of, components = _must_link_components(n, resolved_constraints)

  # ③ 전체 재군집 — 표본이 min_samples·min_cluster_size 미만이면 밀도 군집이 정의되지 않으므로
  # 전원 노이즈로 두고 제약 후처리만 적용한다 (이식본은 min_samples > N에서 raise하므로 사전 분기).
  if n >= max(resolved_config.min_cluster_size, resolved_config.min_samples):
    labels = HDBSCAN(
      min_cluster_size=resolved_config.min_cluster_size,
      min_samples=resolved_config.min_samples,
      metric="cosine",
      cluster_selection_epsilon=resolved_config.cluster_selection_epsilon,
    ).fit_predict(emb)
    labels = np.asarray(labels, dtype=np.int64)
    # 균질 blob 퇴화(클러스터 0개) 교정 — 제약 강제 전에 승격해야 cannot-link가 승격된 라벨을 분리할 수 있다
    _promote_single_blob(labels, emb, resolved_config.merge_centroid_similarity)
  else:
    labels = np.full(n, _NOISE, dtype=np.int64)

  # 사용자 보정 강제 — must-link(병합)를 먼저 적용해야 cannot-link(분리)가 최종 상태에서 위반을 본다
  next_label = int(labels.max()) + 1 if n else 0
  next_label = _enforce_must_link(labels, components, next_label)
  _enforce_cannot_link(labels, comp_of, components, resolved_constraints.cannot_link, emb, next_label)

  # 파편 병합 → 노이즈 구제 → 저신뢰 분리 (순서 중요: 병합된 centroid 기준으로 구제하고,
  # 구제까지 끝난 최종 멤버십에서 저신뢰를 가려낸다)
  _merge_fragments(labels, emb, resolved_constraints.cannot_link, resolved_config.merge_centroid_similarity)
  _rescue_noise(labels, emb, resolved_constraints.cannot_link, resolved_config.rescue_similarity)
  protected = {idx for members in components.values() if len(members) >= 2 for idx in members}
  # cannot-link 당사자도 보호 — split로 생긴 소형 클러스터는 구성상 내부 유사도가 낮을 수 있어,
  # 절대 바닥 축출이 클러스터를 통째로 비워 사용자 분리 결정과 cluster_id를 지우는 것이 리뷰에서
  # 재현됐다. 제약 당사자는 사람이 직접 지목한 얼굴이므로 어느 축출 경로로도 빼지 않는다.
  protected.update(idx for pair in resolved_constraints.cannot_link for idx in pair)
  ambiguous_indices = _evict_ambiguous(labels, emb, resolved_constraints.cannot_link, protected, resolved_config)
  ambiguous_set = set(ambiguous_indices)

  # ID 재조정
  new_clusters = _cluster_groups(labels)
  matched, retired = _match_cluster_ids(new_clusters, previous_cluster_ids, resolved_config.min_match_jaccard)

  # 대표벡터 계산 + 결과 조립 (신규 id 발급은 출력 순서대로 — factory 주입 시 결정성 보장)
  clusters = []
  for new_pos, (_, members) in enumerate(new_clusters):
    inherited = matched.get(new_pos)
    centroid = _normalized_mean(emb, members)
    centroid.flags.writeable = False  # frozen dataclass 출력이 하류에서 변형되지 않도록 보호
    clusters.append(
      PersonCluster(
        cluster_id=inherited if inherited is not None else factory(),
        is_new=inherited is None,
        member_indices=tuple(members),
        membership_similarities=tuple(float(s) for s in _loo_similarities(emb, members)),
        centroid=centroid,
      )
    )
  noise_indices = tuple(int(idx) for idx in np.flatnonzero(labels == _NOISE) if int(idx) not in ambiguous_set)
  return ReclusterResult(
    clusters=tuple(clusters),
    noise_indices=noise_indices,
    ambiguous_indices=ambiguous_indices,
    retired_cluster_ids=retired,
  )


if __name__ == "__main__":
  # SQS/pgvector 없이 파이프라인 파리티를 확인: 로컬 이미지들에서 검출→정렬→임베딩→재군집을
  # 실행해 인물 클러스터 구성을 출력한다 (최초 군집 시나리오 — previous_cluster_ids 전부 None).
  import sys
  import time

  # detect/embed는 onnxruntime·huggingface_hub 임포트 체인을 끌고 오므로 CLI 확인 블록에서만 지연 import한다
  import cv2

  from app.pipeline.align import align_face
  from app.pipeline.detect import FaceDetector
  from app.pipeline.embed import FaceEmbedder

  detector = FaceDetector()
  embedder = FaceEmbedder()
  face_names: list[str] = []
  face_embeddings: list[np.ndarray] = []
  for path in sys.argv[1:]:
    image = cv2.imread(path)
    if image is None:
      print(f"{path}: 건너뜀 (이미지를 읽을 수 없음)")
      continue
    detected = detector.detect(image)
    crops = [(i, align_face(image, face.landmarks)) for i, face in enumerate(detected)]
    valid = [(i, crop) for i, crop in crops if crop is not None]
    for (face_i, _), embedding in zip(valid, embedder.embed_batch([crop for _, crop in valid])):
      if embedding is None:
        continue
      face_names.append(f"{path}#face{face_i}")
      face_embeddings.append(embedding)
    print(f"{path}: {len(detected)} face(s), 임베딩 {len(face_embeddings)}개 누적")

  if not face_embeddings:
    print("클러스터링할 얼굴이 없습니다.")
    sys.exit(0)

  start = time.perf_counter()
  result = recluster(np.stack(face_embeddings), [None] * len(face_embeddings))
  elapsed_ms = (time.perf_counter() - start) * 1000.0

  print(
    f"\n{len(face_embeddings)}개 얼굴 → 클러스터 {len(result.clusters)}개, "
    f"노이즈 {len(result.noise_indices)}개, 저신뢰 {len(result.ambiguous_indices)}개 in {elapsed_ms:.1f} ms"
  )
  for cluster in result.clusters:
    print(f"  [{cluster.cluster_id}] is_new={cluster.is_new}, 멤버 {len(cluster.member_indices)}명")
    for idx in cluster.member_indices:
      print(f"    {face_names[idx]}")
  for idx in result.noise_indices:
    print(f"  노이즈: {face_names[idx]}")
  for idx in result.ambiguous_indices:
    print(f"  저신뢰(ambiguous): {face_names[idx]}")
