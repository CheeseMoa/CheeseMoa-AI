"""
hdbscan_standalone.py
=====================

scikit-learn(`sklearn.cluster.HDBSCAN`)의 HDBSCAN 알고리즘을 **numpy만으로** 재구현한
독립 모듈. sklearn / scipy / hdbscan 패키지 버전과 무관하게 항상 동일한 결과를 내도록
알고리즘 핵심(`_reachability.pyx`, `_linkage.pyx`, `_tree.pyx`, `hdbscan.py`의 brute 경로)을
그대로 옮겨왔다. face-detection-PoC의 검증본 이식이며, 결과 라벨이 비트 동일함을 확인한
안전 수정 3가지만 가했다 (상세: ADR 005): 재귀 → 반복 전환(깊은 체인 tree의 RecursionError 방지),
표본 2개 미만 사전 검증(sklearn 동일 동작), in-place 연산 2건(N×N 임시 행렬 제거).

참고 원본: scikit-learn cc50648cc
    sklearn/cluster/_hdbscan/_reachability.pyx  (mutual reachability graph)
    sklearn/cluster/_hdbscan/_linkage.pyx       (Prim MST, single linkage)
    sklearn/cluster/_hdbscan/_tree.pyx          (condense, stability, EOM, labelling)
    sklearn/cluster/_hdbscan/hdbscan.py         (brute 경로 오케스트레이션)

사용 예
-------
    from app.pipeline.hdbscan_standalone import HDBSCAN    # sklearn 호환 클래스
    labels = HDBSCAN(min_cluster_size=2, min_samples=2,
                     metric='cosine', cluster_selection_epsilon=0.15).fit_predict(X)

    # 또는 함수형
    from app.pipeline.hdbscan_standalone import hdbscan_predict
    labels = hdbscan_predict(X, min_cluster_size=2, metric='cosine')

제약
----
- 입력 X는 유한값(finite)이어야 한다. sklearn처럼 inf/nan 샘플을 -1/-2로 분리하는
  처리는 하지 않는다(프로젝트 임베딩은 항상 L2 정규화된 유한값).
- 지원 metric: 'euclidean'('l2'), 'cosine', 'precomputed'.
- dense(밀집) 경로만 구현. sklearn의 KD/Ball-Tree(prims) 경로와 동일한 MST를 계산하므로
  결과 라벨은 동일하다(부동소수 동률이 없는 한).
"""

import numpy as np

__all__ = ["HDBSCAN", "hdbscan_predict"]

INFTY = np.inf
NOISE = -1

# 구조화 dtype — _tree.pyx의 HIERARCHY_dtype / CONDENSED_dtype와 동일한 의미
HIERARCHY_dtype = np.dtype(
  [
    ("left_node", np.intp),
    ("right_node", np.intp),
    ("value", np.float64),
    ("cluster_size", np.intp),
  ]
)

CONDENSED_dtype = np.dtype(
  [
    ("parent", np.intp),
    ("child", np.intp),
    ("value", np.float64),
    ("cluster_size", np.intp),
  ]
)

MST_edge_dtype = np.dtype(
  [
    ("current_node", np.int64),
    ("next_node", np.int64),
    ("distance", np.float64),
  ]
)


# ---------------------------------------------------------------------------
# 1단계 — pairwise distance
# ---------------------------------------------------------------------------
def _pairwise_distance(X, metric):
  """입력 X로부터 (N, N) 거리 행렬을 계산한다. 대각선은 0으로 둔다.

  sklearn.metrics.pairwise_distances와 동일한 정의를 따른다.
  """
  metric = metric.lower()
  if metric == "precomputed":
    D = np.array(X, dtype=np.float64, copy=True)
    if D.shape[0] != D.shape[1]:
      raise ValueError("precomputed metric에는 정사각 거리 행렬이 필요합니다.")
    np.fill_diagonal(D, 0.0)
    return D

  X = np.asarray(X, dtype=np.float64)

  if metric in ("euclidean", "l2"):
    # ||x - y||^2 = ||x||^2 - 2 x·y + ||y||^2  (sklearn euclidean_distances와 동일한 방식)
    sq_norms = np.einsum("ij,ij->i", X, X)
    D2 = sq_norms[:, None] - 2.0 * (X @ X.T) + sq_norms[None, :]
    np.maximum(D2, 0.0, out=D2)  # 부동소수 오차로 음수가 된 값 보정
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D

  if metric == "cosine":
    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0.0] = 1.0
    Xn = X / norms[:, None]
    S = Xn @ Xn.T
    # sklearn cosine_distances처럼 in-place 변환 — `D = 1.0 - S`는 N×N float64 사본을 하나 더 만들어
    # N=1만에서 피크를 0.8GB 키운다 (PoC 원본의 사본 방식과 결과 비트 동일 검증, ADR 005)
    np.subtract(1.0, S, out=S)
    np.clip(S, 0.0, 2.0, out=S)  # sklearn cosine_distances와 동일하게 [0, 2]로 제한
    np.fill_diagonal(S, 0.0)
    return S

  raise NotImplementedError(f"지원하지 않는 metric='{metric}'. 'euclidean', 'cosine', 'precomputed'만 지원합니다.")


# ---------------------------------------------------------------------------
# 2~3단계 — core distance & mutual reachability graph  (_reachability.pyx)
# ---------------------------------------------------------------------------
def _mutual_reachability_graph(distance_matrix, min_samples):
  """거리 행렬을 mutual reachability 행렬로 변환한다 (in-place 대신 복사본 반환).

  core 거리 = 각 점의 (min_samples)번째 최근접 거리(자기 자신 포함).
  원본: _dense_mutual_reachability_graph, further_neighbor_idx = min_samples - 1
  """
  further_neighbor_idx = min_samples - 1
  # 각 행에서 further_neighbor_idx 위치의 값 = (min_samples)번째로 가까운 거리
  core_distances = np.partition(distance_matrix, further_neighbor_idx, axis=1)[:, further_neighbor_idx]

  # MRD(i, j) = max(core_i, core_j, d_ij)
  mr = np.maximum(distance_matrix, core_distances[:, None])
  np.maximum(mr, core_distances[None, :], out=mr)  # out= 으로 N×N 추가 할당 제거 (결과 비트 동일 검증, ADR 005)
  return mr


# ---------------------------------------------------------------------------
# 4단계 — Prim 알고리즘 MST  (_linkage.pyx: mst_from_mutual_reachability)
# ---------------------------------------------------------------------------
def _mst_from_mutual_reachability(mutual_reachability):
  """dense mutual reachability 행렬에서 Prim 알고리즘으로 MST 간선을 추출한다."""
  n_samples = mutual_reachability.shape[0]
  mst = np.empty(n_samples - 1, dtype=MST_edge_dtype)
  current_labels = np.arange(n_samples, dtype=np.int64)
  current_node = 0
  min_reachability = np.full(n_samples, np.inf, dtype=np.float64)

  for i in range(n_samples - 1):
    label_filter = current_labels != current_node
    current_labels = current_labels[label_filter]
    left = min_reachability[label_filter]
    right = mutual_reachability[current_node][current_labels]
    min_reachability = np.minimum(left, right)

    new_node_index = np.argmin(min_reachability)
    new_node = current_labels[new_node_index]
    mst[i]["current_node"] = current_node
    mst[i]["next_node"] = new_node
    mst[i]["distance"] = min_reachability[new_node_index]
    current_node = new_node

  return mst


# ---------------------------------------------------------------------------
# 5단계 — single linkage tree  (_linkage.pyx: make_single_linkage + UnionFind)
# ---------------------------------------------------------------------------
class _LinkageUnionFind:
  """single linkage 구성용 union-find. union 시 새 클러스터 라벨을 생성한다.
  원본: sklearn.cluster._hierarchical_fast.UnionFind
  """

  def __init__(self, N):
    self.parent = np.full(2 * N - 1, -1, dtype=np.intp)
    self.next_label = N
    self.size = np.concatenate((np.ones(N, dtype=np.intp), np.zeros(N - 1, dtype=np.intp)))

  def union(self, m, n):
    self.parent[m] = self.next_label
    self.parent[n] = self.next_label
    self.size[self.next_label] = self.size[m] + self.size[n]
    self.next_label += 1

  def fast_find(self, n):
    p = n
    while self.parent[n] != -1:
      n = self.parent[n]
    # 경로 압축
    while self.parent[p] != n and self.parent[p] != -1:
      self.parent[p], p = n, self.parent[p]
    return n


def _make_single_linkage(mst):
  """MST(거리 오름차순 정렬됨)로부터 single linkage 계층을 만든다."""
  n_samples = mst.shape[0] + 1
  single_linkage = np.zeros(n_samples - 1, dtype=HIERARCHY_dtype)
  U = _LinkageUnionFind(n_samples)

  for i in range(n_samples - 1):
    current_node = int(mst[i]["current_node"])
    next_node = int(mst[i]["next_node"])
    distance = mst[i]["distance"]

    current_cluster = U.fast_find(current_node)
    next_cluster = U.fast_find(next_node)

    single_linkage[i]["left_node"] = current_cluster
    single_linkage[i]["right_node"] = next_cluster
    single_linkage[i]["value"] = distance
    single_linkage[i]["cluster_size"] = U.size[current_cluster] + U.size[next_cluster]

    U.union(current_cluster, next_cluster)

  return single_linkage


def _process_mst(min_spanning_tree):
  """MST를 거리 기준으로 정렬한 뒤 single linkage 계층으로 변환한다 (hdbscan.py)."""
  row_order = np.argsort(min_spanning_tree["distance"], kind="stable")
  min_spanning_tree = min_spanning_tree[row_order]
  return _make_single_linkage(min_spanning_tree)


# ---------------------------------------------------------------------------
# 6단계 — condensed tree  (_tree.pyx: bfs_from_hierarchy, _condense_tree)
# ---------------------------------------------------------------------------
def _bfs_from_hierarchy(hierarchy, bfs_root):
  """scipy hclust 포맷 계층에서 너비 우선 탐색."""
  n_samples = hierarchy.shape[0] + 1
  process_queue = [bfs_root]
  result = []

  while process_queue:
    result.extend(process_queue)
    process_queue = [x - n_samples for x in process_queue if x >= n_samples]
    if process_queue:
      next_queue = []
      for node in process_queue:
        next_queue.append(int(hierarchy[node]["left_node"]))
        next_queue.append(int(hierarchy[node]["right_node"]))
      process_queue = next_queue
  return result


def _condense_tree(hierarchy, min_cluster_size):
  """min_cluster_size 미만 분기를 흡수하여 condensed tree를 만든다."""
  root = 2 * hierarchy.shape[0]
  n_samples = hierarchy.shape[0] + 1
  next_label = n_samples + 1
  node_list = _bfs_from_hierarchy(hierarchy, root)

  relabel = np.empty(root + 1, dtype=np.intp)
  relabel[root] = n_samples
  result_list = []
  ignore = np.zeros(root + 1, dtype=bool)

  for node in node_list:
    if ignore[node] or node < n_samples:
      continue

    children = hierarchy[node - n_samples]
    left = int(children["left_node"])
    right = int(children["right_node"])
    distance = children["value"]
    if distance > 0.0:
      lambda_value = 1.0 / distance
    else:
      lambda_value = INFTY

    left_count = int(hierarchy[left - n_samples]["cluster_size"]) if left >= n_samples else 1
    right_count = int(hierarchy[right - n_samples]["cluster_size"]) if right >= n_samples else 1

    if left_count >= min_cluster_size and right_count >= min_cluster_size:
      relabel[left] = next_label
      next_label += 1
      result_list.append((relabel[node], relabel[left], lambda_value, left_count))

      relabel[right] = next_label
      next_label += 1
      result_list.append((relabel[node], relabel[right], lambda_value, right_count))

    elif left_count < min_cluster_size and right_count < min_cluster_size:
      for sub_node in _bfs_from_hierarchy(hierarchy, left):
        if sub_node < n_samples:
          result_list.append((relabel[node], sub_node, lambda_value, 1))
        ignore[sub_node] = True
      for sub_node in _bfs_from_hierarchy(hierarchy, right):
        if sub_node < n_samples:
          result_list.append((relabel[node], sub_node, lambda_value, 1))
        ignore[sub_node] = True

    elif left_count < min_cluster_size:
      relabel[right] = relabel[node]
      for sub_node in _bfs_from_hierarchy(hierarchy, left):
        if sub_node < n_samples:
          result_list.append((relabel[node], sub_node, lambda_value, 1))
        ignore[sub_node] = True

    else:
      relabel[left] = relabel[node]
      for sub_node in _bfs_from_hierarchy(hierarchy, right):
        if sub_node < n_samples:
          result_list.append((relabel[node], sub_node, lambda_value, 1))
        ignore[sub_node] = True

  return np.array(result_list, dtype=CONDENSED_dtype)


# ---------------------------------------------------------------------------
# 7단계 — stability  (_tree.pyx: _compute_stability)
# ---------------------------------------------------------------------------
def _compute_stability(condensed_tree):
  parents = condensed_tree["parent"]
  largest_child = int(condensed_tree["child"].max())
  smallest_cluster = int(parents.min())
  num_clusters = int(parents.max()) - smallest_cluster + 1
  largest_child = max(largest_child, smallest_cluster)

  births = np.full(largest_child + 1, np.nan, dtype=np.float64)
  births[condensed_tree["child"]] = condensed_tree["value"]
  births[smallest_cluster] = 0.0

  result = np.zeros(num_clusters, dtype=np.float64)
  for node in condensed_tree:
    parent = int(node["parent"])
    result[parent - smallest_cluster] += (node["value"] - births[parent]) * node["cluster_size"]

  return {idx + smallest_cluster: result[idx] for idx in range(num_clusters)}


# ---------------------------------------------------------------------------
# 8단계 보조 — cluster tree 탐색 / TreeUnionFind / labelling  (_tree.pyx)
# ---------------------------------------------------------------------------
def _bfs_from_cluster_tree(condensed_tree, bfs_root):
  result = []
  parents = condensed_tree["parent"]
  children = condensed_tree["child"]
  process_queue = np.array([bfs_root], dtype=np.intp)
  while len(process_queue) > 0:
    result.extend(process_queue.tolist())
    process_queue = children[np.isin(parents, process_queue)]
  return result


def _max_lambdas(condensed_tree):
  """각 parent 클러스터의 최대 lambda(=death) 값."""
  largest_parent = int(condensed_tree["parent"].max())
  deaths = np.zeros(largest_parent + 1, dtype=np.float64)
  current_parent = int(condensed_tree[0]["parent"])
  max_lambda = condensed_tree[0]["value"]

  for idx in range(1, condensed_tree.shape[0]):
    parent = int(condensed_tree[idx]["parent"])
    lambda_val = condensed_tree[idx]["value"]
    if parent == current_parent:
      max_lambda = max(max_lambda, lambda_val)
    else:
      deaths[current_parent] = max_lambda
      current_parent = parent
      max_lambda = lambda_val
  deaths[current_parent] = max_lambda
  return deaths


class _TreeUnionFind:
  """labelling 용 union-find (union by rank). 원본: _tree.pyx TreeUnionFind"""

  def __init__(self, size):
    self.data = np.zeros((size, 2), dtype=np.intp)
    self.data[:, 0] = np.arange(size)
    self.is_component = np.ones(size, dtype=bool)

  def union(self, x, y):
    x_root = self.find(x)
    y_root = self.find(y)
    if self.data[x_root, 1] < self.data[y_root, 1]:
      self.data[x_root, 0] = y_root
    elif self.data[x_root, 1] > self.data[y_root, 1]:
      self.data[y_root, 0] = x_root
    else:
      self.data[y_root, 0] = x_root
      self.data[x_root, 1] += 1

  def find(self, x):
    if self.data[x, 0] != x:
      self.data[x, 0] = self.find(self.data[x, 0])
      self.is_component[x] = False
    return self.data[x, 0]


def _recurse_leaf_dfs(cluster_tree, current_node):
  # 원본(재귀 DFS)을 명시적 스택으로 전환 — 자식을 역순으로 쌓아 방문 순서까지 동일하다.
  # 깊은 체인형 cluster tree의 RecursionError 방지 (leaf 선택 모드 경로, ADR 005).
  result = []
  stack = [current_node]
  while stack:
    node = stack.pop()
    children = cluster_tree[cluster_tree["parent"] == node]["child"]
    if children.shape[0] == 0:
      result.append(node)
    else:
      stack.extend(int(child) for child in reversed(children.tolist()))
  return result


def _get_cluster_tree_leaves(cluster_tree):
  if cluster_tree.shape[0] == 0:
    return []
  root = int(cluster_tree["parent"].min())
  return _recurse_leaf_dfs(cluster_tree, root)


def _traverse_upwards(cluster_tree, cluster_selection_epsilon, leaf, allow_single_cluster):
  # 원본(꼬리 재귀)을 반복문으로 전환 — 결과 동일. Cython 원본과 달리 순수 파이썬 재귀는 깊은
  # 체인형 cluster tree에서 RecursionError로 죽는 것이 프로덕션 경로(eom+eps)에서 재현됐다 (ADR 005).
  root = int(cluster_tree["parent"].min())
  node = leaf
  while True:
    parent = int(cluster_tree[cluster_tree["child"] == node]["parent"][0])
    if parent == root:
      return parent if allow_single_cluster else node
    parent_eps = 1.0 / cluster_tree[cluster_tree["child"] == parent]["value"][0]
    if parent_eps > cluster_selection_epsilon:
      return parent
    node = parent


def _epsilon_search(leaves, cluster_tree, cluster_selection_epsilon, allow_single_cluster):
  selected_clusters = []
  processed = []
  children = cluster_tree["child"]
  distances = cluster_tree["value"]

  for leaf in leaves:
    eps = 1.0 / distances[children == leaf][0]
    if eps < cluster_selection_epsilon:
      if leaf not in processed:
        epsilon_child = _traverse_upwards(cluster_tree, cluster_selection_epsilon, leaf, allow_single_cluster)
        selected_clusters.append(epsilon_child)
        for sub_node in _bfs_from_cluster_tree(cluster_tree, epsilon_child):
          if sub_node != epsilon_child:
            processed.append(sub_node)
    else:
      selected_clusters.append(leaf)

  return set(selected_clusters)


def _do_labelling(condensed_tree, clusters, cluster_label_map, allow_single_cluster, cluster_selection_epsilon):
  child_array = condensed_tree["child"]
  parent_array = condensed_tree["parent"]
  lambda_array = condensed_tree["value"]

  root_cluster = int(parent_array.min())
  result = np.empty(root_cluster, dtype=np.intp)
  union_find = _TreeUnionFind(int(parent_array.max()) + 1)

  for n in range(condensed_tree.shape[0]):
    child = int(child_array[n])
    parent = int(parent_array[n])
    if child not in clusters:
      union_find.union(parent, child)

  for n in range(root_cluster):
    cluster = int(union_find.find(n))
    label = NOISE
    if cluster != root_cluster:
      label = cluster_label_map[cluster]
    elif len(clusters) == 1 and allow_single_cluster:
      parent_lambda = lambda_array[child_array == n]
      if cluster_selection_epsilon != 0.0:
        threshold = 1.0 / cluster_selection_epsilon
      else:
        threshold = lambda_array[parent_array == cluster].max()
      if parent_lambda >= threshold:
        label = cluster_label_map[cluster]
    result[n] = label

  return result


def _get_probabilities(condensed_tree, cluster_map, labels):
  child_array = condensed_tree["child"]
  parent_array = condensed_tree["parent"]
  lambda_array = condensed_tree["value"]

  result = np.zeros(labels.shape[0], dtype=np.float64)
  deaths = _max_lambdas(condensed_tree)
  root_cluster = int(parent_array.min())

  for n in range(condensed_tree.shape[0]):
    point = int(child_array[n])
    if point >= root_cluster:
      continue
    cluster_num = labels[point]
    if cluster_num == -1:
      continue
    cluster = cluster_map[cluster_num]
    max_lambda = deaths[cluster]
    if max_lambda == 0.0 or np.isinf(lambda_array[n]):
      result[point] = 1.0
    else:
      lambda_val = min(lambda_array[n], max_lambda)
      result[point] = lambda_val / max_lambda

  return result


# ---------------------------------------------------------------------------
# 8단계 — 클러스터 선택 (EOM / leaf)  (_tree.pyx: _get_clusters)
# ---------------------------------------------------------------------------
def _get_clusters(
  condensed_tree,
  stability,
  cluster_selection_method="eom",
  allow_single_cluster=False,
  cluster_selection_epsilon=0.0,
  max_cluster_size=None,
):
  if allow_single_cluster:
    node_list = sorted(stability.keys(), reverse=True)
  else:
    node_list = sorted(stability.keys(), reverse=True)[:-1]  # 루트 제외

  cluster_tree = condensed_tree[condensed_tree["cluster_size"] > 1]
  is_cluster = {cluster: True for cluster in node_list}
  n_samples = int(np.max(condensed_tree[condensed_tree["cluster_size"] == 1]["child"])) + 1

  if max_cluster_size is None:
    max_cluster_size = n_samples + 1  # 절대 발동되지 않는 값
  cluster_sizes = {int(child): int(size) for child, size in zip(cluster_tree["child"], cluster_tree["cluster_size"])}
  if allow_single_cluster:
    cluster_sizes[node_list[-1]] = int(np.sum(cluster_tree[cluster_tree["parent"] == node_list[-1]]["cluster_size"]))

  if cluster_selection_method == "eom":
    for node in node_list:
      child_selection = cluster_tree["parent"] == node
      subtree_stability = np.sum([stability[int(child)] for child in cluster_tree["child"][child_selection]])
      if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size:
        is_cluster[node] = False
        stability[node] = subtree_stability
      else:
        for sub_node in _bfs_from_cluster_tree(cluster_tree, node):
          if sub_node != node:
            is_cluster[sub_node] = False

    if cluster_selection_epsilon != 0.0 and cluster_tree.shape[0] > 0:
      eom_clusters = [c for c in is_cluster if is_cluster[c]]
      selected_clusters = []
      if len(eom_clusters) == 1 and eom_clusters[0] == int(cluster_tree["parent"].min()):
        if allow_single_cluster:
          selected_clusters = eom_clusters
      else:
        selected_clusters = _epsilon_search(
          set(eom_clusters), cluster_tree, cluster_selection_epsilon, allow_single_cluster
        )
      for c in is_cluster:
        is_cluster[c] = c in selected_clusters

  elif cluster_selection_method == "leaf":
    leaves = set(_get_cluster_tree_leaves(cluster_tree))
    if len(leaves) == 0:
      for c in is_cluster:
        is_cluster[c] = False
      is_cluster[int(condensed_tree["parent"].min())] = True

    if cluster_selection_epsilon != 0.0:
      selected_clusters = _epsilon_search(leaves, cluster_tree, cluster_selection_epsilon, allow_single_cluster)
    else:
      selected_clusters = leaves

    for c in is_cluster:
      is_cluster[c] = c in selected_clusters
  else:
    raise ValueError(f"잘못된 cluster_selection_method='{cluster_selection_method}'. 'eom' 또는 'leaf'만 가능합니다.")

  clusters = {c for c in is_cluster if is_cluster[c]}
  cluster_map = {c: n for n, c in enumerate(sorted(clusters))}
  reverse_cluster_map = {n: c for c, n in cluster_map.items()}

  labels = _do_labelling(condensed_tree, clusters, cluster_map, allow_single_cluster, cluster_selection_epsilon)
  probs = _get_probabilities(condensed_tree, reverse_cluster_map, labels)
  return labels, probs


def _tree_to_labels(
  single_linkage_tree,
  min_cluster_size=10,
  cluster_selection_method="eom",
  allow_single_cluster=False,
  cluster_selection_epsilon=0.0,
  max_cluster_size=None,
):
  condensed_tree = _condense_tree(single_linkage_tree, min_cluster_size)
  labels, probabilities = _get_clusters(
    condensed_tree,
    _compute_stability(condensed_tree),
    cluster_selection_method,
    allow_single_cluster,
    cluster_selection_epsilon,
    max_cluster_size,
  )
  return labels, probabilities


# ---------------------------------------------------------------------------
# 공개 인터페이스
# ---------------------------------------------------------------------------
def hdbscan_predict(
  X,
  min_cluster_size=5,
  min_samples=None,
  metric="euclidean",
  cluster_selection_epsilon=0.0,
  cluster_selection_method="eom",
  alpha=1.0,
  max_cluster_size=None,
  allow_single_cluster=False,
):
  """입력 X에 대해 HDBSCAN 클러스터 라벨을 반환한다 (노이즈 = -1).

  sklearn.cluster.HDBSCAN(...).fit_predict(X)와 동일한 결과를 낸다.
  """
  labels, _ = _run_hdbscan(
    X,
    min_cluster_size,
    min_samples,
    metric,
    cluster_selection_epsilon,
    cluster_selection_method,
    alpha,
    max_cluster_size,
    allow_single_cluster,
  )
  return labels


def _run_hdbscan(
  X,
  min_cluster_size,
  min_samples,
  metric,
  cluster_selection_epsilon,
  cluster_selection_method,
  alpha,
  max_cluster_size,
  allow_single_cluster,
):
  if min_cluster_size < 2:
    raise ValueError("min_cluster_size는 2 이상이어야 합니다.")
  if min_samples is None:
    min_samples = min_cluster_size

  X = np.asarray(X)
  n_samples = X.shape[0]
  if n_samples < 2:
    # sklearn과 동일한 사전 검증 — 없으면 n=1·min_samples=1 조합이 빈 condensed tree로 흘러가
    # 불투명한 numpy 예외(zero-size reduction)로 죽는다 (리뷰 재현)
    raise ValueError(f"HDBSCAN에는 표본이 2개 이상 필요합니다. 받은 표본 수: {n_samples}")
  if min_samples > n_samples:
    raise ValueError(f"min_samples({min_samples})는 샘플 수({n_samples}) 이하여야 합니다.")

  # brute 경로 (hdbscan.py: _hdbscan_brute)
  distance_matrix = _pairwise_distance(X, metric)
  distance_matrix /= alpha
  mutual_reachability = _mutual_reachability_graph(distance_matrix, min_samples)
  mst = _mst_from_mutual_reachability(mutual_reachability)
  single_linkage_tree = _process_mst(mst)

  labels, probabilities = _tree_to_labels(
    single_linkage_tree,
    min_cluster_size,
    cluster_selection_method,
    allow_single_cluster,
    cluster_selection_epsilon,
    max_cluster_size,
  )
  return labels, probabilities


class HDBSCAN:
  """sklearn.cluster.HDBSCAN과 호환되는 얇은 래퍼.

  호출부에서 import 한 줄만 바꾸면 그대로 동작하도록 동일한 파라미터/속성 이름을 사용한다.
  """

  def __init__(
    self,
    min_cluster_size=5,
    min_samples=None,
    metric="euclidean",
    cluster_selection_epsilon=0.0,
    cluster_selection_method="eom",
    alpha=1.0,
    max_cluster_size=None,
    allow_single_cluster=False,
    **kwargs,
  ):
    self.min_cluster_size = min_cluster_size
    self.min_samples = min_samples
    self.metric = metric
    self.cluster_selection_epsilon = cluster_selection_epsilon
    self.cluster_selection_method = cluster_selection_method
    self.alpha = alpha
    self.max_cluster_size = max_cluster_size
    self.allow_single_cluster = allow_single_cluster
    # sklearn에만 있는 그 외 인자(algorithm, leaf_size, n_jobs 등)는 무시한다.
    self.labels_ = None
    self.probabilities_ = None

  def fit(self, X, y=None):
    self.labels_, self.probabilities_ = _run_hdbscan(
      X,
      self.min_cluster_size,
      self.min_samples,
      self.metric,
      self.cluster_selection_epsilon,
      self.cluster_selection_method,
      self.alpha,
      self.max_cluster_size,
      self.allow_single_cluster,
    )
    return self

  def fit_predict(self, X, y=None):
    self.fit(X)
    return self.labels_
