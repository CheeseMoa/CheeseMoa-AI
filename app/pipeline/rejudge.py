"""uncertain 얼굴의 Rekognition CompareFaces 보조 재판정 (ADR-030, 2026-07-23 실측 리뷰).

옆얼굴·가림·나이차 하드케이스는 AuraFace 코사인으로 원리적으로 못 가른다(동일인 0.18~0.48 vs
타인 0.20~0.36 완전 겹침). 재군집이 끝나고 uncertain으로 확정되기 직전, 미배정 얼굴을 AuraFace
유사도 상위 top_k개 앨범의 대표 얼굴과 CompareFaces로 비교해 최고점으로 재판정한다:

  - 최고점 ≥ auto_assign_similarity → 자동 편입 후보 (핸들러가 must-link로 기록 후 재군집 2차 패스)
  - 최고점 ∈ [suggest, auto) → "이 앨범 아닐까요?" 제안 (결과 메시지 suggestions)
  - fragment_similarity 이상 복수 매칭 → 같은 인물의 파편 앨범 힌트 (로그 기록, 계약은 후속)

top_k 후보는 전부 호출 후 argmax로 판정한다(조기 종료 없음) — 이 기능이 겨냥하는 대역이 바로
AuraFace 순위를 못 믿는 하드케이스라, 1순위가 임계를 넘겼다고 멈추면 2순위가 진짜 정답일 때
오답 앨범에 붙는다. (face_id, 대표 face_id) 쌍 점수는 호출자에게 캐시 dict로 받고 병합본을
돌려준다 — 재군집·보정마다 재판정이 돌아도 같은 쌍은 재과금이 없다(저장은 handlers의 책임).

이 모듈은 boto3·S3를 모른다: CompareFaces 호출(FaceComparer)·원본 fetch·crop 렌더는 전부
콜러블 주입이다(handlers.FaceExtractor와 같은 구도, 합성은 core.deps의 책임). 덕분에 스모크가
AWS 없이 돈다. rejudge()는 절대 raise하지 않는 계약이 아니다 — best-effort 격리(실패가 job을
죽이지 않게)는 handlers의 몫이고, 여기서는 쌍 단위 실패만 로그 후 건너뛴다.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from app.storage.event_embeddings import EventEmbeddings
  from app.pipeline.cluster import PersonCluster

logger = logging.getLogger(__name__)

# crop JPEG 2장(source, target) → Rekognition Similarity(0~100). "crop에서 얼굴 미검출"은 -1.0
# 센티널로 반환한다(캐싱돼 영구 재과금 방지) — 스로틀·네트워크 등 일시 장애는 raise가 계약이다
# (센티널과 달리 캐싱하면 안 되는 실패). boto3 합성은 core.deps의 책임.
FaceComparer = Callable[[bytes, bytes], float]

# 디코딩된 BGR 원본 + 얼굴 bbox(x, y, w, h) → 재판정용 crop JPEG bytes (퇴화 crop이면 None).
# 실측 근거(2026-07-23 리뷰)와의 crop 파리티가 계약이다 — render_rejudge_crop 참조.
CropRenderer = Callable[[np.ndarray, tuple[float, float, float, float]], "bytes | None"]

# 원본 이미지 fetch — storage.ImageSource.fetch와 동일 시그니처 (실패는 예외로 던진다)
ImageFetcher = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class RejudgeConfig:
  """재판정 임계·상한 — 기본값은 2026-07-23 실측 리뷰의 확정값 (Settings.to_rejudge_config가 주입).

  임계 근거: 실서버 미배정 392개 전수 스윕에서 95는 전수 정답(보수), 90은 실용 권장값,
  85 미만은 남매·타인 아기 오답 대역(82~84 실존)이라 제안조차 하지 않는다.
  """

  auto_assign_similarity: float = 90.0  # 이상이면 자동 편입 (보수 운영 95)
  suggest_similarity: float = 85.0  # [이 값, auto) 대역은 제안 — 미만은 판정 보류(현 동작 유지)
  fragment_similarity: float = 95.0  # 이상 복수 앨범 동시 매칭 = 파편 앨범 힌트
  top_k: int = 3  # 얼굴당 비교할 AuraFace 유사도 상위 앨범 수 (호출 수 상한의 1차 통제)
  max_calls: int = 150  # job당 CompareFaces 호출 상한 — 오염 이벤트의 비용 폭주 안전판
  # 같은 사진 검사의 이중 검출 안전판 — handlers._SAME_FACE_SIMILARITY(ADR-011)와 같은 원리·기본값.
  # 대상 앨범에 같은 사진의 얼굴이 있으면 편입을 기각하되, 그 얼굴이 이 값 이상 닮았으면 타인이
  # 아니라 YuNet 파편 박스라 기각하지 않는다.
  same_face_similarity: float = 0.95

  def __post_init__(self) -> None:
    if not (0.0 < self.suggest_similarity <= self.auto_assign_similarity <= 100.0):
      raise ValueError(
        f"0 < suggest_similarity <= auto_assign_similarity <= 100이어야 합니다. "
        f"받은 값: suggest={self.suggest_similarity}, auto={self.auto_assign_similarity}"
      )
    if not (0.0 < self.fragment_similarity <= 100.0):
      raise ValueError(f"fragment_similarity는 (0, 100] 범위여야 합니다. 받은 값: {self.fragment_similarity}")
    if self.top_k < 1:
      raise ValueError(f"top_k는 1 이상이어야 합니다. 받은 값: {self.top_k}")
    if self.max_calls < 1:
      raise ValueError(f"max_calls는 1 이상이어야 합니다. 받은 값: {self.max_calls}")
    if not (0.0 < self.same_face_similarity <= 1.0):
      raise ValueError(f"same_face_similarity는 (0, 1] 범위여야 합니다. 받은 값: {self.same_face_similarity}")


@dataclass(frozen=True)
class RejudgeAssignment:
  """자동 편입 판정 1건 — 핸들러가 (face_id, rep_face_id)를 must-link로 기록한다."""

  face_id: str
  cluster_id: str
  rep_face_id: str
  similarity: float


@dataclass(frozen=True)
class RejudgeSuggestion:
  """제안 대역 판정 1건 — 결과 메시지의 UncertainSuggestion으로 번역된다 (handlers).

  cluster_id는 판정 시점(1차 패스) 값이라 편입이 2차 패스를 돌리면 신규 발급·병합으로 바뀔 수
  있다 — 최종 앨범은 rep_face_id(대표 얼굴)가 최종 상태에서 속한 클러스터로 재해석하는 것이
  계약이다 (handlers._assemble_result).
  """

  face_id: str
  cluster_id: str
  rep_face_id: str
  similarity: float


@dataclass(frozen=True)
class FragmentHint:
  """복수 앨범 동시 고점 매칭 — 같은 인물의 파편 앨범일 가능성 신호 (유사도 내림차순)."""

  face_id: str
  cluster_ids: tuple[str, ...]
  similarities: tuple[float, ...]


@dataclass(frozen=True)
class RejudgeOutcome:
  """rejudge 1회 실행의 결과 — scores는 (기존 캐시 + 신규 측정) 병합본이며 저장은 핸들러의 책임이다."""

  assignments: tuple[RejudgeAssignment, ...] = ()
  suggestions: tuple[RejudgeSuggestion, ...] = ()
  fragment_hints: tuple[FragmentHint, ...] = ()
  scores: dict[tuple[str, str], float] | None = None
  calls_made: int = 0
  truncated: bool = False  # max_calls 도달로 일부 얼굴을 판정하지 못함


def render_rejudge_crop(
  image: np.ndarray,
  bbox: tuple[float, float, float, float],
  margin_ratio: float = 0.25,
  jpeg_quality: int = 92,
) -> bytes | None:
  """재판정용 얼굴 crop JPEG를 만든다 — 실측(2026-07-23 리뷰)의 crop과 파리티가 계약이다.

  bbox + margin_ratio 여백, JPEG quality 92, 축소 없음: 임계 90/85/95의 근거 데이터가 전부
  이 crop으로 측정됐다. 썸네일 렌더러(1.4배 확장 + max_side 축소)를 쓰면 근거가 무효가 되므로
  의도적으로 별도 구현이다. 퇴화 입력(bbox 미상·빈 crop·인코딩 실패)은 None.
  """
  if margin_ratio < 0:
    raise ValueError(f"margin_ratio는 0 이상이어야 합니다. 받은 값: {margin_ratio}")
  if not (1 <= jpeg_quality <= 100):
    raise ValueError(f"jpeg_quality는 [1, 100] 범위여야 합니다. 받은 값: {jpeg_quality}")
  import cv2  # 지연 임포트 — handlers 스모크가 cv2 없이 이 모듈을 임포트할 수 있게 (deps의 렌더러 합성 시점에만 필요)

  x, y, w, h = bbox
  if w <= 0 or h <= 0:
    return None
  ih, iw = image.shape[:2]
  mx, my = int(w * margin_ratio), int(h * margin_ratio)
  crop = image[max(0, int(y) - my) : min(ih, int(y + h) + my), max(0, int(x) - mx) : min(iw, int(x + w) + mx)]
  if crop.size == 0:
    return None
  ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
  return buf.tobytes() if ok else None


class _FaceUnionFind:
  """face_id 문자열 위의 union-find — must-link 폐포 계산 전용 (handlers의 동명 클래스와 같은 최소 구현)."""

  def __init__(self) -> None:
    self._parent: dict[str, str] = {}

  def find(self, x: str) -> str:
    self._parent.setdefault(x, x)
    root = x
    while self._parent[root] != root:
      root = self._parent[root]
    while self._parent[x] != root:  # 경로 압축
      self._parent[x], x = root, self._parent[x]
    return root

  def union(self, a: str, b: str) -> None:
    root_a, root_b = self.find(a), self.find(b)
    if root_a != root_b:
      self._parent[root_b] = root_a


class UncertainRejudger:
  """uncertain 얼굴 × top-k 앨범 대표의 CompareFaces 재판정기 — 순수 판정 로직 (저장·발행은 handlers)."""

  def __init__(
    self,
    compare: FaceComparer,
    fetch_image: ImageFetcher,
    crop: CropRenderer,
    config: RejudgeConfig | None = None,
  ) -> None:
    self._compare = compare
    self._fetch_image = fetch_image
    self._crop = crop
    self._config = config if config is not None else RejudgeConfig()

  def rejudge(
    self,
    event: "EventEmbeddings",
    clusters: Sequence["PersonCluster"],
    uncertain_indices: Sequence[int],
    cached_scores: dict[tuple[str, str], float],
  ) -> RejudgeOutcome:
    """미배정 얼굴들을 재판정한다 — 결정적: 같은 (event, clusters, 캐시)면 같은 결과.

    한 얼굴의 top-k 쌍 점수가 전부 확보돼야 판정한다(호출 상한·crop 실패·일시 장애로 일부만
    확보되면 그 얼굴은 판정 보류) — 부분 점수의 argmax는 미측정 앨범이 진짜 정답일 때
    오답 편입이 되기 때문. 확보된 점수는 전부 scores에 남아 다음 재군집에서 이어 판정한다.
    """
    config = self._config
    scores = dict(cached_scores)
    calls_made = 0
    truncated = False
    crop_cache: dict[int, bytes | None] = {}

    # 앨범 대표 = 멤버 중 폭 최대 얼굴(bbox·원본 키 유효 행만) — 실측(2026-07-23 리뷰)과 파리티.
    # 썸네일 대표(_select_representative의 membership 최대)와 의도적으로 다르다: 임계의 근거
    # 데이터가 폭 최대 대표로 측정됐다. 동률은 가장 앞 행(결정성).
    reps: list[tuple["PersonCluster", int]] = []
    for cluster in clusters:
      eligible = [i for i in cluster.member_indices if event.bboxes[i][2] > 0 and event.s3_keys[i]]
      if eligible:
        reps.append((cluster, max(eligible, key=lambda i: (event.face_widths[i], -i))))
    if not reps or not uncertain_indices:
      return RejudgeOutcome(scores=scores)

    # 사용자 cannot-link 검사 준비: must-link 폐포(union-find) 위에서 "이 얼굴을 이 앨범에 붙이면
    # 사용자 cannot-link가 깨지는가"를 본다 — 깨질 편입은 2차 패스에서 recluster가 모순으로 거부
    # 하거나 앨범을 쪼개므로, 애초에 만들지 않는다 (사용자 결정 존중 — 제안으로 강등하지 않고 탈락).
    links = _FaceUnionFind()
    for a, b in event.must_link_pairs:
      links.union(a, b)
    member_roots = {
      cluster.cluster_id: {links.find(event.face_ids[i]) for i in cluster.member_indices} for cluster, _ in reps
    }
    cannot_roots = [(links.find(a), links.find(b)) for a, b in event.cannot_link_pairs]

    raw_assignments: list[tuple[str, RejudgeAssignment]] = []  # (photo_id, 편입) — argmax 가드 입력
    suggestions: list[RejudgeSuggestion] = []
    fragment_hints: list[FragmentHint] = []

    for row in sorted(uncertain_indices):
      if event.bboxes[row][2] <= 0 or not event.s3_keys[row]:
        continue  # crop 원천 미상(v2 이하 .npz 행) — 재판정 불가
      embedding = event.embeddings[row]
      # AuraFace 유사도 상위 top_k 앨범 — 동률은 cluster_id 순(결정성)
      ranked = sorted(reps, key=lambda entry: (-float(embedding @ entry[0].centroid), entry[0].cluster_id))
      candidates = ranked[: config.top_k]

      face_scores: list[tuple[float, "PersonCluster", int]] = []
      complete = True
      for cluster, rep_row in candidates:
        pair = (event.face_ids[row], event.face_ids[rep_row])
        if pair not in scores:
          if calls_made >= config.max_calls:
            truncated = True
            complete = False
            continue
          source = self._crop_jpeg(event, row, crop_cache)
          target = self._crop_jpeg(event, rep_row, crop_cache)
          if source is None or target is None:
            complete = False
            continue
          try:
            calls_made += 1
            scores[pair] = float(self._compare(source, target))
          except Exception:
            # 일시 장애(스로틀·네트워크) — 캐싱하지 않고 건너뛴다 (다음 재군집에서 재시도)
            logger.warning("CompareFaces 호출 실패 pair=%s — 이 쌍 건너뜀", pair, exc_info=True)
            complete = False
            continue
        face_scores.append((scores[pair], cluster, rep_row))

      if not complete:
        continue  # top-k 전체 점수 없이는 판정 보류 — 확보분은 캐시에 남아 다음 job에서 이어간다

      # 판정은 전 후보 점수의 argmax — 동률은 AuraFace 순위 우선(max가 첫 최대값을 고른다)
      best_similarity, best_cluster, best_rep = max(face_scores, key=lambda entry: entry[0])
      over_fragment = sorted(
        (entry for entry in face_scores if entry[0] >= config.fragment_similarity), key=lambda e: -e[0]
      )
      if len(over_fragment) >= 2:
        hint = FragmentHint(
          face_id=event.face_ids[row],
          cluster_ids=tuple(entry[1].cluster_id for entry in over_fragment),
          similarities=tuple(entry[0] for entry in over_fragment),
        )
        fragment_hints.append(hint)
        logger.info(
          "파편 앨범 힌트: face_id=%s가 복수 앨범 동시 고점 매칭 %s — 같은 인물의 파편 앨범 가능성",
          hint.face_id,
          list(zip(hint.cluster_ids, hint.similarities)),
        )

      if best_similarity >= config.auto_assign_similarity:
        if self._passes_screens(event, row, best_cluster, links, member_roots, cannot_roots):
          raw_assignments.append(
            (
              event.photo_ids[row],
              RejudgeAssignment(
                face_id=event.face_ids[row],
                cluster_id=best_cluster.cluster_id,
                rep_face_id=event.face_ids[best_rep],
                similarity=best_similarity,
              ),
            )
          )
      elif best_similarity >= config.suggest_similarity:
        if self._passes_screens(event, row, best_cluster, links, member_roots, cannot_roots):
          suggestions.append(
            RejudgeSuggestion(
              face_id=event.face_ids[row],
              cluster_id=best_cluster.cluster_id,
              rep_face_id=event.face_ids[best_rep],
              similarity=best_similarity,
            )
          )

    # 사진·앨범당 argmax 가드: 같은 사진의 두 얼굴이 같은 앨범에 동시 편입되면 두 must-link가 한
    # 컴포넌트로 합쳐지고, 그 사이 같은사진 자동 cannot-link는 사람 결정 우선 규칙으로 탈락해
    # (cluster.Constraints 독스트링) 서로 다른 두 사람이 조용히 융합된다 — 최고점만 남긴다.
    best_by_key: dict[tuple[str, str], RejudgeAssignment] = {}
    for photo_id, assignment in raw_assignments:
      key = (photo_id, assignment.cluster_id)
      incumbent = best_by_key.get(key)
      if incumbent is None or assignment.similarity > incumbent.similarity:
        if incumbent is not None:
          logger.info("같은 사진·앨범 중복 편입 — 최고점만 유지: 탈락 face_id=%s", incumbent.face_id)
        best_by_key[key] = assignment
      else:
        logger.info("같은 사진·앨범 중복 편입 — 최고점만 유지: 탈락 face_id=%s", assignment.face_id)

    assignments = tuple(best_by_key.values())
    if assignments or suggestions or fragment_hints or calls_made:
      logger.info(
        "Rekognition 재판정: 편입 %d건, 제안 %d건, 파편 힌트 %d건 (호출 %d회%s)",
        len(assignments),
        len(suggestions),
        len(fragment_hints),
        calls_made,
        ", 호출 상한 도달로 일부 보류" if truncated else "",
      )
    return RejudgeOutcome(
      assignments=assignments,
      suggestions=tuple(suggestions),
      fragment_hints=tuple(fragment_hints),
      scores=scores,
      calls_made=calls_made,
      truncated=truncated,
    )

  def _crop_jpeg(self, event: "EventEmbeddings", row: int, cache: dict[int, "bytes | None"]) -> "bytes | None":
    """행의 얼굴 crop JPEG를 만든다 — job 내 행 단위 캐시. 원본은 필요 시 1장씩 fetch·폐기한다.

    디코딩 이미지를 쥐고 있지 않는 것은 공유 호스트(t4g.small) 메모리 규율이다 (_make_thumbnail과
    동일). crop bytes(수십 KB)만 캐시한다. 실패(원본 소실·퇴화 crop)는 None — 해당 쌍만 보류된다.
    """
    if row in cache:
      return cache[row]
    result: bytes | None = None
    try:
      image = self._fetch_image(event.s3_keys[row])
      result = self._crop(image, event.bboxes[row])
    except Exception:
      logger.warning("재판정 crop 실패 s3_key=%s — 해당 쌍 보류", event.s3_keys[row], exc_info=True)
    cache[row] = result
    return result

  def _passes_screens(
    self,
    event: "EventEmbeddings",
    row: int,
    cluster: "PersonCluster",
    links: _FaceUnionFind,
    member_roots: dict[str, set[str]],
    cannot_roots: list[tuple[str, str]],
  ) -> bool:
    """편입·제안 사전 검사 2종 — 실패는 강등 없이 탈락(로그)한다."""
    # ① 사용자 cannot-link 존중: 이 얼굴(의 must-link 컴포넌트)이 대상 앨범 멤버(의 컴포넌트)와
    # cannot-link로 묶여 있으면, 사용자가 "다른 사람"이라 결정한 앨범이다 — 점수와 무관하게 탈락.
    face_root = links.find(event.face_ids[row])
    roots = member_roots[cluster.cluster_id]
    for root_a, root_b in cannot_roots:
      if (root_a == face_root and root_b in roots) or (root_b == face_root and root_a in roots):
        logger.info(
          "재판정 탈락(사용자 cannot-link): face_id=%s → cluster_id=%s", event.face_ids[row], cluster.cluster_id
        )
        return False
    # ② 같은 사진 검사: 대상 앨범에 이 얼굴과 같은 사진의 (파편이 아닌) 얼굴이 있으면, 같은 사진에
    # 같은 인물이 두 번 나올 수 없으므로 오판이다 — 편입하면 2차 패스의 같은사진 자동 cannot-link가
    # 앨범을 쪼갠다. 이중 검출 파편(유사도 ≥ same_face_similarity)은 타인 증거가 아니라 예외.
    for member in cluster.member_indices:
      if event.photo_ids[member] != event.photo_ids[row]:
        continue
      if float(event.embeddings[row] @ event.embeddings[member]) < self._config.same_face_similarity:
        logger.info("재판정 탈락(같은 사진 공존): face_id=%s → cluster_id=%s", event.face_ids[row], cluster.cluster_id)
        return False
    return True


if __name__ == "__main__":
  # AWS·cv2 모델 없이 판정 로직 전체를 자가 검증한다 — 페이크 comparer/fetch/crop + 합성 event.
  # TODO(CHMO-165): pytest 도입 시 tests/pipeline/test_rejudge.py로 승격
  from app.pipeline.cluster import PersonCluster
  from app.storage.event_embeddings import EMBED_DIM, EventEmbeddings

  logging.basicConfig(level="ERROR")  # 의도된 실패 케이스의 WARNING 로그 숨김
  passed = 0

  def check(name: str, condition: bool) -> None:
    global passed
    if not condition:
      raise SystemExit(f"실패: {name}")
    passed += 1
    print(f"통과: {name}")

  def axis_vector(axis: int, scale: float = 1.0) -> np.ndarray:
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[axis] = 1.0
    return vector

  # 행 구성: 0·1 = 앨범 A(인물 축 0), 2·3 = 앨범 B(인물 축 2), 4~10 = 미배정(각자 직교 축), 11 = bbox 미상
  rows = [
    ("fa1", "pa1", axis_vector(0), 100.0, (0.0, 0.0, 100.0, 100.0), "s3-0"),
    ("fa2", "pa2", axis_vector(0), 120.0, (0.0, 0.0, 120.0, 120.0), "s3-1"),  # A 대표 (폭 최대)
    ("fb1", "pb1", axis_vector(2), 100.0, (0.0, 0.0, 100.0, 100.0), "s3-2"),  # B 대표 (폭 최대)
    ("fb2", "pb2", axis_vector(2), 90.0, (0.0, 0.0, 90.0, 90.0), "s3-3"),
    ("fu-join", "pu1", axis_vector(10), 80.0, (0.0, 0.0, 80.0, 80.0), "s3-4"),  # 99점 → A 편입
    ("fu-sugg", "pu2", axis_vector(12), 80.0, (0.0, 0.0, 80.0, 80.0), "s3-5"),  # 88점 → A 제안
    ("fu-cl", "pu3", axis_vector(14), 80.0, (0.0, 0.0, 80.0, 80.0), "s3-6"),  # 99점이지만 cannot-link 탈락
    ("fu-same", "pa1", axis_vector(16), 80.0, (0.0, 0.0, 80.0, 80.0), "s3-7"),  # 99점이지만 같은사진(pa1) 탈락
    ("fu-g1", "pu5", axis_vector(18), 80.0, (0.0, 0.0, 80.0, 80.0), "s3-8"),  # 같은 사진 쌍 — 92점 (탈락)
    ("fu-g2", "pu5", axis_vector(20), 80.0, (0.0, 0.0, 80.0, 80.0), "s3-9"),  # 같은 사진 쌍 — 94점 (유지)
    ("fu-frag", "pu6", axis_vector(22), 80.0, (0.0, 0.0, 80.0, 80.0), "s3-10"),  # A 96·B 97 → 힌트 + B 편입
    ("fu-nobox", "pu7", axis_vector(24), 0.0, (0.0, 0.0, 0.0, 0.0), ""),  # crop 원천 미상 — 건너뜀
  ]
  event = EventEmbeddings.empty().append_faces(rows)
  event = event.with_cluster_ids(["A", "A", "B", "B", None, None, None, None, None, None, None, None])
  event = event.with_constraints((), (("fu-cl", "fa1"),))  # 사용자 cannot-link: fu-cl ↔ 앨범 A 멤버

  def cluster_of(cluster_id: str, member_indices: tuple[int, ...], axis: int) -> PersonCluster:
    return PersonCluster(
      cluster_id=cluster_id,
      is_new=False,
      member_indices=member_indices,
      membership_similarities=(1.0,) * len(member_indices),
      centroid=axis_vector(axis),
    )

  clusters = (cluster_of("A", (0, 1), 0), cluster_of("B", (2, 3), 2))
  uncertain = (4, 5, 6, 7, 8, 9, 10, 11)

  SCORES = {
    ("s3-4", "s3-1"): 99.0,
    ("s3-4", "s3-2"): 5.0,
    ("s3-5", "s3-1"): 88.0,
    ("s3-5", "s3-2"): -1.0,  # 얼굴 미검출 센티널 — 캐싱 확인용
    ("s3-6", "s3-1"): 99.0,
    ("s3-6", "s3-2"): 5.0,
    ("s3-7", "s3-1"): 99.0,
    ("s3-7", "s3-2"): 5.0,
    ("s3-8", "s3-2"): 92.0,
    ("s3-8", "s3-1"): 5.0,
    ("s3-9", "s3-2"): 94.0,
    ("s3-9", "s3-1"): 5.0,
    ("s3-10", "s3-1"): 96.0,
    ("s3-10", "s3-2"): 97.0,
  }
  calls: list[tuple[str, str]] = []
  raises: set[tuple[str, str]] = set()

  def fake_compare(source: bytes, target: bytes) -> float:
    key = (source.decode(), target.decode())
    calls.append(key)
    if key in raises:
      raise RuntimeError("throttled")
    return SCORES[key]

  def build(config: RejudgeConfig | None = None) -> UncertainRejudger:
    # fetch가 s3_key를 그대로 태그로 넘기고 crop이 bytes로 되돌린다 — comparer가 쌍을 식별하는 배선
    return UncertainRejudger(
      compare=fake_compare,
      fetch_image=lambda key: np.frombuffer(key.encode(), dtype=np.uint8),
      crop=lambda image, bbox: bytes(image),
      config=config,
    )

  outcome = build().rejudge(event, clusters, uncertain, {})
  check(
    "자동 편입: 99점 얼굴이 앨범 A 대표와의 must-link 쌍으로 판정",
    any(a == RejudgeAssignment("fu-join", "A", "fa2", 99.0) for a in outcome.assignments),
  )
  check(
    "제안 대역 [85, 90): 88점은 편입이 아니라 제안 (대표 face_id 동봉 — 최종 앨범 재해석용)",
    outcome.suggestions == (RejudgeSuggestion("fu-sugg", "A", "fa2", 88.0),)
    and all(a.face_id != "fu-sugg" for a in outcome.assignments),
  )
  check(
    "사용자 cannot-link 탈락: 99점이어도 편입·제안 없음 (강등 아님)",
    all(a.face_id != "fu-cl" for a in outcome.assignments) and all(s.face_id != "fu-cl" for s in outcome.suggestions),
  )
  check(
    "같은 사진 공존 탈락: 앨범에 같은 사진 얼굴이 있으면 99점이어도 편입 없음",
    all(a.face_id != "fu-same" for a in outcome.assignments),
  )
  check(
    "사진·앨범당 argmax 가드: 같은 사진 두 얼굴 중 최고점(94)만 편입",
    any(a == RejudgeAssignment("fu-g2", "B", "fb1", 94.0) for a in outcome.assignments)
    and all(a.face_id != "fu-g1" for a in outcome.assignments),
  )
  check(
    "파편 힌트: 복수 앨범 동시 ≥95 → 힌트 기록 + argmax(B) 편입은 그대로",
    outcome.fragment_hints == (FragmentHint("fu-frag", ("B", "A"), (97.0, 96.0)),)
    and any(a == RejudgeAssignment("fu-frag", "B", "fb1", 97.0) for a in outcome.assignments),
  )
  check("crop 원천 미상 행은 건너뜀 (호출 없음)", not any(k[0] == "" for k in calls))
  check(
    "-1.0 센티널 캐싱: 얼굴 미검출도 점수로 남아 재과금이 없다",
    outcome.scores[("fu-sugg", "fb1")] == -1.0,
  )
  check("편입 3건·호출 14회 정산", len(outcome.assignments) == 3 and outcome.calls_made == len(calls) == 14)

  # 캐시 재사용: 병합본을 그대로 넘기면 호출 0회로 같은 판정
  calls.clear()
  cached_outcome = build().rejudge(event, clusters, uncertain, outcome.scores)
  check(
    "캐시 전량 히트: 호출 0회 + 판정 동일",
    calls == []
    and cached_outcome.calls_made == 0
    and cached_outcome.assignments == outcome.assignments
    and cached_outcome.suggestions == outcome.suggestions,
  )

  # 호출 상한: 1회 예산이면 첫 쌍만 측정하고 그 얼굴은 판정 보류 (부분 점수 argmax 금지)
  calls.clear()
  capped = build(RejudgeConfig(max_calls=1)).rejudge(event, clusters, uncertain, {})
  check(
    "호출 상한: 1회만 호출, truncated 표시, 편입·제안 없음 (부분 점수로 판정하지 않음)",
    len(calls) == 1 and capped.truncated and capped.assignments == () and capped.suggestions == (),
  )
  check("상한 절단분도 측정분은 캐시에 남는다", len(capped.scores) == 1)

  # 일시 장애: 해당 쌍만 건너뛰고(캐싱 없음) 그 얼굴은 판정 보류 — 다른 얼굴 판정은 정상
  calls.clear()
  raises.add(("s3-4", "s3-1"))
  degraded = build().rejudge(event, clusters, uncertain, {})
  check(
    "일시 장애 쌍: 그 얼굴만 판정 보류(미캐싱), 나머지 판정 유지",
    all(a.face_id != "fu-join" for a in degraded.assignments)
    and ("fu-join", "fa2") not in degraded.scores
    and any(a.face_id == "fu-g2" for a in degraded.assignments),
  )
  raises.clear()

  # 대표 없는 앨범(전 행 bbox 미상)·미배정 없음 — 빈 결과
  empty = build().rejudge(event, (), (4,), {})
  check("앨범 대표 부재: 빈 결과 (호출 없음)", empty.assignments == () and empty.calls_made == 0)

  # RejudgeConfig 검증
  for name, kwargs in [
    ("suggest > auto", {"suggest_similarity": 95.0, "auto_assign_similarity": 90.0}),
    ("top_k 0", {"top_k": 0}),
    ("max_calls 0", {"max_calls": 0}),
    ("same_face 0", {"same_face_similarity": 0.0}),
    ("fragment 0", {"fragment_similarity": 0.0}),
  ]:
    try:
      RejudgeConfig(**kwargs)
    except ValueError:
      check(f"거부: RejudgeConfig {name}", True)
    else:
      raise SystemExit(f"실패: RejudgeConfig {name} — ValueError가 발생해야 하는데 통과됨")

  # render_rejudge_crop: 실 cv2 경로 — 실측 crop 파리티 규칙 (0.25 여백 + 클램프 + 퇴화 None)
  canvas = np.zeros((100, 100, 3), dtype=np.uint8)
  jpeg = render_rejudge_crop(canvas, (10.0, 10.0, 40.0, 40.0))
  check("crop 렌더: 정상 bbox → JPEG bytes", isinstance(jpeg, bytes) and jpeg[:2] == b"\xff\xd8")
  check("crop 렌더: 경계 밖 bbox는 클램프", isinstance(render_rejudge_crop(canvas, (-5.0, -5.0, 20.0, 20.0)), bytes))
  check("crop 렌더: 퇴화 bbox(w=0) → None", render_rejudge_crop(canvas, (10.0, 10.0, 0.0, 40.0)) is None)
  check("crop 렌더: 이미지 밖 bbox → None", render_rejudge_crop(canvas, (200.0, 200.0, 40.0, 40.0)) is None)
  try:
    render_rejudge_crop(canvas, (0.0, 0.0, 10.0, 10.0), margin_ratio=-0.1)
  except ValueError:
    check("거부: margin_ratio 음수", True)
  else:
    raise SystemExit("실패: margin_ratio 음수 — ValueError가 발생해야 하는데 통과됨")

  print(f"\n스모크 검증 {passed}건 전부 통과")
