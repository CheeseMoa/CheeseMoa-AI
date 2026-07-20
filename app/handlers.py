"""인바운드 메시지 3종(classify/feedback/delete)의 처리 로직 — 워커 계층의 두뇌.

detect/embed(onnxruntime·cv2 모델 임포트 체인)를 직접 알지 않는다: 얼굴 추출은
`FaceExtractor` 콜러블로 주입받고 합성은 core.deps의 책임이다 — 덕분에 이 모듈의 스모크는
모델 다운로드 없이 돈다. 저장소·이미지 소스도 Protocol로만 알아 페이크 주입이 가능하다.

계약 (ADR-007 재군집 흐름):
  - 핸들러는 store.save(event .npz rewrite)까지 마친 ClassifyResult를 반환한다.
    결과 발행(publish)과 SQS 메시지 삭제는 worker.py의 몫이다 — 저장 → 발행 → 삭제 순서가
    photo_id 멱등 append와 합쳐져 재전달에 안전한 at-least-once를 만든다.
  - image_id ↔ face_id ↔ 행 인덱스 번역, 보정 메시지 → 제약 변환, 보정 간 충돌의
    시간순 해소(later-wins)는 전부 여기서 끝낸다 — recluster에는 일관된 제약 셋만 전달한다
    (cluster.Constraints 독스트링의 요구).
"""

import logging
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import numpy as np

from app.pipeline.cluster import ClusterConfig, Constraints, PersonCluster, recluster
from app.schemas.messages import (
  UNCERTAIN_ALBUM_ID,
  ClassifyRequest,
  ClassifyResult,
  ConfirmDistinctFeedback,
  DeleteRequest,
  FailedImage,
  MergeFeedback,
  ReassignFeedback,
  ResultCluster,
  SplitFeedback,
  UncertainImage,
)
from app.storage.embedding_store import EmbeddingStore
from app.storage.event_embeddings import EventEmbeddings
from app.storage.image_source import ImageSource

logger = logging.getLogger(__name__)


class ExtractedFaces(NamedTuple):
  """한 이미지에서 추출한 결과 — 얼굴 임베딩 + 이미지 단위 품질 원시 판정.

  품질 플래그는 토글 적용 전의 지각(perception) 결과다: 토글(exclude_eyes_closed/blurry)을
  반영해 실제 라우팅을 정하는 정책은 핸들러의 몫 (관심사 분리).
  """

  embeddings: list[np.ndarray]  # 얼굴별 L2 정규화 (512,) float32. 빈 목록 = 얼굴 미검출/전부 퇴화
  eyes_closed: bool = False  # 얼굴 1개라도 양눈 감김 (quality.judge_faces 규칙)
  blurry: bool = False  # 얼굴 1개라도 Laplacian variance < 임계
  face_widths: list[float] | None = None  # embeddings와 같은 순서의 bbox 폭 px — 주 인물 판정용. None = 미상


# 디코딩된 BGR 이미지 → ExtractedFaces. embeddings 빈 목록이면 common_album 라우팅 (feature-spec §6.2).
# detect→align→embed + 품질 판정 합성은 core.deps.build_face_extractor가 만든다.
FaceExtractor = Callable[[np.ndarray], ExtractedFaces]

# classify 처리 중 진행률 보고 콜백 (job_id, event_id, processed, total) → None (CHMO-274).
# 핸들러는 ProgressUpdate 스키마·SQS를 모른다 — 발행 배선은 core.deps가 클로저로 주입한다 (관심사 분리).
ProgressReporter = Callable[[str, str, int, int], None]

# 진행률 발행 간격 — 처리 이미지 이 장수마다 1회 발행한다(루프 진입 시 0/total, 마지막 total/total은
# 별도로 항상 발행). 진행바를 더 촘촘히/성기게 하려면 이 값을 내리/올린다 (트래픽은 반비례).
_PROGRESS_REPORT_EVERY = 3

InboundParsed = (
  ClassifyRequest | MergeFeedback | SplitFeedback | ReassignFeedback | ConfirmDistinctFeedback | DeleteRequest
)

_FaceIdPair = tuple[str, str]


class _FaceUnionFind:
  """face_id 문자열 위의 union-find — later-wins 제약 조정 전용.

  cluster.py의 인덱스판 _UnionFind는 private이고 "모순이면 raise" 계약이라 재사용하지 않는다 —
  여기서는 "연결되는가"를 질의해 어느 쌍을 버릴지 결정해야 한다.
  """

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

  def connected(self, a: str, b: str) -> bool:
    return self.find(a) == self.find(b)

  def would_connect(self, a: str, b: str, x: str, y: str) -> bool:
    """union(a, b)를 수행하면 (수행 전 미연결이던) x-y가 새로 연결되는가."""
    root_x, root_y = self.find(x), self.find(y)
    return root_x != root_y and {root_x, root_y} == {self.find(a), self.find(b)}


def _normalize_pair(pair: _FaceIdPair) -> _FaceIdPair:
  a, b = pair
  return (a, b) if a <= b else (b, a)


def _dedupe_pairs(pairs: Iterable[_FaceIdPair]) -> tuple[_FaceIdPair, ...]:
  """쌍을 (작은 id, 큰 id)로 정규화하고 첫 등장 순서를 보존해 중복을 제거한다."""
  seen: set[_FaceIdPair] = set()
  result: list[_FaceIdPair] = []
  for pair in pairs:
    normalized = _normalize_pair(pair)
    if normalized in seen:
      continue
    seen.add(normalized)
    result.append(normalized)
  return tuple(result)


def _reconcile_constraints(
  old_must: Sequence[_FaceIdPair],
  old_cannot: Sequence[_FaceIdPair],
  new_must: Sequence[_FaceIdPair],
  new_cannot: Sequence[_FaceIdPair],
) -> tuple[tuple[_FaceIdPair, ...], tuple[_FaceIdPair, ...]]:
  """기존 제약과 새 보정을 later-wins(나중 결정 우선)로 조정해 모순 없는 셋을 만든다.

  저장된 셋은 매 보정마다 이 함수를 통과해 항상 자체 일관이므로, 모순은 새 쌍 vs 기존 쌍
  사이에서만 생긴다. 새 쌍은 무조건 수용하고, 기존 쌍은 최신 것부터(배열 뒤부터) 걸러
  나중 결정과 충돌하는 오래된 결정이 먼저 탈락한다. 기존 must/cannot은 종류별 배열이라
  상대 시간순이 없다 — 동률에서는 must(병합 결정)를 먼저 수용한다.
  """
  uf = _FaceUnionFind()
  for a, b in new_must:
    uf.union(a, b)
  for x, y in new_cannot:
    if uf.connected(x, y):
      # 한 메시지의 번역 결과는 구성상 모순이 없어야 한다 — 걸리면 번역 버그이므로 원인 그대로 거부
      raise ValueError(f"번역된 새 제약이 자체 모순입니다: must-link로 연결된 ({x}, {y})에 cannot-link")
  kept_cannot = list(new_cannot)

  surviving_must: list[_FaceIdPair] = []
  for pair in reversed(old_must):
    a, b = pair
    if any(uf.would_connect(a, b, x, y) for x, y in kept_cannot):
      continue  # 이 must를 살리면 더 새로운 cannot이 깨진다 → 오래된 결정 폐기
    uf.union(a, b)
    surviving_must.append(pair)
  surviving_must.reverse()  # 시간순 복원

  surviving_cannot = [pair for pair in reversed(old_cannot) if not uf.connected(*pair)]
  surviving_cannot.reverse()

  return (
    _dedupe_pairs([*surviving_must, *new_must]),
    _dedupe_pairs([*surviving_cannot, *new_cannot]),
  )


# 같은 사진 자동 cannot-link의 이중 검출 안전판 — 같은 사진 두 행의 임베딩이 이 이상 닮으면
# 타인 두 명이 아니라 YuNet이 한 얼굴을 두 박스로 이중 검출한 것으로 보고 제약에서 뺀다
# (잘못된 cannot-link는 같은 사람을 강제로 찢는다). 실 event 실측 같은사진 쌍 최대 0.61,
# 근중복 판정 관례 0.985(ADR-005 burst)와 큰 마진.
_SAME_FACE_SIMILARITY = 0.95


def _same_photo_cannot_links(event: EventEmbeddings) -> tuple[tuple[int, int], ...]:
  """같은 사진에 공존하는 얼굴 행 쌍 = 서로 다른 사람이라는 사실 제약 (ADR-011).

  사람 보정과 달리 저장하지 않고 매 재군집마다 photo_ids에서 유도한다 — 삭제 마스킹·append
  이후에도 항상 현재 행 인덱스와 정합이다. must-link와의 모순 해소(사람 결정 우선)는
  recluster가 자동 쌍을 탈락시키는 방식으로 처리하므로 여기서는 신경 쓰지 않는다.
  """
  rows_by_photo: dict[str, list[int]] = {}
  for row, photo_id in enumerate(event.photo_ids):
    rows_by_photo.setdefault(photo_id, []).append(row)
  pairs = []
  for rows in rows_by_photo.values():
    for a, b in combinations(rows, 2):
      if float(event.embeddings[a] @ event.embeddings[b]) >= _SAME_FACE_SIMILARITY:
        continue  # 이중 검출 의심 — 안전판 (_SAME_FACE_SIMILARITY)
      pairs.append((a, b))
  return tuple(pairs)


@dataclass(frozen=True)
class _ClusterSnapshot:
  """재군집 + 단일 사진 클러스터 강등까지 끝난, 결과 조립에 필요한 event 스냅샷 요약."""

  event: EventEmbeddings  # 저장까지 마친 최종 상태
  clusters: tuple[PersonCluster, ...]  # 인물로 승격된 클러스터만 (강등분 제외)
  unmatched_indices: tuple[int, ...]  # 재군집 노이즈 + 강등된 얼굴
  ambiguous_indices: tuple[int, ...]
  retired_cluster_ids: tuple[str, ...]  # 재군집 은퇴분 + 강등된 기존 id


class JobHandlers:
  """인바운드 메시지 1건 → (event .npz 갱신 저장 + ClassifyResult) 변환기."""

  def __init__(
    self,
    store: EmbeddingStore,
    images: ImageSource,
    extract_faces: FaceExtractor,
    cluster_config: ClusterConfig | None = None,
    new_cluster_id: Callable[[], str] | None = None,
    new_face_id: Callable[[], str] | None = None,
    report_progress: ProgressReporter | None = None,
  ) -> None:
    self._store = store
    self._images = images
    self._extract_faces = extract_faces
    self._cluster_config = cluster_config if cluster_config is not None else ClusterConfig()
    # 기본 uuid4 — 스모크/테스트에서 결정적 팩토리를 주입한다 (recluster.new_id_factory와 같은 패턴)
    self._new_cluster_id = new_cluster_id if new_cluster_id is not None else (lambda: str(uuid.uuid4()))
    self._new_face_id = new_face_id if new_face_id is not None else (lambda: str(uuid.uuid4()))
    # 기본 no-op — progress 큐 미설정 시(또는 스모크) 진행률 발행이 없어도 동작이 같다 (CHMO-274)
    self._report_progress = report_progress if report_progress is not None else (lambda *_args: None)

  def handle(self, message: InboundParsed) -> ClassifyResult:
    """메시지 타입별 핸들러로 분기한다. 예외는 그대로 전파 — 재시도/DLQ 정책은 worker의 몫."""
    match message:
      case ClassifyRequest():
        return self._handle_classify(message)
      case MergeFeedback() | SplitFeedback() | ReassignFeedback() | ConfirmDistinctFeedback():
        return self._handle_feedback(message)
      case DeleteRequest():
        return self._handle_delete(message)
    raise TypeError(f"처리할 수 없는 메시지 타입입니다: {type(message).__name__}")

  # ── classify_request ─────────────────────────────────────────────────────

  def _emit_progress(self, job_id: str, event_id: str, processed: int, total: int) -> None:
    """진행률을 best-effort로 보고한다 (CHMO-274) — 리포터가 던져도 작업을 죽이지 않는다.

    발행 계층(SqsProgressPublisher)도 자체적으로 예외를 삼키지만, 주입된 리포터가 무엇이든
    (커스텀 콜백 포함) 진행률 보고가 classify 본류를 깨지 못하도록 여기서 한 겹 더 격리한다.
    """
    try:
      self._report_progress(job_id, event_id, processed, total)
    except Exception:
      logger.warning("진행률 보고 실패 job_id=%s %d/%d — 무시하고 진행", job_id, processed, total)

  def _handle_classify(self, request: ClassifyRequest) -> ClassifyResult:
    stored = self._store.load(request.event_id) or EventEmbeddings.empty()
    known_photo_ids = set(stored.photo_ids)

    common_album: list[str] = []
    eyes_closed_images: list[str] = []
    blurry_images: list[str] = []
    failed_images: list[FailedImage] = []
    new_rows: list[tuple[str, str, np.ndarray, float]] = []
    # 진행률 발행 (CHMO-274): 이미지 루프가 job 비용의 사실상 전부라 여기서 처리 장수를 흘려보낸다.
    # 루프 진입 전 0/total 1회 — 백엔드가 즉시 QUEUED→PROCESSING으로 바를 띄우게 한다.
    total = len(request.images)
    step = _PROGRESS_REPORT_EVERY
    self._emit_progress(request.job_id, request.event_id, 0, total)
    for index, ref in enumerate(request.images):
      processed = index + 1
      if processed % step == 0 or processed == total:
        self._emit_progress(request.job_id, request.event_id, processed, total)
      if ref.image_id in known_photo_ids:
        continue  # 재전달/중복 요청 멱등: 이미 임베딩된 사진은 건너뛴다 (ADR-007 재군집 흐름 3단계)
      try:
        image = self._images.fetch(ref.s3_key)
        extracted = self._extract_faces(image)
      except Exception as exc:
        # 이미지 단위 격리 (feature-spec §9) — 한 장의 실패가 작업 전체를 죽이지 않는다
        logger.warning("이미지 처리 실패 image_id=%s s3_key=%s", ref.image_id, ref.s3_key, exc_info=exc)
        failed_images.append(FailedImage(image_id=ref.image_id, reason=type(exc).__name__))
        continue
      # 품질 게이트 (feature-spec §6.1·§6.2·§7): 토글 ON이고 해당 사진이면 인물 앨범 대신 품질 앨범으로
      # 분리하고 재군집에서 제외한다(new_rows 미추가) → clusters/common/uncertain과 자동 상호배타.
      # 흔들림을 가장 먼저 본다: 얼굴 미검출이어도(완전 흔들려 검출 실패) extractor가 전체 이미지 흔들림을
      # 반영하므로, common 분기보다 앞서야 흔들린 사진이 공용앨범이 아니라 흔들림 앨범으로 라우팅된다.
      # 또한 blur면 눈 상태 판정을 신뢰할 수 없어 eyes_closed에도 앞선다.
      # 품질 앨범은 이번 요청분만이다 (.npz에 품질 컬럼 미저장 — 얼굴 미검출 common과 같은 request-scoped 비대칭).
      if request.options.exclude_blurry and extracted.blurry:
        blurry_images.append(ref.image_id)
        continue
      if not extracted.embeddings:
        # 얼굴 미검출 + 흔들리지 않음(단체/배경 사진) → 공용 앨범 (feature-spec §6.2 확정: uncertain이 아니다).
        # .npz에는 행이 없으므로 재전달 시 재추출된다 — 결과가 같아 멱등이다.
        common_album.append(ref.image_id)
        continue
      if request.options.exclude_eyes_closed and extracted.eyes_closed:
        eyes_closed_images.append(ref.image_id)
        continue
      widths = extracted.face_widths or [0.0] * len(extracted.embeddings)
      new_rows.extend(
        (self._new_face_id(), ref.image_id, embedding, width) for embedding, width in zip(extracted.embeddings, widths)
      )

    event = stored.append_faces(new_rows)
    snapshot = self._recluster_and_save(request.event_id, event)
    return self._assemble_result(
      request.job_id,
      snapshot,
      common_album=common_album,
      failed_images=failed_images,
      eyes_closed=eyes_closed_images,
      blurry=blurry_images,
    )

  # ── cluster_feedback (merge / split / reassign) ──────────────────────────

  def _handle_feedback(
    self, message: MergeFeedback | SplitFeedback | ReassignFeedback | ConfirmDistinctFeedback
  ) -> ClassifyResult:
    stored = self._store.load(message.event_id)
    if stored is None:
      # 보정 대상 event가 저장된 적 없음 — 재시도가 파일을 만들어주지 않는 결정적 이상이므로
      # failed를 반환해 (worker가 발행 후 삭제) FIFO 그룹이 막히지 않게 한다
      logger.error("보정 대상 event가 없습니다. event_id=%s job_id=%s", message.event_id, message.job_id)
      return ClassifyResult(job_id=message.job_id, status="failed")

    new_must, new_cannot, superseded_faces = self._translate_feedback(message, stored)
    # reassign 이동 얼굴의 기존 must-link는 전부 폐기한다: 체인 구조상 어느 쌍이 "낡은 결정"인지
    # 특정할 수 없어 later-wins에 맡기면 이동 얼굴이 옛 그룹 동료를 새 인물로 끌고 갈 수 있다.
    # 기존 cannot-link는 남긴다 — 새 must와 충돌하면 later-wins가 알아서 폐기한다.
    old_must = [
      pair for pair in stored.must_link_pairs if pair[0] not in superseded_faces and pair[1] not in superseded_faces
    ]
    must, cannot = _reconcile_constraints(old_must, stored.cannot_link_pairs, new_must, new_cannot)
    event = stored.with_constraints(must, cannot)
    snapshot = self._recluster_and_save(message.event_id, event)
    return self._assemble_result(message.job_id, snapshot)

  def _translate_feedback(
    self,
    message: MergeFeedback | SplitFeedback | ReassignFeedback | ConfirmDistinctFeedback,
    stored: EventEmbeddings,
  ) -> tuple[list[_FaceIdPair], list[_FaceIdPair], set[str]]:
    """보정 메시지(image_id·cluster_id 기준)를 face_id 제약 쌍으로 번역한다.

    반환: (새 must 쌍, 새 cannot 쌍, 기존 must-link가 무효화되는 얼굴 — reassign 이동 얼굴만 해당).
    스테일 참조(이미 사라진 cluster_id·사진)는 경고 후 건너뛴다 — 빈 번역이어도 재군집·결과
    발행은 진행해 Spring이 현재 상태 스냅샷을 받게 한다. 대표 얼굴은 행 순서상 첫 얼굴로
    고정해 결정성을 지킨다.
    """
    faces_by_cluster: dict[str, list[str]] = {}
    for face_id, cluster_id in zip(stored.face_ids, stored.cluster_ids):
      if cluster_id is not None:
        faces_by_cluster.setdefault(cluster_id, []).append(face_id)

    match message:
      case MergeFeedback():
        payload = message.merge
        cluster_ids = [payload.target_cluster_id, *payload.source_cluster_ids]
        stale = [cluster_id for cluster_id in cluster_ids if cluster_id not in faces_by_cluster]
        if stale:
          logger.warning("merge 보정의 stale cluster_id 무시: %s (job_id=%s)", stale, message.job_id)
        members = [face_id for cluster_id in cluster_ids for face_id in faces_by_cluster.get(cluster_id, [])]
        if len(members) < 2:
          logger.warning("merge 보정으로 연결할 얼굴이 2개 미만 — 무시 (job_id=%s)", message.job_id)
          return [], [], set()
        # 체인 (f0,f1),(f1,f2),… — must-link 전이성으로 전원이 한 컴포넌트가 된다
        return list(zip(members, members[1:])), [], set()

      case SplitFeedback():
        payload = message.split
        groups_faces: list[list[str]] = []
        for group in payload.groups:
          group_set = set(group)
          faces = [
            face_id
            for face_id, photo_id, cluster_id in zip(stored.face_ids, stored.photo_ids, stored.cluster_ids)
            if cluster_id == payload.cluster_id and photo_id in group_set
          ]
          if faces:
            groups_faces.append(faces)
          else:
            logger.warning("split 보정의 stale 그룹 무시: %s (job_id=%s)", sorted(group_set), message.job_id)
        if len(groups_faces) < 2:
          logger.warning("split 보정의 유효 그룹이 2개 미만 — 무시 (job_id=%s)", message.job_id)
          return [], [], set()
        must = [pair for faces in groups_faces for pair in zip(faces, faces[1:])]
        # 그룹 내부는 must-link로 한 컴포넌트이므로 그룹 쌍마다 대표 간 cannot-link 하나면 충분하다.
        # 그룹을 가로지르는 기존 must-link는 이 대표 간 cannot-link와 충돌해 later-wins가 폐기한다.
        cannot = [
          (groups_faces[i][0], groups_faces[j][0])
          for i in range(len(groups_faces))
          for j in range(i + 1, len(groups_faces))
        ]
        return must, cannot, set()

      case ReassignFeedback():
        payload = message.reassign
        # from_cluster_id가 예약 uncertain 앨범 id면 "미매칭(cluster_id=None) 얼굴"을 대상으로 삼는다.
        # uncertain 얼굴은 실 cluster_id가 없어(.npz엔 None) 일반 reassign(cluster_id 일치)으로는 옮길 수
        # 없기 때문 — 이 가상 앨범을 출처로 인정해 인물 앨범 편입을 가능하게 한다 (feature-spec §6.2·§6.3).
        from_uncertain = payload.from_cluster_id == UNCERTAIN_ALBUM_ID
        moved = [
          face_id
          for face_id, photo_id, cluster_id in zip(stored.face_ids, stored.photo_ids, stored.cluster_ids)
          if photo_id == payload.image_id
          and (cluster_id is None if from_uncertain else cluster_id == payload.from_cluster_id)
        ]
        if not moved:
          logger.warning(
            "reassign 보정 대상 얼굴이 없음 — 무시 (image_id=%s, from=%s, job_id=%s)",
            payload.image_id,
            payload.from_cluster_id,
            message.job_id,
          )
          return [], [], set()
        moved_set = set(moved)
        must: list[_FaceIdPair] = []
        to_faces = faces_by_cluster.get(payload.to_cluster_id, [])
        if to_faces:
          must = [(face_id, to_faces[0]) for face_id in moved]
        else:
          logger.warning("reassign 목적지 클러스터가 비어 있음 — must-link 생략 (to=%s)", payload.to_cluster_id)
        from_remaining = [
          face_id for face_id in faces_by_cluster.get(payload.from_cluster_id, []) if face_id not in moved_set
        ]
        cannot = [(face_id, from_remaining[0]) for face_id in moved] if from_remaining else []
        return must, cannot, moved_set

      case ConfirmDistinctFeedback():
        # merge의 반대 방향: 이미 분리된 클러스터 여러 개를 "서로 다른 사람"으로 확정한다.
        # must-link는 응집만 강제할 뿐 이격은 강제하지 못해, 두 확정 앨범 사이로 유사도가 애매한
        # 신규 사진(다리 사진)이 들어오면 전체 재군집이 둘을 하나로 오병합할 수 있다 — 대표 얼굴
        # 전 쌍(클리크)에 cannot-link를 걸어 향후 어떤 재군집에서도 이 둘이 합쳐지지 않게 한다.
        # 대표 하나만으로 충분한 이유는 _enforce_cannot_link가 위반 라벨을 쪼갠 뒤 제약 없는
        # 나머지 멤버를 최근접 앵커로 재배정하기 때문 — split의 그룹 간 cannot-link와 동일 패턴.
        payload = message.confirm_distinct
        stale = [cluster_id for cluster_id in payload.cluster_ids if cluster_id not in faces_by_cluster]
        if stale:
          logger.warning("confirm_distinct 보정의 stale cluster_id 무시: %s (job_id=%s)", stale, message.job_id)
        representatives = [
          faces_by_cluster[cluster_id][0] for cluster_id in payload.cluster_ids if cluster_id in faces_by_cluster
        ]
        if len(representatives) < 2:
          logger.warning("confirm_distinct 보정으로 분리할 클러스터가 2개 미만 — 무시 (job_id=%s)", message.job_id)
          return [], [], set()
        cannot = [
          (representatives[i], representatives[j])
          for i in range(len(representatives))
          for j in range(i + 1, len(representatives))
        ]
        return [], cannot, set()

    raise TypeError(f"처리할 수 없는 보정 타입입니다: {type(message).__name__}")

  # ── delete_request ───────────────────────────────────────────────────────

  def _handle_delete(self, request: DeleteRequest) -> ClassifyResult:
    stored = self._store.load(request.event_id)
    if stored is None:
      # 저장된 적 없는 event의 삭제 — 지울 것이 없으므로 멱등 no-op 성공
      return ClassifyResult(job_id=request.job_id, status="succeeded")

    masked = stored.masked_by_photo_ids(request.image_ids)
    # 삭제로 행이 전부 사라진 클러스터는 recluster의 previous_cluster_ids에 아예 등장하지 않아
    # retired로 보고되지 않는다 — 여기서 직접 계산해 결과에 합류시킨다
    survivors = {cluster_id for cluster_id in masked.cluster_ids if cluster_id is not None}
    vanished = sorted({cluster_id for cluster_id in stored.cluster_ids if cluster_id is not None} - survivors)

    snapshot = self._recluster_and_save(request.event_id, masked)
    return self._assemble_result(request.job_id, snapshot, extra_retired=vanished)

  # ── 공통 꼬리: 재군집 + .npz rewrite + 결과 조립 ─────────────────────────────

  def _recluster_and_save(self, event_id: str, event: EventEmbeddings) -> _ClusterSnapshot:
    """event 전체를 재군집·강등 판정하고 새 배정을 반영해 저장한다 — 이 저장이 유일한 쓰기이자 마지막 변이다.

    저장 후 크래시(발행·삭제 전)로 메시지가 재전달돼도, photo_id 멱등 append + 결정적 재군집이
    같은 상태에서 같은 결과를 다시 만들므로 안전하다 (오류 매트릭스 참조).
    """
    if not event.face_ids:
      # 전부 삭제된(또는 애초에 빈) event — 재군집할 것이 없어도 빈 파일을 저장해 단일 진실을 유지한다
      self._store.save(event_id, event)
      return _ClusterSnapshot(
        event=event, clusters=(), unmatched_indices=(), ambiguous_indices=(), retired_cluster_ids=()
      )

    row_of = event.row_index_of()
    constraints = Constraints(
      must_link=tuple((row_of[a], row_of[b]) for a, b in event.must_link_pairs),
      cannot_link=tuple((row_of[a], row_of[b]) for a, b in event.cannot_link_pairs),
      auto_cannot_link=_same_photo_cannot_links(event),
    )
    result = recluster(event.embeddings, event.cluster_ids, constraints, self._cluster_config, self._new_cluster_id)

    # 단일 사진 클러스터 강등: 같은 사진에 같은 인물이 두 번 나올 수 없으므로, 한 장의 사진 안
    # 얼굴들로만 구성된 군집은 우연히 닮은 타인들이다(단체 사진에서 재현) — 인물로 승격하지 않고
    # 미매칭으로 되돌린다. 사용자 보정 당사자가 포함된 군집은 사람의 결정이므로 강등하지 않는다.
    constrained_faces = {face_id for pair in event.must_link_pairs + event.cannot_link_pairs for face_id in pair}
    kept: list[PersonCluster] = []
    demoted_indices: list[int] = []
    demoted_retired: list[str] = []
    for cluster in result.clusters:
      photos = {event.photo_ids[index] for index in cluster.member_indices}
      protected = any(event.face_ids[index] in constrained_faces for index in cluster.member_indices)
      if len(photos) >= 2 or protected:
        kept.append(cluster)
        continue
      demoted_indices.extend(cluster.member_indices)
      if not cluster.is_new:
        # 기존에 앨범이 있던 id가 강등되면(예: 삭제로 사진 1장짜리가 된 경우) Spring이 앨범을 정리하게 은퇴 통보
        demoted_retired.append(cluster.cluster_id)

    new_cluster_ids: list[str | None] = [None] * len(event.face_ids)  # 노이즈·ambiguous·강등분은 미배정(None)
    for cluster in kept:
      for index in cluster.member_indices:
        new_cluster_ids[index] = cluster.cluster_id
    saved = event.with_cluster_ids(new_cluster_ids)
    self._store.save(event_id, saved)
    return _ClusterSnapshot(
      event=saved,
      clusters=tuple(kept),
      unmatched_indices=tuple(sorted([*result.noise_indices, *demoted_indices])),
      ambiguous_indices=result.ambiguous_indices,
      retired_cluster_ids=tuple(dict.fromkeys([*result.retired_cluster_ids, *demoted_retired])),
    )

  def _assemble_result(
    self,
    job_id: str,
    snapshot: _ClusterSnapshot,
    *,
    common_album: Sequence[str] = (),
    failed_images: Sequence[FailedImage] = (),
    extra_retired: Sequence[str] = (),
    eyes_closed: Sequence[str] = (),
    blurry: Sequence[str] = (),
  ) -> ClassifyResult:
    """재군집 스냅샷(행 인덱스 세계)을 ClassifyResult(image_id 세계)로 번역한다.

    clusters/uncertain/retired와 미매칭 단체 사진의 common은 event 전체 스냅샷이고,
    얼굴 미검출 common과 failed_images는 이번 요청분만이다 — 미검출 사진은 .npz에 행이 없어
    과거분을 복원할 수 없다 (의도된 비대칭).
    """
    event = snapshot.event
    clusters: list[ResultCluster] = []
    clustered_images: set[str] = set()
    for person in snapshot.clusters:
      # 한 사진에 같은 인물 얼굴이 여러 번 검출돼도 앨범에는 한 번 — 순서 보존 dedupe.
      # 서로 다른 인물이 찍힌 사진은 여러 클러스터에 중복 등장할 수 있다 (N:M, feature-spec §6.2).
      image_ids = list(dict.fromkeys(event.photo_ids[index] for index in person.member_indices))
      clustered_images.update(image_ids)
      clusters.append(
        ResultCluster(
          cluster_id=person.cluster_id,
          is_new=person.is_new,
          image_ids=image_ids,
          representative_vector=[float(value) for value in person.centroid],
        )
      )

    # 미매칭 사진 라우팅 (feature-spec §6.2·§7). 주 인물 얼굴 2명+ 사진은 단체 사진으로 보고 공용 앨범으로,
    # 주 인물 1명(초상·미등록 1인) 미매칭은 uncertain으로 보낸다.
    # 주 인물 판정은 ADR 022와 같은 규칙 — 그 사진 최대 얼굴 폭의 ratio(0.5) 미만은 지나가는 행인으로
    # 보고 세지 않는다: 1인 인물 사진 + 배경 행인이 단체 사진으로 오인돼 공용에 가는 것을 막는다.
    # 폭 미상(0, v1 .npz 행)은 사진 단위로 전원 0이라 최대도 0 → 전원 주 인물 = 종전(전체 얼굴 수)과 동일.
    ratio = self._cluster_config.common_main_face_ratio
    max_width_of: dict[str, float] = {}
    for photo_id, width in zip(event.photo_ids, event.face_widths):
      max_width_of[photo_id] = max(max_width_of.get(photo_id, 0.0), width)
    faces_per_photo = Counter(
      photo_id
      for photo_id, width in zip(event.photo_ids, event.face_widths)
      if ratio <= 0 or width >= max_width_of[photo_id] * ratio
    )
    uncertain: list[UncertainImage] = []
    group_common: list[str] = []
    routed: set[str] = set()
    # 새 정책(group_photo_to_common=True): 얼굴 2명+ 사진은 매칭 여부와 무관하게 공용 앨범에도 노출한다
    # — 인물 앨범과 중복 노출(N:M). 단체 사진은 그 자리에 함께 있던 모두의 사진이라는 제품 결정.
    # event 등장 순서로 안정 정렬. (구 정책은 아래 루프에서 '전원 미매칭'인 2+ 사진만 공용으로 보냈다.)
    if self._cluster_config.group_photo_to_common:
      group_common = [photo_id for photo_id in dict.fromkeys(event.photo_ids) if faces_per_photo[photo_id] >= 2]
    # ambiguous 우선: 한 사진에 ambiguous·unmatched 얼굴이 섞이면 더 정보가 많은 ambiguous로 보고
    for reason, indices in (("ambiguous", snapshot.ambiguous_indices), ("unmatched", snapshot.unmatched_indices)):
      for index in indices:
        photo_id = event.photo_ids[index]
        if photo_id in clustered_images or photo_id in routed:
          continue  # 같은 사진의 다른 얼굴이 인물에 배정됐으면 인물 앨범이 우선한다
        routed.add(photo_id)
        if faces_per_photo[photo_id] >= 2:
          if not self._cluster_config.group_photo_to_common:
            group_common.append(photo_id)  # 구 정책: 전원 미매칭인 단체 사진만 공용
        else:
          uncertain.append(UncertainImage(image_id=photo_id, reason=reason))

    return ClassifyResult(
      job_id=job_id,
      status="partial" if failed_images else "succeeded",
      clusters=clusters,
      common_album=list(dict.fromkeys([*common_album, *group_common])),
      uncertain=uncertain,
      # 품질 게이트로 분리된 사진들 (이번 요청분). 재군집에서 제외됐으므로 clusters/common/uncertain과 겹치지 않는다.
      eyes_closed=list(dict.fromkeys(eyes_closed)),
      blurry=list(dict.fromkeys(blurry)),
      failed_images=list(failed_images),
      retired_cluster_ids=list(dict.fromkeys([*snapshot.retired_cluster_ids, *extra_retired])),
    )


if __name__ == "__main__":
  # AWS·모델 없이 3종 핸들러의 전체 시나리오를 자가 검증한다: 페이크 저장소/이미지소스에
  # "픽셀에 인물 번호를 인코딩한 합성 이미지 → 직교 단위벡터" 추출기를 조합한다.
  # TODO(CHMO-165): pytest 도입 시 tests/test_handlers.py로 승격
  import json
  import math

  from app.schemas.messages import parse_inbound_message
  from app.storage.embedding_store import InMemoryEmbeddingStore
  from app.storage.image_source import InMemoryImageSource
  from app.storage.event_embeddings import EMBED_DIM

  passed = 0

  def check(name: str, condition: bool) -> None:
    global passed
    if not condition:
      raise SystemExit(f"실패: {name}")
    passed += 1
    print(f"통과: {name}")

  def person_vector(person: int, step: int) -> np.ndarray:
    """인물별 평면 위의 단위벡터 — 같은 인물끼리 cos≥0.98, 다른 인물과는 작은 고유 유사도.

    공유 축(마지막 차원)에 인물별로 다른 성분을 실어 교차 유사도가 전부 정확히 0이 되는
    동률을 깬다 — 완전 동률은 HDBSCAN 트리를 퇴화시켜 실데이터에 없는 경로를 태운다.
    """
    theta = math.radians(5.0 * step)
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[2 * person] = math.cos(theta)
    vector[2 * person + 1] = math.sin(theta)
    vector[EMBED_DIM - 1] = 0.1 + 0.02 * person
    return vector / np.linalg.norm(vector)

  SPREAD_COSINES = (0.82, 0.79, 0.76, 0.74, 0.72)

  def spread_person_vector(person: int, step: int) -> np.ndarray:
    """포즈 변화 실사진 대역(동일 인물 쌍 유사도 0.46~0.70)을 모사하는 단위벡터 — 쌍 유사도 = c_i·c_j.

    person_vector(같은 인물 cos≥0.98)와 달리 어느 쌍도 완전 연결 0.7에 못 미쳐, HDBSCAN이 클러스터
    0개를 내는 소규모 단일 인물 이벤트 퇴화를 재현한다 — 연결 성분 부분 승격(ADR-008)만이 앨범을 만든다.
    """
    c = SPREAD_COSINES[step % len(SPREAD_COSINES)]
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[200 + person] = c  # 인물 공유 축 — person_vector의 축 대역과 겹치지 않음
    vector[300 + 8 * (person - 20) + step] = math.sqrt(1.0 - c * c)  # 사진별 고유 직교 축
    return vector

  # confirm_distinct(⑬) 전용 — 같은 2D 부분공간에서 각도로 인물 간 유사도를 정밀 제어한다.
  # 인물 40·41은 0°·62°(cos≈0.47 — 병합 임계 0.55 아래라 사전조건에서 별개 앨범)에 두고, 다리 사진
  # (가상 인물 42)은 이등분각(31°, 각 대표와 cos≈0.86)에 둔다 — 다리가 한쪽에 구제 편입되면 병합 판정이
  # centroid cos≈0.62(≥0.55)·facepair 평균≈0.60(≥0.475)으로 넘어가, confirm_distinct 없이는 두 인물이
  # 실제로 오병합된다(아래 인라인 재검증). 종전 0°·50°는 병합 임계 0.68 시절 기준 — ADR-012 재보정
  # 0.55가 cos 0.64를 삼켜 사전조건이 항상 깨졌다. 현행 보정에 맞게 재배치.
  _CONFIRM_DISTINCT_ANGLES = {40: 0.0, 41: 62.0}
  _CONFIRM_DISTINCT_BRIDGE_ANGLE = 31.0

  def confirm_distinct_vector(person: int, step: int) -> np.ndarray:
    angle = _CONFIRM_DISTINCT_BRIDGE_ANGLE if person == 42 else _CONFIRM_DISTINCT_ANGLES[person]
    theta = math.radians(angle + 0.5 * step)  # step으로 살짝 jitter — 완전 동률 방지
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[400] = math.cos(theta)
    vector[401] = math.sin(theta)
    return vector

  def fake_image(
    faces: list[tuple[int, int]] | list[tuple[int, int, int]], *, eyes_closed: bool = False, blurry: bool = False
  ) -> np.ndarray:
    """(인물 번호, step[, 얼굴 폭 px]) 목록을 픽셀에 인코딩한 합성 BGR 이미지. 품질 플래그는 행 1에 인코딩한다.

    폭 생략 시 100 — 전 얼굴 동일 폭이라 전원 주 인물이 되어 폭 도입 전 시나리오와 동작이 같다.
    """
    image = np.zeros((2, 16, 3), dtype=np.uint8)
    image[0, 0, 0] = len(faces)
    image[1, 0, 0] = int(eyes_closed)  # fake_extractor가 ExtractedFaces 품질 판정으로 되읽는다
    image[1, 0, 1] = int(blurry)
    for slot, face in enumerate(faces):
      image[0, slot + 1, 0] = face[0]
      image[0, slot + 1, 1] = face[1]
      image[0, slot + 1, 2] = face[2] if len(face) > 2 else 100
    return image

  def fake_extractor(image: np.ndarray) -> ExtractedFaces:
    count = int(image[0, 0, 0])
    vectors: list[np.ndarray] = []
    widths: list[float] = []
    for slot in range(count):
      person, step = int(image[0, slot + 1, 0]), int(image[0, slot + 1, 1])
      widths.append(float(image[0, slot + 1, 2]))
      if person >= 40:
        vectors.append(confirm_distinct_vector(person, step))
      elif person >= 20:
        vectors.append(spread_person_vector(person, step))
      else:
        vectors.append(person_vector(person, step))
    return ExtractedFaces(vectors, bool(image[1, 0, 0]), bool(image[1, 0, 1]), widths)

  # 인물 0(A): a1·a2·a3, 인물 1(B): b1·b2, 얼굴 없는 사진, 가져올 수 없는 사진,
  # 미매칭 단체 사진(낯선 2인), 같은 사진 속 닮은 얼굴 쌍(단일 사진 클러스터 강등 대상)
  image_source = InMemoryImageSource(
    {
      "img-a1.jpg": fake_image([(0, 0)]),
      "img-a2.jpg": fake_image([(0, 1)]),
      "img-a3.jpg": fake_image([(0, 2)]),
      "img-b1.jpg": fake_image([(1, 0)]),
      "img-b2.jpg": fake_image([(1, 1)]),
      "img-none.jpg": fake_image([]),
      # 단체 사진: 서로 조금 닮은 타인 2명 (유사도 ≈0.22 — test4.jpg의 낯선 얼굴 수준)
      "img-group.jpg": fake_image([(7, 0), (7, 16)]),
      "img-twins.jpg": fake_image([(6, 0), (6, 1)]),
      # 소규모 단일 인물 이벤트(⑨, event-2)용 — 인물 20은 실측 대역(spread_person_vector) 얼굴
      **{f"img-s{k}.jpg": fake_image([(20, k)]) for k in range(5)},
      # 품질 게이트(⑩)용 — 같은 인물 2(C)의 정상 2장 + 눈감음 1장 + 흔들림 1장
      "img-q-ok1.jpg": fake_image([(2, 0)]),
      "img-q-ok2.jpg": fake_image([(2, 3)]),
      "img-q-eyes.jpg": fake_image([(2, 1)], eyes_closed=True),
      "img-q-blur.jpg": fake_image([(2, 2)], blurry=True),
      # 흔들림 fallback(⑪)용 — 얼굴 미검출 + 흔들림 (완전 흔들려 검출 실패한 사진 모사)
      "img-nf-blur.jpg": fake_image([], blurry=True),
      # uncertain 편입(⑫)용 — 낯선 1인(인물 8) 단일 얼굴 → uncertain(unmatched)
      "img-stranger.jpg": fake_image([(8, 0)]),
      # confirm_distinct(⑬)용 — 확정 인물 40·41 + 둘 사이 다리 사진(인물 42)
      "img-cd-40-1.jpg": fake_image([(40, 0)]),
      "img-cd-40-2.jpg": fake_image([(40, 1)]),
      "img-cd-41-1.jpg": fake_image([(41, 0)]),
      "img-cd-41-2.jpg": fake_image([(41, 1)]),
      "img-cd-bridge.jpg": fake_image([(42, 0)]),
    }
  )
  store = InMemoryEmbeddingStore()

  def counter(prefix: str) -> Callable[[], str]:
    state = {"n": 0}

    def next_id() -> str:
      state["n"] += 1
      return f"{prefix}-{state['n']}"

    return next_id

  handlers = JobHandlers(
    store=store,
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("person"),
    new_face_id=counter("face"),
  )

  def image_ids_of(result: ClassifyResult, *, containing: str) -> set[str]:
    for cluster in result.clusters:
      if containing in cluster.image_ids:
        return set(cluster.image_ids)
    return set()

  classify_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c1",
      "group_id": "group-1",
      "event_id": "event-1",
      "images": [
        {"image_id": "img-a1", "s3_key": "img-a1.jpg"},
        {"image_id": "img-a2", "s3_key": "img-a2.jpg"},
        {"image_id": "img-a3", "s3_key": "img-a3.jpg"},
        {"image_id": "img-b1", "s3_key": "img-b1.jpg"},
        {"image_id": "img-b2", "s3_key": "img-b2.jpg"},
        {"image_id": "img-none", "s3_key": "img-none.jpg"},
        {"image_id": "img-broken", "s3_key": "img-broken.jpg"},
        {"image_id": "img-group", "s3_key": "img-group.jpg"},
        {"image_id": "img-twins", "s3_key": "img-twins.jpg"},
      ],
    }
  )

  # ① 최초 분류: 인물 2명 + 공용 앨범 1장 + 실패 1장 → partial
  result = handlers.handle(parse_inbound_message(classify_body))
  check(
    "classify: 인물 2명, 신규 id",
    len(result.clusters) == 2 and all(cluster.is_new for cluster in result.clusters),
  )
  check(
    "classify: 클러스터 구성",
    image_ids_of(result, containing="img-a1") == {"img-a1", "img-a2", "img-a3"}
    and image_ids_of(result, containing="img-b1") == {"img-b1", "img-b2"},
  )
  check(
    "classify: 공용 앨범(미검출+미매칭 단체)·실패·부분 성공",
    set(result.common_album) == {"img-none", "img-group", "img-twins"}
    and [failed.image_id for failed in result.failed_images] == ["img-broken"]
    and result.status == "partial",
  )
  check(
    "단일 사진 클러스터 강등: 같은 사진 속 닮은 쌍은 인물 승격 안 함, uncertain도 아님",
    all("img-group" not in c.image_ids and "img-twins" not in c.image_ids for c in result.clusters)
    and all(u.image_id not in {"img-group", "img-twins"} for u in result.uncertain),
  )
  twins_assignments = [
    cid for pid, cid in zip(store.load("event-1").photo_ids, store.load("event-1").cluster_ids) if pid == "img-twins"
  ]
  check("강등 얼굴은 .npz에 미배정(None)으로 저장", twins_assignments == [None, None])
  check(
    "classify: 대표벡터는 512차원 단위벡터",
    all(
      len(cluster.representative_vector) == EMBED_DIM
      and abs(float(np.linalg.norm(np.array(cluster.representative_vector))) - 1.0) < 1e-3
      for cluster in result.clusters
    ),
  )
  cluster_a_id = next(c.cluster_id for c in result.clusters if "img-a1" in c.image_ids)
  cluster_b_id = next(c.cluster_id for c in result.clusters if "img-b1" in c.image_ids)
  check("classify: .npz 저장 (행 9개 — 인물 5 + 단체 2 + 강등 쌍 2)", len(store.load("event-1").face_ids) == 9)

  # ② 같은 job 재전달(재시도 모사): 저장된 사진은 스킵되고 결과는 동일해야 한다 (멱등)
  retry = handlers.handle(parse_inbound_message(classify_body))
  check(
    "classify 재시도: 행 수 불변 + 동일 구성 + id 승계",
    len(store.load("event-1").face_ids) == 9
    and image_ids_of(retry, containing="img-a1") == {"img-a1", "img-a2", "img-a3"}
    and {c.cluster_id for c in retry.clusters} == {cluster_a_id, cluster_b_id}
    and not any(c.is_new for c in retry.clusters),
  )

  # ③ merge: B를 A로 병합 → 클러스터 1개, 한 id는 은퇴
  merge_body = json.dumps(
    {
      "type": "cluster_feedback",
      "job_id": "job-f1",
      "event_id": "event-1",
      "action": "merge",
      "merge": {"target_cluster_id": cluster_a_id, "source_cluster_ids": [cluster_b_id]},
    }
  )
  merged = handlers.handle(parse_inbound_message(merge_body))
  check(
    "merge: 클러스터 1개로 병합, 나머지 id 은퇴",
    len(merged.clusters) == 1
    and set(merged.clusters[0].image_ids) == {"img-a1", "img-a2", "img-a3", "img-b1", "img-b2"}
    and {merged.clusters[0].cluster_id, *merged.retired_cluster_ids} == {cluster_a_id, cluster_b_id},
  )
  check("merge: must-link 체인 저장 (4쌍)", len(store.load("event-1").must_link_pairs) == 4)
  merged_id = merged.clusters[0].cluster_id

  # ④ split: 병합을 다시 원래 두 그룹으로 분리 — later-wins가 그룹 간 옛 must-link를 폐기해야 한다
  split_body = json.dumps(
    {
      "type": "cluster_feedback",
      "job_id": "job-f2",
      "event_id": "event-1",
      "action": "split",
      "split": {"cluster_id": merged_id, "groups": [["img-a1", "img-a2", "img-a3"], ["img-b1", "img-b2"]]},
    }
  )
  split_result = handlers.handle(parse_inbound_message(split_body))
  check(
    "split: 두 그룹으로 재분리 (한쪽은 신규 id)",
    len(split_result.clusters) == 2
    and image_ids_of(split_result, containing="img-a1") == {"img-a1", "img-a2", "img-a3"}
    and image_ids_of(split_result, containing="img-b1") == {"img-b1", "img-b2"}
    and sum(1 for cluster in split_result.clusters if cluster.is_new) == 1,
  )
  after_split = store.load("event-1")
  face_of = dict(zip(after_split.photo_ids, after_split.face_ids))
  check(
    "split: later-wins — 그룹을 가로지르는 옛 must-link가 저장소에서 사라짐",
    all(
      {a, b} <= {face_of["img-a1"], face_of["img-a2"], face_of["img-a3"]}
      or {a, b} <= {face_of["img-b1"], face_of["img-b2"]}
      for a, b in after_split.must_link_pairs
    )
    and len(after_split.cannot_link_pairs) == 1,
  )
  cluster_a_id = next(c.cluster_id for c in split_result.clusters if "img-a1" in c.image_ids)
  cluster_b_id = next(c.cluster_id for c in split_result.clusters if "img-b1" in c.image_ids)

  # ⑤ reassign: img-a3를 A→B로 이동 — 기하(임베딩)상 A와 가깝지만 사용자 결정이 이겨야 한다
  reassign_body = json.dumps(
    {
      "type": "cluster_feedback",
      "job_id": "job-f3",
      "event_id": "event-1",
      "action": "reassign",
      "reassign": {"image_id": "img-a3", "from_cluster_id": cluster_a_id, "to_cluster_id": cluster_b_id},
    }
  )
  reassigned = handlers.handle(parse_inbound_message(reassign_body))
  check(
    "reassign: 사용자 결정 강제 — a3는 a1과 분리되고 b1과 묶인다",
    image_ids_of(reassigned, containing="img-a1").isdisjoint({"img-a3"})
    and "img-a3" in image_ids_of(reassigned, containing="img-b1"),
  )

  # ⑥ delete: img-a3(제약 당사자)·img-b2 삭제 → b1 홀로 남아 노이즈, B id 은퇴, a3 제약 프루닝
  delete_body = json.dumps(
    {"type": "delete_request", "job_id": "job-d1", "event_id": "event-1", "image_ids": ["img-a3", "img-b2"]}
  )
  deleted = handlers.handle(parse_inbound_message(delete_body))
  after_delete = store.load("event-1")
  check(
    "delete: 행 제거 + 댕글링 제약 프루닝",
    set(after_delete.photo_ids) == {"img-a1", "img-a2", "img-b1", "img-group", "img-twins"}
    and all(face_of["img-a3"] not in pair for pair in after_delete.must_link_pairs + after_delete.cannot_link_pairs),
  )
  check(
    "delete: 홀로 남은 b1은 unmatched, B id 은퇴",
    [u.image_id for u in deleted.uncertain if u.reason == "unmatched"] == ["img-b1"]
    and cluster_b_id in deleted.retired_cluster_ids,
  )

  # ⑦ 전체 삭제 → 빈 .npz, 남은 인물 id 전부 은퇴 (recluster 없이 vanished 경로)
  delete_all_body = json.dumps(
    {
      "type": "delete_request",
      "job_id": "job-d2",
      "event_id": "event-1",
      "image_ids": ["img-a1", "img-a2", "img-b1", "img-group", "img-twins"],
    }
  )
  emptied = handlers.handle(parse_inbound_message(delete_all_body))
  check(
    "전체 delete: 빈 event 저장 + 전 인물 은퇴",
    emptied.status == "succeeded"
    and emptied.clusters == []
    and emptied.retired_cluster_ids == [cluster_a_id]
    and store.load("event-1") == EventEmbeddings.empty(),
  )

  # ⑧ 경계: 저장된 적 없는 event의 feedback은 failed, delete는 멱등 성공
  ghost_merge = json.dumps(
    {
      "type": "cluster_feedback",
      "job_id": "job-f9",
      "event_id": "event-유령",
      "action": "merge",
      "merge": {"target_cluster_id": "person-x", "source_cluster_ids": ["person-y"]},
    }
  )
  ghost_delete = json.dumps(
    {"type": "delete_request", "job_id": "job-d9", "event_id": "event-유령", "image_ids": ["img-1"]}
  )
  check("유령 event feedback → failed", handlers.handle(parse_inbound_message(ghost_merge)).status == "failed")
  check(
    "유령 event delete → 멱등 succeeded", handlers.handle(parse_inbound_message(ghost_delete)).status == "succeeded"
  )

  # ⑨ 소규모 단일 인물 이벤트(실측 유사도 대역): 구 전체 일괄 승격(완전 연결 0.7)이면 전원
  # uncertain(unmatched)이 되어 앨범이 안 생기던 케이스 — 연결 성분 부분 승격(ADR-008)의 종단 회귀 고정
  single_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c2",
      "group_id": "group-1",
      "event_id": "event-2",
      "images": [{"image_id": f"img-s{k}", "s3_key": f"img-s{k}.jpg"} for k in range(5)],
    }
  )
  single = handlers.handle(parse_inbound_message(single_body))
  check(
    "단일 인물 5장(쌍 유사도 0.53~0.65): 인물 앨범 1개에 전원 소속, uncertain 없음",
    len(single.clusters) == 1
    and set(single.clusters[0].image_ids) == {f"img-s{k}" for k in range(5)}
    and single.uncertain == []
    and single.common_album == []
    and single.status == "succeeded",
  )

  # ⑩ 품질 게이트: 눈감음/흔들림 사진을 인물 앨범 대신 품질 앨범으로 분리 (토글 ON), OFF면 분리 안 함
  quality_on_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c3",
      "group_id": "group-1",
      "event_id": "event-3",
      "images": [
        {"image_id": "img-q-ok1", "s3_key": "img-q-ok1.jpg"},
        {"image_id": "img-q-ok2", "s3_key": "img-q-ok2.jpg"},
        {"image_id": "img-q-eyes", "s3_key": "img-q-eyes.jpg"},
        {"image_id": "img-q-blur", "s3_key": "img-q-blur.jpg"},
      ],
    }
  )
  quality_on = handlers.handle(parse_inbound_message(quality_on_body))
  routed_elsewhere = set(quality_on.common_album) | {u.image_id for u in quality_on.uncertain}
  for cluster in quality_on.clusters:
    routed_elsewhere |= set(cluster.image_ids)
  check(
    "품질 ON: 눈감음/흔들림 분리, 정상 2장만 인물 군집",
    quality_on.eyes_closed == ["img-q-eyes"]
    and quality_on.blurry == ["img-q-blur"]
    and len(quality_on.clusters) == 1
    and set(quality_on.clusters[0].image_ids) == {"img-q-ok1", "img-q-ok2"},
  )
  check(
    "품질 ON: 분리된 사진은 인물/공용/uncertain 어디에도 없음(상호배타) + .npz엔 정상 2행만",
    routed_elsewhere.isdisjoint({"img-q-eyes", "img-q-blur"}) and len(store.load("event-3").face_ids) == 2,
  )

  # 같은 3종을 토글 OFF로 — 품질 분리 없이 전원 인물 군집에 남는다 (event-4로 격리)
  quality_off_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c4",
      "group_id": "group-1",
      "event_id": "event-4",
      "options": {"exclude_eyes_closed": False, "exclude_blurry": False},
      "images": [
        {"image_id": "img-q-ok1", "s3_key": "img-q-ok1.jpg"},
        {"image_id": "img-q-eyes", "s3_key": "img-q-eyes.jpg"},
        {"image_id": "img-q-blur", "s3_key": "img-q-blur.jpg"},
      ],
    }
  )
  quality_off = handlers.handle(parse_inbound_message(quality_off_body))
  check(
    "품질 OFF: 분리 안 함 — 눈감음/흔들림도 인물 군집에 포함",
    quality_off.eyes_closed == []
    and quality_off.blurry == []
    and len(quality_off.clusters) == 1
    and set(quality_off.clusters[0].image_ids) == {"img-q-ok1", "img-q-eyes", "img-q-blur"}
    and len(store.load("event-4").face_ids) == 3,
  )

  # ⑪ 흔들림 fallback: 얼굴 미검출 + 흔들림 → 공용앨범이 아니라 흔들림 앨범 (완전 흔들려 검출 실패한 사진 구제).
  #    얼굴 미검출 + 선명(img-none)은 그대로 공용앨범. (전체 이미지 흔들림 측정은 extractor의 몫 — 핸들러 라우팅만 검증)
  nofaceblur_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c5",
      "group_id": "group-1",
      "event_id": "event-5",
      "images": [
        {"image_id": "img-nf-blur", "s3_key": "img-nf-blur.jpg"},  # 얼굴X, 흔들림O → blurry
        {"image_id": "img-nf-ok", "s3_key": "img-none.jpg"},  # 얼굴X, 흔들림X → common
      ],
    }
  )
  nfb = handlers.handle(parse_inbound_message(nofaceblur_body))
  check(
    "흔들림 fallback: 얼굴 미검출+흔들림 → blurry, 얼굴 미검출+선명 → common",
    nfb.blurry == ["img-nf-blur"] and nfb.common_album == ["img-nf-ok"] and len(store.load("event-5").face_ids) == 0,
  )

  # ⑫ uncertain 사진의 인물 앨범 편입: 예약 앨범 id(UNCERTAIN_ALBUM_ID)를 from_cluster_id로 하는 reassign으로
  #    미매칭(cluster_id=None) 얼굴을 must-link 편입한다 (feature-spec §6.2·§6.3 — 계약 확장)
  uncertain_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c6",
      "group_id": "group-1",
      "event_id": "event-6",
      "images": [
        {"image_id": "img-u1", "s3_key": "img-a1.jpg"},  # 인물 0
        {"image_id": "img-u2", "s3_key": "img-a2.jpg"},  # 인물 0 (같은 사람) → 인물 앨범 형성
        {"image_id": "img-stranger", "s3_key": "img-stranger.jpg"},  # 낯선 1인 → uncertain(unmatched)
      ],
    }
  )
  uncertain_result = handlers.handle(parse_inbound_message(uncertain_body))
  check(
    "uncertain 편입 전: stranger는 uncertain(unmatched) + 예약 앨범 id 부여",
    len(uncertain_result.clusters) == 1
    and any(
      u.image_id == "img-stranger" and u.reason == "unmatched" and u.album_id == UNCERTAIN_ALBUM_ID
      for u in uncertain_result.uncertain
    ),
  )
  cluster_u_id = next(c.cluster_id for c in uncertain_result.clusters if "img-u1" in c.image_ids)
  reassign_uncertain_body = json.dumps(
    {
      "type": "cluster_feedback",
      "job_id": "job-f6",
      "event_id": "event-6",
      "action": "reassign",
      "reassign": {
        "image_id": "img-stranger",
        "from_cluster_id": UNCERTAIN_ALBUM_ID,  # 예약 앨범 id — uncertain 얼굴을 출처로 인정
        "to_cluster_id": cluster_u_id,
      },
    }
  )
  joined = handlers.handle(parse_inbound_message(reassign_uncertain_body))
  check(
    "uncertain 편입: 예약 앨범 id reassign으로 stranger가 인물 앨범에 편입 + uncertain에서 제거",
    "img-stranger" in image_ids_of(joined, containing="img-u1")
    and all(u.image_id != "img-stranger" for u in joined.uncertain),
  )
  after_join = store.load("event-6")
  check(
    "uncertain 편입: stranger가 .npz에 인물 앨범으로 배정 저장 (must-link 지속)",
    dict(zip(after_join.photo_ids, after_join.cluster_ids))["img-stranger"] == cluster_u_id,
  )

  # ⑬ confirm_distinct: 확정된 두 인물 앨범 사이 다리(bridge) 사진의 오병합을 cannot-link로 차단.
  #    must-link(각 앨범 내부 응집)만으로는 이 시나리오를 못 막는다 — 인물 40·41은 확실히 다른
  #    사람(cos≈0.64)이지만 다리 사진(인물 42, 두 대표와 cos≈0.906)이 들어오면 confirm_distinct
  #    없이는 전체 재군집이 둘을 하나로 오병합한다(실측 확인). confirm_distinct로 대표 얼굴 간
  #    cannot-link를 걸어두면 이후 다리 사진이 추가돼도 두 앨범이 분리 유지된다.
  confirm_setup_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c7",
      "group_id": "group-1",
      "event_id": "event-7",
      "images": [
        {"image_id": "img-cd-40-1", "s3_key": "img-cd-40-1.jpg"},
        {"image_id": "img-cd-40-2", "s3_key": "img-cd-40-2.jpg"},
        {"image_id": "img-cd-41-1", "s3_key": "img-cd-41-1.jpg"},
        {"image_id": "img-cd-41-2", "s3_key": "img-cd-41-2.jpg"},
      ],
    }
  )
  confirm_setup = handlers.handle(parse_inbound_message(confirm_setup_body))
  check(
    "confirm_distinct 사전조건: 두 인물이 별개 앨범으로 형성됨",
    len(confirm_setup.clusters) == 2,
  )
  cluster_40_id = next(c.cluster_id for c in confirm_setup.clusters if "img-cd-40-1" in c.image_ids)
  cluster_41_id = next(c.cluster_id for c in confirm_setup.clusters if "img-cd-41-1" in c.image_ids)

  confirm_distinct_body = json.dumps(
    {
      "type": "cluster_feedback",
      "job_id": "job-f7",
      "event_id": "event-7",
      "action": "confirm_distinct",
      "confirm_distinct": {"cluster_ids": [cluster_40_id, cluster_41_id]},
    }
  )
  handlers.handle(parse_inbound_message(confirm_distinct_body))
  after_confirm = store.load("event-7")
  check(
    "confirm_distinct: 대표 얼굴 간 cannot-link 1쌍 저장",
    len(after_confirm.cannot_link_pairs) == 1,
  )

  bridge_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c8",
      "group_id": "group-1",
      "event_id": "event-7",
      "images": [{"image_id": "img-cd-bridge", "s3_key": "img-cd-bridge.jpg"}],
    }
  )
  bridged = handlers.handle(parse_inbound_message(bridge_body))
  check(
    "confirm_distinct: 다리 사진 추가 후에도 두 앨범이 분리 유지 (오병합 없음)",
    len(bridged.clusters) == 2
    and {cluster_40_id, cluster_41_id} == {c.cluster_id for c in bridged.clusters}
    and bridged.retired_cluster_ids == [],
  )
  # 적대적 성질 재검증: 같은 기하를 제약 없이 재군집하면 실제로 오병합(1 클러스터)돼야 위 검증이
  # 유의미하다 — 임계 재보정으로 기하가 무해해지면(2 클러스터) 여기서 즉시 표면화된다.
  unconstrained = recluster(
    np.stack([confirm_distinct_vector(p, s) for p, s in ((40, 0), (40, 1), (41, 0), (41, 1), (42, 0))]),
    [None] * 5,
    Constraints(),
    handlers._cluster_config,
    counter("검증"),
  )
  check("confirm_distinct 기하: 제약 없으면 오병합 재현 (검증의 적대적 성질 유지)", len(unconstrained.clusters) == 1)

  # ⑭ 같은 사진 자동 cannot-link 유도 (ADR-011): 다인 사진의 얼굴 쌍은 제약이 되고,
  # 임베딩이 사실상 동일(≥ _SAME_FACE_SIMILARITY)한 쌍은 YuNet 이중 검출로 보고 제외한다
  base_vec = np.zeros(512, dtype=np.float32)
  base_vec[0] = 1.0
  dup_vec = base_vec.copy()  # 이중 검출 재현 — 같은 얼굴의 두 박스는 임베딩이 사실상 동일
  other_vec = np.zeros(512, dtype=np.float32)
  other_vec[1] = 1.0
  auto_event = EventEmbeddings.empty().append_faces(
    [
      ("f-1", "p-multi", base_vec),
      ("f-2", "p-multi", dup_vec),
      ("f-3", "p-multi", other_vec),
      ("f-4", "p-solo", base_vec),
    ]
  )
  check(
    "같은사진 자동 cannot-link: 타인 쌍만 유도, 이중 검출 쌍·다른 사진 쌍은 제외",
    _same_photo_cannot_links(auto_event) == ((0, 2), (1, 2)),
  )

  # ⑮ 주 인물 카운트 라우팅 (CHMO-330): 얼굴 2개라도 하나가 행인(최대 폭의 50% 미만)이면 단체 사진이
  # 아니다 — 미매칭 주 인물 1명이므로 공용이 아니라 uncertain. 폭이 대등한 2인 미매칭은 공용(①에서 검증).
  image_source.images["img-walkin.jpg"] = fake_image([(9, 0, 120), (10, 0, 40)])  # 낯선 주 인물 + 행인
  walkin_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c9",
      "group_id": "group-1",
      "event_id": "event-9",
      "images": [{"image_id": "img-walkin", "s3_key": "img-walkin.jpg"}],
    }
  )
  walkin = handlers.handle(parse_inbound_message(walkin_body))
  check(
    "주 인물 1명+행인: 공용 아님, uncertain(unmatched)",
    walkin.common_album == [] and [(u.image_id, u.reason) for u in walkin.uncertain] == [("img-walkin", "unmatched")],
  )
  check(
    "얼굴 폭이 .npz에 저장·왕복됨",
    store.load("event-9").face_widths == (120.0, 40.0),
  )

  print(f"\n스모크 검증 {passed}건 전부 통과")
