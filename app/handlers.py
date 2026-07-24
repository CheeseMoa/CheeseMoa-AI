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
from dataclasses import dataclass, replace
from itertools import combinations
from typing import NamedTuple

import numpy as np

from app.pipeline.cluster import ClusterConfig, Constraints, PersonCluster, recluster
from app.pipeline.rejudge import (
  ClusterPairRejudger,
  NonhumanFaceGate,
  NonhumanOutcome,
  NonhumanVerdict,
  PairRejudgeOutcome,
  RejudgeOutcome,
  UncertainRejudger,
)
from app.schemas.messages import (
  UNCERTAIN_ALBUM_ID,
  ClassifyRequest,
  ClassifyResult,
  ConfirmDistinctFeedback,
  DeleteRequest,
  FaceBox,
  FailedImage,
  MergeFeedback,
  ReassignFeedback,
  ResultCluster,
  SplitFeedback,
  UncertainImage,
  UncertainSuggestion,
)
from app.storage.embedding_store import EmbeddingStore
from app.storage.event_embeddings import EventEmbeddings
from app.storage.image_source import ImageSource
from app.storage.nonhuman_verdicts import NonhumanVerdictRecord, NonhumanVerdictStore
from app.storage.rekognition_scores import RekognitionScoreStore
from app.storage.thumbnail_store import ThumbnailStore

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
  bboxes: list[tuple[int, int, int, int]] | None = None  # 같은 순서의 (x, y, w, h) 원본 px — 썸네일 crop용. None = 미상


# 디코딩된 BGR 이미지 → ExtractedFaces. embeddings 빈 목록이면 common_album 라우팅 (feature-spec §6.2).
# detect→align→embed + 품질 판정 합성은 core.deps.build_face_extractor가 만든다.
FaceExtractor = Callable[[np.ndarray], ExtractedFaces]

# classify 처리 중 진행률 보고 콜백 (job_id, event_id, processed, total) → None (CHMO-274).
# 핸들러는 ProgressUpdate 스키마·SQS를 모른다 — 발행 배선은 core.deps가 클로저로 주입한다 (관심사 분리).
ProgressReporter = Callable[[str, str, int, int], None]

# 디코딩된 BGR 원본 + 얼굴 bbox(x, y, w, h) → 썸네일 JPEG bytes (CHMO-335). 실패는 예외로 던진다.
# 핸들러는 cv2를 모른다 — pipeline.thumbnail과의 합성은 core.deps의 책임이다 (FaceExtractor와 같은 구도).
ThumbnailRenderer = Callable[[np.ndarray, tuple[float, float, float, float]], bytes]

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
# 근중복 판정 관례 0.985(ADR-005 burst)와 큰 마진. 재실측(2026-07-21, 24개 event 758쌍):
# 이중 검출 0.978~0.979 vs 타인 최고 0.756 — 여전히 빈 구간 안. 같은 원리·같은 기본값을 머릿수
# 붕괴(ClusterConfig.common_duplicate_face_similarity, ADR-027)도 쓴다.
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


def _uncertain_face_boxes(event: EventEmbeddings, indices: Iterable[int]) -> list[FaceBox]:
  """uncertain 사진의 상세 화면 얼굴 crop용 bbox 목록 (계약 교체 CHMO-407, feature-spec §6.2).

  폭 내림차순(주 얼굴 먼저), 동률은 event 행 순. bbox 미상(w·h≤0 — v2 이하 .npz 행)은 제외한다.
  """
  boxes: list[FaceBox] = []
  for index in sorted(indices, key=lambda i: (-event.bboxes[i][2], i)):
    x, y, w, h = event.bboxes[index]
    if round(w) <= 0 or round(h) <= 0:
      continue
    boxes.append(FaceBox(x=round(x), y=round(y), w=round(w), h=round(h)))
  return boxes


def _uncertain_suggestions(
  event: EventEmbeddings, indices: Iterable[int], suggestion_of: dict[int, tuple[str, float]]
) -> list[UncertainSuggestion]:
  """uncertain 사진의 재판정 제안 목록 (ADR-030) — 유사도 내림차순, 동률은 event 행 순.

  face_bbox는 _uncertain_face_boxes와 동일한 반올림 값이라 앱이 face_bboxes 원소와 int 동등
  비교로 결속한다 — 그래서 대상도 같은 집합(주 인물 자격 crop 얼굴)으로 제한하고, bbox 미상
  행(face_bboxes에서 빠지는 행)은 제안도 싣지 않는다(결속 불가).
  """
  items: list[tuple[float, int, UncertainSuggestion]] = []
  for index in indices:
    matched = suggestion_of.get(index)
    if matched is None:
      continue
    x, y, w, h = event.bboxes[index]
    if round(w) <= 0 or round(h) <= 0:
      continue
    cluster_id, similarity = matched
    items.append(
      (
        similarity,
        index,
        UncertainSuggestion(
          face_bbox=FaceBox(x=round(x), y=round(y), w=round(w), h=round(h)),
          cluster_id=cluster_id,
          similarity=similarity,
        ),
      )
    )
  items.sort(key=lambda entry: (-entry[0], entry[1]))
  return [entry[2] for entry in items]


@dataclass(frozen=True)
class _ClusterPass:
  """재군집 + 단일 사진 클러스터 강등 1회 실행의 결과 — 저장 없는 순수 계산 (재판정 2차 패스 재사용)."""

  clusters: tuple[PersonCluster, ...]  # 인물로 승격된 클러스터만 (강등분 제외)
  unmatched_indices: tuple[int, ...]  # 재군집 노이즈 + 강등된 얼굴
  ambiguous_indices: tuple[int, ...]
  retired_cluster_ids: tuple[str, ...]  # 재군집 은퇴분 + 강등된 기존 id
  new_cluster_ids: tuple[str | None, ...]  # 행별 새 배정 (.npz 갱신 원천)


@dataclass(frozen=True)
class _ClusterSnapshot:
  """재군집 + 단일 사진 클러스터 강등까지 끝난, 결과 조립에 필요한 event 스냅샷 요약."""

  event: EventEmbeddings  # 저장까지 마친 최종 상태
  clusters: tuple[PersonCluster, ...]  # 인물로 승격된 클러스터만 (강등분 제외)
  unmatched_indices: tuple[int, ...]  # 재군집 노이즈 + 강등된 얼굴
  ambiguous_indices: tuple[int, ...]
  retired_cluster_ids: tuple[str, ...]  # 재군집 은퇴분 + 강등된 기존 id
  rejudge: RejudgeOutcome | None = None  # Rekognition 재판정 결과 — 제안이 _assemble_result로 흐르는 통로 (ADR-030)


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
    render_thumbnail: ThumbnailRenderer | None = None,
    thumbnails: ThumbnailStore | None = None,
    rejudger: UncertainRejudger | None = None,
    rejudge_scores: RekognitionScoreStore | None = None,
    pair_rejudger: ClusterPairRejudger | None = None,
    apply_pair_merges: bool = False,
    nonhuman_gate: NonhumanFaceGate | None = None,
    nonhuman_verdicts: NonhumanVerdictStore | None = None,
  ) -> None:
    self._store = store
    self._images = images
    self._extract_faces = extract_faces
    self._cluster_config = cluster_config if cluster_config is not None else ClusterConfig()
    # 둘 다 주입돼야 썸네일 활성 (CHMO-335) — 하나라도 None이면 전 경로에서 생략 = 종전 동작 (롤백 경로)
    self._render_thumbnail = render_thumbnail
    self._thumbnails = thumbnails
    # 둘 다 주입돼야 Rekognition 재판정 활성 (ADR-030) — 하나라도 None이면 생략 = 종전 동작 (REJUDGE_ENABLED=false 롤백 경로)
    self._rejudger = rejudger
    self._rejudge_scores = rejudge_scores
    # 앨범 쌍 병합 재판정 (ADR-031) — 판정(호출·로그)과 반영(soft_merge 기록)이 분리된 2단 스위치다.
    # apply_pair_merges=False면 판정 로그만 남고 파티션은 변하지 않는다(관측 모드 — 실사용자 이벤트에서
    # 회색지대 비율·통과율을 확인한 뒤 켠다, ADR-031 §롤아웃). 점수 캐시 저장소는 얼굴 단위와 공유한다.
    self._pair_rejudger = pair_rejudger
    self._apply_pair_merges = apply_pair_merges
    # 둘 다 주입돼야 비인간 얼굴 게이트 활성 (ADR-032) — 하나라도 None이면 전 경로 생략 = 종전 동작
    # (NONHUMAN_GATE_ENABLED=false 롤백 경로). 판정 저장소가 필수인 이유: 통과 판정은 npz에 안 남아
    # 캐시 없이는 매 재군집마다 같은 얼굴을 DetectLabels로 재과금한다. 비활성이면 npz에 이미 남은
    # 강등 기록(nonhuman_face_ids)도 무시한다 — 스위치 한 번으로 강등 얼굴이 전부 되살아나는 것이
    # 롤백 계약이다(기록 자체는 npz에 남아 재활성 시 다시 적용된다).
    self._nonhuman_gate = nonhuman_gate
    self._nonhuman_verdicts = nonhuman_verdicts
    self._nonhuman_active = nonhuman_gate is not None and nonhuman_verdicts is not None
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
    new_rows: list[tuple[str, str, np.ndarray, float, tuple[float, float, float, float], str, float]] = []
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
      bboxes = extracted.bboxes or [(0, 0, 0, 0)] * len(extracted.embeddings)
      long_side = float(
        max(image.shape[0], image.shape[1])
      )  # 원본 긴 변 px — uncertain 저해상도 원인 판정용 (CHMO-404)
      new_rows.extend(
        (self._new_face_id(), ref.image_id, embedding, width, tuple(float(v) for v in bbox), ref.s3_key, long_side)
        for embedding, width, bbox in zip(extracted.embeddings, widths, bboxes)
      )

    event = stored.append_faces(new_rows)
    snapshot = self._recluster_and_save(request.event_id, event)
    return self._assemble_result(
      request.job_id,
      request.event_id,
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
    # 사람이 옮긴 얼굴(superseded)의 재판정 편입은 폐기한다 — 사람 결정이 재판정보다 우선이므로
    # 낡은 soft-attach가 이후 재군집에서 그 얼굴을 옛 대표 앨범으로 되돌리지 못하게 한다 (ADR-030 개정).
    kept_soft_attach = [
      pair for pair in stored.soft_attach_pairs if pair[0] not in superseded_faces and pair[1] not in superseded_faces
    ]
    # 앨범 쌍 병합도 같은 원리로 폐기한다 (ADR-031 ④): 사용자가 split·reassign으로 병합의 근거였던
    # 대표 얼굴을 옮겼다면, 그 병합 기록이 살아남아 다음 재군집에서 앨범을 다시 붙이면 안 된다.
    kept_soft_merge = [
      pair for pair in stored.soft_merge_pairs if pair[0] not in superseded_faces and pair[1] not in superseded_faces
    ]
    # 비인간 강등 해제 (ADR-032): 사용자 보정이 강등 얼굴을 건드리면 오판으로 보고 목록에서 제거한다 —
    # 오판이 사용자 조작 한 번으로 영구 복구되는 경로. 다음 재군집에서 그 행이 되살아나고, 이후에는
    # 보정 당사자 면제(NonhumanFaceGate의 must/cannot-link 생략)가 재강등을 영구히 막는다.
    touched_faces = {face_id for pair in (*new_must, *new_cannot) for face_id in pair} | superseded_faces
    kept_nonhuman = [face_id for face_id in stored.nonhuman_face_ids if face_id not in touched_faces]
    event = (
      stored.with_constraints(must, cannot)
      .with_soft_attach_pairs(kept_soft_attach)
      .with_soft_merge_pairs(kept_soft_merge)
      .with_nonhuman_face_ids(kept_nonhuman)
    )
    snapshot = self._recluster_and_save(message.event_id, event)
    return self._assemble_result(message.job_id, message.event_id, snapshot)

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
    return self._assemble_result(request.job_id, request.event_id, snapshot, extra_retired=vanished)

  # ── 공통 꼬리: 재군집 + .npz rewrite + 결과 조립 ─────────────────────────────

  def _cluster_pass(self, event: EventEmbeddings) -> _ClusterPass:
    """재군집 + 단일 사진 클러스터 강등 1회 — 저장 없는 순수 계산 (재판정 편입 반영 시 2차 패스로 재실행)."""
    row_of = event.row_index_of()
    constraints = Constraints(
      must_link=tuple((row_of[a], row_of[b]) for a, b in event.must_link_pairs),
      cannot_link=tuple((row_of[a], row_of[b]) for a, b in event.cannot_link_pairs),
      auto_cannot_link=_same_photo_cannot_links(event),
      # 재판정 편입(ADR-030 개정): 응집 게이트 이후 F를 R의 최종 클러스터에 부착만 한다 (must-link 강제로
      # 대상 클러스터가 재파편화되던 회귀 교정). 방향(F→R) 보존 — 저장 순서가 (F, R)이다.
      soft_attach=tuple((row_of[f], row_of[r]) for f, r in event.soft_attach_pairs),
      # 앨범 쌍 병합(ADR-031): 두 대표가 최종적으로 속한 앨범을 응집 게이트 이후 union한다. 판정
      # (전원 합의·완전 연결)은 rejudge에서 끝났고 여기서는 반영만 한다 — soft_attach와 같은 구도.
      soft_merge=tuple((row_of[a], row_of[b]) for a, b in event.soft_merge_pairs),
    )
    # 비인간 강등 행(ADR-032)은 재군집 입력에서 통째로 뺀다 — clusters/noise/ambiguous 어디에도
    # 등장하지 않아 new_cluster_ids가 None(미배정)으로 남고, 전량 강등 사진의 공용 앨범 라우팅은
    # _assemble_result의 책임이다. 게이트 비활성(롤백)이면 npz 기록을 무시해 강등 얼굴이 되살아난다.
    result = recluster(
      event.embeddings,
      event.cluster_ids,
      constraints,
      self._cluster_config,
      self._new_cluster_id,
      excluded_rows=tuple(row_of[face_id] for face_id in event.nonhuman_face_ids) if self._nonhuman_active else (),
    )

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
    return _ClusterPass(
      clusters=tuple(kept),
      unmatched_indices=tuple(sorted([*result.noise_indices, *demoted_indices])),
      ambiguous_indices=result.ambiguous_indices,
      retired_cluster_ids=tuple(dict.fromkeys([*result.retired_cluster_ids, *demoted_retired])),
      new_cluster_ids=tuple(new_cluster_ids),
    )

  def _recluster_and_save(self, event_id: str, event: EventEmbeddings) -> _ClusterSnapshot:
    """event 전체를 재군집·강등 판정하고 새 배정을 반영해 저장한다 — 이 저장이 유일한 쓰기이자 마지막 변이다.

    저장 후 크래시(발행·삭제 전)로 메시지가 재전달돼도, photo_id 멱등 append + 결정적 재군집이
    같은 상태에서 같은 결과를 다시 만들므로 안전하다 (오류 매트릭스 참조). Rekognition 판정이
    활성이면 1차 패스가 끝난 상태에서 세 가지를 판정한다 — 비인간 얼굴 강등(ADR-032, 가장 먼저:
    강등 얼굴을 뒤 재판정의 후보에서 뺀다), 미배정 얼굴의 앨범 편입(ADR-030), 회색지대 앨범 쌍의
    병합(ADR-031). 전부 npz 기록(nonhuman_face_ids·soft-attach·soft-merge)으로 남기고 재군집을 한 번
    더 돌린다 — 스냅샷 직접 수정 대신 2차 패스를 쓰는 이유는 같은사진 cannot-link·강등·병합 게이트 등
    모든 후처리 불변식이 자동 재적용되고, 저장된 .npz를 재군집한 결과와 발행 결과가 항상 일치해
    재전달 안전성이 유지되기 때문이다 (군집 비용 <0.1%).
    """
    if not event.face_ids:
      # 전부 삭제된(또는 애초에 빈) event — 재군집할 것이 없어도 빈 파일을 저장해 단일 진실을 유지한다.
      # 재판정 점수·비인간 판정 캐시도 함께 지운다 — 생체 파생 정보가 원본 얼굴보다 오래 남지 않게 (ADR-030·032)
      self._store.save(event_id, event)
      self._delete_rejudge_scores(event_id)
      self._delete_nonhuman_verdicts(event_id)
      return _ClusterSnapshot(
        event=event, clusters=(), unmatched_indices=(), ambiguous_indices=(), retired_cluster_ids=()
      )

    current = self._cluster_pass(event)
    outcome: RejudgeOutcome | None = None
    pair_outcome: PairRejudgeOutcome | None = None
    nonhuman_outcome: NonhumanOutcome | None = None
    gate_active = self._nonhuman_active
    rejudge_active = self._rejudge_scores is not None and (
      self._rejudger is not None or self._pair_rejudger is not None
    )
    if gate_active or rejudge_active:
      # best-effort 격리 (썸네일과 동일 정책): 게이트·재판정·캐시의 어떤 실패도 job을 죽이지 않는다 —
      # 실패 시 1차 패스 결과 그대로(현 동작 유지). event 교체는 2차 패스 성공 후에만 커밋한다: 강등·
      # 편입 soft-attach·병합 soft-merge가 예상 밖 모순으로 재군집을 깨뜨리는 경우 그 기록이 저장되면
      # 이후 모든 재군집이 죽는다(오염 방지). 세 판정은 같은 1차 패스 위에서 끝내고 2차 패스는 한 번만 돈다.
      try:
        augmented = event
        # ① 비인간 얼굴 게이트 (ADR-032) — 두 재판정보다 먼저가 계약: 강등 얼굴을 CompareFaces의
        # source·대표 후보에서 배제해 호출 낭비를 없앤다 (실 event 139에서 인형이 앨범 대표로 뽑혀
        # 미배정 4명과 비교된 4쌍이 전부 -1.0으로 버려진 실기록이 근거).
        demoted_rows: set[int] = set()
        if gate_active:
          nonhuman_outcome = self._nonhuman_gate.judge(
            event,
            current.clusters,
            (*current.ambiguous_indices, *current.unmatched_indices),
            {
              face_id: NonhumanVerdict(
                face_id=face_id,
                nonhuman=record.nonhuman,
                rule=record.rule,
                labels=dict(record.labels),
                n_faces=record.n_faces,
              )
              for face_id, record in self._nonhuman_verdicts.load(event_id).items()
            },
          )
          if nonhuman_outcome.demoted_face_ids:
            augmented = augmented.with_nonhuman_face_ids(
              tuple(dict.fromkeys(augmented.nonhuman_face_ids + nonhuman_outcome.demoted_face_ids))
            )
            row_of = event.row_index_of()
            demoted_rows = {row_of[face_id] for face_id in nonhuman_outcome.demoted_face_ids}
        # ② 강등분을 제거한 뷰로 두 재판정 — 클러스터 멤버에서 강등 행을 빼고(전원 강등 앨범은 드롭),
        # uncertain 목록에서도 강등 행을 뺀다. centroid는 판정 시점 값 그대로 둔다 — 후보 순위용이라
        # 재계산 불필요, 정확한 값은 2차 패스가 다시 만든다.
        live_clusters = current.clusters
        live_uncertain = (*current.ambiguous_indices, *current.unmatched_indices)
        if demoted_rows:
          trimmed: list[PersonCluster] = []
          for cluster in current.clusters:
            members = [
              (i, s) for i, s in zip(cluster.member_indices, cluster.membership_similarities) if i not in demoted_rows
            ]
            if not members:
              continue  # 전원 강등된 앨범 — 재판정 후보가 아니다 (소멸은 2차 패스가 확정)
            trimmed.append(
              replace(
                cluster,
                member_indices=tuple(i for i, _ in members),
                membership_similarities=tuple(s for _, s in members),
              )
            )
          live_clusters = tuple(trimmed)
          live_uncertain = tuple(i for i in live_uncertain if i not in demoted_rows)
        if rejudge_active:
          scores = self._rejudge_scores.load(event_id)
          if self._rejudger is not None and live_uncertain:
            outcome = self._rejudger.rejudge(event, live_clusters, live_uncertain, scores)
            scores = outcome.scores if outcome.scores is not None else scores
          if self._pair_rejudger is not None and len(live_clusters) >= 2:
            # 같은 점수 dict를 이어 쓴다 — 얼굴 단위와 겹치는 (얼굴, 대표) 쌍은 재과금되지 않는다
            pair_outcome = self._pair_rejudger.judge(event, live_clusters, scores)
            scores = pair_outcome.scores if pair_outcome.scores is not None else scores

        if outcome is not None and outcome.assignments:
          # 편입은 must-link가 아니라 soft-attach로 기록한다 (ADR-030 개정): 저응집 F를 must-link로 강제하면
          # 대상 클러스터의 응집이 내려가 파편병합 게이트 탈락·기존 멤버 축출을 유발했다(실 event 134 회귀).
          # soft-attach는 모든 응집 게이트 이후 F를 대표 R의 최종 클러스터에 부착만 해 클러스터를 안 흔든다.
          augmented = augmented.with_soft_attach_pairs(
            augmented.soft_attach_pairs + tuple((a.face_id, a.rep_face_id) for a in outcome.assignments)
          )
        if pair_outcome is not None and pair_outcome.merges:
          if self._apply_pair_merges:
            augmented = augmented.with_soft_merge_pairs(
              tuple(dict.fromkeys(augmented.soft_merge_pairs + pair_outcome.soft_merge_pairs))
            )
          else:
            logger.info(
              "앨범 쌍 병합 판정 %d건 — 관측 모드(REJUDGE_PAIR_APPLY=false)라 반영하지 않음: %s",
              len(pair_outcome.merges),
              [merge.cluster_ids for merge in pair_outcome.merges],
            )
        if augmented is not event:
          current = self._cluster_pass(augmented)
          event = augmented  # 2차 패스 성공 — 강등·재판정 기록을 저장 대상으로 확정 (이후 재군집에서 유지)
      except Exception:
        logger.warning(
          "Rekognition 판정(게이트·재판정) 실패 event_id=%s — 무시하고 진행 (현 동작 유지)", event_id, exc_info=True
        )
        outcome = None
        pair_outcome = None
        nonhuman_outcome = None

    saved = event.with_cluster_ids(current.new_cluster_ids)
    self._store.save(event_id, saved)
    measured = [o.scores for o in (outcome, pair_outcome) if o is not None and o.scores is not None]
    if measured:
      # 점수 캐시 갱신 — best-effort. 죽은 face_id 항목은 프루닝한다 (.npz의 제약 프루닝과 같은 위생)
      try:
        alive = set(saved.face_ids)
        merged_scores = {pair: sim for scores in measured for pair, sim in scores.items()}
        self._rejudge_scores.save(
          event_id, {pair: sim for pair, sim in merged_scores.items() if pair[0] in alive and pair[1] in alive}
        )
      except Exception:
        logger.warning("재판정 점수 캐시 저장 실패 event_id=%s — 무시하고 진행", event_id, exc_info=True)
    if nonhuman_outcome is not None and nonhuman_outcome.verdicts is not None:
      # 비인간 판정 캐시 갱신 — best-effort. 통과 판정의 캐싱이 이 저장의 존재 이유다(강등분은 npz에
      # 남지만 통과분은 안 남는다). 죽은 face_id 항목은 프루닝한다 (점수 캐시와 같은 위생)
      try:
        alive = set(saved.face_ids)
        self._nonhuman_verdicts.save(
          event_id,
          {
            face_id: NonhumanVerdictRecord(
              nonhuman=verdict.nonhuman, rule=verdict.rule, labels=dict(verdict.labels), n_faces=verdict.n_faces
            )
            for face_id, verdict in nonhuman_outcome.verdicts.items()
            if face_id in alive
          },
        )
      except Exception:
        logger.warning("비인간 판정 캐시 저장 실패 event_id=%s — 무시하고 진행", event_id, exc_info=True)
    return _ClusterSnapshot(
      event=saved,
      clusters=current.clusters,
      unmatched_indices=current.unmatched_indices,
      ambiguous_indices=current.ambiguous_indices,
      retired_cluster_ids=current.retired_cluster_ids,
      rejudge=outcome,
    )

  def _delete_rejudge_scores(self, event_id: str) -> None:
    """재판정 점수 캐시를 지운다 — best-effort (실패해도 고아 객체일 뿐, 다음 활성 job의 프루닝이 정리)."""
    if self._rejudge_scores is None:
      return
    try:
      self._rejudge_scores.delete(event_id)
    except Exception:
      logger.warning("재판정 점수 캐시 삭제 실패 event_id=%s — 무시하고 진행", event_id, exc_info=True)

  def _delete_nonhuman_verdicts(self, event_id: str) -> None:
    """비인간 판정 캐시를 지운다 — best-effort (_delete_rejudge_scores와 같은 정책, ADR-032)."""
    if self._nonhuman_verdicts is None:
      return
    try:
      self._nonhuman_verdicts.delete(event_id)
    except Exception:
      logger.warning("비인간 판정 캐시 삭제 실패 event_id=%s — 무시하고 진행", event_id, exc_info=True)

  # ── 대표 얼굴 썸네일 (CHMO-335) ──────────────────────────────────────────────

  @staticmethod
  def _select_representative(person: PersonCluster, event: EventEmbeddings) -> int | None:
    """썸네일 대표 얼굴 행을 고른다 — LOO centroid 유사도(membership) 최대 = 그 인물다움이 가장 확실한 얼굴.

    bbox·원본 키 미상 행(v2 이하 .npz 폴백)은 crop이 불가능해 후보에서 뺀다. strict 비교라 동률은
    가장 앞 행이 이긴다(member_indices 오름차순 계약) — 같은 멤버십이면 항상 같은 대표(결정성).
    """
    best: int | None = None
    best_similarity = float("-inf")
    for index, similarity in zip(person.member_indices, person.membership_similarities):
      if event.bboxes[index][2] <= 0 or not event.s3_keys[index]:
        continue
      if similarity > best_similarity:
        best, best_similarity = index, similarity
    return best

  def _make_thumbnail(self, event_id: str, person: PersonCluster, event: EventEmbeddings) -> str | None:
    """대표 얼굴을 원본에서 crop·업로드하고 저장 키를 반환한다 — best-effort (실패가 job을 죽이지 않는다).

    원본은 대표 1장만 재fetch한다 — 요청 전체의 디코딩 이미지를 쥐고 있는 설계는 공유 호스트
    (t4g.small) 메모리를 위협해 금지. 키는 (event_id, cluster_id) 고정이라 재군집으로 대표가
    바뀌면 같은 키에 덮어써진다(멱등 — Spring은 presigned URL 발급 시점의 최신 객체를 서빙).
    """
    if self._render_thumbnail is None or self._thumbnails is None:
      return None  # 비활성 (미주입) — 종전 동작
    try:
      representative = self._select_representative(person, event)
      if representative is None:
        return None  # v2 이하 데이터만으로 구성된 클러스터 — 새 사진이 classify되면 자연 회복
      image = self._images.fetch(event.s3_keys[representative])
      jpeg = self._render_thumbnail(image, event.bboxes[representative])
      return self._thumbnails.put(event_id, person.cluster_id, jpeg)
    except Exception:
      logger.warning(
        "썸네일 생성 실패 event_id=%s cluster_id=%s — 무시하고 진행", event_id, person.cluster_id, exc_info=True
      )
      return None

  def _delete_retired_thumbnails(self, event_id: str, cluster_ids: Sequence[str]) -> None:
    """은퇴 클러스터의 썸네일을 정리한다 — best-effort (실패해도 고아 객체일 뿐 계약상 무해)."""
    if self._thumbnails is None:
      return
    for cluster_id in cluster_ids:
      try:
        self._thumbnails.delete(event_id, cluster_id)
      except Exception:
        logger.warning(
          "은퇴 썸네일 삭제 실패 event_id=%s cluster_id=%s — 무시하고 진행", event_id, cluster_id, exc_info=True
        )

  def _assemble_result(
    self,
    job_id: str,
    event_id: str,
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
          thumbnail_s3_key=self._make_thumbnail(event_id, person, event),
        )
      )

    # 미매칭 사진 라우팅 (feature-spec §6.2·§7). 주 인물 얼굴 2명+ 사진은 단체 사진으로 보고 공용 앨범으로,
    # 주 인물 1명(초상·미등록 1인) 미매칭은 uncertain으로 보낸다.
    # 주 인물 판정은 ADR 022와 같은 규칙 — 그 사진 최대 얼굴 폭의 ratio(0.5) 미만은 지나가는 행인으로
    # 보고 세지 않는다: 1인 인물 사진 + 배경 행인이 단체 사진으로 오인돼 공용에 가는 것을 막는다.
    # 폭 미상(0, v1 .npz 행)은 사진 단위로 전원 0이라 최대도 0 → 전원 주 인물 = 종전(전체 얼굴 수)과 동일.
    ratio = self._cluster_config.common_main_face_ratio
    counted = self._headcount_eligible(event)
    max_width_of: dict[str, float] = {}
    for index, (photo_id, width) in enumerate(zip(event.photo_ids, event.face_widths)):
      if counted[index]:
        max_width_of[photo_id] = max(max_width_of.get(photo_id, 0.0), width)
    # uncertain 품질 원인(CHMO-404)용: 사진별 원본 긴 변 px (같은 사진 행은 같은 값, 0=미상이라 제외 = v3 이하 .npz)
    long_side_of: dict[str, float] = {}
    for photo_id, side in zip(event.photo_ids, event.image_long_sides):
      if side > 0:
        long_side_of[photo_id] = side
    main_face = [
      counted[index] and (ratio <= 0 or width >= max_width_of[photo_id] * ratio)
      for index, (photo_id, width) in enumerate(zip(event.photo_ids, event.face_widths))
    ]
    faces_per_photo = Counter(photo_id for index, photo_id in enumerate(event.photo_ids) if main_face[index])
    uncertain: list[UncertainImage] = []
    group_common: list[str] = []
    routed: set[str] = set()
    # 새 정책(group_photo_to_common=True): 주 인물 얼굴 2명+ 사진은 매칭 여부와 무관하게 공용 앨범에도
    # 노출한다 — 인물 앨범과 중복 노출(N:M). 단체 사진은 그 자리에 함께 있던 모두의 사진이라는 제품 결정.
    # OR 서로 다른 인물 앨범 2개+에 속한 사진도 단체다 — 뒤에 작게 찍혔어도 인식된 일행이면 행인이 아니라서
    # 크기 게이트(주 인물 카운트)가 놓친다. 이 조건이 Spring의 종전 "중복 사진 공용 복제" 규칙과 정확히
    # 같아, 이 정책이 켜져 있는 한 AI 결과가 그 규칙의 상위집합이다(백엔드 복제 로직 제거 가능 근거).
    # event 등장 순서로 안정 정렬. (구 정책은 아래 루프에서 '전원 미매칭'인 2+ 사진만 공용으로 보냈다.)
    if self._cluster_config.group_photo_to_common:
      albums_per_photo = Counter(photo_id for person in clusters for photo_id in person.image_ids)
      group_common = [
        photo_id
        for photo_id in dict.fromkeys(event.photo_ids)
        if faces_per_photo[photo_id] >= 2 or albums_per_photo[photo_id] >= 2
      ]
    # 전량 강등 사진 → 공용 앨범 (ADR-032): 강등 행은 clusters/unmatched/ambiguous 어디에도 없어
    # 그대로 두면 그 사진이 결과 메시지에서 통째로 사라진다(앱에서 사진 증발). "얼굴 0개" 경로와 같은
    # 목적지다. event 스코프로 계산해 재전달·재분류에도 같은 결과이며, 얼굴 미검출 common(요청 스코프)과
    # 달리 과거분도 복원된다 — 강등 행이 npz에 남아 있어 가능한 비대칭.
    demoted_common: list[str] = []
    if self._nonhuman_active and event.nonhuman_face_ids:
      nonhuman_faces = set(event.nonhuman_face_ids)
      human_photos = {
        photo_id for face_id, photo_id in zip(event.face_ids, event.photo_ids) if face_id not in nonhuman_faces
      }
      demoted_common = [photo_id for photo_id in dict.fromkeys(event.photo_ids) if photo_id not in human_photos]
    # 상세 화면 얼굴 crop용 bbox (계약 교체 CHMO-407, BE#107): 그 사진의 uncertain 얼굴 중 주 인물
    # 자격(counted — ADR 025·027 AND 폭 게이트 — ADR 022) 통과 얼굴 전부. 행인·오검출·파편은 싣지 않는다
    # — 자격 얼굴이 없으면(오검출 전용 사진) 빈 배열: 박스를 그릴 주 인물이 없다.
    crop_faces_of: dict[str, list[int]] = {}
    for index in (*snapshot.ambiguous_indices, *snapshot.unmatched_indices):
      if main_face[index]:
        crop_faces_of.setdefault(event.photo_ids[index], []).append(index)
    # Rekognition 재판정 제안 (ADR-030): face_id → (앨범, 유사도). 제안의 앨범은 판정 시점 id가 아니라
    # "비교했던 대표 얼굴이 최종 상태에서 속한 클러스터"로 재해석한다 — 편입이 2차 패스를 돌리면
    # 신규 발급·병합으로 id가 바뀔 수 있기 때문 (RejudgeSuggestion 독스트링). 대표가 최종적으로
    # 미배정이면(극단) 제안을 버리고, 편입돼 uncertain을 벗어난 얼굴의 제안은 아래 결합에서 자연 탈락한다.
    suggestion_of: dict[int, tuple[str, float]] = {}
    if snapshot.rejudge is not None and snapshot.rejudge.suggestions:
      row_of = event.row_index_of()
      for suggestion in snapshot.rejudge.suggestions:
        row = row_of.get(suggestion.face_id)
        rep_row = row_of.get(suggestion.rep_face_id)
        final_cluster = event.cluster_ids[rep_row] if rep_row is not None else None
        if row is not None and final_cluster is not None:
          suggestion_of[row] = (final_cluster, suggestion.similarity)
    # ambiguous 우선: 한 사진에 ambiguous·unmatched 얼굴이 섞이면 더 정보가 많은 ambiguous로 보고
    for reason, indices in (("ambiguous", snapshot.ambiguous_indices), ("unmatched", snapshot.unmatched_indices)):
      for index in indices:
        photo_id = event.photo_ids[index]
        if photo_id in routed:
          continue
        if photo_id in clustered_images:
          # 계약 확장(결정 2026-07-21): 인물 앨범에 배정된 사진이라도 주 인물 미매칭 얼굴이 남아 있으면
          # uncertain에도 싣는다 — 미등록 인물의 수동 편입(__uncertain__ reassign) 진입점. 행인·오검출·
          # 파편(main_face 미달) 미매칭은 종전대로 인물 앨범이 우선한다(숨김).
          if not (self._cluster_config.unmatched_main_to_uncertain and main_face[index]):
            continue
        elif faces_per_photo[photo_id] >= 2:
          routed.add(photo_id)
          if not self._cluster_config.group_photo_to_common:
            group_common.append(photo_id)  # 구 정책: 전원 미매칭인 단체 사진만 공용
          continue
        routed.add(photo_id)
        uncertain.append(
          UncertainImage(
            image_id=photo_id,
            reason=reason,
            face_bboxes=_uncertain_face_boxes(event, crop_faces_of.get(photo_id, ())),
            causes=self._uncertain_causes(max_width_of.get(photo_id, 0.0), long_side_of.get(photo_id, 0.0), reason),
            suggestions=_uncertain_suggestions(event, crop_faces_of.get(photo_id, ()), suggestion_of),
          )
        )

    retired_cluster_ids = list(dict.fromkeys([*snapshot.retired_cluster_ids, *extra_retired]))
    self._delete_retired_thumbnails(event_id, retired_cluster_ids)
    return ClassifyResult(
      job_id=job_id,
      status="partial" if failed_images else "succeeded",
      clusters=clusters,
      common_album=list(dict.fromkeys([*common_album, *group_common, *demoted_common])),
      uncertain=uncertain,
      # 품질 게이트로 분리된 사진들 (이번 요청분). 재군집에서 제외됐으므로 clusters/common/uncertain과 겹치지 않는다.
      eyes_closed=list(dict.fromkeys(eyes_closed)),
      blurry=list(dict.fromkeys(blurry)),
      failed_images=list(failed_images),
      retired_cluster_ids=retired_cluster_ids,
    )

  def _uncertain_causes(self, max_main_face_width: float, long_side: float, reason: str) -> list[str]:
    """uncertain 사진이 왜 분류가 어려웠는지 원인 코드 (CHMO-404) — 앱이 "분류가 어려워요" 화면에 설명·안내를 띄우는 근거.

    reason(군집에서 무슨 일: ambiguous/unmatched)과 직교하는 '왜' 축이다. 세 원인:
      small_faces       = 주 얼굴(counted 최대 폭)이 small_face_px 미만 — 멀리·작게 찍힘(참고용)
      low_resolution    = 그중 원본 긴 변이 low_res_long_side 미만 — 저해상도가 작은 얼굴을 유발("원본으로 다시",
                          유일 actionable). small_faces 없이는 실리지 않는다(멀쩡한 사진에 "저해상도" 오안내 차단).
      single_appearance = 주 얼굴이 충분히 크고(품질 정상) 아무와도 매칭 안 됨(unmatched) — 이 인물이 이벤트에
                          한 번만 등장해 묶을 짝이 없음(앨범은 2장+ 필요, 화질 정상이라 재업로드가 아니라 "더
                          나오면 자동 앨범/직접 지정"이 안내다). ambiguous(두 인물 사이)는 한 번 등장이 아니라 제외.
    폭·긴 변 미상(0, v1/v3 이하 .npz 행)이면 판정 불가로 그 원인은 빠진다. small_face_px=0이면 기능 전체 비활성.
    """
    small_px = self._cluster_config.uncertain_small_face_px
    if small_px <= 0 or max_main_face_width <= 0.0:
      return []  # 비활성 또는 폭 미상(v1 .npz) → 판단 근거 없음
    if max_main_face_width < small_px:  # 작은 얼굴 → 품질이 원인
      low_res = self._cluster_config.uncertain_low_res_long_side
      if low_res > 0 and 0.0 < long_side < low_res:
        return ["low_resolution", "small_faces"]  # 저해상도가 작은 얼굴을 유발 — 재업로드가 actionable(우선 안내)
      return ["small_faces"]  # 해상도는 충분하나 멀리·작게 찍힘 — 재업로드로 해결 안 됨(참고용)
    if reason == "unmatched":  # 선명한 얼굴(품질 정상)인데 아무와도 매칭 안 됨 = 이 이벤트에 한 번만 등장
      return ["single_appearance"]
    return []  # 큰 얼굴 + ambiguous(두 인물 사이) → 깔끔한 원인 없음

  def _headcount_eligible(self, event: EventEmbeddings) -> list[bool]:
    """단체 판정 머릿수 자격 — 이중 검출(ADR 027)과 오검출(ADR 025)을 머릿수에서 뺀다.

    ① 이중 검출 붕괴: 같은 사진의 행들이 common_duplicate_face_similarity 이상으로 닮으면 타인이
    아니라 YuNet이 한 얼굴에 그린 파편 박스들이다(같은사진 자동 cannot-link의 _SAME_FACE_SIMILARITY
    안전판과 같은 원리 — 그 안전판 덕에 파편 행들은 같은 인물 앨범에 얌전히 들어가지만, 머릿수가
    행 수를 그대로 세면 1인 셀피가 "주 인물 2명 단체"로 공용 앨범에 노출된다. event 105 실측 파편 쌍
    0.978~0.979 vs 같은사진 타인 최고 0.756). 근중복 연결 그룹마다 폭 최대 행만 자격을 유지한다.
    ② 실인물 자격: 미배정 얼굴이 event 내 어떤 얼굴과도 유사도 바닥 미만이면 오검출로 보고 뺀다.
    털·사물 오검출은 쓰레기 임베딩이라 모든 얼굴과 바닥 유사도인데(event 93 퍼 후드 0.183), 주 인물
    크기로 검출되면 역시 단체로 오판된다 (ADR 025). 최근접 비교 상대에서 같은 사진의 근중복 행만
    뺀다 — 파편은 독립 증거가 아니라서, 넣으면 오검출이 이중 검출됐을 때 파편끼리 ~0.98로 바닥을
    뚫는다(ADR 027). 근중복이 아닌 같은사진 타인은 종전대로 증거다(전면 제외는 그 사진에만 등장하는
    낯선 단체를 오검출로 오판한다). 클러스터 배정 얼굴(실측 전역 최근접 최저 0.407)과, 비교 상대가
    없는 단독 얼굴은 판단 근거가 없어 항상 센다.
    ③ 비인간 강등 (ADR 032): 인형·조형물로 강등된 행은 처음부터 자격이 없다 — 아이 1명 + 인형 1개
    사진이 "주 인물 2명 단체"로 공용 앨범에 노출되던 것을 고친다. ①의 파편 대표 선정에서도 제외한다.
    """
    floor = self._cluster_config.common_face_min_similarity
    duplicate_floor = self._cluster_config.common_duplicate_face_similarity
    total = len(event.face_ids)
    nonhuman = set(event.nonhuman_face_ids) if self._nonhuman_active else set()
    eligible = [face_id not in nonhuman for face_id in event.face_ids]
    if (floor <= 0 and duplicate_floor <= 0) or total < 2:
      return eligible
    embeddings = np.asarray(event.embeddings, dtype=np.float32)
    similarities = embeddings @ embeddings.T
    rows_by_photo: dict[str, list[int]] = {}
    for index, photo_id in enumerate(event.photo_ids):
      rows_by_photo.setdefault(photo_id, []).append(index)

    if duplicate_floor > 0:
      for rows in rows_by_photo.values():
        remaining = list(rows)
        while remaining:
          # 시드에서 근중복(≥ duplicate_floor)으로 이어지는 연결 그룹을 모은다 — 파편 3개+도 한 명
          group = [remaining.pop(0)]
          frontier = list(group)
          while frontier:
            current = frontier.pop()
            linked = [row for row in remaining if similarities[current, row] >= duplicate_floor]
            for row in linked:
              remaining.remove(row)
            group.extend(linked)
            frontier.extend(linked)
          if len(group) > 1:
            # 강등 행(③)은 파편 대표가 될 수 없다 — 전원 강등 그룹이면 대표 없음 (자격 전원 상실 유지)
            candidates = [row for row in group if eligible[row]]
            keep = max(candidates, key=lambda row: event.face_widths[row]) if candidates else -1
            for row in group:
              eligible[row] = row == keep

    if floor > 0:
      comparator_sim = similarities.copy()
      np.fill_diagonal(comparator_sim, -np.inf)
      if duplicate_floor > 0:
        for rows in rows_by_photo.values():
          for position, row_a in enumerate(rows):
            for row_b in rows[position + 1 :]:
              if similarities[row_a, row_b] >= duplicate_floor:
                comparator_sim[row_a, row_b] = comparator_sim[row_b, row_a] = -np.inf
      for index, cluster_id in enumerate(event.cluster_ids):
        if cluster_id is not None or not eligible[index]:
          continue
        best = float(comparator_sim[index].max())
        if np.isfinite(best) and best < floor:
          eligible[index] = False
    return eligible


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
  from app.storage.thumbnail_store import InMemoryThumbnailStore

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
  # 인물 40·41은 0°·62°(centroid cos≈0.47 — 병합 임계 0.55 아래라 사전조건에서 별개 앨범)에 두고,
  # 다리 사진(가상 인물 42)은 31°(두 인물 centroid와 cos 0.91/0.80)에 둔다 — 다리가 한쪽에 구제
  # 편입되면 병합 판정이 centroid cos≈0.59(≥0.55)·facepair 평균≈0.57(≥0.475)으로 넘어가,
  # confirm_distinct 없이는 두 인물이 실제로 오병합된다(아래 인라인 재검증). 종전 0°·50°는 병합 임계
  # 0.68 시절 기준 — ADR-012 재보정 0.55가 cos 0.64를 삼켜 사전조건이 항상 깨졌다. 현행 보정에 맞게
  # 재배치. step 지터는 12° — 같은 인물 사진 쌍 cos 0.978로, 근중복 붕괴 임계(0.985, ADR-029) 아래의
  # 실사진 대역을 모사한다 (종전 0.5°는 cos 0.99996 = 재업로드 복제 수준이라 ⓪ 붕괴가 접는다).
  _CONFIRM_DISTINCT_ANGLES = {40: 0.0, 41: 62.0}
  _CONFIRM_DISTINCT_BRIDGE_ANGLE = 31.0

  def confirm_distinct_vector(person: int, step: int) -> np.ndarray:
    angle = _CONFIRM_DISTINCT_BRIDGE_ANGLE if person == 42 else _CONFIRM_DISTINCT_ANGLES[person]
    theta = math.radians(angle + 12.0 * step)  # step 지터 — 완전 동률 방지 + 붕괴 임계 아래 실사진 대역
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[400] = math.cos(theta)
    vector[401] = math.sin(theta)
    return vector

  def garbage_vector(step: int) -> np.ndarray:
    """오검출(털·사물)의 쓰레기 임베딩 모사 — 전용 축이라 모든 인물 벡터와 유사도 0 (ADR 025 실측 0.183 이하 대역)."""
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[450 + step] = 1.0
    return vector

  def stranger_vector(person: int, step: int) -> np.ndarray:
    """미등록 실인물(낯선 일행) 모사 — 인물 16 평면에 0.3 성분을 실어 인물 16 얼굴과 유사도 ≈0.28.

    실인물 자격 바닥(0.185, ADR 025)은 넘되(오검출과의 판별축 — 실인물 미배정 실측 최저 0.191 대역)
    구제(0.6)·소속(0.4)·blob 승격(0.45)에는 전부 못 미쳐 미매칭 노이즈로 남는다. person_vector의
    교차 인물 유사도(≈0.16)는 바닥 미달이라 오검출로 오인돼 이 대역을 모사할 수 없다."""
    theta = math.radians(5.0 * step)
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[2 * 16] = 0.3 * math.cos(theta)
    vector[2 * 16 + 1] = 0.3 * math.sin(theta)
    vector[460 + 8 * (person - 17) + step] = math.sqrt(1.0 - 0.09)
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

  def rj_stranger_vector(tag: int) -> np.ndarray:
    """재판정(㉓) 전용 미등록 인물 모사 — 인물 4 평면에 0.3 성분: FP 게이트(0.185)는 넘되 구제 전부 미달.

    AuraFace로는 절대 앨범에 못 붙는 하드케이스 대역이다 — Rekognition 재판정만이 회수 경로다."""
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[2 * 4] = 0.3
    vector[490 + tag] = math.sqrt(1.0 - 0.09)  # 490+ 축 — garbage(450대)·stranger(460~483) 축과 불겹침
    return vector

  def fake_extractor(image: np.ndarray) -> ExtractedFaces:
    count = int(image[0, 0, 0])
    vectors: list[np.ndarray] = []
    widths: list[float] = []
    for slot in range(count):
      person, step = int(image[0, slot + 1, 0]), int(image[0, slot + 1, 1])
      widths.append(float(image[0, slot + 1, 2]))
      if person >= 60:
        vectors.append(rj_stranger_vector(person - 60))
      elif person >= 50:
        vectors.append(garbage_vector(step))
      elif person >= 40:
        vectors.append(confirm_distinct_vector(person, step))
      elif person >= 20:
        vectors.append(spread_person_vector(person, step))
      elif person >= 17:
        vectors.append(stranger_vector(person, step))
      else:
        vectors.append(person_vector(person, step))
    # 합성 bbox: 폭을 정사각 (0, 0, w, w)로 — 썸네일 대표 선정(w>0)·페이크 렌더러 경로를 태운다
    bboxes = [(0, 0, int(width), int(width)) for width in widths]
    return ExtractedFaces(vectors, bool(image[1, 0, 0]), bool(image[1, 0, 1]), widths, bboxes)

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

  thumb_store = InMemoryThumbnailStore()
  handlers = JobHandlers(
    store=store,
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("person"),
    new_face_id=counter("face"),
    # 페이크 렌더러 — 픽셀 처리 없이 결정적 bytes만 만든다 (실렌더는 pipeline.thumbnail __main__이 검증)
    render_thumbnail=lambda image, bbox: b"jpeg:" + repr(bbox).encode(),
    thumbnails=thumb_store,
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
    set(result.common_album) == {"img-none", "img-group"}
    and [failed.image_id for failed in result.failed_images] == ["img-broken"]
    and result.status == "partial",
  )
  # img-twins(사실상 동일 임베딩 쌍 0.996)는 cannot-link 안전판과 같은 원리로 머릿수도 1명(ADR 027) —
  # 이중 검출 1인 사진이므로 공용(단체)이 아니라 uncertain(미매칭 1인)이다.
  check(
    "단일 사진 클러스터 강등: 같은 사진 속 닮은 쌍은 인물 승격 안 함, 근중복 쌍은 단체도 아님(uncertain)",
    all("img-group" not in c.image_ids and "img-twins" not in c.image_ids for c in result.clusters)
    and all(u.image_id != "img-group" for u in result.uncertain)
    and any(u.image_id == "img-twins" and u.reason == "unmatched" for u in result.uncertain),
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
  check(
    "classify: bbox·s3_key가 .npz에 저장됨 (v3)",
    all(bbox[2] > 0 for bbox in store.load("event-1").bboxes) and store.load("event-1").s3_keys[0] == "img-a1.jpg",
  )
  check(
    "classify: 클러스터마다 대표 얼굴 썸네일 업로드 + 키 동봉 (CHMO-335)",
    all(
      c.thumbnail_s3_key == f"thumbnails/events/event-1/{c.cluster_id}.jpg" and c.thumbnail_s3_key in thumb_store.blobs
      for c in result.clusters
    ),
  )

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
  check(
    "merge: 생존 앨범 썸네일 유지(덮어쓰기) + 은퇴 앨범 썸네일 삭제",
    merged.clusters[0].thumbnail_s3_key in thumb_store.blobs
    and all(f"thumbnails/events/event-1/{cid}.jpg" not in thumb_store.blobs for cid in merged.retired_cluster_ids),
  )
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
    # img-twins(근중복 쌍 = 1인, ADR 027)는 ①과 동일하게 unmatched로 계속 표류한다
    [u.image_id for u in deleted.uncertain if u.reason == "unmatched"] == ["img-b1", "img-twins"]
    and cluster_b_id in deleted.retired_cluster_ids,
  )
  check("delete: 은퇴 앨범 썸네일 삭제", f"thumbnails/events/event-1/{cluster_b_id}.jpg" not in thumb_store.blobs)

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
  check(
    "uncertain face_bboxes: bbox 미상 행(v2 이하 .npz)은 배열에서 제외",
    _uncertain_face_boxes(auto_event, [0]) == [],
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
    # 합성 orthogonal 벡터는 두 얼굴 다 ADR-025 FP 게이트에 걸려 counted가 없다(아래 causes 체크와 같은
    # 성질) → 주 인물 자격 얼굴 0 = 빈 배열. 오검출·행인에 박스를 그리지 않는다는 CHMO-407 의도 그 자체.
    "uncertain face_bboxes: 주 인물 자격 얼굴 없는 사진(FP 게이트 전원 탈락)은 빈 배열",
    walkin.uncertain[0].face_bboxes == [],
  )
  check(
    # single_appearance는 counted(실인물) 얼굴에만 붙는다 — 합성 orthogonal 벡터는 최근접 유사도 0이라
    # ADR-025 FP 게이트에 걸려 counted 얼굴이 없다(→ max_width 0 → causes []). 실 데이터의 외톨이 실얼굴은
    # 최근접 ~0.3(> 0.185)이라 counted → single_appearance (아래 FP 게이트 off 종단 테스트로 재현).
    "uncertain causes: FP 게이트에 걸린 합성 얼굴은 counted 아님 → 원인 없음 (CHMO-404)",
    walkin.uncertain[0].causes == [],
  )
  check(
    "얼굴 폭이 .npz에 저장·왕복됨",
    store.load("event-9").face_widths == (120.0, 40.0),
  )

  # ⑯ 인물 앨범 2개+ 사진 = 단체 (CHMO-330): 뒤에 작게 찍혀 주 인물 카운트에서 빠져도, 인식된 일행이면
  # 행인이 아니다 — 서로 다른 두 인물 앨범에 속한 사진은 공용에도 노출된다 (Spring 종전 복제 규칙의 이식).
  image_source.images.update(
    {
      "img-m1.jpg": fake_image([(11, 0)]),
      "img-m2.jpg": fake_image([(11, 1)]),
      "img-s1.jpg": fake_image([(12, 0)]),
      "img-s2.jpg": fake_image([(12, 1)]),
      "img-mix.jpg": fake_image([(11, 2, 200), (12, 2, 60)]),  # 주 인물 11 + 작게 찍힌 일행 12(매칭)
    }
  )
  mix_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c10",
      "group_id": "group-1",
      "event_id": "event-10",
      "images": [
        {"image_id": "img-m1", "s3_key": "img-m1.jpg"},
        {"image_id": "img-m2", "s3_key": "img-m2.jpg"},
        {"image_id": "img-s1", "s3_key": "img-s1.jpg"},
        {"image_id": "img-s2", "s3_key": "img-s2.jpg"},
        {"image_id": "img-mix", "s3_key": "img-mix.jpg"},
      ],
    }
  )
  mixed = handlers.handle(parse_inbound_message(mix_body))
  check(
    "작게 찍힌 일행 매칭: 사진이 두 인물 앨범 모두에 소속",
    sum("img-mix" in c.image_ids for c in mixed.clusters) == 2,
  )
  check(
    "인물 앨범 2개+ 사진은 주 인물 1명이어도 공용에 노출",
    mixed.common_album == ["img-mix"],
  )

  # ⑰ 썸네일 best-effort (CHMO-335): 대표 원본 fetch 실패는 job을 죽이지 않고 해당 키만 None,
  # 렌더러 미주입(비활성)이면 전 클러스터 None — 종전 동작과 동일한 롤백 경로.
  image_source.images["img-t1.jpg"] = fake_image([(14, 0)])
  image_source.images["img-t2.jpg"] = fake_image([(14, 1)])
  thumb_setup_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c11",
      "group_id": "group-1",
      "event_id": "event-11",
      "images": [
        {"image_id": "img-t1", "s3_key": "img-t1.jpg"},
        {"image_id": "img-t2", "s3_key": "img-t2.jpg"},
      ],
    }
  )
  thumb_setup = handlers.handle(parse_inbound_message(thumb_setup_body))
  check(
    "썸네일: 신규 event에도 정상 생성",
    len(thumb_setup.clusters) == 1
    and thumb_setup.clusters[0].thumbnail_s3_key
    == f"thumbnails/events/event-11/{thumb_setup.clusters[0].cluster_id}.jpg"
    and thumb_setup.clusters[0].thumbnail_s3_key in thumb_store.blobs,
  )
  del image_source.images["img-t1.jpg"], image_source.images["img-t2.jpg"]  # 원본 소실 모사
  refetch_fail = handlers.handle(
    parse_inbound_message(
      json.dumps({"type": "delete_request", "job_id": "job-d11", "event_id": "event-11", "image_ids": ["img-없음"]})
    )
  )
  check(
    "썸네일 best-effort: 대표 원본 fetch 실패 → job은 성공, 해당 키만 None",
    refetch_fail.status == "succeeded"
    and len(refetch_fail.clusters) == 1
    and refetch_fail.clusters[0].thumbnail_s3_key is None,
  )
  image_source.images["img-t1.jpg"] = fake_image([(14, 0)])
  image_source.images["img-t2.jpg"] = fake_image([(14, 1)])
  plain_handlers = JobHandlers(store=InMemoryEmbeddingStore(), images=image_source, extract_faces=fake_extractor)
  plain = plain_handlers.handle(parse_inbound_message(thumb_setup_body))
  check(
    "썸네일 비활성 (렌더러·스토어 미주입): 전 클러스터 None — 종전 동작",
    len(plain.clusters) == 1 and plain.clusters[0].thumbnail_s3_key is None,
  )

  # ⑱ 오검출 머릿수 게이트 (ADR 025): 인물 사진에 주 인물 크기의 오검출(모두와 바닥 유사도)이 붙어도
  # 단체 사진이 아니다 — 인물 앨범에만 소속, 공용 노출 없음. event 93 퍼 후드 셀피 회귀 고정.
  image_source.images.update(
    {
      "img-f1.jpg": fake_image([(15, 0)]),
      "img-f2.jpg": fake_image([(15, 1)]),
      "img-fur.jpg": fake_image([(15, 2, 200), (50, 0, 150)]),  # 인물 15 + 주 인물 크기 오검출(인물 50)
    }
  )
  fur_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c12",
      "group_id": "group-1",
      "event_id": "event-12",
      "images": [
        {"image_id": "img-f1", "s3_key": "img-f1.jpg"},
        {"image_id": "img-f2", "s3_key": "img-f2.jpg"},
        {"image_id": "img-fur", "s3_key": "img-fur.jpg"},
      ],
    }
  )
  fur = handlers.handle(parse_inbound_message(fur_body))
  check(
    "오검출 머릿수 게이트: 인물+오검출 사진은 단체 아님 — 인물 앨범만, 공용·uncertain 없음",
    sum("img-fur" in c.image_ids for c in fur.clusters) == 1 and fur.common_album == [] and fur.uncertain == [],
  )
  gate_off_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    cluster_config=ClusterConfig(common_face_min_similarity=0.0),
  )
  gate_off = gate_off_handlers.handle(parse_inbound_message(fur_body))
  check(
    "오검출 머릿수 게이트 비활성(0): 오검출이 주 인물로 세어져 공용 노출 재현 (검증의 적대적 성질 유지)",
    gate_off.common_album == ["img-fur"],
  )

  # ⑲ 이중 검출 머릿수 붕괴 (ADR 027): 초대형 얼굴의 파편 박스 2행은 cannot-link 안전판 덕에 인물
  # 앨범에는 한 사람으로 들어가면서, 머릿수로는 2명으로 세어져 1인 셀피가 공용에 노출되던 회귀 고정
  # (event 105). 붕괴는 라우팅 전용 — .npz 행·군집은 불변이다.
  image_source.images.update(
    {
      "img-dd1.jpg": fake_image([(3, 0)]),
      "img-dd2.jpg": fake_image([(3, 3)]),
      "img-ddup.jpg": fake_image([(3, 1, 120), (3, 2, 110)]),  # 같은 인물 파편 2박스 — 유사도 ≈0.996
    }
  )
  dup_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c13",
      "group_id": "group-1",
      "event_id": "event-13",
      "images": [
        {"image_id": "img-dd1", "s3_key": "img-dd1.jpg"},
        {"image_id": "img-dd2", "s3_key": "img-dd2.jpg"},
        {"image_id": "img-ddup", "s3_key": "img-ddup.jpg"},
      ],
    }
  )
  dup = handlers.handle(parse_inbound_message(dup_body))
  check(
    "이중 검출 붕괴: 파편 2행 사진은 단체 아님 — 인물 앨범만, 공용·uncertain 없음, .npz 행은 2개 유지",
    sum("img-ddup" in c.image_ids for c in dup.clusters) == 1
    and dup.common_album == []
    and dup.uncertain == []
    and store.load("event-13").photo_ids.count("img-ddup") == 2,
  )
  dup_off_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    cluster_config=ClusterConfig(common_duplicate_face_similarity=0.0),
  )
  dup_off = dup_off_handlers.handle(parse_inbound_message(dup_body))
  check(
    "이중 검출 붕괴 비활성(0): 파편 2행이 2명으로 세어져 공용 노출 재현 (검증의 적대적 성질 유지)",
    dup_off.common_album == ["img-ddup"],
  )

  # ⑳ 매칭 사진의 미매칭 주 인물 → uncertain 동시 노출 (계약 확장, 결정 2026-07-21): 2명 인식 사진에서
  # 한 명만 매칭되면 종전엔 인물 앨범(+공용)에만 실려, 미등록 인물을 "분류가 어려워요 → 인물 앨범 편입"
  # (__uncertain__ reassign)으로 수동 구제할 진입점이 없었다. 주 인물 미매칭 얼굴이 남은 사진은 인물·공용과
  # 중복으로 uncertain에도 싣는다 — 행인(주 인물 크기 미달) 미매칭은 종전대로 숨긴다.
  image_source.images.update(
    {
      "img-x1.jpg": fake_image([(16, 0)]),
      "img-x2.jpg": fake_image([(16, 1)]),
      "img-mixu.jpg": fake_image([(16, 2, 200), (17, 0, 180)]),  # 인물 16 + 미등록 주 인물 17(미매칭)
      "img-mixw.jpg": fake_image([(16, 3, 200), (18, 0, 60)]),  # 인물 16 + 행인 18(60 < 200×0.5)
    }
  )
  mixu_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c14",
      "group_id": "group-1",
      "event_id": "event-14",
      "images": [
        {"image_id": "img-x1", "s3_key": "img-x1.jpg"},
        {"image_id": "img-x2", "s3_key": "img-x2.jpg"},
        {"image_id": "img-mixu", "s3_key": "img-mixu.jpg"},
        {"image_id": "img-mixw", "s3_key": "img-mixw.jpg"},
      ],
    }
  )
  mixu = handlers.handle(parse_inbound_message(mixu_body))
  check(
    "미매칭 주 인물 동시 노출: 인물 앨범 + 공용 + uncertain(unmatched), bbox는 미매칭 얼굴",
    sum("img-mixu" in c.image_ids for c in mixu.clusters) == 1
    and "img-mixu" in mixu.common_album
    and [(u.image_id, u.reason) for u in mixu.uncertain] == [("img-mixu", "unmatched")]
    and mixu.uncertain[0].face_bboxes == [FaceBox(x=0, y=0, w=180, h=180)],
  )
  check(
    "미매칭 행인은 종전대로 숨김: 인물 앨범만 — 공용·uncertain 없음",
    sum("img-mixw" in c.image_ids for c in mixu.clusters) == 1
    and "img-mixw" not in mixu.common_album
    and all(u.image_id != "img-mixw" for u in mixu.uncertain),
  )
  legacy_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    cluster_config=ClusterConfig(unmatched_main_to_uncertain=False),
  )
  legacy = legacy_handlers.handle(parse_inbound_message(mixu_body))
  check(
    "동시 노출 비활성(False): 매칭 얼굴 있는 사진은 uncertain 제외 — 구 정책 재현 (검증의 적대적 성질 유지)",
    legacy.uncertain == [],
  )

  # ㉑ uncertain 품질 원인 (CHMO-404): 저해상도(긴 변 < 2000)로 주 얼굴이 작게(< 100px) 잡힌 미매칭 사진은
  # uncertain에 low_resolution·small_faces 원인을 실어, 앱이 "원본으로 다시 올리세요" 안내를 띄우게 한다.
  # fake_image는 shape (2,16,3)이라 긴 변 16 < 2000 — 작은 얼굴이면 저해상도 분기가 자동으로 잡힌다.
  image_source.images["img-lowres.jpg"] = fake_image([(13, 7, 80)])  # 미등록 1인, 작은 얼굴(80px < 100)
  lowres_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-404",
      "group_id": "group-1",
      "event_id": "event-404",
      "images": [{"image_id": "img-lowres", "s3_key": "img-lowres.jpg"}],
    }
  )
  lowres = handlers.handle(parse_inbound_message(lowres_body))
  check(
    "uncertain 품질 원인: 저해상도+작은 얼굴 → causes=[low_resolution, small_faces]",
    [(u.image_id, u.reason, u.causes) for u in lowres.uncertain]
    == [("img-lowres", "unmatched", ["low_resolution", "small_faces"])],
  )
  check("이미지 긴 변이 .npz에 저장·왕복됨 (v4)", store.load("event-404").image_long_sides == (16.0,))
  check(
    "uncertain causes 판정 분기 (single_appearance·품질·ambiguous·폭미상)",
    handlers._uncertain_causes(120.0, 16.0, "unmatched") == ["single_appearance"]  # 선명+매칭없음 → 한 번 등장
    and handlers._uncertain_causes(120.0, 16.0, "ambiguous") == []  # 선명하지만 두 인물 사이 → 원인 없음
    and handlers._uncertain_causes(80.0, 16.0, "unmatched") == ["low_resolution", "small_faces"]  # 작은 얼굴+저해상도
    and handlers._uncertain_causes(80.0, 4000.0, "unmatched") == ["small_faces"]  # 작은 얼굴+고해상도 → 재업로드 무효
    and handlers._uncertain_causes(80.0, 0.0, "unmatched")
    == ["small_faces"]  # 긴 변 미상(v3 이하) → low_resolution 미판정
    and handlers._uncertain_causes(0.0, 16.0, "unmatched") == [],  # 폭 미상(v1) → 판단 근거 없음
  )
  disabled_causes = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    cluster_config=ClusterConfig(uncertain_small_face_px=0.0),
  )
  check(
    "uncertain causes 비활성(small_face_px=0): 항상 빈 배열 (롤백 스위치)",
    disabled_causes._uncertain_causes(80.0, 16.0, "unmatched") == []
    and disabled_causes._uncertain_causes(120.0, 16.0, "unmatched") == [],
  )
  # single_appearance 종단: 선명한 대형 얼굴 1인이 한 번만 등장하면 앨범이 안 생긴다(2장+ 필요) — 화질은
  # 정상이라 재업로드가 아니라 "더 나오면 자동 앨범/직접 지정" 안내다. counted 실인물 얼굴에만 붙으므로
  # FP 게이트를 끄고(실 데이터 외톨이 실얼굴의 counted 상태 재현) 검증한다.
  solo_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    cluster_config=ClusterConfig(common_face_min_similarity=0.0),  # FP 게이트 off = 모든 얼굴 counted
  )
  image_source.images["img-solo.jpg"] = fake_image([(14, 3, 200)])  # 선명한 대형 얼굴(200px) 1인, 한 번 등장
  solo_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-solo",
      "group_id": "group-1",
      "event_id": "event-solo",
      "images": [{"image_id": "img-solo", "s3_key": "img-solo.jpg"}],
    }
  )
  solo = solo_handlers.handle(parse_inbound_message(solo_body))
  check(
    "single_appearance 종단: 선명한 대형 얼굴 1인(한 번 등장) → causes=[single_appearance]",
    [(u.image_id, u.reason, u.causes) for u in solo.uncertain] == [("img-solo", "unmatched", ["single_appearance"])],
  )

  # ㉒ uncertain face_bboxes 배열 (계약 교체 CHMO-407, BE#107): 주 인물 자격 uncertain 얼굴이 여럿이면
  # 전부 싣는다(폭 내림차순). 자격 2개+가 나오는 유일한 경로는 "매칭 사진 + 미매칭 주 인물 복수"다 —
  # 비매칭 사진은 주 인물 2명+이면 단체 규칙으로 공용에 빠져 uncertain에 못 들어온다(⑳ 경로 재사용).
  image_source.images.update(
    {
      "img-y1.jpg": fake_image([(16, 4)]),
      "img-y2.jpg": fake_image([(16, 5)]),
      # 인물 16(매칭, 220) + 미등록 주 인물 17·18(180·150 ≥ 220×0.5=110) + 행인 19(60 < 110, counted지만 폭 미달)
      "img-multi.jpg": fake_image([(16, 6, 220), (17, 2, 180), (18, 1, 150), (19, 0, 60)]),
    }
  )
  multi_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-c15",
      "group_id": "group-1",
      "event_id": "event-15",
      "images": [
        {"image_id": "img-y1", "s3_key": "img-y1.jpg"},
        {"image_id": "img-y2", "s3_key": "img-y2.jpg"},
        {"image_id": "img-multi", "s3_key": "img-multi.jpg"},
      ],
    }
  )
  multi = handlers.handle(parse_inbound_message(multi_body))
  check(
    # 배열 길이 2가 곧 행인 제외 검증이다 — 19도 counted(stranger 유사도 ≈0.3 > FP 바닥)라 크기 게이트만이 거른다
    "uncertain face_bboxes: 미매칭 주 인물 2명 전부, 폭 내림차순 — 행인은 제외",
    [(u.image_id, u.reason) for u in multi.uncertain] == [("img-multi", "unmatched")]
    and multi.uncertain[0].face_bboxes == [FaceBox(x=0, y=0, w=180, h=180), FaceBox(x=0, y=0, w=150, h=150)],
  )
  tie_event = EventEmbeddings.empty().append_faces(
    [
      ("f-t1", "p-tie", base_vec, 100.0, (0.0, 0.0, 100.0, 100.0)),
      ("f-t2", "p-tie", other_vec, 100.0, (10.0, 0.0, 100.0, 100.0)),
    ]
  )
  check(
    # 입력을 역순으로 줘 정렬이 실제로 개입함을 보장 — 폭 동률은 event 행 순이 계약이다
    "uncertain face_bboxes: 폭 동률은 event 행 순 — 입력 순서 무관",
    _uncertain_face_boxes(tie_event, (1, 0)) == [FaceBox(x=0, y=0, w=100, h=100), FaceBox(x=10, y=0, w=100, h=100)],
  )

  # ㉓ Rekognition 재판정 (ADR-030): AuraFace로는 회수 불가한 하드케이스(유사도 0.295) 미배정 얼굴을
  # CompareFaces 점수로 자동 편입(≥90)·제안(85~90)한다. 편입은 must-link 기록 + 재군집 2차 패스,
  # 점수는 (face, 대표) 쌍으로 캐싱돼 재군집·보정에서 재과금이 없다.
  from app.storage.rekognition_scores import InMemoryRekognitionScoreStore

  image_source.images.update(
    {
      # step 간격 3(15°, 동일인 쌍 cos≈0.967)로 근중복 붕괴 임계(0.985, ADR-029) 아래의 실사진 대역을
      # 모사한다 — 간격 1(cos≈0.996)이면 ⓪ 붕괴가 앨범 쌍을 대표 1행으로 접어, 편입 must-link가 2차
      # 패스에서 클러스터를 만들 때 고아 근중복 승격이 꺼져 무관한 앨범이 와해된다 (confirm_distinct_vector
      # 의 12° 지터와 같은 이유).
      "rj-a1.jpg": fake_image([(4, 0)]),
      "rj-a2.jpg": fake_image([(4, 3, 150)]),  # 폭 최대 — 앨범 A의 재판정 대표
      "rj-b1.jpg": fake_image([(5, 0)]),
      "rj-b2.jpg": fake_image([(5, 3)]),
      "rj-u-join.jpg": fake_image([(60, 0)]),  # 하드케이스 미등록 — 99점 → A 편입
      "rj-u-sugg.jpg": fake_image([(61, 0)]),  # 하드케이스 미등록 — 88점 → A 제안
      # cannot-link 케이스는 전 얼굴과 직교(garbage 대역)로 둔다 — 0.3 성분 대역이면 HDBSCAN이 일단
      # A에 붙였다가 cannot-link 강제 분리 + 제약 당사자 보호로 단독 앨범이 되어 uncertain을 벗어난다
      "rj-u-cl.jpg": fake_image([(50, 5)]),  # 미등록 — 99점이지만 사용자 cannot-link 탈락
    }
  )
  # 페이크 crop 태그 = "p{인물}-{step}" — comparer가 어떤 쌍인지 식별하는 배선 (fetch는 실제 image_source 경유)
  rj_crop = lambda image, bbox: f"p{int(image[0, 1, 0])}-{int(image[0, 1, 1])}".encode()  # noqa: E731
  rj_scores_by_pair = {
    ("p60-0", "p4-3"): 99.0,
    ("p61-0", "p4-3"): 88.0,
  }
  rj_calls: list[tuple[str, str]] = []
  rj_raises: set[tuple[str, str]] = set()

  def rj_compare(source: bytes, target: bytes) -> float:
    key = (source.decode(), target.decode())
    rj_calls.append(key)
    if key in rj_raises:
      raise RuntimeError("throttled")
    return rj_scores_by_pair.get(key, 5.0)

  rj_score_store = InMemoryRekognitionScoreStore()
  rj_store = InMemoryEmbeddingStore()
  rj_handlers = JobHandlers(
    store=rj_store,
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("rj-person"),
    new_face_id=counter("rj-face"),
    rejudger=UncertainRejudger(compare=rj_compare, fetch_image=image_source.fetch, crop=rj_crop),
    rejudge_scores=rj_score_store,
  )
  rj_body = {
    "type": "classify_request",
    "job_id": "job-rj1",
    "group_id": "group-1",
    "event_id": "event-rj",
    "images": [
      {"image_id": "rj-a1", "s3_key": "rj-a1.jpg"},
      {"image_id": "rj-a2", "s3_key": "rj-a2.jpg"},
      {"image_id": "rj-b1", "s3_key": "rj-b1.jpg"},
      {"image_id": "rj-b2", "s3_key": "rj-b2.jpg"},
      {"image_id": "rj-u-join", "s3_key": "rj-u-join.jpg"},
      {"image_id": "rj-u-sugg", "s3_key": "rj-u-sugg.jpg"},
    ],
  }
  rj1 = rj_handlers.handle(parse_inbound_message(json.dumps(rj_body)))
  rj_cluster_a = next(c for c in rj1.clusters if "rj-a1" in c.image_ids)
  check(
    "재판정 자동 편입: 99점 하드케이스가 앨범 A에 편입 + uncertain에서 제거",
    "rj-u-join" in rj_cluster_a.image_ids and all(u.image_id != "rj-u-join" for u in rj1.uncertain),
  )
  rj_event = rj_store.load("event-rj")
  rj_face_of = dict(zip(rj_event.photo_ids, rj_event.face_ids))
  check(
    "재판정 편입은 (얼굴, 대표) soft-attach로 .npz에 기록 — 이후 재군집에서 유지되는 근거 (ADR-030 개정)",
    rj_event.soft_attach_pairs == ((rj_face_of["rj-u-join"], rj_face_of["rj-a2"]),)
    and rj_event.must_link_pairs == ()
    and dict(zip(rj_event.photo_ids, rj_event.cluster_ids))["rj-u-join"] == rj_cluster_a.cluster_id,
  )
  rj_sugg = next(u for u in rj1.uncertain if u.image_id == "rj-u-sugg")
  check(
    "재판정 제안 [85, 90): suggestions 동봉 + face_bbox가 face_bboxes 원소와 동일 값 (결속 계약)",
    [(s.cluster_id, s.similarity) for s in rj_sugg.suggestions] == [(rj_cluster_a.cluster_id, 88.0)]
    and rj_sugg.suggestions[0].face_bbox == rj_sugg.face_bboxes[0],
  )
  check(
    "재판정 호출 정산: 미배정 2명 × top-k 2앨범 = 4회 + 점수 캐시 저장",
    len(rj_calls) == 4 and len(rj_score_store.load("event-rj")) == 4,
  )

  # 같은 event 재분류(재업로드·보정 재군집 모사): 편입 얼굴은 uncertain에 재진입하지 않고,
  # 남은 미배정의 쌍은 캐시 히트 — CompareFaces 추가 호출 0회 (재과금 방지의 종단 검증)
  rj2 = rj_handlers.handle(parse_inbound_message(json.dumps({**rj_body, "job_id": "job-rj2"})))
  check(
    "재판정 재실행: 편입 유지 + 제안 재현 + 추가 호출 0회 (캐시 전량 히트)",
    "rj-u-join" in next(c for c in rj2.clusters if "rj-a1" in c.image_ids).image_ids
    and any(u.image_id == "rj-u-sugg" and len(u.suggestions) == 1 for u in rj2.uncertain)
    and len(rj_calls) == 4,
  )

  # 사용자 cannot-link 존중 종단: 99점이어도 사용자가 "다른 사람"으로 확정한 앨범엔 편입하지 않는다.
  # u-cl 첫 분류는 일시 장애로 판정 보류(미캐싱)시켜 두고, cannot-link를 기록한 뒤 99점을 노출한다.
  rj_raises.update({("p50-5", "p4-3"), ("p50-5", "p5-0")})
  rj3 = rj_handlers.handle(
    parse_inbound_message(
      json.dumps(
        {
          **rj_body,
          "job_id": "job-rj3",
          "images": [*rj_body["images"], {"image_id": "rj-u-cl", "s3_key": "rj-u-cl.jpg"}],
        }
      )
    )
  )
  check(
    "재판정 일시 장애: 해당 얼굴만 판정 보류(uncertain 유지) + 실패 쌍 미캐싱",
    any(u.image_id == "rj-u-cl" for u in rj3.uncertain)
    and all(pair[0] != rj_face_of.get("rj-u-cl", "") for pair in rj_score_store.load("event-rj")),
  )
  rj_raises.clear()
  rj_scores_by_pair[("p50-5", "p4-3")] = 99.0
  rj_event = rj_store.load("event-rj")
  rj_face_of = dict(zip(rj_event.photo_ids, rj_event.face_ids))
  rj_store.save(  # 사용자 결정 모사: u-cl ↔ 앨범 A 멤버 cannot-link (split/confirm_distinct 번역 결과와 동일 형태)
    "event-rj",
    rj_event.with_constraints(
      rj_event.must_link_pairs, rj_event.cannot_link_pairs + ((rj_face_of["rj-u-cl"], rj_face_of["rj-a1"]),)
    ),
  )
  rj4 = rj_handlers.handle(
    parse_inbound_message(
      json.dumps(
        {
          **rj_body,
          "job_id": "job-rj4",
          "images": [*rj_body["images"], {"image_id": "rj-u-cl", "s3_key": "rj-u-cl.jpg"}],
        }
      )
    )
  )
  check(
    "재판정 cannot-link 존중: 99점이어도 편입·제안 없음 — uncertain 유지 (사용자 결정 우선)",
    any(u.image_id == "rj-u-cl" and u.suggestions == [] for u in rj4.uncertain)
    and all("rj-u-cl" not in c.image_ids for c in rj4.clusters),
  )

  # 전체 삭제 시 점수 캐시도 삭제 (생체 파생 정보 위생)
  rj_handlers.handle(
    parse_inbound_message(
      json.dumps(
        {
          "type": "delete_request",
          "job_id": "job-rj-del",
          "event_id": "event-rj",
          "image_ids": ["rj-a1", "rj-a2", "rj-b1", "rj-b2", "rj-u-join", "rj-u-sugg", "rj-u-cl"],
        }
      )
    )
  )
  check(
    "전체 delete: 재판정 점수 캐시 함께 삭제", rj_score_store.blobs == {} and rj_store.load("event-rj").face_ids == ()
  )

  # 재판정 장애 = 종전 동작: comparer가 전 쌍에서 던져도 결과가 비활성(미주입) 실행과 완전히 같다
  def rj_always_raise(source: bytes, target: bytes) -> float:
    raise RuntimeError("rekognition down")

  rj_err_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("rj2-person"),
    new_face_id=counter("rj2-face"),
    rejudger=UncertainRejudger(compare=rj_always_raise, fetch_image=image_source.fetch, crop=rj_crop),
    rejudge_scores=InMemoryRekognitionScoreStore(),
  )
  rj_off_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("rj2-person"),
    new_face_id=counter("rj2-face"),
  )
  check(
    "재판정 전면 장애: 결과가 비활성 실행과 동일 (best-effort — 현 동작 유지)",
    rj_err_handlers.handle(parse_inbound_message(json.dumps(rj_body)))
    == rj_off_handlers.handle(parse_inbound_message(json.dumps(rj_body))),
  )

  # 콜드 핸들러 + 웜 캐시: 다른 워커/재기동이 캐시만 물려받아도 호출 0회로 같은 판정 (재과금 방지)
  pre_calls: list[tuple[bytes, bytes]] = []

  def pre_compare(source: bytes, target: bytes) -> float:
    pre_calls.append((source, target))
    return 5.0

  pre_score_store = InMemoryRekognitionScoreStore()
  pre_score_store.save(
    "event-rj",
    {
      ("pre-face-5", "pre-face-2"): 99.0,  # u-join ↔ A 대표 (append 순서 기반 결정적 face id)
      ("pre-face-5", "pre-face-3"): 5.0,
      ("pre-face-6", "pre-face-2"): 88.0,
      ("pre-face-6", "pre-face-3"): 5.0,
    },
  )
  pre_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("pre-person"),
    new_face_id=counter("pre-face"),
    rejudger=UncertainRejudger(compare=pre_compare, fetch_image=image_source.fetch, crop=rj_crop),
    rejudge_scores=pre_score_store,
  )
  pre_result = pre_handlers.handle(parse_inbound_message(json.dumps(rj_body)))
  check(
    "웜 캐시 콜드 시작: 편입·제안 재현 + CompareFaces 호출 0회",
    "rj-u-join" in next(c for c in pre_result.clusters if "rj-a1" in c.image_ids).image_ids
    and any(u.image_id == "rj-u-sugg" and len(u.suggestions) == 1 for u in pre_result.uncertain)
    and pre_calls == [],
  )

  # ㉔ Rekognition 앨범 쌍 병합 재판정 (ADR-031): 같은 인물이 두 앨범으로 갈라진 것을 회수한다.
  # 인물 40(0°)·41(62°)은 centroid cos≈0.47 — 회색지대(≥probe_floor 0.35)지만 로컬 병합 임계(0.55)
  # 미달이라 파편병합이 절대 붙이지 못하는, 실측이 말한 바로 그 대역이다.
  from app.pipeline.rejudge import ClusterPairRejudger

  image_source.images.update(
    {
      "pm-a1.jpg": fake_image([(40, 0)]),
      "pm-a2.jpg": fake_image([(40, 1)]),
      "pm-b1.jpg": fake_image([(41, 0)]),
      "pm-b2.jpg": fake_image([(41, 1)]),
    }
  )
  pm_calls: list[tuple[str, str]] = []

  def pm_compare(source: bytes, target: bytes) -> float:
    pm_calls.append((source.decode(), target.decode()))
    return 99.0  # 전원 합의 통과 대역 (산포 0)

  def pm_body(job_id: str) -> str:
    return json.dumps(
      {
        "type": "classify_request",
        "job_id": job_id,
        "group_id": "group-1",
        "event_id": "event-pm",
        "images": [
          {"image_id": "pm-a1", "s3_key": "pm-a1.jpg"},
          {"image_id": "pm-a2", "s3_key": "pm-a2.jpg"},
          {"image_id": "pm-b1", "s3_key": "pm-b1.jpg"},
          {"image_id": "pm-b2", "s3_key": "pm-b2.jpg"},
        ],
      }
    )

  def pm_handlers_of(store, *, apply_merges: bool, compare=pm_compare) -> JobHandlers:
    return JobHandlers(
      store=store,
      images=image_source,
      extract_faces=fake_extractor,
      new_cluster_id=counter("pm-person"),
      new_face_id=counter("pm-face"),
      pair_rejudger=ClusterPairRejudger(compare=compare, fetch_image=image_source.fetch, crop=rj_crop),
      rejudge_scores=InMemoryRekognitionScoreStore(),
      apply_pair_merges=apply_merges,
    )

  pm_off = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("pm-person"),
    new_face_id=counter("pm-face"),
  ).handle(parse_inbound_message(pm_body("job-pm0")))
  check("(㉔) 전제: 회색지대 두 앨범은 로컬 게이트로 병합되지 않는다", len(pm_off.clusters) == 2)

  pm_calls.clear()
  pm_observe_store = InMemoryEmbeddingStore()
  pm_observe = pm_handlers_of(pm_observe_store, apply_merges=False).handle(parse_inbound_message(pm_body("job-pm1")))
  check(
    "(㉔) 관측 모드(APPLY=false): 판정은 하되(호출 4회) 파티션·.npz는 그대로",
    len(pm_observe.clusters) == 2 and len(pm_calls) == 4 and pm_observe_store.load("event-pm").soft_merge_pairs == (),
  )

  pm_calls.clear()
  pm_store = InMemoryEmbeddingStore()
  pm_handlers = pm_handlers_of(pm_store, apply_merges=True)
  pm1 = pm_handlers.handle(parse_inbound_message(pm_body("job-pm2")))
  pm_event = pm_store.load("event-pm")
  pm_face_of = dict(zip(pm_event.photo_ids, pm_event.face_ids))
  check(
    "(㉔) 앨범 쌍 병합: 회색지대 두 앨범이 한 인물로 합쳐진다 (전원 합의 99점)",
    len(pm1.clusters) == 1 and {"pm-a1", "pm-a2", "pm-b1", "pm-b2"} == set(pm1.clusters[0].image_ids),
  )
  check(
    "(㉔) 병합은 (대표, 대표) soft-merge로 .npz에 기록 — must-link는 쓰지 않는다 (ADR-031 ③)",
    pm_event.soft_merge_pairs == ((pm_face_of["pm-a1"], pm_face_of["pm-b1"]),) and pm_event.must_link_pairs == (),
  )

  pm_calls.clear()
  pm2 = pm_handlers.handle(parse_inbound_message(pm_body("job-pm3")))
  check(
    "(㉔) 재분류: 저장된 soft-merge로 병합 유지 + 캐시 전량 히트로 추가 호출 0회",
    len(pm2.clusters) == 1 and pm_calls == [],
  )

  # 사용자 split이 병합을 되돌린다 — 사람 결정이 기계 증거에 우선 (두 앨범 사이 cannot-link가 union을 막는다)
  pm_split = pm_handlers.handle(
    parse_inbound_message(
      json.dumps(
        {
          "type": "cluster_feedback",
          "job_id": "job-pm4",
          "event_id": "event-pm",
          "action": "split",
          "split": {
            "cluster_id": pm2.clusters[0].cluster_id,
            "groups": [["pm-a1", "pm-a2"], ["pm-b1", "pm-b2"]],
          },
        }
      )
    )
  )
  check("(㉔) 사용자 split이 병합을 되돌림 (soft-merge는 cannot-link를 못 덮는다)", len(pm_split.clusters) == 2)

  def pm_always_raise(source: bytes, target: bytes) -> float:
    raise RuntimeError("rekognition down")

  pm_broken = pm_handlers_of(InMemoryEmbeddingStore(), apply_merges=True, compare=pm_always_raise).handle(
    parse_inbound_message(pm_body("job-pm5"))
  )
  check(
    "(㉔) 앨범 쌍 재판정 전면 장애: 비활성 실행과 동일 결과 (best-effort)",
    [sorted(c.image_ids) for c in pm_broken.clusters] == [sorted(c.image_ids) for c in pm_off.clusters],
  )

  # ㉕ 비인간 얼굴 게이트 (ADR-032): 인형·조형물이 앨범을 만들지도, uncertain으로 새지도 않는다.
  # 인물 9 = 인형(Doll 99·DetectFaces 0), 인물 10 = 실제 아이, 인물 11 = 조형물(Sculpture 92.5).
  from app.pipeline.rejudge import NonhumanFaceGate
  from app.storage.nonhuman_verdicts import InMemoryNonhumanVerdictStore

  image_source.images.update(
    {
      "nh-d1.jpg": fake_image([(9, 0)]),
      "nh-d2.jpg": fake_image([(9, 1)]),
      "nh-d3.jpg": fake_image([(9, 2)]),
      "nh-k1.jpg": fake_image([(10, 0)]),
      "nh-k2.jpg": fake_image([(10, 1)]),
      "nh-kd.jpg": fake_image([(10, 2), (9, 3)]),  # 아이 1명 + 인형 1개 한 장
      "nh-s1.jpg": fake_image([(11, 0)]),
      "nh-s2.jpg": fake_image([(11, 1)]),
    }
  )
  NH_LABELS: dict[str, dict[str, float]] = {  # 페이크 crop 태그("p{인물}-{step}") → DetectLabels 응답
    **{f"p9-{k}": {"Doll": 99.0, "Toy": 98.0, "Person": 95.1} for k in range(4)},  # 인형의 Person 95.1 실측 재현
    **{f"p10-{k}": {"Person": 99.0} for k in range(3)},
    **{f"p11-{k}": {"Sculpture": 92.5} for k in range(2)},
  }
  NH_FACES: dict[str, int] = {f"p9-{k}": 0 for k in range(4)}  # 인형 crop은 Rekognition도 얼굴 0개
  nh_label_calls: list[str] = []
  nh_gate_down = {"flag": False}

  def nh_detect_labels(crop: bytes) -> dict[str, float]:
    if nh_gate_down["flag"]:
      raise RuntimeError("rekognition down")
    tag = crop.decode()
    nh_label_calls.append(tag)
    return NH_LABELS.get(tag, {})

  def nh_count_faces(crop: bytes) -> int:
    return NH_FACES.get(crop.decode(), 1)

  def nh_gate() -> NonhumanFaceGate:
    return NonhumanFaceGate(
      detect_labels=nh_detect_labels, count_faces=nh_count_faces, fetch_image=image_source.fetch, crop=rj_crop
    )

  nh_verdict_store = InMemoryNonhumanVerdictStore()
  nh_store = InMemoryEmbeddingStore()
  nh_handlers = JobHandlers(
    store=nh_store,
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("nh-person"),
    new_face_id=counter("nh-face"),
    nonhuman_gate=nh_gate(),
    nonhuman_verdicts=nh_verdict_store,
  )
  NH_BASE = ["nh-d1", "nh-d2", "nh-d3", "nh-k1", "nh-k2", "nh-kd"]

  def nh_body(job_id: str, image_ids: Sequence[str]) -> str:
    return json.dumps(
      {
        "type": "classify_request",
        "job_id": job_id,
        "group_id": "group-1",
        "event_id": "event-nh",
        "images": [{"image_id": name, "s3_key": f"{name}.jpg"} for name in image_ids],
      }
    )

  nh1 = nh_handlers.handle(parse_inbound_message(nh_body("job-nh1", NH_BASE)))
  check(
    "(㉕) 인형 3장: 1차 패스 앨범이 강등으로 소멸 + 세 사진 공용 앨범행 + npz에 강등 4건(인형 4얼굴) 기록",
    all("nh-d1" not in c.image_ids for c in nh1.clusters)
    and {"nh-d1", "nh-d2", "nh-d3"} <= set(nh1.common_album)
    and all(not u.image_id.startswith("nh-d") for u in nh1.uncertain)
    and len(nh_store.load("event-nh").nonhuman_face_ids) == 4,
  )
  check(
    "(㉕) 아이 1명 + 인형 1개: 아이 앨범 유지 + 인형이 머릿수에서 빠져 '단체'로 공용에 노출되지 않음",
    any({"nh-k1", "nh-k2", "nh-kd"} == set(c.image_ids) for c in nh1.clusters)
    and "nh-kd" not in nh1.common_album
    and all(u.image_id != "nh-kd" for u in nh1.uncertain),
  )

  nh_calls_before = len(nh_label_calls)
  nh2 = nh_handlers.handle(parse_inbound_message(nh_body("job-nh2", NH_BASE)))
  check(
    "(㉕) 재분류: 강등 행이 재군집 입력에서 빠져 Rekognition 추가 호출 0회 + 라우팅 재현",
    len(nh_label_calls) == nh_calls_before and {"nh-d1", "nh-d2", "nh-d3"} <= set(nh2.common_album),
  )

  # 게이트 배포 전(또는 장애 중) 형성된 조형물 앨범 모사 — 장애 플래그로 판정만 무력화해 앨범을 심는다
  nh_gate_down["flag"] = True
  nh_handlers.handle(parse_inbound_message(nh_body("job-nh-pre", [*NH_BASE, "nh-s1", "nh-s2"])))
  nh_gate_down["flag"] = False
  nh3 = nh_handlers.handle(parse_inbound_message(nh_body("job-nh3", [*NH_BASE, "nh-s1", "nh-s2"])))
  check(
    "(㉕) 승계 앨범은 판정하지 않음: 이미 형성된 조형물 앨범은 유지 + 호출 0회 (소급 정리는 범위 밖)",
    any({"nh-s1", "nh-s2"} == set(c.image_ids) for c in nh3.clusters)
    and all(not tag.startswith("p11") for tag in nh_label_calls),
  )

  # 삭제로 조형물 앨범이 1장이 되면 남은 얼굴이 미배정으로 떨어져 게이트가 잡는다 → 앨범 은퇴 통보
  nh_sculpt_id = next(c.cluster_id for c in nh3.clusters if "nh-s1" in c.image_ids)
  nh_del = nh_handlers.handle(
    parse_inbound_message(
      json.dumps({"type": "delete_request", "job_id": "job-nh-del1", "event_id": "event-nh", "image_ids": ["nh-s2"]})
    )
  )
  check(
    "(㉕) 강등이 기존 앨범을 비우면 retired 통보 + 남은 조형물 사진은 공용 앨범 (규칙 B)",
    nh_sculpt_id in nh_del.retired_cluster_ids and "nh-s1" in nh_del.common_album,
  )

  # 사용자 보정이 강등 얼굴을 건드리면 영구 복구된다 — reassign(nh-d1을 아이 앨범으로)
  nh_event = nh_store.load("event-nh")
  nh_face_of = dict(zip(nh_event.photo_ids, nh_event.face_ids))
  nh_kid_id = next(c.cluster_id for c in nh3.clusters if "nh-k1" in c.image_ids)
  nh_fb = nh_handlers.handle(
    parse_inbound_message(
      json.dumps(
        {
          "type": "cluster_feedback",
          "job_id": "job-nh-fb",
          "event_id": "event-nh",
          "action": "reassign",
          "reassign": {"image_id": "nh-d1", "from_cluster_id": UNCERTAIN_ALBUM_ID, "to_cluster_id": nh_kid_id},
        }
      )
    )
  )
  check(
    "(㉕) 사용자 보정이 강등을 해제: nonhuman 목록에서 제거 + 다음 재군집에서 얼굴이 되살아남",
    "nh-d1" in next(c for c in nh_fb.clusters if "nh-k1" in c.image_ids).image_ids
    and nh_face_of["nh-d1"] not in nh_store.load("event-nh").nonhuman_face_ids,
  )
  nh4 = nh_handlers.handle(parse_inbound_message(nh_body("job-nh4", [*NH_BASE, "nh-s1"])))
  check(
    "(㉕) 되살린 얼굴은 재강등되지 않음 (보정 당사자 면제 — must-link가 판정 자체를 생략)",
    "nh-d1" in next(c for c in nh4.clusters if "nh-k1" in c.image_ids).image_ids
    and nh_face_of["nh-d1"] not in nh_store.load("event-nh").nonhuman_face_ids,
  )

  # 게이트 전면 장애 = 비활성과 완전히 같은 결과 (best-effort 격리)
  nh_gate_down["flag"] = True
  nh_err = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("nhe-person"),
    new_face_id=counter("nhe-face"),
    nonhuman_gate=nh_gate(),
    nonhuman_verdicts=InMemoryNonhumanVerdictStore(),
  ).handle(parse_inbound_message(nh_body("job-nh-e", NH_BASE)))
  nh_gate_down["flag"] = False
  nh_off = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("nhe-person"),
    new_face_id=counter("nhe-face"),
  ).handle(parse_inbound_message(nh_body("job-nh-e", NH_BASE)))
  check("(㉕) 게이트 전면 장애: 비활성 실행과 동일 결과 (best-effort — 현 동작 유지)", nh_err == nh_off)

  # 전체 delete 시 비인간 판정 캐시도 삭제 (생체 파생 정보 위생)
  nh_handlers.handle(
    parse_inbound_message(
      json.dumps(
        {
          "type": "delete_request",
          "job_id": "job-nh-del2",
          "event_id": "event-nh",
          "image_ids": [*NH_BASE, "nh-s1", "nh-s2"],
        }
      )
    )
  )
  check(
    "(㉕) 전체 delete: 비인간 판정 캐시 함께 삭제",
    nh_verdict_store.blobs == {} and nh_store.load("event-nh").face_ids == (),
  )

  # 순서 계약: 게이트가 재판정보다 먼저라, 강등된 인형 앨범은 CompareFaces 대표 후보에서 배제된다
  # (실 event 139에서 인형 대표와의 비교 4쌍이 전부 버려진 낭비의 재발 방지)
  image_source.images.update(
    {
      "nh8-a1.jpg": fake_image([(12, 0)]),
      "nh8-a2.jpg": fake_image([(12, 1, 150)]),  # 폭 최대 — 아이 앨범의 재판정 대표
      "nh8-d1.jpg": fake_image([(13, 0)]),
      "nh8-d2.jpg": fake_image([(13, 1, 150)]),
      "nh8-u.jpg": fake_image([(63, 0)]),  # 하드케이스 미등록 — 재판정 대상
    }
  )
  NH_LABELS.update(
    {
      **{f"p12-{k}": {} for k in range(2)},
      **{f"p13-{k}": {"Doll": 99.0} for k in range(2)},
      "p63-0": {},
    }
  )
  NH_FACES.update({f"p13-{k}": 0 for k in range(2)})
  nh8_compare_calls: list[tuple[str, str]] = []

  def nh8_compare(source: bytes, target: bytes) -> float:
    nh8_compare_calls.append((source.decode(), target.decode()))
    return 5.0

  nh8_handlers = JobHandlers(
    store=InMemoryEmbeddingStore(),
    images=image_source,
    extract_faces=fake_extractor,
    new_cluster_id=counter("nh8-person"),
    new_face_id=counter("nh8-face"),
    rejudger=UncertainRejudger(compare=nh8_compare, fetch_image=image_source.fetch, crop=rj_crop),
    rejudge_scores=InMemoryRekognitionScoreStore(),
    nonhuman_gate=nh_gate(),
    nonhuman_verdicts=InMemoryNonhumanVerdictStore(),
  )
  nh8_handlers.handle(
    parse_inbound_message(
      json.dumps(
        {
          "type": "classify_request",
          "job_id": "job-nh8",
          "group_id": "group-1",
          "event_id": "event-nh8",
          "images": [
            {"image_id": name, "s3_key": f"{name}.jpg"} for name in ["nh8-a1", "nh8-a2", "nh8-d1", "nh8-d2", "nh8-u"]
          ],
        }
      )
    )
  )
  check(
    "(㉕) 게이트가 재판정보다 먼저: 강등된 인형 앨범은 CompareFaces 대표 후보에서 배제",
    all("p13" not in tag for pair in nh8_compare_calls for tag in pair)
    and any(pair[1] == "p12-1" for pair in nh8_compare_calls),
  )

  print(f"\n스모크 검증 {passed}건 전부 통과")
