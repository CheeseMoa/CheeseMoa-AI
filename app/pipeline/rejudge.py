"""Rekognition 보조 판정 3종 — 얼굴 편입(ADR-030)·앨범 쌍 병합(ADR-031)·비인간 얼굴 게이트(ADR-032).

**얼굴 단위 (`UncertainRejudger`, ADR-030 · 2026-07-23 실측 리뷰)**

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

**앨범 쌍 단위 (`ClusterPairRejudger`, ADR-031 · 2026-07-24 실측 리뷰)**

같은 인물이 두 앨범으로 갈라지는 문제(실측 34개 이벤트 중 12개)는 위 훅으로 닿지 않는다 — 얼굴 단위
재판정은 미배정 얼굴만 보므로 앨범 A·B가 둘 다 형성되면 실행조차 안 된다. 로컬 게이트가 거절한
회색지대 앨범 쌍(centroid ≥ probe_floor)을 앨범당 대표 K장씩 K×K로 비교해 **전원이 merge_similarity
이상이고 산포가 max_spread 이하**일 때만 병합한다:

  - 대표 1쌍 argmax는 실측 기각 — 확실한 타인 쌍이 `[99.99, 33.74, 1.96, 1.56]`이라 앨범 두 개가
    통째로 융합됐다. 전원 합의로 잃는 회수량은 1쌍뿐이고 그마저 오병합으로 확정된 쌍이었다.
  - 같은 사진 공존(ADR-011)·사용자 cannot-link 쌍은 **점수를 묻기 전에** 탈락한다 — 공짜 정밀도이자
    비용 절감이다.
  - 3개 이상이 합쳐질 땐 컴포넌트 완전 연결을 요구한다 — A–B·B–C만 보고 합치면 아무도 검사하지 않은
    A–C가 남아 체인 융합(이 레포에서 가장 많이 데인 실패 모드)이 된다.

**비인간 얼굴 게이트 (`NonhumanFaceGate`, ADR-032 · 2026-07-24 실측 스윕)**

인형·조형물 얼굴은 YuNet score(0.89~0.92)가 실얼굴보다 오히려 높아 로컬 신호로는 원리적으로 못
거른다(2026-07-20 조사 전면 기각) — 인형 3장만으로 앨범이 만들어지고(실 event 139) 조형물이
uncertain으로 샌다. 얼굴 crop에 `DetectLabels` 1콜을 물어 **2신호 AND**로만 강등한다:
`Sculpture|Statue ≥ 70`이면 즉시 강등(규칙 B), `Doll|Toy ≥ 90`이면 `DetectFaces` 2콜째로 확인해
얼굴 0개일 때만 강등(규칙 A). `DetectFaces` 미검출 단독 규칙은 금지다 — 실제 아이 4/17(24%)을
오거부한다(실측). 강등 = npz `nonhuman_face_ids` 기록 → 재군집 입력 제외(반영은 handlers).

세 판정기는 crop 렌더·행 단위 crop 캐시를 `_RekognitionProbe`로 공유하고, CompareFaces 쌍 점수
캐시·예산 소비는 두 재판정기만의 부품(`_CompareFacesProbe`)이다 — 같은 event의 두 재판정이 한
점수 dict를 쓰므로 겹치는 쌍은 한 번만 과금된다. 재판정 반영은 둘 다 재군집의 응집 게이트 이후
단계다(`Constraints.soft_attach` / `soft_merge`) — must-link 강제는 대상 앨범을 흔든다(실 event
134 회귀).

이 모듈은 boto3·S3를 모른다: Rekognition 호출(FaceComparer·LabelDetector·FaceCounter)·원본
fetch·crop 렌더는 전부 콜러블 주입이다(handlers.FaceExtractor와 같은 구도, 합성은 core.deps의
책임). 덕분에 스모크가 AWS 없이 돈다. 판정 메서드들은 절대 raise하지 않는 계약이 아니다 —
best-effort 격리(실패가 job을 죽이지 않게)는 handlers의 몫이고, 여기서는 건 단위 실패만 로그 후
건너뛴다.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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

# crop JPEG → DetectLabels 응답 {레이블 이름: 신뢰도 0~100} (ADR-032). MinConfidence 필터는 deps
# 클로저가 소비한다. 스로틀·권한 없음 등 실패는 raise가 계약 — 게이트가 그 얼굴만 판정 보류한다.
LabelDetector = Callable[[bytes], "dict[str, float]"]

# crop JPEG → DetectFaces FaceDetails 개수 (ADR-032 규칙 A의 2콜째). 실패는 raise가 계약.
FaceCounter = Callable[[bytes], int]


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


@dataclass(frozen=True)
class PairRejudgeConfig:
  """앨범 쌍 재판정의 임계·상한 — 기본값은 2026-07-24 실측 리뷰의 확정값 (Settings.to_pair_rejudge_config).

  임계 근거: 통과 13쌍은 전부 98점 이상이라 90과 95의 결과가 같다 — 실제 방어선은 임계가 아니라
  K×K 전원 합의다. 산포는 통과쌍 0.0~1.9(중앙 0.2) vs 대조군 15.0~98.4로 갈려 비용 0의 2차 안전판이다.
  """

  probe_floor: float = 0.35  # 이 centroid 미만 앨범 쌍은 묻지 않는다 — 정확도가 아니라 비용 통제용 바닥
  merge_similarity: float = 90.0  # K×K 전원이 이 값 이상이어야 병합 (얼굴 단위 편입과 같은 값·더 엄격한 조건)
  max_spread: float = 5.0  # K×K 산포(max−min) 상한 — 타인 오답은 한두 쌍만 튀어오른다
  reps: int = 2  # 앨범당 대표 수 K (쌍당 K² 호출). K=1은 실측 기각 (argmax 융합)
  max_calls: int = 200  # job당 호출 상한 — 앨범 쌍 수는 인물 수의 제곱으로 는다
  same_face_similarity: float = 0.95  # 이 값 이상 닮은 같은 사진 얼굴은 타인 증거가 아니라 YuNet 파편 박스

  def __post_init__(self) -> None:
    if not (0.0 <= self.probe_floor <= 1.0):
      raise ValueError(f"probe_floor는 [0, 1] 범위여야 합니다. 받은 값: {self.probe_floor}")
    if not (0.0 < self.merge_similarity <= 100.0):
      raise ValueError(f"merge_similarity는 (0, 100] 범위여야 합니다. 받은 값: {self.merge_similarity}")
    if not (0.0 <= self.max_spread <= 100.0):
      raise ValueError(f"max_spread는 [0, 100] 범위여야 합니다. 받은 값: {self.max_spread}")
    if self.reps < 1:
      raise ValueError(f"reps는 1 이상이어야 합니다. 받은 값: {self.reps}")
    if self.max_calls < 1:
      raise ValueError(f"max_calls는 1 이상이어야 합니다. 받은 값: {self.max_calls}")
    if not (0.0 < self.same_face_similarity <= 1.0):
      raise ValueError(f"same_face_similarity는 (0, 1] 범위여야 합니다. 받은 값: {self.same_face_similarity}")


@dataclass(frozen=True)
class ClusterPairMerge:
  """앨범 쌍 병합 판정 1건 — 핸들러가 (rep_a, rep_b)를 soft_merge로 기록한다."""

  cluster_ids: tuple[str, str]  # 판정 시점(1차 패스) id — 최종 id는 재군집의 Jaccard 승계가 정한다
  rep_face_ids: tuple[str, str]  # 각 앨범의 1순위 대표 (face_id 사전순)
  min_similarity: float
  max_similarity: float


@dataclass(frozen=True)
class PairRejudgeOutcome:
  """cluster pair judge 1회 실행의 결과 — scores는 (기존 캐시 + 신규 측정) 병합본 (저장은 핸들러의 책임)."""

  merges: tuple[ClusterPairMerge, ...] = ()
  scores: dict[tuple[str, str], float] | None = None
  calls_made: int = 0
  truncated: bool = False  # max_calls 도달로 일부 앨범 쌍을 판정하지 못함

  @property
  def soft_merge_pairs(self) -> tuple[tuple[str, str], ...]:
    """병합 판정을 재군집 제약(Constraints.soft_merge)의 원천인 face_id 쌍으로 낸다."""
    return tuple(merge.rep_face_ids for merge in self.merges)


# 판정 규칙의 관심 레이블 (ADR-032 실측 확정 — 바꾸면 근거가 무효). `Art`·`Person`·`Painting`은
# 쓰지 않는다: 흐린 실제 아이가 `Art` 55.1을 받고 인형이 `Person` 95.1을 받는다(오판 축).
_DOLL_LABELS = ("Doll", "Toy")
_SCULPTURE_LABELS = ("Sculpture", "Statue")


@dataclass(frozen=True)
class NonhumanConfig:
  """비인간 얼굴 게이트의 임계·상한 — 기본값은 2026-07-24 실측 스윕의 확정값 (Settings.to_nonhuman_config).

  임계 근거: 인형 3종 `Doll` 96.9~99.7 vs 실인물 최고 오탐 대역 없음(0/54), 조형물 2종
  `Sculpture` 77.1~92.5 vs 실인물 0/55. `DetectFaces` 미검출 단독은 실제 아이 4/17 오거부라 금지 —
  `Doll|Toy` 레이블과의 AND(규칙 A)만 허용된다.
  """

  doll_confidence: float = 90.0  # Doll|Toy 임계 — 이상이면 DetectFaces 2콜째로 확인 (규칙 A)
  sculpture_confidence: float = 70.0  # Sculpture|Statue 임계 — 이상이면 즉시 강등 (규칙 B, 2콜 없음)
  label_min_confidence: float = 50.0  # DetectLabels MinConfidence — deps의 detect_labels 클로저가 소비
  cluster_probe_faces: int = 3  # 신규 앨범 후보당 판정 얼굴 수 (폭 내림차순) — 과반 규칙의 분모 상한
  max_calls: int = 100  # job당 Rekognition 호출 상한 (DetectLabels+DetectFaces 합산) — 비용 폭주 안전판

  def __post_init__(self) -> None:
    for name, value in (
      ("doll_confidence", self.doll_confidence),
      ("sculpture_confidence", self.sculpture_confidence),
    ):
      if not (0.0 < value <= 100.0):
        raise ValueError(f"{name}는 (0, 100] 범위여야 합니다. 받은 값: {value}")
    if not (0.0 <= self.label_min_confidence <= 100.0):
      raise ValueError(f"label_min_confidence는 [0, 100] 범위여야 합니다. 받은 값: {self.label_min_confidence}")
    if self.cluster_probe_faces < 1:
      raise ValueError(f"cluster_probe_faces는 1 이상이어야 합니다. 받은 값: {self.cluster_probe_faces}")
    if self.max_calls < 1:
      raise ValueError(f"max_calls는 1 이상이어야 합니다. 받은 값: {self.max_calls}")


@dataclass(frozen=True)
class NonhumanVerdict:
  """얼굴 1개의 비인간 판정 — 통과(nonhuman=False)도 판정이다(캐싱돼 재호출을 막는다).

  n_faces는 DetectFaces FaceDetails 개수, -1 = 미호출(규칙 B 강등·레이블 미달 통과는 2콜째가
  없다). labels는 DetectLabels 응답 전체 — 오판 사후 분석(ADR-032 §롤아웃 감시)의 근거.
  """

  face_id: str
  nonhuman: bool
  rule: str = ""  # "doll" | "sculpture" | "" (통과)
  labels: "dict[str, float]" = field(default_factory=dict)
  n_faces: int = -1


@dataclass(frozen=True)
class NonhumanOutcome:
  """비인간 게이트 1회 실행의 결과 — verdicts는 (기존 캐시 + 신규 측정) 병합본 (저장은 핸들러의 책임)."""

  demoted_face_ids: tuple[str, ...] = ()
  verdicts: "dict[str, NonhumanVerdict] | None" = None
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


class _CallBudget:
  """job당 CompareFaces 호출 예산 — 소진 시 이후 측정은 판정 보류가 되고 truncated로 보고된다."""

  def __init__(self, limit: int) -> None:
    self.limit = limit
    self.made = 0
    self.truncated = False

  def exhausted(self) -> bool:
    return self.made >= self.limit


def _cluster_representatives(event: "EventEmbeddings", cluster: "PersonCluster", count: int) -> tuple[int, ...]:
  """앨범 대표 행을 폭 최대 count개 고른다 (bbox·원본 키 유효 행만) — 실측 리뷰들과 파리티가 계약이다.

  썸네일 대표(_select_representative의 membership 최대)와 의도적으로 다르다: 임계 90/85/95의 근거
  데이터가 전부 '폭 최대' 대표로 측정됐다. 동률은 앞 행이 이긴다(결정성 — 같은 멤버십이면 같은 대표).
  """
  eligible = [i for i in cluster.member_indices if event.bboxes[i][2] > 0 and event.s3_keys[i]]
  return tuple(sorted(eligible, key=lambda i: (-event.face_widths[i], i))[:count])


class _RekognitionProbe:
  """Rekognition 판정의 공통 부품 — 원본 fetch·crop 렌더·행 단위 crop 캐시 (세 판정기가 상속).

  이 계층은 boto3·S3를 모른다: 원본 fetch·crop 렌더는 전부 콜러블 주입이다
  (합성은 core.deps의 책임). 덕분에 스모크가 AWS 없이 돈다.
  """

  def __init__(self, fetch_image: ImageFetcher, crop: CropRenderer) -> None:
    self._fetch_image = fetch_image
    self._crop = crop

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


class _CompareFacesProbe(_RekognitionProbe):
  """CompareFaces 채점 부품 — crop 부품 위에 쌍 점수 캐시·예산 소비를 얹는다 (두 재판정기가 상속).

  비인간 게이트(NonhumanFaceGate)는 CompareFaces를 쓰지 않으므로 이 계층에 불참한다 (ADR-032).
  """

  def __init__(self, compare: FaceComparer, fetch_image: ImageFetcher, crop: CropRenderer) -> None:
    super().__init__(fetch_image, crop)
    self._compare = compare

  def _pair_score(
    self,
    event: "EventEmbeddings",
    source_row: int,
    target_row: int,
    scores: dict[tuple[str, str], float],
    crop_cache: dict[int, "bytes | None"],
    budget: _CallBudget,
  ) -> float | None:
    """(source, target) 쌍의 Similarity 1건 — 캐시 우선, 없으면 예산 안에서 측정한다.

    확보 실패(예산 소진·crop 실패·일시 장애)는 전부 None이고, 호출자는 그 판정을 보류한다.
    일시 장애는 캐싱하지 않아 다음 재군집에서 재시도된다 — "얼굴 미검출" 센티널(-1.0)만이 캐싱돼
    영구 재과금을 막는다(그 구분은 FaceComparer 구현의 계약).
    """
    pair = (event.face_ids[source_row], event.face_ids[target_row])
    if pair in scores:
      return scores[pair]
    if budget.exhausted():
      budget.truncated = True
      return None
    source = self._crop_jpeg(event, source_row, crop_cache)
    target = self._crop_jpeg(event, target_row, crop_cache)
    if source is None or target is None:
      return None
    try:
      budget.made += 1
      scores[pair] = float(self._compare(source, target))
    except Exception:
      logger.warning("CompareFaces 호출 실패 pair=%s — 이 쌍 건너뜀", pair, exc_info=True)
      return None
    return scores[pair]


class UncertainRejudger(_CompareFacesProbe):
  """uncertain 얼굴 × top-k 앨범 대표의 CompareFaces 재판정기 — 순수 판정 로직 (저장·발행은 handlers)."""

  def __init__(
    self,
    compare: FaceComparer,
    fetch_image: ImageFetcher,
    crop: CropRenderer,
    config: RejudgeConfig | None = None,
  ) -> None:
    super().__init__(compare, fetch_image, crop)
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
    budget = _CallBudget(config.max_calls)
    crop_cache: dict[int, bytes | None] = {}

    # 앨범 대표 = 멤버 중 폭 최대 얼굴 1장 (_cluster_representatives 독스트링 — 실측과 파리티)
    reps: list[tuple["PersonCluster", int]] = []
    for cluster in clusters:
      selected = _cluster_representatives(event, cluster, 1)
      if selected:
        reps.append((cluster, selected[0]))
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
        # 방향은 (미배정 얼굴 → 대표) 고정 — 캐시 키가 재군집마다 흔들리지 않게 (재과금 방지)
        similarity = self._pair_score(event, row, rep_row, scores, crop_cache, budget)
        if similarity is None:
          complete = False
          continue
        face_scores.append((similarity, cluster, rep_row))

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
    if assignments or suggestions or fragment_hints or budget.made:
      logger.info(
        "Rekognition 재판정: 편입 %d건, 제안 %d건, 파편 힌트 %d건 (호출 %d회%s)",
        len(assignments),
        len(suggestions),
        len(fragment_hints),
        budget.made,
        ", 호출 상한 도달로 일부 보류" if budget.truncated else "",
      )
    return RejudgeOutcome(
      assignments=assignments,
      suggestions=tuple(suggestions),
      fragment_hints=tuple(fragment_hints),
      scores=scores,
      calls_made=budget.made,
      truncated=budget.truncated,
    )

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


class ClusterPairRejudger(_CompareFacesProbe):
  """회색지대 앨범 쌍의 CompareFaces 병합 재판정기 (ADR-031) — 순수 판정 로직 (반영·저장은 handlers).

  로컬 게이트(파편 병합)가 거절한 앨범 쌍만 본다: 통과했다면 이미 합쳐졌을 것이고, 확정적 반증
  (같은 사진 공존·사용자 cannot-link)이 있는 쌍은 점수를 묻지 않고 탈락시킨다. 판정은 대표 K×K
  전원 합의이며, 3개 이상이 한 앨범으로 합쳐질 때는 컴포넌트 완전 연결까지 요구한다.
  """

  def __init__(
    self,
    compare: FaceComparer,
    fetch_image: ImageFetcher,
    crop: CropRenderer,
    config: PairRejudgeConfig | None = None,
  ) -> None:
    super().__init__(compare, fetch_image, crop)
    self._config = config if config is not None else PairRejudgeConfig()

  def judge(
    self,
    event: "EventEmbeddings",
    clusters: Sequence["PersonCluster"],
    cached_scores: dict[tuple[str, str], float],
  ) -> PairRejudgeOutcome:
    """앨범 쌍들을 재판정한다 — 결정적: 같은 (event, clusters, 캐시)면 같은 결과.

    후보는 centroid 유사도 내림차순으로 처리하고, 각 쌍은 K×K 점수가 전부 확보돼야 판정한다
    (일부만 확보되면 그 쌍은 미달 — 부분 점수 판정 금지는 얼굴 단위와 같은 원칙). 확보된 점수는
    scores에 남아 다음 재군집에서 이어 판정한다.
    """
    config = self._config
    scores = dict(cached_scores)
    budget = _CallBudget(config.max_calls)
    crop_cache: dict[int, bytes | None] = {}
    if len(clusters) < 2:
      return PairRejudgeOutcome(scores=scores)

    reps = {cluster.cluster_id: _cluster_representatives(event, cluster, config.reps) for cluster in clusters}

    # ① 후보 선정 — 회색지대만, 확정적 반증은 호출 전에 탈락 (ADR-031 ①)
    candidates: list[tuple[float, str, str]] = []  # (-centroid 유사도, cluster_id, cluster_id) 사전순 동률 처리
    for i, cluster_a in enumerate(clusters):
      for cluster_b in clusters[i + 1 :]:
        key = self._pair_key(cluster_a.cluster_id, cluster_b.cluster_id)
        similarity = float(cluster_a.centroid @ cluster_b.centroid)
        if similarity < config.probe_floor:
          continue  # 비용 통제 바닥 미만 — 묻지 않는다 (완전 연결 검사에서는 '미달'로 취급된다)
        if len(reps[cluster_a.cluster_id]) < config.reps or len(reps[cluster_b.cluster_id]) < config.reps:
          logger.info("앨범 쌍 재판정 탈락(대표 부족): %s ↔ %s", *key)
          continue  # 대표 K장을 못 채우는 앨범(원본 소실·v2 이하 행) — 전원 합의를 성립시킬 수 없다
        if self._same_photo_conflict(event, cluster_a, cluster_b):
          logger.info("앨범 쌍 재판정 탈락(같은 사진 공존 = 확정적 타인): %s ↔ %s — 호출 없음", *key)
          continue
        if self._user_separated(event, cluster_a, cluster_b):
          logger.info("앨범 쌍 재판정 탈락(사용자 cannot-link): %s ↔ %s — 호출 없음", *key)
          continue
        candidates.append((-similarity, *key))

    # ② 판정 — 대표 K×K 전원 합의 + 산포 (ADR-031 ②)
    verdicts: dict[tuple[str, str], bool] = {}
    stats: dict[tuple[str, str], tuple[float, float]] = {}  # 통과 쌍의 (min, max) — 로그·결과용
    for negative_similarity, id_a, id_b in sorted(candidates):
      pair_scores = self._score_grid(event, reps[id_a], reps[id_b], scores, crop_cache, budget)
      if pair_scores is None:
        verdicts[(id_a, id_b)] = False  # 점수 미확보 — 판정 보류 = 미달 (다음 job에서 캐시로 이어감)
        logger.info("앨범 쌍 재판정 보류(점수 미확보): %s ↔ %s", id_a, id_b)
        continue
      low, high = min(pair_scores), max(pair_scores)
      passed = low >= config.merge_similarity and (high - low) <= config.max_spread
      verdicts[(id_a, id_b)] = passed
      stats[(id_a, id_b)] = (low, high)
      logger.info(
        "앨범 쌍 재판정: %s ↔ %s centroid=%.3f 점수=%s → %s",
        id_a,
        id_b,
        -negative_similarity,
        [round(value, 2) for value in pair_scores],
        "병합" if passed else f"기각(최저 {low:.1f}, 산포 {high - low:.1f})",
      )

    # ②-b 완전 연결 — 합쳐질 컴포넌트 안의 모든 쌍이 통과해야 한다 (체인 융합 차단)
    components = self._link_components(candidates, verdicts)

    merges = tuple(
      ClusterPairMerge(
        cluster_ids=(id_a, id_b),
        rep_face_ids=self._pair_key(event.face_ids[reps[id_a][0]], event.face_ids[reps[id_b][0]]),
        min_similarity=stats[(id_a, id_b)][0],
        max_similarity=stats[(id_a, id_b)][1],
      )
      for component in components
      for i, id_a in enumerate(component)
      for id_b in component[i + 1 :]
    )
    if merges or budget.made:
      logger.info(
        "Rekognition 앨범 쌍 재판정: 병합 %d쌍 (앨범 %d개 → %d묶음, 호출 %d회%s)",
        len(merges),
        sum(len(component) for component in components),
        len(components),
        budget.made,
        ", 호출 상한 도달로 일부 보류" if budget.truncated else "",
      )
    return PairRejudgeOutcome(merges=merges, scores=scores, calls_made=budget.made, truncated=budget.truncated)

  @staticmethod
  def _pair_key(a: str, b: str) -> tuple[str, str]:
    """두 id를 사전순으로 정규화한다 — 판정 키와 점수 캐시 키가 앨범 나열 순서에 흔들리지 않게.

    캐시 키가 재군집마다 뒤집히면 같은 대표 쌍을 다시 과금한다("대표가 바뀌지 않는 한 0원"이 깨진다).
    """
    return (a, b) if a <= b else (b, a)

  def _score_grid(
    self,
    event: "EventEmbeddings",
    reps_a: Sequence[int],
    reps_b: Sequence[int],
    scores: dict[tuple[str, str], float],
    crop_cache: dict[int, "bytes | None"],
    budget: _CallBudget,
  ) -> list[float] | None:
    """대표 K×K 점수 전량. 한 쌍이라도 확보 실패면 None (부분 점수로는 판정하지 않는다)."""
    grid: list[float] = []
    for row_a in reps_a:
      for row_b in reps_b:
        source, target = (row_a, row_b) if event.face_ids[row_a] <= event.face_ids[row_b] else (row_b, row_a)
        similarity = self._pair_score(event, source, target, scores, crop_cache, budget)
        if similarity is None:
          return None
        grid.append(similarity)
    return grid

  def _same_photo_conflict(
    self, event: "EventEmbeddings", cluster_a: "PersonCluster", cluster_b: "PersonCluster"
  ) -> bool:
    """두 앨범에 같은 사진의 (파편이 아닌) 얼굴이 각각 있는가 = 확정적으로 다른 사람 (ADR-011).

    같은 사진에 같은 인물이 두 번 나올 수 없다. 이중 검출 파편(유사도 ≥ same_face_similarity)은
    타인 증거가 아니라 YuNet이 한 얼굴에 그린 박스 두 개이므로 예외다.
    """
    rows_by_photo: dict[str, list[int]] = {}
    for row in cluster_a.member_indices:
      rows_by_photo.setdefault(event.photo_ids[row], []).append(row)
    for row_b in cluster_b.member_indices:
      for row_a in rows_by_photo.get(event.photo_ids[row_b], ()):
        if float(event.embeddings[row_a] @ event.embeddings[row_b]) < self._config.same_face_similarity:
          return True
    return False

  @staticmethod
  def _user_separated(event: "EventEmbeddings", cluster_a: "PersonCluster", cluster_b: "PersonCluster") -> bool:
    """사용자가 split으로 갈라놓은 앨범 쌍인가 — 점수와 무관하게 탈락(사람 결정이 기계 증거에 우선)."""
    faces_a = {event.face_ids[row] for row in cluster_a.member_indices}
    faces_b = {event.face_ids[row] for row in cluster_b.member_indices}
    return any((a in faces_a and b in faces_b) or (b in faces_a and a in faces_b) for a, b in event.cannot_link_pairs)

  def _link_components(
    self, candidates: Sequence[tuple[float, str, str]], verdicts: dict[tuple[str, str], bool]
  ) -> list[tuple[str, ...]]:
    """통과 쌍을 유사도 내림차순 greedy로 합치되, 컴포넌트 완전 연결을 요구한다 (ADR-031 ②-b).

    판정은 쌍 단위지만 결과는 전이적이다: A–B와 B–C가 통과하면 A·B·C가 한 앨범이 되는데 A–C는
    아무도 검사하지 않았다. 이 레포에서 가장 많이 데인 실패 모드가 정확히 이 체인 융합이라
    (event 105의 5그룹 융합, ADR-024), 두 컴포넌트를 합치기 전 교차하는 모든 앨범 쌍이 통과인지
    확인하고 하나라도 미달·미측정(probe_floor 미만이라 후보가 아니었던 경우 포함)이면 건너뛴다.
    반환: 크기 2 이상인 컴포넌트들의 cluster_id 튜플(내부 사전순, 컴포넌트 간에도 결정적 순서).
    """
    members: dict[str, list[str]] = {}
    root_of: dict[str, str] = {}
    for _, id_a, id_b in candidates:
      for cluster_id in (id_a, id_b):
        root_of.setdefault(cluster_id, cluster_id)
        members.setdefault(cluster_id, [cluster_id])
    for _, id_a, id_b in sorted(candidates):
      if not verdicts.get((id_a, id_b)):
        continue
      root_a, root_b = root_of[id_a], root_of[id_b]
      if root_a == root_b:
        continue
      if not all(verdicts.get(self._pair_key(x, y)) for x in members[root_a] for y in members[root_b]):
        logger.info("앨범 쌍 병합 보류(컴포넌트 완전 연결 미달 — 체인 융합 차단): %s ↔ %s", id_a, id_b)
        continue
      if root_b < root_a:  # 사전순 앞선 쪽이 루트 (결정성)
        root_a, root_b = root_b, root_a
      members[root_a].extend(members.pop(root_b))
      for cluster_id in members[root_a]:
        root_of[cluster_id] = root_a
    return sorted(tuple(sorted(group)) for group in members.values() if len(group) >= 2)


class NonhumanFaceGate(_RekognitionProbe):
  """인형·조형물 얼굴의 DetectLabels 2신호 판정기 (ADR-032) — 순수 판정 로직 (강등 반영·저장은 handlers).

  판정 대상은 두 가지뿐이다: ① 신규 앨범 후보(`is_new`) — 폭 상위 대표들을 판정해 과반이
  비인간이면 클러스터 전 멤버 강등(작고 흐린 실얼굴 1장이 앨범을 통째로 날리는 사고 방지),
  ② 미배정 얼굴(노이즈+ambiguous) — 단건 판정. 승계 앨범은 이미 사람 앨범으로 확정된
  역사(과거 재군집 통과)가 있어 판정하지 않는다 — 호출 0회.

  사람 결정이 기계 증거에 우선한다: 얼굴 자신 또는 그 클러스터 멤버가 must/cannot-link에
  등장하면 판정 자체를 생략한다 (UncertainRejudger._passes_screens ①과 같은 철학).
  """

  def __init__(
    self,
    detect_labels: LabelDetector,
    count_faces: FaceCounter,
    fetch_image: ImageFetcher,
    crop: CropRenderer,
    config: NonhumanConfig | None = None,
  ) -> None:
    super().__init__(fetch_image, crop)
    self._detect_labels = detect_labels
    self._count_faces = count_faces
    self._config = config if config is not None else NonhumanConfig()

  def judge(
    self,
    event: "EventEmbeddings",
    clusters: Sequence["PersonCluster"],
    uncertain_indices: Sequence[int],
    cached_verdicts: "dict[str, NonhumanVerdict]",
  ) -> NonhumanOutcome:
    """얼굴들의 비인간 여부를 판정한다 — 결정적: 같은 (event, clusters, 캐시)면 같은 결과.

    캐시 히트는 호출 0회다 — 통과 판정도 캐싱돼(verdicts 병합본) 매 재군집의 재과금을 막는다.
    얼굴 단위 실패(crop 실패·일시 장애·예산 소진)는 그 얼굴만 판정 보류(강등 없음 = 종전 동작)다.
    """
    config = self._config
    verdicts: dict[str, NonhumanVerdict] = dict(cached_verdicts)
    budget = _CallBudget(config.max_calls)
    crop_cache: dict[int, bytes | None] = {}
    constrained = {face_id for pair in event.must_link_pairs + event.cannot_link_pairs for face_id in pair}
    demoted: list[str] = []

    # ① 신규 앨범 후보 — 대표 폭 상위 최대 cluster_probe_faces장 판정, 과반 규칙으로 전 멤버 강등
    for cluster in clusters:
      if not cluster.is_new:
        continue
      if any(event.face_ids[i] in constrained for i in cluster.member_indices):
        continue  # 사람이 지목한 얼굴이 낀 앨범 — 기계 증거로 뒤집지 않는다 (호출 0회)
      judged = 0
      nonhuman = 0
      for row in _cluster_representatives(event, cluster, config.cluster_probe_faces):
        verdict = self._face_verdict(event, row, verdicts, crop_cache, budget)
        if verdict is None:
          continue  # 판정 확보 실패분은 분모에서 뺀다
        judged += 1
        nonhuman += int(verdict.nonhuman)
      if judged and nonhuman * 2 > judged:  # 과반 (2개 판정이면 2개 모두 필요) — 판정 0개면 강등 없음
        demoted.extend(event.face_ids[i] for i in cluster.member_indices)
        logger.info(
          "비인간 앨범 강등: cluster_id=%s 멤버 %d명 (판정 %d 중 비인간 %d)",
          cluster.cluster_id,
          len(cluster.member_indices),
          judged,
          nonhuman,
        )

    # ② 미배정 얼굴 — uncertain 확정 직전의 노이즈+ambiguous 단건 판정
    for row in sorted(uncertain_indices):
      if event.face_ids[row] in constrained:
        continue  # 사람 결정 존중 — 보정 당사자는 판정하지 않는다 (해제 후 재강등 방지도 이 규칙)
      verdict = self._face_verdict(event, row, verdicts, crop_cache, budget)
      if verdict is not None and verdict.nonhuman:
        demoted.append(event.face_ids[row])

    if demoted or budget.made:
      logger.info(
        "비인간 얼굴 게이트: 강등 %d건 (호출 %d회%s)",
        len(demoted),
        budget.made,
        ", 호출 상한 도달로 일부 보류" if budget.truncated else "",
      )
    return NonhumanOutcome(
      demoted_face_ids=tuple(dict.fromkeys(demoted)),
      verdicts=verdicts,
      calls_made=budget.made,
      truncated=budget.truncated,
    )

  def _face_verdict(
    self,
    event: "EventEmbeddings",
    row: int,
    verdicts: "dict[str, NonhumanVerdict]",
    crop_cache: dict[int, "bytes | None"],
    budget: _CallBudget,
  ) -> NonhumanVerdict | None:
    """행 1개의 판정 — 캐시 우선, 없으면 예산 안에서 측정한다. 확보 실패(None)는 판정 보류.

    규칙 A(인형)는 레이블과 DetectFaces가 **둘 다** 확보돼야 판정한다 — 레이블만 있고 2콜째가
    실패하면 통과·강등 어느 쪽도 캐싱하지 않는다(부분 증거 판정 금지, 두 재판정기와 같은 원칙).
    """
    face_id = event.face_ids[row]
    if face_id in verdicts:
      return verdicts[face_id]
    if event.bboxes[row][2] <= 0 or not event.s3_keys[row]:
      return None  # crop 원천 미상(v2 이하 .npz 행) — 판정 불가
    crop = self._crop_jpeg(event, row, crop_cache)
    if crop is None:
      return None
    if budget.exhausted():
      budget.truncated = True
      return None
    try:
      budget.made += 1
      labels = {str(name): float(confidence) for name, confidence in self._detect_labels(crop).items()}
    except Exception:
      logger.warning("DetectLabels 호출 실패 face_id=%s — 이 얼굴 판정 보류", face_id, exc_info=True)
      return None

    config = self._config
    # 규칙 B 먼저 — 조형물은 Rekognition도 얼굴로 검출하므로 DetectFaces 확인이 무의미하다 (2콜 없음)
    if max((labels.get(name, 0.0) for name in _SCULPTURE_LABELS), default=0.0) >= config.sculpture_confidence:
      verdict = NonhumanVerdict(face_id=face_id, nonhuman=True, rule="sculpture", labels=labels)
    elif max((labels.get(name, 0.0) for name in _DOLL_LABELS), default=0.0) >= config.doll_confidence:
      # 규칙 A — Doll|Toy 단독으로는 강등하지 않는다: DetectFaces가 얼굴을 보면 인형을 든 실인물이다
      if budget.exhausted():
        budget.truncated = True
        return None
      try:
        budget.made += 1
        n_faces = int(self._count_faces(crop))
      except Exception:
        logger.warning("DetectFaces 호출 실패 face_id=%s — 이 얼굴 판정 보류", face_id, exc_info=True)
        return None
      verdict = NonhumanVerdict(
        face_id=face_id, nonhuman=n_faces == 0, rule="doll" if n_faces == 0 else "", labels=labels, n_faces=n_faces
      )
    else:
      verdict = NonhumanVerdict(face_id=face_id, nonhuman=False, labels=labels)
    if verdict.nonhuman:
      logger.info("비인간 얼굴 판정: face_id=%s rule=%s labels=%s", face_id, verdict.rule, verdict.labels)
    verdicts[face_id] = verdict
    return verdict


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

  # ── 앨범 쌍 병합 재판정 (ADR-031) ──────────────────────────────────────────
  # 행 구성: 앨범당 2행(폭 120/100 — 앞 행이 1순위 대표). s3_key = face_id라 comparer가 쌍을 식별한다.
  pair_rows = [
    ("ca1", "p-a1", axis_vector(0), 120.0, (0.0, 0.0, 120.0, 120.0), "ca1"),
    ("ca2", "p-a2", axis_vector(1), 100.0, (0.0, 0.0, 100.0, 100.0), "ca2"),
    ("cb1", "p-b1", axis_vector(2), 120.0, (0.0, 0.0, 120.0, 120.0), "cb1"),
    ("cb2", "p-b2", axis_vector(3), 100.0, (0.0, 0.0, 100.0, 100.0), "cb2"),
    ("cc1", "p-c1", axis_vector(4), 120.0, (0.0, 0.0, 120.0, 120.0), "cc1"),
    ("cc2", "p-c2", axis_vector(5), 100.0, (0.0, 0.0, 100.0, 100.0), "cc2"),
    ("cd1", "p-a1", axis_vector(6), 120.0, (0.0, 0.0, 120.0, 120.0), "cd1"),  # 앨범 A와 같은 사진(p-a1)
    ("cd2", "p-d2", axis_vector(7), 100.0, (0.0, 0.0, 100.0, 100.0), "cd2"),
    ("ce1", "p-e1", axis_vector(8), 120.0, (0.0, 0.0, 120.0, 120.0), ""),  # 원본 미상 — 대표 자격 없음
    ("ce2", "p-e2", axis_vector(9), 100.0, (0.0, 0.0, 100.0, 100.0), "ce2"),
  ]
  pair_event = EventEmbeddings.empty().append_faces(pair_rows)

  def planar(degrees: float) -> np.ndarray:
    """평면 위 단위 centroid — 두 앨범의 centroid 유사도를 각도 차의 코사인으로 정확히 지정한다."""
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    radians = np.deg2rad(degrees)
    vector[0], vector[1] = float(np.cos(radians)), float(np.sin(radians))
    return vector

  def album(cluster_id: str, member_indices: tuple[int, ...], degrees: float) -> PersonCluster:
    return PersonCluster(
      cluster_id=cluster_id,
      is_new=False,
      member_indices=member_indices,
      membership_similarities=(1.0,) * len(member_indices),
      centroid=planar(degrees),
    )

  pair_table: dict[tuple[str, str], float] = {}
  pair_calls: list[tuple[str, str]] = []

  def set_grid(id_a: str, id_b: str, values: Sequence[float]) -> None:
    """앨범 id_a × id_b의 대표 K×K(=4) 점수를 채운다 (id_a가 사전순 앞 — 호출 방향과 일치)."""
    keys = [(f"{id_a}{i}", f"{id_b}{j}") for i in (1, 2) for j in (1, 2)]
    pair_table.update(dict(zip(keys, values)))

  def pair_compare(source: bytes, target: bytes) -> float:
    key = (source.decode(), target.decode())
    pair_calls.append(key)
    return pair_table[key]

  def build_pair(config: PairRejudgeConfig | None = None) -> ClusterPairRejudger:
    return ClusterPairRejudger(
      compare=pair_compare,
      fetch_image=lambda key: np.frombuffer(key.encode(), dtype=np.uint8),
      crop=lambda image, bbox: bytes(image),
      config=config,
    )

  album_a, album_b = album("A", (0, 1), 0.0), album("B", (2, 3), 45.0)  # centroid 0.707 — 회색지대 후보
  set_grid("ca", "cb", [99.0, 98.5, 99.2, 98.8])
  pair_calls.clear()
  pair_outcome = build_pair().judge(pair_event, (album_a, album_b), {})
  check(
    "전원 합의 통과: K×K 4점 전부 ≥90·산포 ≤5 → 대표 쌍 soft_merge 1건",
    pair_outcome.merges == (ClusterPairMerge(("A", "B"), ("ca1", "cb1"), 98.5, 99.2),)
    and pair_outcome.soft_merge_pairs == (("ca1", "cb1"),)
    and pair_outcome.calls_made == 4,
  )
  check(
    "호출 방향은 face_id 사전순 고정 (앨범 나열 순서와 무관 — 캐시 재과금 방지)",
    all(source <= target for source, target in pair_calls),
  )

  cached_pair = build_pair().judge(pair_event, (album_b, album_a), pair_outcome.scores)  # 순서 뒤집어 재호출
  check(
    "캐시 전량 히트: 앨범 순서를 뒤집어도 호출 0회 + 같은 판정",
    cached_pair.calls_made == 0 and cached_pair.soft_merge_pairs == (("ca1", "cb1"),),
  )

  set_grid("ca", "cb", [99.0, 98.5, 99.2, 88.0])  # 1점만 미달
  check(
    "1개 미달 기각: 나머지가 99점이어도 병합 없음", build_pair().judge(pair_event, (album_a, album_b), {}).merges == ()
  )
  set_grid("ca", "cb", [99.0, 98.5, 99.2, 93.0])  # 전원 ≥90이지만 산포 6.2
  check(
    "산포 초과 기각: 전원 90 이상이어도 max−min > 5면 병합 없음",
    build_pair().judge(pair_event, (album_a, album_b), {}).merges == (),
  )
  set_grid("ca", "cb", [99.0, 98.5, 99.2, 98.8])  # 이후 시나리오를 위해 통과값 복원

  pair_calls.clear()
  far = build_pair().judge(pair_event, (album_a, album("B", (2, 3), 80.0)), {})  # centroid 0.17
  check("probe_floor 미만은 후보 제외 — 호출 0회", far.merges == () and pair_calls == [])

  pair_calls.clear()
  album_d = album("D", (6, 7), 10.0)  # centroid 0.985지만 A와 같은 사진(p-a1) 공존
  same_photo = build_pair().judge(pair_event, (album_a, album_d), {})
  check("같은 사진 공존은 호출 전 탈락 (확정적 타인, ADR-011) — 호출 0회", same_photo.merges == () and pair_calls == [])

  pair_calls.clear()
  split_event = pair_event.with_constraints((), (("ca1", "cb1"),))  # 사용자가 갈라놓은 앨범 쌍
  user_split = build_pair().judge(split_event, (album_a, album_b), {})
  check("사용자 cannot-link 앨범 쌍은 호출 전 탈락 — 호출 0회", user_split.merges == () and pair_calls == [])

  pair_calls.clear()
  short_reps = build_pair().judge(pair_event, (album_a, album("E", (8, 9), 45.0)), {})  # 유효 대표 1장뿐
  check("대표가 K장 미만인 앨범은 판정 불가 — 호출 0회", short_reps.merges == () and pair_calls == [])

  # 체인 융합 차단: A–B·B–C가 통과해도 A–C가 미달·미측정이면 셋이 한 앨범이 되지 않는다 (ADR-031 ②-b)
  album_c = album("C", (4, 5), 60.0)  # A와 0.5, B와 0.966
  set_grid("cb", "cc", [99.0, 99.0, 99.0, 99.0])
  set_grid("ca", "cc", [20.0, 20.0, 20.0, 20.0])  # A–C는 측정됐지만 미달
  chain = build_pair().judge(pair_event, (album_a, album_b, album_c), {})
  check(
    "체인 차단(A–C 미달): 통과 쌍 B–C만 병합, A는 합류하지 않음",
    chain.soft_merge_pairs == (("cb1", "cc1"),),
  )
  set_grid("ca", "cc", [99.0, 99.0, 99.0, 99.0])
  clique = build_pair().judge(pair_event, (album_a, album_b, album_c), {})
  check(
    "완전 연결 충족: 세 앨범 전 쌍 통과 → 컴포넌트 내 3쌍 전부 기록",
    clique.soft_merge_pairs == (("ca1", "cb1"), ("ca1", "cc1"), ("cb1", "cc1")),
  )
  album_c_far = album("C", (4, 5), 80.0)  # A와 0.17(후보 아님) · B와 0.906
  unmeasured = build_pair().judge(pair_event, (album_a, album_b, album_c_far), {})
  check(
    "체인 차단(A–C가 probe_floor 미만이라 미측정): 미달로 취급 — B–C만 병합",
    unmeasured.soft_merge_pairs == (("cb1", "cc1"),),
  )

  pair_calls.clear()
  capped = build_pair(PairRejudgeConfig(max_calls=2)).judge(pair_event, (album_a, album_b, album_c), {})
  check(
    "호출 상한: 예산 소진 시 truncated + 부분 점수로 판정하지 않음",
    capped.truncated and capped.merges == () and len(pair_calls) == 2,
  )
  check("상한 절단분도 측정분은 캐시에 남는다", len(capped.scores) == 2)

  pair_calls.clear()
  no_pair = build_pair().judge(pair_event, (album_a,), {})
  check("앨범이 1개면 판정할 쌍이 없다 — 호출 0회", no_pair.merges == () and pair_calls == [])

  # ── 비인간 얼굴 게이트 (ADR-032) ──────────────────────────────────────────
  # 행 구성: 앨범 4개(NH 인형 과반·HM 실인물 다수·OLD 승계·CON 보정 당사자) + 미배정 9행.
  # s3_key = face_id라 페이크 detect_labels/count_faces가 얼굴을 식별한다 (위 페이크들과 같은 배선).
  nh_rows = [
    ("nd1", "np-0", axis_vector(0), 120.0, (0.0, 0.0, 120.0, 120.0), "nd1"),  # NH — Doll 99.6·얼굴 0
    ("nd2", "np-1", axis_vector(1), 110.0, (0.0, 0.0, 110.0, 110.0), "nd2"),  # NH — Doll 96.9·얼굴 0
    ("nd3", "np-2", axis_vector(2), 100.0, (0.0, 0.0, 100.0, 100.0), "nd3"),  # NH — 관심 레이블 없음 (통과)
    ("nh1", "np-3", axis_vector(3), 120.0, (0.0, 0.0, 120.0, 120.0), "nh1"),  # HM — Doll 99.0·얼굴 0
    ("nh2", "np-4", axis_vector(4), 110.0, (0.0, 0.0, 110.0, 110.0), "nh2"),  # HM — 통과
    ("nh3", "np-5", axis_vector(5), 100.0, (0.0, 0.0, 100.0, 100.0), "nh3"),  # HM — 통과
    ("no1", "np-6", axis_vector(6), 120.0, (0.0, 0.0, 120.0, 120.0), "no1"),  # OLD(승계) — 판정 대상 아님
    ("nc1", "np-7", axis_vector(7), 120.0, (0.0, 0.0, 120.0, 120.0), "nc1"),  # CON — must-link 당사자 면제
    ("nc2", "np-8", axis_vector(8), 100.0, (0.0, 0.0, 100.0, 100.0), "nc2"),  # 미배정·must-link 당사자 면제
    ("ns1", "np-9", axis_vector(9), 100.0, (0.0, 0.0, 100.0, 100.0), "ns1"),  # Sculpture 77.1 → 규칙 B 강등
    ("nb1", "np-10", axis_vector(10), 100.0, (0.0, 0.0, 100.0, 100.0), "nb1"),  # Doll 89.9 → 경계 통과
    ("nb2", "np-11", axis_vector(11), 100.0, (0.0, 0.0, 100.0, 100.0), "nb2"),  # Doll 90.0·얼굴 0 → 강등
    ("nb3", "np-12", axis_vector(12), 100.0, (0.0, 0.0, 100.0, 100.0), "nb3"),  # Sculpture 69.9 → 경계 통과
    ("nb4", "np-13", axis_vector(13), 100.0, (0.0, 0.0, 100.0, 100.0), "nb4"),  # Sculpture 70.0 → 강등
    ("nr1", "np-14", axis_vector(14), 100.0, (0.0, 0.0, 100.0, 100.0), "nr1"),  # Art 55.1 흐린 아이 → 통과
    ("nf1", "np-15", axis_vector(15), 100.0, (0.0, 0.0, 100.0, 100.0), "nf1"),  # Doll 99.0·얼굴 1 → 통과
    ("nx1", "np-16", axis_vector(16), 0.0, (0.0, 0.0, 0.0, 0.0), ""),  # crop 원천 미상 — 건너뜀
  ]
  nh_event = EventEmbeddings.empty().append_faces(nh_rows)
  nh_event = nh_event.with_cluster_ids(["NH", "NH", "NH", "HM", "HM", "HM", "OLD", "CON"] + [None] * 9)
  nh_event = nh_event.with_constraints([("nc1", "nc2")], ())

  def nh_album(cluster_id: str, member_indices: tuple[int, ...], is_new: bool) -> PersonCluster:
    return PersonCluster(
      cluster_id=cluster_id,
      is_new=is_new,
      member_indices=member_indices,
      membership_similarities=(1.0,) * len(member_indices),
      centroid=axis_vector(0),
    )

  nh_clusters = (
    nh_album("NH", (0, 1, 2), True),
    nh_album("HM", (3, 4, 5), True),
    nh_album("OLD", (6,), False),
    nh_album("CON", (7,), True),
  )
  nh_uncertain = (8, 9, 10, 11, 12, 13, 14, 15, 16)

  NH_LABELS: dict[str, dict[str, float]] = {
    "nd1": {"Doll": 99.6, "Toy": 99.6, "Person": 95.1},  # 인형이 Person 95.1을 받는 실측 재현 — Person은 안 쓴다
    "nd2": {"Toy": 96.9},
    "nd3": {"Person": 99.0},
    "nh1": {"Doll": 99.0},
    "nh2": {"Person": 99.0},
    "nh3": {},
    "no1": {"Doll": 99.0},  # 승계 앨범 — 호출되면 안 되는 값
    "nc1": {"Doll": 99.0},  # 면제 — 호출되면 안 되는 값
    "nc2": {"Doll": 99.0},  # 면제 — 호출되면 안 되는 값
    "ns1": {"Sculpture": 77.1, "Art": 60.0},
    "nb1": {"Doll": 89.9},
    "nb2": {"Doll": 90.0},
    "nb3": {"Statue": 69.9},
    "nb4": {"Statue": 70.0},
    "nr1": {"Art": 55.1, "Painting": 53.2},  # 흐린 실제 아이의 실측 레이블 — Art·Painting은 판정에 안 쓴다
    "nf1": {"Doll": 99.0},  # 인형을 든 실인물 — DetectFaces가 얼굴 1개를 본다
  }
  NH_FACES: dict[str, int] = {"nd1": 0, "nd2": 0, "nh1": 0, "nb2": 0, "nf1": 1}
  nh_label_calls: list[str] = []
  nh_count_calls: list[str] = []

  def nh_detect_labels(crop: bytes) -> dict[str, float]:
    tag = crop.decode()
    nh_label_calls.append(tag)
    return NH_LABELS[tag]

  def nh_count_faces(crop: bytes) -> int:
    tag = crop.decode()
    nh_count_calls.append(tag)
    return NH_FACES[tag]

  def build_gate(config: NonhumanConfig | None = None) -> NonhumanFaceGate:
    return NonhumanFaceGate(
      detect_labels=nh_detect_labels,
      count_faces=nh_count_faces,
      fetch_image=lambda key: np.frombuffer(key.encode(), dtype=np.uint8),
      crop=lambda image, bbox: bytes(image),
      config=config,
    )

  nh_outcome = build_gate().judge(nh_event, nh_clusters, nh_uncertain, {})
  check(
    "신규 앨범 과반 강등: 3장 중 2장 인형(규칙 A) → 클러스터 전 멤버 강등",
    {"nd1", "nd2", "nd3"} <= set(nh_outcome.demoted_face_ids),
  )
  check(
    "과반 미달 유지: 3장 중 1장만 비인간 → 앨범 강등 없음",
    not ({"nh1", "nh2", "nh3"} & set(nh_outcome.demoted_face_ids)),
  )
  check("승계 앨범(is_new=False)은 판정 대상 아님 — 호출 0회", "no1" not in nh_label_calls)
  check(
    "면제: must-link 당사자(클러스터 멤버·미배정 모두)는 호출 0회",
    "nc1" not in nh_label_calls and "nc2" not in nh_label_calls,
  )
  check(
    "규칙 B: Sculpture 77.1 → 강등, DetectFaces 2콜째 없음",
    "ns1" in nh_outcome.demoted_face_ids and "ns1" not in nh_count_calls,
  )
  check(
    "경계값: Doll 89.9 통과 / 90.0 강등, Sculpture 69.9 통과 / 70.0 강등",
    "nb1" not in nh_outcome.demoted_face_ids
    and "nb2" in nh_outcome.demoted_face_ids
    and "nb3" not in nh_outcome.demoted_face_ids
    and "nb4" in nh_outcome.demoted_face_ids,
  )
  check("경계 미달 Doll(89.9)은 DetectFaces도 안 나간다", "nb1" not in nh_count_calls)
  check(
    "실인물 대조: 관심 레이블 없음·Art 55.1+Painting 53.2 전부 통과 판정 (Art·Person·Painting 미사용)",
    "nr1" not in nh_outcome.demoted_face_ids
    and nh_outcome.verdicts["nr1"].nonhuman is False
    and nh_outcome.verdicts["nd3"].nonhuman is False,  # nd3 강등은 개별 판정이 아니라 앨범 과반 규칙의 결과
  )
  check(
    "규칙 A 미발동: Doll 99.0이어도 DetectFaces가 얼굴을 보면 실인물(인형 든 아이)",
    "nf1" not in nh_outcome.demoted_face_ids and "nf1" in nh_count_calls,
  )
  check("crop 원천 미상 행은 건너뜀 (호출 없음)", "" not in nh_label_calls)
  check(
    "통과 판정도 캐시에 남는다 (재과금 방지의 핵심)",
    nh_outcome.verdicts is not None and nh_outcome.verdicts["nr1"].nonhuman is False,
  )

  nh_label_calls.clear()
  nh_count_calls.clear()
  nh_cached = build_gate().judge(nh_event, nh_clusters, nh_uncertain, nh_outcome.verdicts)
  check(
    "캐시 전량 히트: 호출 0회 + 같은 판정",
    nh_label_calls == []
    and nh_count_calls == []
    and nh_cached.calls_made == 0
    and nh_cached.demoted_face_ids == nh_outcome.demoted_face_ids,
  )

  nh_label_calls.clear()
  nh_count_calls.clear()
  nh_capped = build_gate(NonhumanConfig(max_calls=2)).judge(nh_event, nh_clusters, nh_uncertain, {})
  check(
    "호출 상한: 2콜(nd1 레이블+얼굴)로 소진 → truncated + 측정분(판정 1건)은 캐시에 잔존",
    nh_capped.truncated and nh_capped.calls_made == 2 and len(nh_capped.verdicts) == 1,
  )

  # NonhumanConfig 검증
  for name, kwargs in [
    ("doll_confidence 0", {"doll_confidence": 0.0}),
    ("sculpture_confidence 범위 밖", {"sculpture_confidence": 150.0}),
    ("label_min_confidence 음수", {"label_min_confidence": -1.0}),
    ("cluster_probe_faces 0", {"cluster_probe_faces": 0}),
    ("max_calls 0", {"max_calls": 0}),
  ]:
    try:
      NonhumanConfig(**kwargs)
    except ValueError:
      check(f"거부: NonhumanConfig {name}", True)
    else:
      raise SystemExit(f"실패: NonhumanConfig {name} — ValueError가 발생해야 하는데 통과됨")

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

  # PairRejudgeConfig 검증
  for name, kwargs in [
    ("probe_floor 범위 밖", {"probe_floor": 1.5}),
    ("merge_similarity 0", {"merge_similarity": 0.0}),
    ("max_spread 음수", {"max_spread": -1.0}),
    ("reps 0", {"reps": 0}),
    ("max_calls 0", {"max_calls": 0}),
    ("same_face 0", {"same_face_similarity": 0.0}),
  ]:
    try:
      PairRejudgeConfig(**kwargs)
    except ValueError:
      check(f"거부: PairRejudgeConfig {name}", True)
    else:
      raise SystemExit(f"실패: PairRejudgeConfig {name} — ValueError가 발생해야 하는데 통과됨")

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
