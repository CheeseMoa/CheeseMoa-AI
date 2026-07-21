"""순수 파이프라인 단계로서의 YuNet 얼굴 검출.

디코딩된 BGR 이미지를 받아 `DetectedFace` 리스트로 변환한다. S3 다운로드와 `cv2.imdecode`는
호출자의 책임이며, 이 모듈은 S3나 huggingface_hub를 직접 다루지 않는다. 모델 획득은
`ModelSource` 뒤로 완전히 캡슐화되어 있다.
"""

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.core.model_source import ModelSource, default_yunet_source

DEFAULT_SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000
DEFAULT_MAX_SIDE = 2000
# 스윕 확정값(measure_landmark_jitter.py --sweep, 2026-07-15): 폭 {128,160,192,224} × 마진 {0.3,0.5,0.75}
# 중 224/0.75가 전 지표 최고 — 같은 얼굴 유사도 평균 0.9596·최저 0.6881, 랜드마크 이동 평균 3.07%
DEFAULT_REFINE_NORM_FACE_WIDTH = 224
DEFAULT_REFINE_MARGIN_RATIO = 0.75
# 분포 측정 확정값(ADR-013, 2026-07-15): 실사진 16개 이벤트 210 얼굴에서 배경 행인 얼굴은
# rel_w(bbox 폭/이미지 긴 변) 최대 0.82%, 앨범 배정 얼굴은 최소 3.29% — 사이가 빈 구간이라
# 1.0~3.0% 어디를 골라도 앨범 손실 0(고원). 고원 상단 쪽 2.5%를 채택해 배경 노이즈 제거를
# 늘렸다(측정 데이터에서 노이즈 63/121 제거, 앨범 손실 0). 앨범 최소값까지 마진은 1.3배로
# 얇은 편 — 앨범 사진 누락 리포트가 오면 이 값부터 내려볼 것.
DEFAULT_MIN_FACE_REL_WIDTH = 0.025
# 대형 오검출 결합 필터(survey 2026-07-15): 팔짱 낀 팔 등 진짜 얼굴 크기의 오검출은 ADR-013
# 크기 필터를 통과한다. 단일 축으로는 못 거른다 — 진짜 앨범 얼굴도 score 최저 0.616(초근접
# 대형)·종횡비(w/h) 최저 0.563(기울인 셀카)까지 내려간다. 그러나 두 축이 동시에 낮은 진짜
# 얼굴은 없어(score<0.80이면 종횡비 ≥0.76), 결합 규칙이 실사진 앨범 얼굴 손실 0/114로 오검출을
# 제거한다. 이 미만 AND 종횡비 미만이면 제거.
DEFAULT_FP_SCORE_THRESHOLD = 0.78
DEFAULT_FP_ASPECT_THRESHOLD = 0.70  # 종횡비 = bbox 폭/높이 (w/h)
# 대형 근접 얼굴 재검출 회복 (ADR-017): YuNet은 WIDER FACE(작은 얼굴)로 학습돼 초근접 대형 얼굴에
# 낮은 score를 주고(실측 최저 0.22), score gate 0.6에 걸려 사라진다 → 공통 사진첩 직행. 크기(rel_w)
# 하한을 넘는 저score 후보를 "정규 크기로 축소해 재검출"하고, 재검출 score가 이 값 이상이면 되살린다.
# 실얼굴은 너무 커서 저score였을 뿐이라 정규 스케일에서 score가 오르지만(median 0.44), 진짜 오검출
# (아웃포커스 배경·신체일부)은 어느 스케일에서도 낮다(median 0.00) — score·선명도로는 안 갈리지만
# 재검출은 깨끗이 가른다(재검출≥0.80에서 실얼굴 회복·오검출 0/41, survey 2026-07-16).
# rel_w 하한 0.30→0.20 재보정(ADR-017 §재보정, 2026-07-17): 화면을 다 덮는 얼굴은 YuNet이 bbox를
# 실제보다 훨씬 작게(파편으로) 그려 rel_w 0.280·0.295로 게이트에 미달했다(event 60 등 미검출 2장).
# 게이트 스윕 실측(783장): 0.20으로 내려도 오검출 통과 0·기존 검출 손실 0 — 판별은 재검출 score가 한다.
DEFAULT_BIG_FACE_REL_WIDTH = 0.20  # 이 rel_w(bbox폭/긴변) 이상의 저score 얼굴만 재검출 회복 대상
DEFAULT_BIG_FACE_REDETECT_SCORE = 0.80  # 정규 스케일 재검출 score가 이 이상이면 실얼굴로 보고 회복
# 재검출 랜드마크 신뢰 임계 (survey_refine_shift.py, 2026-07-17): 이동량 가드(_REFINE_MAX_SHIFT_RATIO)의
# 기준이 원 bbox 폭인데, 초대형 얼굴은 YuNet이 bbox를 파편으로 그려 진짜 얼굴이 파편보다 훨씬 크다 —
# 올바른 교정도 "파편 폭 × 0.5"를 넘어 가드에 걸리고, 깨진 원 랜드마크가 유지돼 쓰레기 임베딩이 된다
# (event 73: 공통첩 유출 + 쓰레기끼리 뭉친 유령 앨범). 34개 이벤트 520장 실측에서 가드 발동 33건 중
# 재검출 score는 좋은 교정 전부 ≥0.86, 무익한 후보(양쪽 다 쓰레기) 전부 ≤0.39로 갈린다(빈 구간
# [0.39, 0.86], 오매칭 점프 0건) — 회복 임계와 같은 0.80을 채택. 재검출 score가 이 값 이상이면
# 이동량 가드를 무시하고 재검출 랜드마크를 채택한다. 0 = 비활성(종전 가드만).
DEFAULT_REFINE_TRUST_REDETECT_SCORE = 0.80
# confident 파편 디둡 (ADR-027, survey 2026-07-21): 초대형 얼굴의 파편 박스 2개가 둘 다 score gate를
# 통과하면 회복 경로가 아니라 confident로 들어와 ADR-017 디둡이 닿지 않는다 — YuNet 자체 NMS도 파편 쌍
# IoU 0.297<0.3으로 미발동, 1인 셀피가 "주 인물 2명 단체"로 세어져 공용 앨범에 노출된다(event 105).
# 원본 1,078장 전수 재검출: 진짜 파편 쌍은 정제 랜드마크 중심거리가 얼굴폭의 0.019·0.024, 실제 타인
# 겹침 쌍은 최저 0.436(뒤 인물에 겹친 옆얼굴) — 빈 구간 [0.024, 0.436]의 기하 중앙 0.1 채택. 회복
# 경로의 _DEDUP_LANDMARK_RATIO(0.5)는 그 실제 타인 쌍을 삼키므로 confident에는 별도 임계가 필요하다.
# 파편화는 정제 대상 대형 얼굴에서만 관측돼(파편 쌍 폭 1201·1079 / 535·559 vs 타인 쌍 180·162) 검사를
# 양쪽 폭 > refine_norm_face_width로 한정한다 — 위험한 소형 실얼굴 쌍은 원리적으로 제외. 0 = 비활성.
DEFAULT_CONFIDENT_DEDUP_LANDMARK_RATIO = 0.1
_BIG_FACE_MODEL_FLOOR = 0.2  # 회복 활성 시 YuNet 모델 score 바닥 — 저score 대형 후보를 표면화 (실측 최저 0.22 포섭)
_YUNET_INIT_SIZE = (320, 320)  # 초기 placeholder; 이미지마다 setInputSize()로 덮어씀
_NUM_LANDMARKS = 5
_BBOX_SLICE = slice(0, 4)  # face[0:4]  = x, y, w, h
_LMK_SLICE = slice(4, 14)  # face[4:14] = 5점 랜드마크 (x,y)x5, YuNet 원본 순서
_SCORE_IDX = 14  # face[14]   = 신뢰도
_REFINE_MATCH_IOU = 0.3  # 재검출 결과가 원 얼굴과 같은 얼굴인지 판정하는 하한 (미달 시 원 랜드마크 유지)
_REFINE_MIN_CROP_SIDE = 32  # 리사이즈된 크롭이 이보다 작으면 재검출 의미가 없어 건너뜀
_REFINE_MAX_SHIFT_RATIO = 0.5  # 정상 지터는 얼굴폭 15.8% 이내 — 이보다 크면 오매칭으로 보고 폐기
_DEDUP_IOU = 0.5  # 회복된 대형 후보가 같은 얼굴을 중복 검출할 때 억제하는 bbox IoU 하한 (ADR-017)
# 회복 박스 디둡의 랜드마크 중심 근접 하한(얼굴폭 대비). 저바닥에서 한 대형 얼굴이 offset 박스로
# 여러 개 되살아나면 bbox IoU는 낮아도(~0.2) 정제 랜드마크는 같은 얼굴을 가리킨다 — 중심 거리가
# 이 비율×얼굴폭 미만이면 같은 얼굴로 본다. 서로 다른 얼굴의 중심은 대개 1얼굴폭 이상 떨어져 안전.
_DEDUP_LANDMARK_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class DetectedFace:
  """원본 이미지 픽셀 좌표로 표현된, 검출된 얼굴 1개."""

  bbox: tuple[int, int, int, int]  # (x, y, w, h), 원본 이미지 픽셀 좌표
  # frozen 데이터클래스가 ndarray 필드로 자동 생성하는 __eq__/__hash__는 예외를 던진다
  # (배열 == 비교의 진리값 모호성, ndarray unhashable). compare=False로 eq·hash 대상에서
  # 제외해 값 동등성을 bbox+score로만 판단한다 — set/dict 키·중복 제거·테스트 비교를 안전하게.
  landmarks: np.ndarray = field(compare=False)  # shape (5, 2), float32 — YuNet 원본 순서, 재배열 금지
  score: float  # 신뢰도 [0, 1]


@dataclass(frozen=True)
class DetectorConfig:
  """`FaceDetector`의 튜닝 파라미터."""

  model_source: ModelSource | None = None
  score_threshold: float = DEFAULT_SCORE_THRESHOLD
  nms_threshold: float = NMS_THRESHOLD
  top_k: int = TOP_K
  max_side: int | None = DEFAULT_MAX_SIDE
  # 2단계 랜드마크 정제 (2026-07-14 리뷰): YuNet은 WIDER FACE(작은 얼굴)로 학습돼 학습 분포 밖의
  # 대형 얼굴에서 랜드마크가 불안정하다(얼굴폭 대비 최대 15.8% 지터). 대형 얼굴만 골라 정규 스케일로
  # 축소한 크롭에서 랜드마크를 재추출해 지터를 줄인다.
  refine_landmarks: bool = True
  refine_norm_face_width: int = DEFAULT_REFINE_NORM_FACE_WIDTH  # 재검출 크롭에서의 얼굴 목표 폭(px)
  refine_margin_ratio: float = DEFAULT_REFINE_MARGIN_RATIO  # bbox 대비 여유 크롭 비율
  # 배경 인물 필터 (ADR-013): 멀리 배경에 찍힌 얼굴이 사진 2장에 반복 등장하면 행인 앨범이
  # 만들어진다. 상대 크기 기준인 이유는 절대 px 기준이 저해상도 업로드(558×418 이미지의 21px
  # 앨범 얼굴 실측)를 자르기 때문. 0.0 = 비활성.
  min_face_rel_width: float = DEFAULT_MIN_FACE_REL_WIDTH
  # 대형 오검출 결합 필터 (survey 2026-07-15): score와 종횡비가 동시에 낮은 검출만 오검출로 제거.
  # 둘 중 하나만 0이면 (score < S AND aspect < A)가 항상 거짓 → 필터 전체 비활성.
  fp_score_threshold: float = DEFAULT_FP_SCORE_THRESHOLD
  fp_aspect_threshold: float = DEFAULT_FP_ASPECT_THRESHOLD
  # 대형 근접 얼굴 재검출 회복 (ADR-017): rel_w가 이 값 이상인 저score(<score_threshold) 얼굴을
  # 정규 스케일 재검출해 score가 big_face_redetect_score 이상이면 되살린다. 둘 중 하나라도 0이면 비활성.
  big_face_rel_width: float = DEFAULT_BIG_FACE_REL_WIDTH
  big_face_redetect_score: float = DEFAULT_BIG_FACE_REDETECT_SCORE
  # 재검출 랜드마크 신뢰 임계: 재검출 score가 이 값 이상이면 이동량 가드를 무시하고 재검출
  # 랜드마크를 채택한다 — 파편 bbox의 초대형 얼굴 교정 (DEFAULT 주석 참고). 0 = 비활성.
  refine_trust_redetect_score: float = DEFAULT_REFINE_TRUST_REDETECT_SCORE
  # confident 얼굴 간 파편 디둡 (ADR-027): 양쪽 폭이 refine_norm_face_width 초과인 confident 쌍의
  # 정제 랜드마크 중심거리가 이 비율×min(얼굴폭) 미만이면 같은 얼굴의 파편으로 보고 score 높은
  # 박스만 남긴다 (DEFAULT 주석 참고). 0 = 비활성.
  confident_dedup_landmark_ratio: float = DEFAULT_CONFIDENT_DEDUP_LANDMARK_RATIO

  def __post_init__(self) -> None:
    # None만 다운스케일 비활성화를 의미한다. 0/음수는 scale=0 → 1/scale ZeroDivisionError를
    # 유발하므로 생성 시점에 거부한다.
    if self.max_side is not None and self.max_side <= 0:
      raise ValueError(f"max_side는 양의 정수 또는 None이어야 합니다. 받은 값: {self.max_side}")
    if self.refine_norm_face_width <= 0:
      raise ValueError(f"refine_norm_face_width는 양의 정수여야 합니다. 받은 값: {self.refine_norm_face_width}")
    if self.refine_margin_ratio < 0:
      raise ValueError(f"refine_margin_ratio는 0 이상이어야 합니다. 받은 값: {self.refine_margin_ratio}")
    # 1.0 이상은 모든 얼굴을 제거하는 설정 실수라 생성 시점에 거부한다 (0.0 = 비활성은 허용)
    if not 0.0 <= self.min_face_rel_width < 1.0:
      raise ValueError(f"min_face_rel_width는 [0, 1) 범위여야 합니다. 받은 값: {self.min_face_rel_width}")
    if not 0.0 <= self.fp_score_threshold <= 1.0:
      raise ValueError(f"fp_score_threshold는 [0, 1] 범위여야 합니다. 받은 값: {self.fp_score_threshold}")
    if self.fp_aspect_threshold < 0.0:
      raise ValueError(f"fp_aspect_threshold는 0 이상이어야 합니다. 받은 값: {self.fp_aspect_threshold}")
    if not 0.0 <= self.big_face_rel_width < 1.0:
      raise ValueError(f"big_face_rel_width는 [0, 1) 범위여야 합니다. 받은 값: {self.big_face_rel_width}")
    if not 0.0 <= self.big_face_redetect_score <= 1.0:
      raise ValueError(f"big_face_redetect_score는 [0, 1] 범위여야 합니다. 받은 값: {self.big_face_redetect_score}")
    if not 0.0 <= self.refine_trust_redetect_score <= 1.0:
      raise ValueError(
        f"refine_trust_redetect_score는 [0, 1] 범위여야 합니다. 받은 값: {self.refine_trust_redetect_score}"
      )
    # 1.0 이상은 서로 다른 얼굴(중심 거리 대개 1폭 이상)까지 삼키는 설정 실수라 생성 시점에 거부한다
    if not 0.0 <= self.confident_dedup_landmark_ratio < 1.0:
      raise ValueError(
        f"confident_dedup_landmark_ratio는 [0, 1) 범위여야 합니다. 받은 값: {self.confident_dedup_landmark_ratio}"
      )


def _clamp_bbox(bbox: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
  """(x, y, w, h) 박스를 이미지 경계로 클램프한다. 랜드마크는 클램프하지 않는다."""
  # 클램프된 경계에서 폭/높이를 다시 계산해, 이미지 밖 영역이 네 변 모두에서 잘리게 한다.
  # 상단/좌측을 벗어난 박스의 원점을 0으로 스냅할 때는 박스를 이동만 하지 말고 축소해야 한다.
  x0 = max(0, int(bbox[0]))
  y0 = max(0, int(bbox[1]))
  x1 = min(w, int(bbox[0]) + int(bbox[2]))
  y1 = min(h, int(bbox[1]) + int(bbox[3]))
  return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
  """(x, y, w, h) 박스 두 개의 IoU. 퇴화 박스(넓이 0)는 0을 반환한다."""
  ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
  ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
  inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
  union = a[2] * a[3] + b[2] * b[3] - inter
  return inter / union if union > 0 else 0.0


def _to_contiguous_bgr(image: np.ndarray) -> np.ndarray:
  """YuNet 입력 계약(C-연속 메모리의 3채널 BGR)에 맞게 정규화한다.

  호출자는 보통 cv2.imdecode(IMREAD_COLOR)로 연속 BGR을 넘기지만, 그레이스케일·알파 채널·
  RGB→BGR 뷰(`img[..., ::-1]` 등 비연속) 같은 변형 입력이 와도 detect()가 예외 없이 동작하도록
  방어한다. 비연속 버퍼는 DNN 순전파에 그대로 넘기면 오검출/에러를 낳으므로 마지막에 연속성을
  보장한다. dtype 변환(uint16/float→uint8)은 스케일 의미가 모호해 여기서 손대지 않는다.
  """
  if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # (H,W) / (H,W,1) 그레이스케일 → BGR
  elif image.ndim == 3 and image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # 알파 채널 제거
  return np.ascontiguousarray(image)  # 3채널 통과 경로 포함, 비연속 뷰를 연속 버퍼로 확정


class FaceDetector:
  """YuNet 얼굴 검출기.

  인스턴스는 스레드 안전하지 않다: 내부 cv2 검출기에 대한 `setInputSize()` + `detect()`가
  상태를 공유하는 2단계 호출이라 동시 호출 시 레이스가 발생한다. 스레드/프로세스당 하나씩
  사용하라. 전역 설정인 `cv2.setNumThreads()`는 여기가 아니라 워커 부트스트랩에서 호출한다.
  """

  def __init__(self, config: DetectorConfig | None = None) -> None:
    resolved_config = config or DetectorConfig()
    source = resolved_config.model_source or default_yunet_source()
    model_path = source.resolve()  # 생성 시 모델을 1회만 로딩
    self._score_threshold = resolved_config.score_threshold  # 기본(정상 크기 얼굴) score gate
    # 대형 근접 얼굴 재검출 회복(ADR-017) 활성 시, 모델 score 바닥을 내려 저score 대형 후보를
    # 표면화한다 — 실제 keep/drop은 detect()가 정규 스케일 재검출로 판정한다(모델 gate는 후보 노출용).
    self._big_face_enabled = resolved_config.big_face_rel_width > 0.0 and resolved_config.big_face_redetect_score > 0.0
    self._big_face_rel_width = resolved_config.big_face_rel_width
    self._big_face_redetect_score = resolved_config.big_face_redetect_score
    model_score_threshold = (
      min(resolved_config.score_threshold, _BIG_FACE_MODEL_FLOOR)
      if self._big_face_enabled
      else resolved_config.score_threshold
    )
    self._detector = cv2.FaceDetectorYN.create(
      model=model_path,
      config="",
      input_size=_YUNET_INIT_SIZE,
      score_threshold=model_score_threshold,
      nms_threshold=resolved_config.nms_threshold,
      top_k=resolved_config.top_k,
    )
    self._max_side = resolved_config.max_side
    self._min_face_rel_width = resolved_config.min_face_rel_width
    self._fp_score_threshold = resolved_config.fp_score_threshold
    self._fp_aspect_threshold = resolved_config.fp_aspect_threshold
    self._refine_landmarks = resolved_config.refine_landmarks
    self._refine_norm_face_width = resolved_config.refine_norm_face_width
    self._refine_margin_ratio = resolved_config.refine_margin_ratio
    self._refine_trust_score = resolved_config.refine_trust_redetect_score
    self._confident_dedup_ratio = resolved_config.confident_dedup_landmark_ratio

  def detect(self, image: np.ndarray) -> list[DetectedFace]:
    """디코딩된 BGR 이미지에서 얼굴을 검출한다. 정상 ndarray에는 예외를 던지지 않는다."""
    image = _to_contiguous_bgr(image)  # 채널/연속성 정규화 후 원본 좌표계(h, w)를 확정
    h, w = image.shape[:2]
    scale = 1.0
    frame = image
    if self._max_side is not None and max(h, w) > self._max_side:
      # 축소 전용(scale < 1.0): 큰 이미지의 검출 비용만 낮추고 작은 이미지는 확대하지 않는다.
      # 좌표는 inv = 1/scale 로 원본 좌표계에 되돌린다.
      scale = self._max_side / max(h, w)
      frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    self._detector.setInputSize((frame.shape[1], frame.shape[0]))  # (width, height)
    _, raw = self._detector.detect(frame)
    if raw is None:
      return []

    inv = 1.0 / scale
    # 배경 인물 필터의 기준은 원본 좌표계 bbox 폭 — 검출용 축소(max_side)와 무관하게 동작한다
    long_side = max(h, w)
    min_face_w_px = self._min_face_rel_width * long_side
    confident: list[DetectedFace] = []
    pending: list[DetectedFace] = []  # 저score 대형 후보 — 정규 스케일 재검출로 keep/drop 판정 (ADR-017)
    for row in raw:
      face = self._to_detected_face(row, inv, w, h)
      if face is None or face.bbox[2] < min_face_w_px or self._is_large_false_positive(face):
        continue
      if face.score >= self._score_threshold:
        confident.append(face)
      elif self._big_face_enabled and face.bbox[2] >= self._big_face_rel_width * long_side:
        pending.append(face)
      # else: 정상 크기 저score → 폐기 (모델 gate가 회복 활성 시 바닥까지 내려가 여기 도달)

    # 정제·회복은 raw 순회 종료 후에 시작한다 — _refine_face/_recover_large_face의 setInputSize/detect
    # 재호출이 위 검출 상태를 덮어쓰므로 순서를 바꾸면 안 된다. 축소본 frame이 아니라 원본 image를 넘겨
    # max_side 축소를 우회한 원본 해상도 크롭에서 재검출한다.
    faces = [self._refine_face(image, face) for face in confident] if self._refine_landmarks else list(confident)
    if self._confident_dedup_ratio > 0.0 and len(faces) > 1:
      # 정제 후에 실행해야 한다 — 파편 박스들의 랜드마크는 정제가 같은 진짜 얼굴로 교정해야 중심이 모인다
      faces = self._dedup_confident_fragments(faces)
    if self._big_face_enabled and pending:
      # 회복 후보를 재검출 판정한 뒤, 이미 확정된 얼굴(confident)이나 먼저 채택된 회복 얼굴과 중복이
      # 아닌 것만 추가한다. confident는 절대 억제하지 않아 기존 검출 동작을 보존한다. score 내림차순
      # 처리로 같은 얼굴의 여러 offset 박스 중 최상 박스를 남긴다.
      recovered = sorted(
        (f for f in (self._recover_large_face(image, p) for p in pending) if f is not None),
        key=lambda f: f.score,
        reverse=True,
      )
      for face in recovered:
        if not self._duplicates_existing(face, faces):
          faces.append(face)
    return faces

  def _normal_scale_redetect(self, image: np.ndarray, face: DetectedFace) -> tuple[float, np.ndarray] | None:
    """얼굴 주변을 잘라 얼굴폭을 정규 폭으로 리사이즈해 재검출한 (score, 원좌표 랜드마크)를 돌려준다.

    YuNet 학습 분포(작은 얼굴) 밖의 대형 얼굴을 정규 스케일로 되돌려 재검출한다 — 랜드마크 정제
    (_refine_face)와 저score 대형 후보의 실얼굴 판정(_recover_large_face) 공용 코어. IoU로 원 얼굴과
    같은 얼굴을 고르고(크롭에 걸친 옆 사람 배제), 미검출·크롭 실패·오매칭이면 None. score는 재검출
    결과의 신뢰도(정규 스케일이라 원 저score보다 신뢰할 만하다)다.
    """
    x, y, w, h = face.bbox
    if w <= 0 or h <= 0:
      return None
    # bbox에 여유 마진을 두고 원본에서 크롭한다 (경계 클램프는 마진만 깎을 뿐 얼굴 자체는 보존)
    margin = max(w, h) * self._refine_margin_ratio
    cx0 = max(0, int(x - margin))
    cy0 = max(0, int(y - margin))
    cx1 = min(image.shape[1], int(x + w + margin))
    cy1 = min(image.shape[0], int(y + h + margin))
    if cx1 <= cx0 or cy1 <= cy0:
      return None
    # 얼굴폭이 정규 폭이 되도록 스케일한다. 축소는 INTER_AREA(박스필터 저역통과 내장), 확대(작은 이미지의
    # 대형 얼굴)는 INTER_LINEAR. 회복 경로는 정규 폭 미만 얼굴도 재검출해야 하므로 확대를 허용한다.
    r = self._refine_norm_face_width / w
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image[cy0:cy1, cx0:cx1], None, fx=r, fy=r, interpolation=interp)
    rh, rw = resized.shape[:2]
    if min(rh, rw) < _REFINE_MIN_CROP_SIDE:
      return None

    self._detector.setInputSize((rw, rh))
    _, raw = self._detector.detect(resized)
    if raw is None or len(raw) == 0:
      return None

    ref_bbox = ((x - cx0) * r, (y - cy0) * r, w * r, h * r)  # 원 bbox를 크롭-리사이즈 좌표계로 변환
    best_row = max(raw, key=lambda row: _bbox_iou(ref_bbox, tuple(row[_BBOX_SLICE])))
    if _bbox_iou(ref_bbox, tuple(best_row[_BBOX_SLICE])) < _REFINE_MATCH_IOU:
      return None
    score = float(best_row[_SCORE_IDX])
    landmarks = (best_row[_LMK_SLICE].reshape(_NUM_LANDMARKS, 2) / r + np.array([cx0, cy0])).astype(np.float32)
    if not np.isfinite(landmarks).all() or not np.isfinite(score):
      return None
    return score, landmarks

  def _refine_face(self, image: np.ndarray, face: DetectedFace) -> DetectedFace:
    """대형 얼굴의 랜드마크를 정규 스케일 크롭 재검출로 정제한다. 어떤 실패든 원본 face를 반환한다.

    bbox·score는 원본을 유지한다 — 정제 목적은 랜드마크뿐이고, bbox는 하류(품질 게이트 크롭)에 이미
    전파되는 계약이다.
    """
    x, y, w, h = face.bbox
    if w <= self._refine_norm_face_width:
      return face  # 정규 폭 이하 얼굴은 학습 분포 안이라 문제없고, 확대 재검출은 비용만 든다
    result = self._normal_scale_redetect(image, face)
    if result is None:
      return face
    score, refined = result
    # 정상 지터는 얼굴폭 15.8% 이내 — 그보다 훨씬 큰 이동은 오매칭 폭주로 보고 폐기한다.
    # 단 재검출 score가 신뢰 임계 이상이면 채택한다 — 초대형 얼굴은 원 bbox가 파편이라 올바른
    # 교정도 파편 폭 기준 가드를 넘는다 (event 73 공통첩 유출, DEFAULT_REFINE_TRUST_REDETECT_SCORE 주석).
    if float(np.abs(refined - face.landmarks).max()) > _REFINE_MAX_SHIFT_RATIO * w and not self._trusted_redetect(
      score
    ):
      return face
    refined.flags.writeable = False  # frozen dataclass 출력이 하류에서 변형되지 않도록 보호
    return replace(face, landmarks=refined)

  def _recover_large_face(self, image: np.ndarray, face: DetectedFace) -> DetectedFace | None:
    """저score 대형 후보를 정규 스케일 재검출로 판정한다 (ADR-017).

    YuNet이 학습 분포 밖 대형 얼굴에 낮은 score를 주는 약점 때문에 score gate에서 사라진 실얼굴을,
    정규 스케일 재검출 score가 임계 이상이면 되살린다. 실얼굴은 정규 스케일에서 score가 오르지만
    오검출(아웃포커스 배경·신체일부)은 어느 스케일에서도 낮다. bbox·score는 원본을 유지하고
    (하류 계약), 재검출 랜드마크가 정상 지터 범위면 그것으로 정제한다. 임계 미달이면 None(폐기).
    """
    result = self._normal_scale_redetect(image, face)
    if result is None:
      return None
    score, refined = result
    if score < self._big_face_redetect_score:
      return None  # 재검출에서도 저score → 오검출로 폐기
    # 신뢰 임계 이상이면 이동량 가드를 무시하고 재검출 랜드마크를 채택한다 — 여기서 원 랜드마크를
    # 유지하면 offset 파편 박스가 쓰레기 임베딩으로 살아남아 디둡(랜드마크 중심)도 비껴가고,
    # 사진 3장 이상에서 반복되면 쓰레기끼리 뭉친 유령 인물 앨범이 된다 (event 73 실측 741d1aef).
    if float(np.abs(refined - face.landmarks).max()) <= _REFINE_MAX_SHIFT_RATIO * face.bbox[
      2
    ] or self._trusted_redetect(score):
      refined.flags.writeable = False
      return replace(face, landmarks=refined)
    return face  # 실얼굴이나 랜드마크 매칭이 불안정 → 원 랜드마크 유지하고 얼굴은 살린다

  def _trusted_redetect(self, score: float) -> bool:
    """재검출 score가 신뢰 임계 이상인지 — 이동량 가드를 무시하고 재검출 랜드마크를 채택할 자격."""
    return self._refine_trust_score > 0.0 and score >= self._refine_trust_score

  def _duplicates_existing(self, face: DetectedFace, existing: list[DetectedFace]) -> bool:
    """회복된 face가 이미 채택된 얼굴 중 하나와 같은 얼굴인지 판정한다 (ADR-017 디둡).

    bbox IoU가 높거나(_DEDUP_IOU), 정제 랜드마크 중심 거리가 얼굴폭의 _DEDUP_LANDMARK_RATIO 미만이면
    같은 얼굴이다 — 저바닥에서 한 대형 얼굴이 offset 박스로 여러 개 되살아나면 IoU는 낮아도 랜드마크는
    같은 얼굴을 가리키므로 두 신호를 함께 본다.
    """
    return any(self._same_face_geometry(face, other, _DEDUP_LANDMARK_RATIO) for other in existing)

  @staticmethod
  def _same_face_geometry(a: DetectedFace, b: DetectedFace, landmark_ratio: float) -> bool:
    """두 검출이 같은 얼굴을 가리키는지 — bbox IoU 또는 랜드마크 중심 근접, 둘 중 하나면 같은 얼굴."""
    if _bbox_iou(a.bbox, b.bbox) >= _DEDUP_IOU:
      return True
    scale = min(a.bbox[2], b.bbox[2])
    if scale <= 0:
      return False
    return float(np.hypot(*(a.landmarks.mean(axis=0) - b.landmarks.mean(axis=0)))) < landmark_ratio * scale

  def _dedup_confident_fragments(self, faces: list[DetectedFace]) -> list[DetectedFace]:
    """confident 얼굴 사이의 초대형 얼굴 파편 중복을 제거한다 (ADR-027).

    파편 박스 여러 개가 전부 score gate를 통과하면 회복 경로의 ADR-017 디둡이 닿지 않는다 — score
    내림차순으로 훑어 최상 박스만 남긴다(회복 경로의 "같은 얼굴의 여러 offset 박스 중 최상 박스"
    관례와 동일). 검사는 양쪽 폭이 refine_norm_face_width 초과일 때만: 파편화는 학습 분포 밖 대형
    얼굴 현상이고, 그 크기에서만 정제가 랜드마크 중심을 진짜 얼굴로 교정해 중심 근접이 신뢰할 수
    있는 신호가 된다 (DEFAULT_CONFIDENT_DEDUP_LANDMARK_RATIO 주석의 실측). 생존자는 원 순서 유지.
    """
    survivors: list[DetectedFace] = []
    for face in sorted(faces, key=lambda f: f.score, reverse=True):
      if not any(
        face.bbox[2] > self._refine_norm_face_width
        and kept.bbox[2] > self._refine_norm_face_width
        and self._same_face_geometry(face, kept, self._confident_dedup_ratio)
        for kept in survivors
      ):
        survivors.append(face)
    kept_ids = {id(face) for face in survivors}
    return [face for face in faces if id(face) in kept_ids]

  def _is_large_false_positive(self, face: DetectedFace) -> bool:
    """score와 종횡비(w/h)가 동시에 낮은 대형 오검출인지 판정한다 (survey 2026-07-15)."""
    # 둘 중 하나라도 0이면 비활성 — 음수 불가라 결합 조건이 항상 거짓이 되는 것과 동치지만
    # 명시적으로 조기 반환한다.
    if self._fp_score_threshold <= 0.0 or self._fp_aspect_threshold <= 0.0:
      return False
    _, _, bw, bh = face.bbox  # bh > 0 은 _to_detected_face가 보장 → 0 나눗셈 없음
    return face.score < self._fp_score_threshold and (bw / bh) < self._fp_aspect_threshold

  def _to_detected_face(self, row: np.ndarray, inv: float, w: int, h: int) -> DetectedFace | None:
    bbox_f = row[_BBOX_SLICE] * inv
    landmarks = (row[_LMK_SLICE].reshape(_NUM_LANDMARKS, 2) * inv).astype(np.float32)  # 원본 순서 유지
    score = float(row[_SCORE_IDX])
    if not (np.isfinite(bbox_f).all() and np.isfinite(landmarks).all() and np.isfinite(score)):
      return None
    x, y, bw, bh = _clamp_bbox(bbox_f, w, h)
    if bw <= 0 or bh <= 0:
      return None
    landmarks.flags.writeable = False  # frozen dataclass 출력이 하류에서 변형되지 않도록 보호
    return DetectedFace((x, y, bw, bh), landmarks, score)


if __name__ == "__main__":
  # SQS/S3 없이 PoC 레시피와의 파리티를 확인: 로컬 이미지 경로를 그대로 FaceDetector에 넣어
  # 검출 결과를 출력한다.
  import sys
  import time

  detector = FaceDetector()
  for path in sys.argv[1:]:
    image = cv2.imread(path)
    if image is None:
      print(f"{path}: 건너뜀 (이미지를 읽을 수 없음)")
      continue

    start = time.perf_counter()
    detected = detector.detect(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"{path}: {len(detected)} face(s) in {elapsed_ms:.1f} ms")
    for face in detected:
      aspect = face.bbox[2] / face.bbox[3]  # w/h — 결합 오검출 필터 튜닝 재현용
      print(f"  bbox={face.bbox} score={face.score:.3f} aspect={aspect:.3f} first_landmark={tuple(face.landmarks[0])}")
