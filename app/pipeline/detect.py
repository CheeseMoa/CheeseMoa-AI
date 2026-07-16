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
_YUNET_INIT_SIZE = (320, 320)  # 초기 placeholder; 이미지마다 setInputSize()로 덮어씀
_NUM_LANDMARKS = 5
_BBOX_SLICE = slice(0, 4)  # face[0:4]  = x, y, w, h
_LMK_SLICE = slice(4, 14)  # face[4:14] = 5점 랜드마크 (x,y)x5, YuNet 원본 순서
_SCORE_IDX = 14  # face[14]   = 신뢰도
_REFINE_MATCH_IOU = 0.3  # 재검출 결과가 원 얼굴과 같은 얼굴인지 판정하는 하한 (미달 시 원 랜드마크 유지)
_REFINE_MIN_CROP_SIDE = 32  # 리사이즈된 크롭이 이보다 작으면 재검출 의미가 없어 건너뜀
_REFINE_MAX_SHIFT_RATIO = 0.5  # 정상 지터는 얼굴폭 15.8% 이내 — 이보다 크면 오매칭으로 보고 폐기


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
    self._detector = cv2.FaceDetectorYN.create(
      model=model_path,
      config="",
      input_size=_YUNET_INIT_SIZE,
      score_threshold=resolved_config.score_threshold,
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
    min_face_w_px = self._min_face_rel_width * max(h, w)
    faces: list[DetectedFace] = []
    for row in raw:
      face = self._to_detected_face(row, inv, w, h)
      if face is not None and face.bbox[2] >= min_face_w_px and not self._is_large_false_positive(face):
        faces.append(face)
    if self._refine_landmarks:
      # 본검출 raw 순회가 끝난 뒤에 정제를 시작한다 — _refine_face의 setInputSize/detect 재호출이
      # 위 검출 상태를 덮어쓰므로 순서를 바꾸면 안 된다. 축소본 frame이 아니라 원본 image를 넘겨
      # max_side 축소를 우회한 원본 해상도 크롭에서 재검출하는 것이 정제의 요점이다.
      faces = [self._refine_face(image, face) for face in faces]
    return faces

  def _refine_face(self, image: np.ndarray, face: DetectedFace) -> DetectedFace:
    """대형 얼굴의 랜드마크를 정규 스케일 크롭 재검출로 정제한다. 어떤 실패든 원본 face를 반환한다.

    YuNet의 학습 분포(작은 얼굴)에 맞게 "얼굴 주변을 잘라 → 얼굴폭을 정규 폭으로 축소 → 재검출"한
    랜드마크를 원본 좌표계로 되돌린다. bbox·score는 원본을 유지한다 — 정제 목적은 랜드마크뿐이고,
    bbox는 하류(품질 게이트 크롭)에 이미 전파되는 계약이다.
    """
    x, y, w, h = face.bbox
    if w <= self._refine_norm_face_width:
      return face  # 정규 폭 이하 얼굴은 학습 분포 안이라 문제없고, 확대 재검출은 비용만 든다

    # bbox에 여유 마진을 두고 원본에서 크롭한다 (경계 클램프는 마진만 깎을 뿐 얼굴 자체는 보존)
    margin = max(w, h) * self._refine_margin_ratio
    cx0 = max(0, int(x - margin))
    cy0 = max(0, int(y - margin))
    cx1 = min(image.shape[1], int(x + w + margin))
    cy1 = min(image.shape[0], int(y + h + margin))
    if cx1 <= cx0 or cy1 <= cy0:
      return face

    # 얼굴폭이 정규 폭이 되도록 축소 — INTER_AREA는 박스필터 저역통과를 내장해 별도 프리블러가 불필요
    r = self._refine_norm_face_width / w  # 게이트(w > 정규 폭)로 r < 1 보장
    resized = cv2.resize(image[cy0:cy1, cx0:cx1], None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    if min(rh, rw) < _REFINE_MIN_CROP_SIDE:
      return face

    self._detector.setInputSize((rw, rh))
    _, raw = self._detector.detect(resized)
    if raw is None or len(raw) == 0:
      return face

    # 재검출 후보 중 원 얼굴과 같은 얼굴을 IoU로 고른다 — score argmax는 크롭에 걸친 옆 사람을
    # 잡을 수 있다. 미검출·다인 크롭 케이스를 이 한 경로로 처리한다.
    ref_bbox = ((x - cx0) * r, (y - cy0) * r, w * r, h * r)  # 원 bbox를 크롭-리사이즈 좌표계로 변환
    best_row = max(raw, key=lambda row: _bbox_iou(ref_bbox, tuple(row[_BBOX_SLICE])))
    if _bbox_iou(ref_bbox, tuple(best_row[_BBOX_SLICE])) < _REFINE_MATCH_IOU:
      return face

    refined = (best_row[_LMK_SLICE].reshape(_NUM_LANDMARKS, 2) / r + np.array([cx0, cy0])).astype(np.float32)
    if not np.isfinite(refined).all():
      return face
    # 정상 지터는 얼굴폭 15.8% 이내 — 그보다 훨씬 큰 이동은 오매칭 폭주로 보고 폐기한다
    if float(np.abs(refined - face.landmarks).max()) > _REFINE_MAX_SHIFT_RATIO * w:
      return face
    refined.flags.writeable = False  # frozen dataclass 출력이 하류에서 변형되지 않도록 보호
    return replace(face, landmarks=refined)

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
