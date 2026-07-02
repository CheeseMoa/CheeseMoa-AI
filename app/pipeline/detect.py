"""순수 파이프라인 단계로서의 YuNet 얼굴 검출.

디코딩된 BGR 이미지를 받아 `DetectedFace` 리스트로 변환한다. S3 다운로드와 `cv2.imdecode`는
호출자의 책임이며, 이 모듈은 S3나 huggingface_hub를 직접 다루지 않는다. 모델 획득은
`ModelSource` 뒤로 완전히 캡슐화되어 있다.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.model_source import ModelSource, default_yunet_source

DEFAULT_SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000
DEFAULT_MAX_SIDE = 2000
_YUNET_INIT_SIZE = (320, 320)  # 초기 placeholder; 이미지마다 setInputSize()로 덮어씀
_NUM_LANDMARKS = 5
_BBOX_SLICE = slice(0, 4)  # face[0:4]  = x, y, w, h
_LMK_SLICE = slice(4, 14)  # face[4:14] = 5점 랜드마크 (x,y)x5, YuNet 원본 순서
_SCORE_IDX = 14  # face[14]   = 신뢰도


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

  def __post_init__(self) -> None:
    # None만 다운스케일 비활성화를 의미한다. 0/음수는 scale=0 → 1/scale ZeroDivisionError를
    # 유발하므로 생성 시점에 거부한다.
    if self.max_side is not None and self.max_side <= 0:
      raise ValueError(f"max_side는 양의 정수 또는 None이어야 합니다. 받은 값: {self.max_side}")


def _clamp_bbox(bbox: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
  """(x, y, w, h) 박스를 이미지 경계로 클램프한다. 랜드마크는 클램프하지 않는다."""
  # 클램프된 경계에서 폭/높이를 다시 계산해, 이미지 밖 영역이 네 변 모두에서 잘리게 한다.
  # 상단/좌측을 벗어난 박스의 원점을 0으로 스냅할 때는 박스를 이동만 하지 말고 축소해야 한다.
  x0 = max(0, int(bbox[0]))
  y0 = max(0, int(bbox[1]))
  x1 = min(w, int(bbox[0]) + int(bbox[2]))
  y1 = min(h, int(bbox[1]) + int(bbox[3]))
  return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


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
    faces: list[DetectedFace] = []
    for row in raw:
      face = self._to_detected_face(row, inv, w, h)
      if face is not None:
        faces.append(face)
    return faces

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
      print(f"  bbox={face.bbox} score={face.score:.3f} first_landmark={tuple(face.landmarks[0])}")
