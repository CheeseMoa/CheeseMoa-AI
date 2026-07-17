"""MediaPipe Face Landmarker blendshape 기반 눈감음 점수 — litert 이식 (ADR 021).

눈감음 CNN(open-closed-eye-0001)의 도메인 실패(유아 오탐·보정 이미지 미탐·수면 미탐)를 A/B
실측으로 해결한 MediaPipe Face Landmarker의 eyeBlink blendshape를, mediapipe pip 패키지 없이
실행한다 — 공식 패키지는 linux aarch64 휠이 없어 EC2(t4g)에 배포할 수 없으므로,
face_landmarker.task(zip, Apache 2.0) 내부의 tflite 2개만 ai-edge-litert로 직접 돌리고
그래프 글루(RoI 구성 → 랜드마크 → blendshape 입력 변환)를 numpy/cv2로 이식했다
(HDBSCAN·face_align 이식과 같은 패턴, 파리티는 macOS mediapipe 참조 대비 코퍼스 실측으로 검증).

이식 근거 (google-ai-edge/mediapipe, Apache 2.0):
  - RoI: 회전 = 두 눈 keypoint가 수평이 되는 각(face_detector_graph.cc — start 우안/end 좌안,
    target 0°), 박스 = 검출 rect의 1.5배(RectTransformationCalculator scale 1.5). 원본과 달리
    우리 rect는 BlazeFace가 아니라 YuNet bbox라서 배율·시프트는 참조 파리티 스윕으로 재보정한다
    (_ROI_SCALE·_ROI_Y_SHIFT 주석 참고).
  - 랜드마크 모델: 256x256 RGB [0,1] 입력 → 478점(crop 픽셀 좌표)·presence 로짓
    (face_landmarks_detector_graph.cc — TensorsToFloats SIGMOID).
  - blendshape 모델: 478점 중 146점 서브셋(face_blendshapes_graph.cc kLandmarksSubsetIdxs)을
    "이미지 픽셀 좌표" 그대로 [1,146,2]로 입력(LandmarksToTensor X·Y + image_size — 정규화는
    모델 내부가 수행) → 52 blendshape. eyeBlinkLeft/Right = kBlendshapeNames 인덱스 9·10.

모델 로딩(다운로드 포함)은 FaceBlinkScorer 생성 시 1회만 일어난다 (EyeStateClassifier와 동일 계약).
"""

import math
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.model_source import ModelSource, default_face_landmarker_source

_LANDMARKER_MEMBER = "face_landmarks_detector.tflite"
_BLENDSHAPES_MEMBER = "face_blendshapes.tflite"
_CROP_SIZE = 256  # 랜드마크 모델 입력 한 변
_NUM_LANDMARKS = 478
_EYE_BLINK_LEFT = 9  # face_blendshapes_graph.cc kBlendshapeNames 순서
_EYE_BLINK_RIGHT = 10

# RoI 기하 — mediapipe 원본은 BlazeFace 검출 rect의 1.5배 정사각(회전 = 눈 라인 수평화)인데,
# 우리 rect는 YuNet 5점의 회전좌표계 bbox라 기하가 다르다(5점 스팬은 눈~입꼬리라 얼굴보다 작다).
# 값은 참조(macOS mediapipe 0.10.35) blink 점수와의 파리티 스윕으로 확정 (ADR 021 §파리티):
# 604 참조 얼굴에서 3.0/-0.05가 |Δmin_blink| 중앙값 0.0082·p95 0.0577·t=0.40 판정 뒤집힘 0·
# 알려진 감음 6/6 재현(3.5는 안경 감은 눈이 0.25 이하로 무너짐). 시프트는 회전 +y(턱 방향) 비율.
_ROI_SCALE = 3.0
_ROI_Y_SHIFT = -0.05

# face_blendshapes_graph.cc kLandmarksSubsetIdxs — blendshape 모델(HUND)이 요구하는 146점 서브셋
_BLENDSHAPE_IDXS = np.array(
  [
    0,
    1,
    4,
    5,
    6,
    7,
    8,
    10,
    13,
    14,
    17,
    21,
    33,
    37,
    39,
    40,
    46,
    52,
    53,
    54,
    55,
    58,
    61,
    63,
    65,
    66,
    67,
    70,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    93,
    95,
    103,
    105,
    107,
    109,
    127,
    132,
    133,
    136,
    144,
    145,
    146,
    148,
    149,
    150,
    152,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    168,
    172,
    173,
    176,
    178,
    181,
    185,
    191,
    195,
    197,
    234,
    246,
    249,
    251,
    263,
    267,
    269,
    270,
    276,
    282,
    283,
    284,
    285,
    288,
    291,
    293,
    295,
    296,
    297,
    300,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    323,
    324,
    332,
    334,
    336,
    338,
    356,
    361,
    362,
    365,
    373,
    374,
    375,
    377,
    378,
    379,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    397,
    398,
    400,
    402,
    405,
    409,
    415,
    454,
    466,
    468,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    477,
  ],
  dtype=np.int64,
)


@dataclass(frozen=True)
class BlinkConfig:
  """`FaceBlinkScorer`의 설정. 판정 임계(blink_threshold 등)는 QualityConfig 소관이라 여기 없다."""

  model_source: ModelSource | None = None
  # litert 인트라 op 스레드. 0 = 런타임 기본. 코어 수보다 크게 잡으면 오버서브스크립션으로
  # 급격히 느려진다 (onnxruntime과 동일한 실측 교훈 — deps가 가용 코어 수를 주입한다).
  num_threads: int = 0


def _output_index(details: list[dict], *, size: int, ndim: int) -> int:
  """출력 텐서를 (원소 수, 차원 수)로 식별한다 — 이름(IdentityN)은 변환 산물이라 신뢰할 수 없다."""
  for d in details:
    shape = d["shape"]
    if int(np.prod(shape)) == size and len(shape) == ndim:
      return d["index"]
  raise ValueError(f"기대한 출력 텐서(size={size}, ndim={ndim})가 없습니다: {[d['shape'] for d in details]}")


class FaceBlinkScorer:
  """face_landmarker.task의 랜드마크+blendshape 모델로 얼굴별 (presence, blinkL, blinkR)을 계산한다.

  litert Interpreter는 스레드 안전하지 않지만 단일 스레드 워커 전제라 무해하다 (FaceDetector와 동일).
  """

  def __init__(self, config: BlinkConfig | None = None) -> None:
    resolved = config or BlinkConfig()
    source = resolved.model_source or default_face_landmarker_source()
    task_path = source.resolve()  # 생성 시 1회만 다운로드/캐시
    with zipfile.ZipFile(task_path) as bundle:
      landmarker_bytes = bundle.read(_LANDMARKER_MEMBER)
      blendshapes_bytes = bundle.read(_BLENDSHAPES_MEMBER)

    # litert는 무거운 의존이 아니지만 지연 import를 유지한다 — blink 비활성(threshold=0) 조립이
    # 이 모듈을 아예 만들지 않으므로, import 실패가 비활성 운영을 못 막게 하는 방어는 deps 쪽에 있다.
    from ai_edge_litert.interpreter import Interpreter

    threads = {"num_threads": resolved.num_threads} if resolved.num_threads > 0 else {}
    self._landmarker = Interpreter(model_content=landmarker_bytes, **threads)
    self._landmarker.allocate_tensors()
    self._lm_input = self._landmarker.get_input_details()[0]["index"]
    lm_outputs = self._landmarker.get_output_details()
    self._lm_points = _output_index(lm_outputs, size=_NUM_LANDMARKS * 3, ndim=4)  # [1,1,1,1434]
    self._lm_presence = _output_index(lm_outputs, size=1, ndim=4)  # [1,1,1,1] 로짓 (ndim 2는 별개 신호)

    self._blendshaper = Interpreter(model_content=blendshapes_bytes, **threads)
    self._blendshaper.allocate_tensors()
    self._bs_input = self._blendshaper.get_input_details()[0]["index"]
    self._bs_output = self._blendshaper.get_output_details()[0]["index"]  # [52]

  def blink_scores(self, image: np.ndarray, landmarks: np.ndarray) -> tuple[float, float, float] | None:
    """원본 이미지 + YuNet 5점 → (presence, blinkL, blinkR). RoI 퇴화 시 None.

    presence는 랜드마크 모델의 얼굴 존재 확신(시그모이드) — 호출자가 바닥(blink_presence_floor)
    미달이면 blink를 버리고 CNN 폴백으로 처리한다.
    """
    corners = self._roi_corners(landmarks)
    if corners is None:
      return None
    p0, p1, p2 = corners
    src = np.array([p0, p1, p2], dtype=np.float32)
    dst = np.array([[0.0, 0.0], [_CROP_SIZE, 0.0], [0.0, _CROP_SIZE]], dtype=np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(image, matrix, (_CROP_SIZE, _CROP_SIZE), borderValue=0.0)
    blob = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)[np.newaxis]

    self._landmarker.set_tensor(self._lm_input, blob)
    self._landmarker.invoke()
    presence = float(1.0 / (1.0 + math.exp(-float(self._landmarker.get_tensor(self._lm_presence).ravel()[0]))))
    points = self._landmarker.get_tensor(self._lm_points).reshape(_NUM_LANDMARKS, 3)[:, :2] / _CROP_SIZE

    # crop 정규화 좌표 → 원본 이미지 픽셀 좌표 (blendshape 모델은 픽셀 좌표를 받아 내부 정규화)
    origin = np.asarray(p0, dtype=np.float32)
    axis_x = np.asarray(p1, dtype=np.float32) - origin
    axis_y = np.asarray(p2, dtype=np.float32) - origin
    image_points = origin + points[:, :1] * axis_x + points[:, 1:2] * axis_y

    subset = image_points[_BLENDSHAPE_IDXS].astype(np.float32)[np.newaxis]  # [1,146,2]
    self._blendshaper.set_tensor(self._bs_input, subset)
    self._blendshaper.invoke()
    scores = self._blendshaper.get_tensor(self._bs_output).ravel()
    return presence, float(scores[_EYE_BLINK_LEFT]), float(scores[_EYE_BLINK_RIGHT])

  def _roi_corners(self, landmarks: np.ndarray) -> tuple | None:
    """YuNet 5점 → 회전 정사각 RoI의 (좌상, 우상, 좌하) 꼭짓점. 퇴화 기하는 None."""
    if landmarks.shape != (5, 2):
      raise ValueError(f"landmarks는 shape (5, 2)여야 합니다. 받은 shape: {landmarks.shape}")
    pts = landmarks.astype(np.float64)
    right_eye, left_eye = pts[0], pts[1]
    eye_vec = left_eye - right_eye
    if float(np.hypot(*eye_vec)) < 1e-6:
      return None
    rot = math.atan2(float(eye_vec[1]), float(eye_vec[0]))  # 눈 라인 수평화 (target 0°)
    # rect: 5점 전체의 회전좌표계 bbox — YuNet bbox(축 정렬)는 기운 얼굴에서 부풀어 배율이 안 맞는다
    ux = np.array([math.cos(rot), math.sin(rot)])
    uy = np.array([-math.sin(rot), math.cos(rot)])
    local = np.stack([pts @ ux, pts @ uy], axis=1)
    lo, hi = local.min(axis=0), local.max(axis=0)
    side = float(max(hi - lo)) * _ROI_SCALE
    if side < 2.0:
      return None
    center_local = (lo + hi) / 2.0
    center = center_local[0] * ux + center_local[1] * uy + _ROI_Y_SHIFT * side * uy
    half = side / 2.0
    p0 = center - half * ux - half * uy
    p1 = center + half * ux - half * uy
    p2 = center - half * ux + half * uy
    return tuple(map(tuple, (p0, p1, p2)))


if __name__ == "__main__":
  # SQS/S3 없이 로컬 이미지에서 얼굴별 (presence, blinkL, blinkR)를 출력한다 — 임계 보정·파리티 확인용.
  import sys

  from app.pipeline.detect import FaceDetector

  detector = FaceDetector()
  scorer = FaceBlinkScorer()
  for path in sys.argv[1:]:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
      print(f"{path}: 건너뜀 (이미지를 읽을 수 없음)")
      continue
    for i, face in enumerate(detector.detect(image)):
      result = scorer.blink_scores(image, face.landmarks)
      if result is None:
        print(f"{path} face{i}: RoI 퇴화")
        continue
      presence, left, right = result
      print(f"{path} face{i} w={face.bbox[2]}: presence={presence:.3f} blink=[{left:.3f}, {right:.3f}]")
