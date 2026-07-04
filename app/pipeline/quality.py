"""순수 파이프라인 단계로서의 사진 품질 판정 (눈감음 CNN + 흔들림 Laplacian).

두 게이트를 제공한다:
  - 눈감음: align.py가 5점(눈·코·입꼬리)으로 정렬한 112x112 crop의 고정 눈 좌표에서 양눈을 잘라
    open-closed-eye-0001 CNN으로 open/closed 분류. 양눈 모두 closed면 그 얼굴은 눈감음.
    입꼬리 랜드마크는 align의 Umeyama 변환을 통해 롤·스케일 정규화에 기여하므로, 기운 얼굴에서도
    눈 crop이 일정하게 프레이밍된다.
  - 흔들림: 원본 bbox 얼굴 crop의 Laplacian variance가 임계 미만이면 흔들림 (모델 불필요, OpenCV만).
    정렬 crop은 warpAffine 보간이 고주파를 뭉개 variance를 왜곡하므로 blur 판정엔 쓰지 않는다.

이미지 단위 판정은 "얼굴 1개라도 해당하면 그 사진을 분리" 규칙으로 집계한다.
모델 로딩(다운로드 포함)은 EyeStateClassifier 생성 시 1회만 일어난다 — 워커 부트스트랩에서
분류기를 생성해 모델을 적재한 뒤 SQS 폴링을 시작한다 (detect/embed와 동일).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from app.core.model_source import ModelSource, default_eye_source

_EYE_INPUT_SIZE = 32  # open-closed-eye-0001 입력 = 1x3x32x32 (NCHW, BGR)
_EYE_PIXEL_MEAN = 127.0  # model.yml: mean 127.0 / scale 255.0 (채널 반전 없음, BGR 그대로)
_EYE_PIXEL_SCALE = 255.0
# softmax 출력의 closed 클래스 인덱스. OMZ 문서는 [open, closed]로 표기하나, face-test 실측에서
# 뜬 눈이 index 1에 ~1.0을 내므로 실제 순서는 [closed, open]이다 → closed는 index 0. (검증: docs 표기 반대)
_CLOSED_INDEX = 0
_EYE_CLASSES = 2

# 정렬 crop(112x112) 안의 고정 눈 중심 = align._ARCFACE_DST의 앞 두 점(우안, 좌안)과 일치해야 한다.
# align은 순수 수학 모듈이나 _ARCFACE_DST는 module-private이라, 좌표를 여기 재선언하고 계약으로 고정한다.
_EYE_CENTERS = ((38.2946, 51.6963), (73.5318, 51.5014))  # (우안, 좌안)


@dataclass(frozen=True)
class QualityConfig:
  """`judge_faces`·`EyeStateClassifier`의 튜닝 파라미터. 기본값은 초기값이며 face-test 실측으로 보정한다."""

  # 흔들림: 얼굴 bbox crop의 Laplacian variance가 이 값 미만이면 흔들림. 절대 스케일이라 [0,1] 아님 —
  # 100.0은 임시 초기값이고, 선명/흔들림 샘플 분포를 보고 확정한다 (검증 2단계).
  blur_threshold: float = 100.0
  # 눈감음: closed 클래스 softmax 확률이 이 값 이상이면 그 눈을 감은 것으로 본다. face-test 실측 보정값 0.85 —
  # 진짜 감은 눈은 min 확률 ≥0.8인데, 뒤통수 오검출(0.65)·안경 실눈(0.52) 같은 약한 오탐이 그 아래로 떨어진다.
  eye_closed_confidence: float = 0.85
  # 정렬 crop에서 고정 눈 좌표 둘레로 자를 정사각 한 변(px). 모델 입력 32로 리사이즈하기 전의 원본 창.
  eye_box_px: int = 24

  def __post_init__(self) -> None:
    # DetectorConfig/ClusterConfig와 같은 정책: 무의미한 값은 생성 시점에 거부한다.
    if self.blur_threshold <= 0.0:
      raise ValueError(f"blur_threshold는 양수여야 합니다. 받은 값: {self.blur_threshold}")
    if not 0.0 <= self.eye_closed_confidence <= 1.0:
      raise ValueError(f"eye_closed_confidence는 [0, 1] 범위여야 합니다. 받은 값: {self.eye_closed_confidence}")
    if self.eye_box_px <= 1:
      raise ValueError(f"eye_box_px는 2 이상이어야 합니다. 받은 값: {self.eye_box_px}")


@dataclass(frozen=True)
class EyeConfig:
  """`EyeStateClassifier`의 튜닝 파라미터. 모델은 32x32라 스레드 설정은 무의미해 노출하지 않는다."""

  model_source: ModelSource | None = None


def blur_variance(crop: np.ndarray) -> float:
  """얼굴 crop의 Laplacian variance — 낮을수록 흔들림/뭉개짐. cv2만 사용(모델 불필요).

  이미 그레이스케일(2D)이면 변환 없이 쓰고, 3채널이면 BGR→GRAY 변환한다 (detect의 방어적 입력 정규화와 같은 철학).
  """
  gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_eye(aligned: np.ndarray, center: tuple[float, float], box_px: int) -> np.ndarray | None:
  """정렬 crop(112x112)에서 center 둘레 정사각을 잘라 32x32 BGR로 리사이즈한다.

  경계에 걸려 유효 영역이 너무 작아지면 None을 반환한다 (얼굴 단위 None 스킵 정책과 일관).
  """
  half = box_px / 2.0
  cx, cy = center
  h, w = aligned.shape[:2]
  x0 = max(0, int(round(cx - half)))
  y0 = max(0, int(round(cy - half)))
  x1 = min(w, int(round(cx + half)))
  y1 = min(h, int(round(cy + half)))
  if x1 - x0 < 2 or y1 - y0 < 2:
    return None
  eye = aligned[y0:y1, x0:x1]
  return cv2.resize(eye, (_EYE_INPUT_SIZE, _EYE_INPUT_SIZE), interpolation=cv2.INTER_AREA)


class EyeStateClassifier:
  """open-closed-eye-0001 눈 상태 분류기.

  `ort.InferenceSession.run`은 스레드 안전하므로 인스턴스 하나를 공유해도 된다.
  모델 파일 획득·세션 생성은 생성자에서 1회만 수행한다 (FaceEmbedder와 동일 계약).
  """

  def __init__(self, config: EyeConfig | None = None) -> None:
    resolved_config = config or EyeConfig()
    source = resolved_config.model_source or default_eye_source()
    model_path = source.resolve()  # 생성 시 모델을 1회만 로딩(필요 시 URL 다운로드)
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # ERROR 미만 로그 억제 (embed.py와 동일)
    self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    model_input = self._session.get_inputs()[0]
    model_output = self._session.get_outputs()[0]
    self._input_name = model_input.name
    self._output_name = model_output.name

    # 잘못된 모델 파일 주입(예: EYE_MODEL_PATH 오설정)을 첫 추론이 아니라 로딩 시점에 잡는다.
    # 출력은 [1, 2, 1, 1] softmax(open, closed) — 배치 외 원소 수가 2가 아니면 다른 모델이다.
    non_batch = model_output.shape[1:]
    if all(isinstance(dim, int) for dim in non_batch):
      elems = int(np.prod(non_batch)) if non_batch else 0
      if elems != _EYE_CLASSES:
        raise ValueError(
          f"눈 상태 모델 출력이 2클래스(open, closed)가 아닙니다. 받은 출력 shape: "
          f"{model_output.shape} (경로: {model_path})"
        )

    # 배치 축이 정수(고정)면 단건 루프, 심볼릭이면 배치 1회 추론 (embed.py 패턴). 이 모델은 보통 고정 1.
    batch_axis = model_input.shape[0]
    self._supports_batch = not isinstance(batch_axis, int)

  def _preprocess(self, eye_crop: np.ndarray) -> np.ndarray:
    """32x32 BGR uint8 눈 crop → (3, 32, 32) float32 블롭. (x-127)/255, 채널 반전 없음(BGR), NCHW."""
    if eye_crop.shape != (_EYE_INPUT_SIZE, _EYE_INPUT_SIZE, 3):
      raise ValueError(f"eye_crop은 shape (32, 32, 3)이어야 합니다. 받은 shape: {eye_crop.shape}")
    blob = np.transpose(eye_crop, (2, 0, 1)).astype(np.float32)
    return (blob - _EYE_PIXEL_MEAN) / _EYE_PIXEL_SCALE

  def closed_prob(self, eye_crops: Sequence[np.ndarray]) -> list[float]:
    """눈 crop들의 closed(감음) 확률을 입력 순서대로 반환한다."""
    if not eye_crops:
      return []
    blobs = [self._preprocess(crop) for crop in eye_crops]
    if self._supports_batch:
      raw = self._session.run([self._output_name], {self._input_name: np.stack(blobs)})[0]
    else:
      raw = np.concatenate(
        [self._session.run([self._output_name], {self._input_name: blob[np.newaxis]})[0] for blob in blobs]
      )
    probs = raw.reshape(raw.shape[0], -1)  # (N, 2) — [1,2,1,1] 등 잉여 축을 평탄화
    return [float(row[_CLOSED_INDEX]) for row in probs]


# 얼굴 1개 = (정렬 crop 또는 None, 원본 bbox crop). 정렬 실패 얼굴은 aligned=None으로 눈감음 판정 제외.
FacePair = tuple[np.ndarray | None, np.ndarray]


def judge_faces(faces: Sequence[FacePair], classifier: EyeStateClassifier, config: QualityConfig) -> tuple[bool, bool]:
  """얼굴별 (정렬 crop, bbox crop) 목록 → 이미지 단위 (eyes_closed, blurry) 판정.

  "얼굴 1개라도" 규칙: 어느 한 얼굴이라도 양눈 감김이면 eyes_closed, 어느 한 얼굴이라도 blur면 blurry.
  양눈이 모두 잡히는 얼굴만 눈감음 후보다 — 옆얼굴 등 한쪽 눈만 잡히면 보수적으로 미판정.
  """
  eyes_closed = False
  blurry = False
  for aligned, bbox_crop in faces:
    if not eyes_closed and aligned is not None:
      eye_crops = [crop_eye(aligned, center, config.eye_box_px) for center in _EYE_CENTERS]
      if all(crop is not None for crop in eye_crops):
        probs = classifier.closed_prob(eye_crops)  # type: ignore[arg-type]  # 위에서 None 배제 확인
        if all(prob >= config.eye_closed_confidence for prob in probs):
          eyes_closed = True
    if not blurry and bbox_crop is not None and bbox_crop.size > 0:
      if blur_variance(bbox_crop) < config.blur_threshold:
        blurry = True
    if eyes_closed and blurry:
      break  # 둘 다 확정되면 나머지 얼굴은 볼 필요 없다
  return eyes_closed, blurry


if __name__ == "__main__":
  # SQS/S3·모델 없이 임계값을 보정한다: 로컬 이미지에서 얼굴별 양눈 closed 확률·blur variance·최종
  # 판정을 출력한다. 눈뜬/감은 샘플에서 closed 확률이 갈리는지, blur 분포가 선명/흔들림을 가르는지 확인.
  # TODO(CHMO-165): pytest 도입 시 tests/로 승격 + 임계값 확정
  import sys

  from app.pipeline.align import align_face
  from app.pipeline.detect import FaceDetector

  detector = FaceDetector()
  classifier = EyeStateClassifier()
  config = QualityConfig()
  for path in sys.argv[1:]:
    # Windows 한글 경로 대응: cv2.imread는 비ASCII 경로에서 None만 반환하므로 fromfile+imdecode를 쓴다
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
      print(f"{path}: 건너뜀 (이미지를 읽을 수 없음)")
      continue

    detected = detector.detect(image)
    faces: list[FacePair] = []
    for i, face in enumerate(detected):
      x, y, bw, bh = face.bbox
      bbox_crop = image[y : y + bh, x : x + bw]
      aligned = align_face(image, face.landmarks)
      faces.append((aligned, bbox_crop))

      probs: list[float] = []
      if aligned is not None:
        eye_crops = [crop_eye(aligned, center, config.eye_box_px) for center in _EYE_CENTERS]
        if all(crop is not None for crop in eye_crops):
          probs = classifier.closed_prob(eye_crops)  # type: ignore[arg-type]
      var = blur_variance(bbox_crop) if bbox_crop.size else float("nan")
      probs_str = ", ".join(f"{p:.3f}" for p in probs) if probs else "n/a"
      print(f"  {path} face{i}: closed_prob=[{probs_str}] blur_var={var:.1f}")

    eyes_closed, blurry = judge_faces(faces, classifier, config)
    print(f"{path}: {len(detected)} face(s) → eyes_closed={eyes_closed}, blurry={blurry}")
