"""순수 파이프라인 단계로서의 사진 품질 판정 (눈감음 CNN + 흔들림 Laplacian).

두 게이트를 제공한다:
  - 눈감음: align.py가 5점(눈·코·입꼬리)으로 정렬한 112x112 crop의 고정 눈 좌표에서 양눈을 잘라
    open-closed-eye-0001 CNN으로 open/closed 분류. 양눈 모두 closed면 그 얼굴은 눈감음.
    입꼬리 랜드마크는 align의 Umeyama 변환을 통해 롤·스케일 정규화에 기여하므로, 기운 얼굴에서도
    눈 crop이 일정하게 프레이밍된다.
  - 흔들림: 원본 bbox 얼굴 crop을 112x112로 리사이즈 + 3x3 가우시안 후 Laplacian variance가 임계 미만이면
    흔들림 (모델 불필요, OpenCV만). 리사이즈는 해상도 의존성 제거(같은 얼굴도 crop이 클수록 variance가
    낮아짐), 가우시안은 고감도 노이즈가 고주파로 잡혀 흔들린 얼굴을 선명으로 오판하는 것을 막는다.
    극소 얼굴(min_blur_face_px 미만)은 정보가 부족해 판정에서 제외하고 전체 이미지 fallback에 맡긴다.
    판정 대상은 주 인물 얼굴뿐이다(가장 큰 얼굴 폭 대비 blur_main_face_ratio 이상) — 배경 인물은
    아웃포커스(피사계 심도)로 뭉개지는 것이 정상 촬영이라 사진 전체의 흔들림 증거가 못 되고, 실제로
    주 인물이 선명한 사진이 배경 얼굴 하나 때문에 blurry로 오분류됐다(event 30 실측, 2026-07-15).
    손떨림이라면 주 인물까지 전부 뭉개지므로 주 인물만 봐도 놓치지 않는다.
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

# 흔들림 판정 전 얼굴 crop을 이 크기로 리사이즈한다 — Laplacian variance는 스케일 불변이 아니라서
# (고해상도일수록 같은 얼굴의 variance가 낮게 나옴) 고정 크기로 정규화해야 단일 임계가 성립한다.
_BLUR_NORM_SIZE = 112
_BLUR_DENOISE_KERNEL = (3, 3)  # 고감도 노이즈 억제 — 야간 사진의 노이즈가 variance를 뻥튀기하는 것 방지

# 정렬 crop(112x112) 안의 고정 눈 중심 = align._ARCFACE_DST의 앞 두 점(우안, 좌안)과 일치해야 한다.
# align은 순수 수학 모듈이나 _ARCFACE_DST는 module-private이라, 좌표를 여기 재선언하고 계약으로 고정한다.
_EYE_CENTERS = ((38.2946, 51.6963), (73.5318, 51.5014))  # (우안, 좌안)

# shake_signals의 측정 정규화: 전체 이미지를 긴 변 이 크기로 축소해 해상도 의존을 없앤다
# (_BLUR_NORM_SIZE와 같은 이유 — 원본 크기로 재면 12MP와 스크린샷에 단일 임계가 성립하지 않는다).
_SHAKE_NORM_MAX_SIDE = 1024
# 구조 텐서 평균에 넣을 그라디언트 크기 상위 백분위 — 약한 그라디언트(민무늬 영역·노이즈)는 방향
# 정보가 무의미해 쏠림을 희석하므로 강한 에지 픽셀만 쓴다 (라벨셋 검증값, ADR 014).
_SHAKE_GRAD_PERCENTILE = 90


@dataclass(frozen=True)
class QualityConfig:
  """`judge_faces`·`EyeStateClassifier`의 튜닝 파라미터. 기본값은 초기값이며 face-test 실측으로 보정한다."""

  # 흔들림: 얼굴 bbox crop의 정규화 Laplacian variance(face_blur_variance)가 이 값 미만이면 흔들림.
  # 절대 스케일이라 [0,1] 아님 — test2 라벨셋 실측 보정값 25.0 (선명 최솟값 28.7 vs 흔들림 최댓값 22.3의
  # 중간, 2026-07-14). 마진이 얇아 실서비스 오탐/미탐 사례가 쌓이면 라벨셋에 추가해 재보정한다.
  blur_threshold: float = 25.0
  # 흔들림 판정 자격의 최소 얼굴 크기(bbox 짧은 변, px). 이보다 작은 얼굴은 픽셀 정보가 부족해
  # variance가 양방향으로 신뢰 불가(선명한 극소 얼굴이 7까지 떨어지거나 노이즈로 186까지 튐 — test2 실측)
  # → 판정에서 제외한다. 판정 자격 얼굴이 하나도 없으면 judge_faces가 blurry=None을 반환하고
  # 호출자가 전체 이미지 fallback으로 처리한다.
  min_blur_face_px: int = 64
  # 흔들림 판정 자격의 상대 크기 하한: 사진에서 가장 큰 얼굴 bbox 폭 대비 이 비율 미만인 얼굴은 주 인물이
  # 아니라고 보고 blur 판정에서 제외한다. 배경 인물의 아웃포커스가 선명한 사진을 blurry로 오분류하는 것
  # 방지 (event 30 실측: 주 인물 variance 62~157 선명, 배경 얼굴 9.2·18.5로 오탐). 0 = 비활성(모든 얼굴 판정).
  blur_main_face_ratio: float = 0.5
  # 얼굴 미검출 시 전체 이미지 Laplacian variance로 흔들림을 판정하는 fallback 임계값 (완전 흔들려
  # 얼굴 검출조차 실패한 사진 구제). 얼굴 crop과 측정 스케일이 달라 별도 설정값이다 — 실측에서 완전
  # 흔들린 전체 이미지는 variance ~1~7(선명 300+)로 폭락하나, 단순한 선명 장면(민무늬 벽 등)은 낮게
  # 나올 수 있어 오탐 방지 차원에서 보수적으로 튜닝한다. 초기값은 blur_threshold와 동일(실측 보정 필요).
  whole_image_blur_threshold: float = 100.0
  # 얼굴 미검출 fallback의 2차 신호 — 방향성 블러 (ADR 014). variance는 텍스처 양을 재는 지표라
  # 배경 무늬가 많은 흔들린 사진을 놓치는데, 손떨림은 모든 에지가 한 방향으로 번져 그라디언트
  # 방향이 쏠린다(구조 텐서 coherence 0~1). 쏠림이 이 값 이상이면 흔들림 후보. 라벨셋 실측:
  # 흔들림 0.397~0.631 vs 선명 최고 0.312 — 중간이 아닌 0.40을 쓰는 이유는 샘플 15장의 얇은
  # 마진에 과적합하지 않기 위한 보수 선택(0.397짜리 1장은 알려진 미탐). 0 = 비활성.
  shake_coherence_threshold: float = 0.40
  # 방향 쏠림이 높아도 정규화 variance(긴 변 1024 축소 후 측정)가 이 값 이상이면 선명으로 본다 —
  # 구도가 단순해 에지 방향이 우연히 쏠린 선명한 사진의 오탐 가드 (라벨셋 child4: 쏠림 0.491이지만
  # variance 236.9로 명백히 선명). 흔들린 사진의 실측 최고는 55.5.
  shake_max_norm_variance: float = 60.0
  # 흔들림 재확인 게이트 — variance가 임계 미달(흔들림)이어도 전체 이미지 방향 쏠림이 이 값 미만이면
  # 손떨림이 아니라 원판 자체가 소프트한 사진(옛날 인화 재촬영·앱 스무딩)으로 보고 blurry를 해제한다.
  # variance는 잔결의 양만 재서 "디테일이 원래 없는 사진"과 "흔들려서 디테일이 뭉개진 사진"을 구분
  # 못하는데, 손떨림은 모든 에지가 한 방향으로 번져 쏠림이 높다(얼굴 crop 쏠림은 판별력 없음 — 실측
  # 흔들림 0.132~0.349 vs 옛날사진 0.044~0.454 겹침). event 50 실측: 옛날 사진 오탐 최고 0.268 vs
  # 흔들림 라벨셋 최저 0.397(얼굴 미검출)·0.444(얼굴 검출) — 빈 구간 중 미탐 리스크가 적은 쪽인
  # 0.35 채택. 2차 신호(shake_coherence_threshold ≥ 이 값)로 잡힌 사진은 정의상 게이트를 통과한다.
  # 한계: 등방성 블러(아웃포커스 주 인물·회전 손떨림)는 쏠림이 낮아 함께 해제된다. 0 = 비활성.
  shake_coherence_floor: float = 0.35
  # 재확인 게이트의 면제 조건(fallback 한정) — 전체 이미지 raw variance가 이 값 미만이면 잔결 붕괴
  # 수준의 블러라 쏠림과 무관하게 흔들림을 유지한다. 고스팅형 손떨림(겹침 번짐)은 에지 방향이 다양해
  # 쏠림이 낮게 나오는데(event 55 실측 13.5/0.306 — 게이트가 오해제), 소프트 원판은 얼굴 미검출이어도
  # variance가 이만큼 붕괴하지 않는다(event 50 무얼굴 옛날 사진 98.9). 13.5와 98.9 사이 40 채택.
  # 얼굴 경로에는 적용하지 않는다 — 블러 프레임을 두른 옛날 사진은 전체 variance가 1.7~24까지
  # 떨어져(합성 블러 배경) 면제가 오탐을 되살린다. 0 = 비활성(게이트 항상 적용).
  whole_image_collapse_variance: float = 40.0
  # 눈감음: closed 클래스 softmax 확률이 이 값 이상이면 그 눈을 감은 것으로 본다. face-test 실측 보정값 0.85 —
  # 진짜 감은 눈은 min 확률 ≥0.8인데, 뒤통수 오검출(0.65)·안경 실눈(0.52) 같은 약한 오탐이 그 아래로 떨어진다.
  eye_closed_confidence: float = 0.85
  # 정렬 crop에서 고정 눈 좌표 둘레로 자를 정사각 한 변(px). 모델 입력 32로 리사이즈하기 전의 원본 창.
  eye_box_px: int = 24

  def __post_init__(self) -> None:
    # DetectorConfig/ClusterConfig와 같은 정책: 무의미한 값은 생성 시점에 거부한다.
    if self.blur_threshold <= 0.0:
      raise ValueError(f"blur_threshold는 양수여야 합니다. 받은 값: {self.blur_threshold}")
    if self.min_blur_face_px <= 0:
      raise ValueError(f"min_blur_face_px는 양수여야 합니다. 받은 값: {self.min_blur_face_px}")
    if not 0.0 <= self.blur_main_face_ratio <= 1.0:
      raise ValueError(f"blur_main_face_ratio는 [0, 1] 범위여야 합니다. 받은 값: {self.blur_main_face_ratio}")
    if self.whole_image_blur_threshold <= 0.0:
      raise ValueError(f"whole_image_blur_threshold는 양수여야 합니다. 받은 값: {self.whole_image_blur_threshold}")
    if not 0.0 <= self.shake_coherence_threshold <= 1.0:
      raise ValueError(f"shake_coherence_threshold는 [0, 1] 범위여야 합니다. 받은 값: {self.shake_coherence_threshold}")
    if self.shake_max_norm_variance <= 0.0:
      raise ValueError(f"shake_max_norm_variance는 양수여야 합니다. 받은 값: {self.shake_max_norm_variance}")
    if not 0.0 <= self.shake_coherence_floor <= 1.0:
      raise ValueError(f"shake_coherence_floor는 [0, 1] 범위여야 합니다. 받은 값: {self.shake_coherence_floor}")
    if self.whole_image_collapse_variance < 0.0:
      raise ValueError(
        f"whole_image_collapse_variance는 0 이상이어야 합니다. 받은 값: {self.whole_image_collapse_variance}"
      )
    if not 0.0 <= self.eye_closed_confidence <= 1.0:
      raise ValueError(f"eye_closed_confidence는 [0, 1] 범위여야 합니다. 받은 값: {self.eye_closed_confidence}")
    if self.eye_box_px <= 1:
      raise ValueError(f"eye_box_px는 2 이상이어야 합니다. 받은 값: {self.eye_box_px}")


@dataclass(frozen=True)
class EyeConfig:
  """`EyeStateClassifier`의 튜닝 파라미터. 모델은 32x32라 스레드 설정은 무의미해 노출하지 않는다."""

  model_source: ModelSource | None = None


def blur_variance(crop: np.ndarray) -> float:
  """이미지의 원시 Laplacian variance — 낮을수록 흔들림/뭉개짐. cv2만 사용(모델 불필요).

  이미 그레이스케일(2D)이면 변환 없이 쓰고, 3채널이면 BGR→GRAY 변환한다 (detect의 방어적 입력 정규화와 같은 철학).
  전체 이미지 fallback 판정용 — 얼굴 crop 판정은 정규화가 들어간 face_blur_variance를 쓴다.
  """
  gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_blur_variance(crop: np.ndarray) -> float:
  """얼굴 bbox crop의 정규화 Laplacian variance — 112x112 리사이즈 + 3x3 가우시안 후 측정.

  원시 variance는 crop 해상도에 반비례해(큰 선명 얼굴이 작은 흔들린 얼굴보다 낮게 나옴) 단일 임계가
  성립하지 않는다 — test2 라벨셋에서 원시값은 선명 21 vs 흔들림 186으로 역전됐으나, 정규화 후
  선명 최솟값 28.7 vs 흔들림 최댓값 22.3으로 분리됐다 (2026-07-14 실측).
  """
  gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  resized = cv2.resize(gray, (_BLUR_NORM_SIZE, _BLUR_NORM_SIZE), interpolation=cv2.INTER_AREA)
  denoised = cv2.GaussianBlur(resized, _BLUR_DENOISE_KERNEL, 0)
  return float(cv2.Laplacian(denoised, cv2.CV_64F).var())


def shake_signals(image: np.ndarray) -> tuple[float, float]:
  """전체 이미지의 (정규화 variance, 방향 쏠림)을 잰다 — 얼굴 미검출 fallback의 방향성 블러 검출용 (ADR 014).

  정규화 variance: 긴 변 1024 축소 + 3x3 가우시안 후 Laplacian variance — 원본 크기로 재는
  blur_variance와 달리 해상도 의존이 없어 단일 임계가 성립한다.
  방향 쏠림: 그라디언트 크기 상위 10% 픽셀의 구조 텐서 평균에서 sqrt((Jxx-Jyy)² + 4·Jxy²)/(Jxx+Jyy).
  0 = 에지 방향이 제각각(등방, 선명한 일반 사진), 1 = 전부 한 방향(손떨림이 모든 에지를 같은
  방향으로 번지게 한 사진). 텍스처 양과 무관한 신호라 variance가 놓치는 사진을 잡는다.
  """
  gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  scale = _SHAKE_NORM_MAX_SIDE / max(gray.shape)
  if scale < 1.0:
    size = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
  denoised = cv2.GaussianBlur(gray, _BLUR_DENOISE_KERNEL, 0)
  norm_var = float(cv2.Laplacian(denoised, cv2.CV_64F).var())

  gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  magnitude = gx * gx + gy * gy
  strong = magnitude >= np.percentile(magnitude, _SHAKE_GRAD_PERCENTILE)
  jxx = float((gx * gx)[strong].mean())
  jyy = float((gy * gy)[strong].mean())
  jxy = float((gx * gy)[strong].mean())
  trace = jxx + jyy
  coherence = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / trace) if trace > 0 else 0.0
  return norm_var, coherence


def shake_confirmed(image: np.ndarray, config: QualityConfig) -> bool:
  """variance가 흔들림이라 한 사진을 전체 이미지 방향 쏠림으로 재확인한다 (shake_coherence_floor 주석 참고).

  True = 쏠림이 바닥값 이상(손떨림 정합) → blurry 유지. False = 등방(원판이 소프트한 사진) → 해제.
  얼굴 판정·전체 이미지 fallback 공통 최종 게이트로, blurry=True일 때만 호출한다.
  """
  if config.shake_coherence_floor <= 0:
    return True
  _, coherence = shake_signals(image)
  return coherence >= config.shake_coherence_floor


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


def judge_faces(
  faces: Sequence[FacePair], classifier: EyeStateClassifier, config: QualityConfig
) -> tuple[bool, bool | None]:
  """얼굴별 (정렬 crop, bbox crop) 목록 → 이미지 단위 (eyes_closed, blurry) 판정.

  "얼굴 1개라도" 규칙: 어느 한 얼굴이라도 양눈 감김이면 eyes_closed, 어느 한 얼굴이라도 blur면 blurry.
  양눈이 모두 잡히는 얼굴만 눈감음 후보다 — 옆얼굴 등 한쪽 눈만 잡히면 보수적으로 미판정.
  blurry는 주 인물 얼굴만 본다 — 판정 자격은 bbox 짧은 변 ≥ min_blur_face_px 그리고 bbox 폭이 사진 내
  가장 큰 얼굴 폭의 blur_main_face_ratio 이상. 배경 인물의 아웃포커스는 흔들림 증거가 아니다(모듈 주석).
  판정 자격 얼굴이 하나도 없으면 None — 얼굴 미검출과 같은 "얼굴로는 알 수 없음"이므로 호출자가
  전체 이미지 fallback으로 판정한다.
  """
  widest = max(
    (crop.shape[1] for _, crop in faces if crop is not None and crop.size > 0),
    default=0,
  )
  min_main_width = config.blur_main_face_ratio * widest
  eyes_closed = False
  blurry: bool | None = None
  for aligned, bbox_crop in faces:
    if not eyes_closed and aligned is not None:
      eye_crops = [crop_eye(aligned, center, config.eye_box_px) for center in _EYE_CENTERS]
      if all(crop is not None for crop in eye_crops):
        probs = classifier.closed_prob(eye_crops)  # type: ignore[arg-type]  # 위에서 None 배제 확인
        if all(prob >= config.eye_closed_confidence for prob in probs):
          eyes_closed = True
    if blurry is not True and bbox_crop is not None and bbox_crop.size > 0:
      if min(bbox_crop.shape[:2]) >= config.min_blur_face_px and bbox_crop.shape[1] >= min_main_width:
        blurry = face_blur_variance(bbox_crop) < config.blur_threshold
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
    for face in detected:
      x, y, bw, bh = face.bbox
      faces.append((align_face(image, face.landmarks), image[y : y + bh, x : x + bw]))

    widest = max((crop.shape[1] for _, crop in faces if crop.size > 0), default=0)
    for i, (aligned, bbox_crop) in enumerate(faces):
      probs: list[float] = []
      if aligned is not None:
        eye_crops = [crop_eye(aligned, center, config.eye_box_px) for center in _EYE_CENTERS]
        if all(crop is not None for crop in eye_crops):
          probs = classifier.closed_prob(eye_crops)  # type: ignore[arg-type]
      var = face_blur_variance(bbox_crop) if bbox_crop.size else float("nan")
      note = ""
      if not (bbox_crop.size > 0 and min(bbox_crop.shape[:2]) >= config.min_blur_face_px):
        note = " (극소 얼굴 — blur 판정 제외)"
      elif bbox_crop.shape[1] < config.blur_main_face_ratio * widest:
        note = " (배경 얼굴 — blur 판정 제외)"
      probs_str = ", ".join(f"{p:.3f}" for p in probs) if probs else "n/a"
      bh, bw = bbox_crop.shape[:2]
      print(f"  {path} face{i} bbox={bw}x{bh}: closed_prob=[{probs_str}] blur_var={var:.1f}{note}")

    eyes_closed, blurry = judge_faces(faces, classifier, config)
    gate_exempt = False
    if blurry is None:
      # deps.build_face_extractor와 같은 fallback — CLI 판정이 프로덕션 라우팅과 일치해야 보정에 쓸 수 있다
      whole_var = blur_variance(image)
      blurry = whole_var < config.whole_image_blur_threshold
      gate_exempt = blurry and whole_var < config.whole_image_collapse_variance
      norm_var, coherence = shake_signals(image)
      if not blurry and config.shake_coherence_threshold > 0:
        blurry = coherence >= config.shake_coherence_threshold and norm_var < config.shake_max_norm_variance
      print(
        f"  {path}: blur 판정 자격 얼굴 없음 → 전체 이미지 fallback "
        f"(whole_var={whole_var:.1f} norm_var={norm_var:.1f} coherence={coherence:.3f})"
      )
      if gate_exempt:
        print(f"  {path}: variance 붕괴 수준 → 쏠림 재확인 면제 (흔들림 확정)")
    if blurry and not gate_exempt and not shake_confirmed(image, config):
      blurry = False
      print(f"  {path}: variance는 흔들림이나 방향 쏠림 미달 → 소프트 원판으로 보고 해제")
    print(f"{path}: {len(detected)} face(s) → eyes_closed={eyes_closed}, blurry={blurry}")
