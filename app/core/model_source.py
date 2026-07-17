"""AI 모델 파일 위치의 단일 진실 소스(single source of truth).

모든 파이프라인 코드는 경로를 하드코딩하지 않고 `ModelSource`를 통해 모델 파일을
얻는다. 모델을 얻는 새로운 방식(예: S3, 공유 네트워크 마운트, 번들 리소스)을 추가하려면
`resolve()` 하나만 가진 `ModelSource` 프로토콜을 구현하고 여기에 팩토리를 노출하면 된다.
모델 위치가 바뀌어도 이 모듈(또는 해당 환경변수)만 수정하면 된다.
"""

import os
from typing import Protocol, runtime_checkable

YUNET_REPO_ID = "opencv/face_detection_yunet"
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH_ENV = "YUNET_MODEL_PATH"

# PoC는 snapshot_download로 레포 전체(~427MB)를 받았지만, 임베딩에는 glintr100.onnx(~261MB)만
# 필요하므로 단일 파일 다운로드를 사용한다 — 받은 파일 내용은 동일하다.
AURAFACE_REPO_ID = "fal/AuraFace-v1"
AURAFACE_MODEL_FILENAME = "glintr100.onnx"
AURAFACE_MODEL_PATH_ENV = "AURAFACE_MODEL_PATH"

# 눈감음 판정 CNN(open-closed-eye-0001). YuNet/AuraFace와 달리 Hugging Face Hub에 상업용 라이선스로
# 존재하지 않는다 — Apache-2.0 ONNX는 OpenVINO Open Model Zoo 스토리지 직접 URL에만 있다
# (HF 미러는 cc-by-nc-sa 비상업이라 제외). 그래서 HybridModelSource(HF)가 아니라 UrlModelSource로 받는다.
EYE_MODEL_URL = (
  "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/"
  "open-closed-eye-0001/open_closed_eye.onnx"
)
EYE_MODEL_FILENAME = "open_closed_eye.onnx"
EYE_MODEL_PATH_ENV = "EYE_MODEL_PATH"

# MediaPipe Face Landmarker 번들(.task = tflite 3개를 담은 zip, Apache 2.0) — 눈감음 하이브리드
# 판정(ADR 021)이 내부의 face_landmarks_detector.tflite + face_blendshapes.tflite만 litert로 실행한다
# (mediapipe pip 패키지는 linux aarch64 휠이 없어 EC2 배포 불가 — 런타임은 ai-edge-litert).
# 버전 고정 경로(/1/)를 쓴다 — latest는 모델 교체 시 파리티 검증 없이 동작이 바뀐다.
FACELANDMARKER_MODEL_URL = (
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACELANDMARKER_MODEL_FILENAME = "face_landmarker.task"
FACELANDMARKER_MODEL_PATH_ENV = "FACELANDMARKER_MODEL_PATH"


@runtime_checkable
class ModelSource(Protocol):
  """로컬 파일시스템에서 모델 파일을 찾아내는 인터페이스.

  유일한 메서드 `resolve()`가 확장 지점이다. 새로운 획득 전략을 지원하려면 이 메서드만
  구현하면 되고, 파이프라인 코드는 전혀 손대지 않는다.
  """

  def resolve(self) -> str: ...


class LocalModelSource:
  """고정된 로컬 경로에 이미 존재하는 모델 파일."""

  def __init__(self, path: str) -> None:
    self._path = path

  def resolve(self) -> str:
    if not os.path.isfile(self._path):
      raise FileNotFoundError(
        f"모델 파일을 '{self._path}' 에서 찾을 수 없습니다. 올바른 경로를 지정하거나, "
        f"오버라이드를 해제해 기본 다운로드 소스로 폴백하세요."
      )
    return self._path


class HuggingFaceModelSource:
  """Hugging Face Hub에서 내려받아(캐시 포함) 사용하는 모델 파일."""

  def __init__(self, repo_id: str, filename: str) -> None:
    self._repo_id = repo_id
    self._filename = filename

  def resolve(self) -> str:
    # 지연 import: 다운로드 경로를 실제로 탈 때(로컬 오버라이드가 없을 때)만 필요하도록
    # huggingface_hub를 선택적 의존성으로 유지한다.
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=self._repo_id, filename=self._filename)


class UrlModelSource:
  """직접 URL에서 내려받아 로컬 캐시에 저장해 재사용하는 모델 파일 (Hub에 없는 모델용).

  HuggingFaceModelSource가 hf_hub_download에 위임하는 캐싱을, HF에 없는 모델(open-closed-eye-0001)에
  대해 직접 구현한다. 최초 1회만 다운로드하고 이후엔 캐시 히트로 즉시 반환한다.
  """

  def __init__(self, url: str, filename: str, cache_dir: str | None = None) -> None:
    self._url = url
    self._filename = filename
    # 기본 캐시 위치는 사용자 홈의 ~/.cache/cheesemoa (Windows에서도 expanduser로 해석된다).
    # CHEESEMOA_CACHE_DIR로 오버라이드 가능 — 컨테이너/CI에서 캐시 볼륨을 지정하기 위함.
    self._cache_dir = cache_dir or os.path.join(
      os.getenv("CHEESEMOA_CACHE_DIR") or os.path.expanduser("~/.cache/cheesemoa"), "models"
    )

  def resolve(self) -> str:
    # 지연 import: 실제 다운로드 경로를 탈 때만 필요하다 (로컬 오버라이드·캐시 히트 시엔 불필요).
    import shutil
    import ssl
    import tempfile
    import urllib.request

    target = os.path.join(self._cache_dir, self._filename)
    if os.path.isfile(target):
      return target  # 캐시 히트 — 재다운로드 없음

    # huggingface_hub(requests)와 동일하게 certifi CA 번들을 쓴다 — OS 스토어에 없는 공개 CA도 검증되도록.
    # certifi가 없으면 urllib 기본 컨텍스트로 폴백한다 (선택적 의존성).
    try:
      import certifi

      context: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
      context = None

    os.makedirs(self._cache_dir, exist_ok=True)
    # 부분 다운로드가 캐시로 남지 않도록 임시파일에 받은 뒤 원자적 rename으로 확정한다.
    # os.fdopen을 바깥 with로 두어, 안쪽 urlopen이 예외를 던져도 fd가 확실히 닫히게 한다
    # (Windows에서 열린 fd가 남으면 아래 os.remove가 PermissionError로 실패한다).
    fd, tmp_path = tempfile.mkstemp(dir=self._cache_dir, suffix=".part")
    try:
      with os.fdopen(fd, "wb") as tmp_file:
        with urllib.request.urlopen(self._url, timeout=60, context=context) as response:
          shutil.copyfileobj(response, tmp_file)
      os.replace(tmp_path, target)  # 같은 디렉터리 내 rename이라 원자적
    except BaseException:
      # 실패 시 임시파일 흔적을 남기지 않는다 (다음 시도가 깨끗한 상태에서 재다운로드하도록)
      if os.path.exists(tmp_path):
        os.remove(tmp_path)
      raise
    return target


class HybridModelSource:
  """로컬 오버라이드가 있으면 우선 사용하고, 없으면 Hub에서 다운로드한다."""

  def __init__(self, local_path: str | None, repo_id: str, filename: str) -> None:
    self._local_path = local_path
    self._repo_id = repo_id
    self._filename = filename

  def resolve(self) -> str:
    if self._local_path:
      return LocalModelSource(self._local_path).resolve()
    return HuggingFaceModelSource(self._repo_id, self._filename).resolve()


def default_yunet_source() -> ModelSource:
  """YuNet 검출기가 모델 파일을 얻을 때 사용하는 기본 소스."""
  return HybridModelSource(os.getenv(YUNET_MODEL_PATH_ENV), YUNET_REPO_ID, YUNET_MODEL_FILENAME)


def default_auraface_source() -> ModelSource:
  """AuraFace 임베더가 모델 파일을 얻을 때 사용하는 기본 소스."""
  return HybridModelSource(os.getenv(AURAFACE_MODEL_PATH_ENV), AURAFACE_REPO_ID, AURAFACE_MODEL_FILENAME)


def default_eye_source() -> ModelSource:
  """눈감음 판정 CNN(open-closed-eye-0001)이 모델 파일을 얻을 때 사용하는 기본 소스.

  default_yunet/auraface와 동형(로컬 오버라이드 우선)이지만, Hub 대신 UrlModelSource로 URL 다운로드한다.
  """
  local_path = os.getenv(EYE_MODEL_PATH_ENV)
  if local_path:
    return LocalModelSource(local_path)
  return UrlModelSource(EYE_MODEL_URL, EYE_MODEL_FILENAME)


def default_face_landmarker_source() -> ModelSource:
  """FaceBlinkScorer가 face_landmarker.task 번들을 얻을 때 사용하는 기본 소스.

  default_eye_source와 동형(로컬 오버라이드 → URL 다운로드). .task는 zip 그대로 캐시하고
  내부 tflite 추출은 소비자(app/pipeline/blink.py)의 책임이다.
  """
  local_path = os.getenv(FACELANDMARKER_MODEL_PATH_ENV)
  if local_path:
    return LocalModelSource(local_path)
  return UrlModelSource(FACELANDMARKER_MODEL_URL, FACELANDMARKER_MODEL_FILENAME)
