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
