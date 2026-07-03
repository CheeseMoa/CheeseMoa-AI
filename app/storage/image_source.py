"""원본 이미지 획득 인터페이스 — S3 구현과 인메모리 페이크.

classify 요청의 s3_key로 디코딩된 BGR ndarray를 돌려준다 (detect.py 입력 계약).
원본 버킷은 Spring 소유라 워커는 읽기 전용이다. 획득·디코드 실패는 `ImageFetchError`
하나로 수렴한다 — 핸들러가 이미지 단위로 격리해 failed_images로 보고하는 계약 (feature-spec §9).
"""

from typing import Protocol

import cv2
import numpy as np


class ImageFetchError(Exception):
  """이미지 1장을 가져오거나 디코드하지 못함 — 작업 전체가 아니라 해당 이미지만 실패 처리한다."""


class ImageSource(Protocol):
  """s3_key → 디코딩된 BGR ndarray."""

  def fetch(self, s3_key: str) -> np.ndarray: ...


class S3ImageSource:
  """S3 원본 버킷에서 이미지를 내려받아 디코딩하는 프로덕션 구현."""

  def __init__(self, s3_client, bucket: str) -> None:
    self._client = s3_client
    self._bucket = bucket

  def fetch(self, s3_key: str) -> np.ndarray:
    try:
      body = self._client.get_object(Bucket=self._bucket, Key=s3_key)["Body"].read()
    except Exception as exc:
      # 미존재 키·권한·네트워크 등 원인 유형과 무관하게 이미지 단위 실패로 수렴한다 — 전 이미지가
      # 실패하는 인프라 장애도 partial + 전건 failed_images로 표면화되어 재시도 대상이 된다.
      raise ImageFetchError(f"이미지를 가져올 수 없습니다: {s3_key} ({type(exc).__name__}: {exc})") from exc
    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
      raise ImageFetchError(f"이미지를 디코드할 수 없습니다: {s3_key}")
    return image


class InMemoryImageSource:
  """스모크/테스트용 페이크 — s3_key → 사전 등록된 BGR ndarray."""

  def __init__(self, images: dict[str, np.ndarray] | None = None) -> None:
    self.images: dict[str, np.ndarray] = dict(images or {})

  def fetch(self, s3_key: str) -> np.ndarray:
    if s3_key not in self.images:
      raise ImageFetchError(f"이미지를 가져올 수 없습니다: {s3_key} (등록되지 않은 키)")
    return self.images[s3_key]


if __name__ == "__main__":
  source = InMemoryImageSource({"a.jpg": np.zeros((4, 4, 3), dtype=np.uint8)})
  if source.fetch("a.jpg").shape != (4, 4, 3):
    raise SystemExit("실패: 등록된 이미지 fetch")
  try:
    source.fetch("없음.jpg")
  except ImageFetchError:
    print("통과: InMemoryImageSource fetch + 미등록 키 거부")
  else:
    raise SystemExit("실패: 미등록 키 — ImageFetchError가 발생해야 하는데 통과됨")
