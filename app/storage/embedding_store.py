"""event .npz의 저장소 인터페이스 — S3 구현과 인메모리 페이크 (ADR-007).

핸들러는 이 Protocol만 알고 S3를 모른다. 버킷·프리픽스 등 실주소는 core.deps가
Settings에서 읽어 S3 구현에 주입한다. 페이크는 스모크/테스트가 직접 조립한다.
"""

from typing import Protocol

from app.storage.event_embeddings import EventEmbeddings


class EmbeddingStore(Protocol):
  """event_id 단위 EventEmbeddings 로드/저장 인터페이스."""

  def load(self, event_id: str) -> EventEmbeddings | None:
    """저장된 event를 복원한다. 저장된 적 없으면 None (= 최초 분류)."""
    ...

  def save(self, event_id: str, data: EventEmbeddings) -> None:
    """event 전체를 덮어쓴다 — 단일 FIFO 큐(messageGroupId=event_id)가 동시 쓰기를 직렬화하므로
    이 워커가 event 파일의 단일 writer다 (ADR-007)."""
    ...


class S3EmbeddingStore:
  """`s3://{bucket}/{prefix}{event_id}.npz`를 읽고 쓰는 프로덕션 구현."""

  def __init__(self, s3_client, bucket: str, prefix: str = "embeddings/") -> None:
    self._client = s3_client
    self._bucket = bucket
    self._prefix = prefix

  def _key(self, event_id: str) -> str:
    return f"{self._prefix}{event_id}.npz"

  def load(self, event_id: str) -> EventEmbeddings | None:
    try:
      response = self._client.get_object(Bucket=self._bucket, Key=self._key(event_id))
    except self._client.exceptions.NoSuchKey:
      return None
    return EventEmbeddings.from_npz_bytes(response["Body"].read())

  def save(self, event_id: str, data: EventEmbeddings) -> None:
    self._client.put_object(Bucket=self._bucket, Key=self._key(event_id), Body=data.to_npz_bytes())


class InMemoryEmbeddingStore:
  """스모크/테스트용 페이크 — bytes로 저장해 직렬화 코덱·로드 불변식까지 실제로 태운다."""

  def __init__(self) -> None:
    self.blobs: dict[str, bytes] = {}

  def load(self, event_id: str) -> EventEmbeddings | None:
    blob = self.blobs.get(event_id)
    if blob is None:
      return None
    return EventEmbeddings.from_npz_bytes(blob)

  def save(self, event_id: str, data: EventEmbeddings) -> None:
    self.blobs[event_id] = data.to_npz_bytes()


if __name__ == "__main__":
  # 페이크가 코덱을 실제로 왕복시키는지 확인한다 (S3 구현은 AWS 없이 검증 불가 — worker 스모크에서 제외).
  import numpy as np

  from app.storage.event_embeddings import EMBED_DIM

  store = InMemoryEmbeddingStore()
  if store.load("event-1") is not None:
    raise SystemExit("실패: 저장 전 load는 None이어야 함")

  vector = np.zeros(EMBED_DIM, dtype=np.float32)
  vector[0] = 1.0
  event = EventEmbeddings.empty().append_faces([("face-1", "img-1", vector)])
  store.save("event-1", event)
  if store.load("event-1") != event:
    raise SystemExit("실패: 저장-로드 왕복이 원본과 달라짐")
  print("통과: InMemoryEmbeddingStore 왕복 (코덱 경유)")
