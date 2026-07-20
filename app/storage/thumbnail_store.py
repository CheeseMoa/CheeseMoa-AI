"""인물 앨범 대표 얼굴 썸네일 JPEG의 저장소 인터페이스 — S3 구현과 인메모리 페이크 (CHMO-335).

핸들러는 이 Protocol만 알고 S3를 모른다 (embedding_store와 같은 구도). 키는
`{prefix}{event_id}/{cluster_id}.jpg` 고정 — 재군집으로 대표가 바뀌면 같은 키에 덮어쓴다
(Spring은 매 조회 presigned URL을 발급하므로 항상 최신 객체를 서빙한다). 예외는 삼키지 않고
전파한다 — 썸네일 실패가 job을 죽이지 않게 하는 best-effort 격리는 handlers의 책임이다.
"""

from typing import Protocol


class ThumbnailStore(Protocol):
  """(event_id, cluster_id) 단위 썸네일 JPEG 저장/삭제 인터페이스."""

  def put(self, event_id: str, cluster_id: str, jpeg: bytes) -> str:
    """썸네일을 덮어쓰고 저장된 키를 반환한다 — 반환 키가 결과 메시지의 thumbnail_s3_key가 된다."""
    ...

  def delete(self, event_id: str, cluster_id: str) -> None:
    """은퇴한 클러스터의 썸네일을 지운다. 없는 키 삭제도 성공(멱등 — S3 delete_object 시맨틱스)."""
    ...


class S3ThumbnailStore:
  """`s3://{bucket}/{prefix}{event_id}/{cluster_id}.jpg`를 쓰고 지우는 프로덕션 구현 (워커 소유 버킷)."""

  def __init__(self, s3_client, bucket: str, prefix: str = "thumbnails/") -> None:
    self._client = s3_client
    self._bucket = bucket
    self._prefix = prefix

  def _key(self, event_id: str, cluster_id: str) -> str:
    return f"{self._prefix}{event_id}/{cluster_id}.jpg"

  def put(self, event_id: str, cluster_id: str, jpeg: bytes) -> str:
    key = self._key(event_id, cluster_id)
    self._client.put_object(Bucket=self._bucket, Key=key, Body=jpeg, ContentType="image/jpeg")
    return key

  def delete(self, event_id: str, cluster_id: str) -> None:
    self._client.delete_object(Bucket=self._bucket, Key=self._key(event_id, cluster_id))


class InMemoryThumbnailStore:
  """스모크/테스트용 페이크 — 키 규칙은 S3 구현과 동일하게 유지한다 (결과 메시지 검증에 쓰인다)."""

  def __init__(self, prefix: str = "thumbnails/") -> None:
    self._prefix = prefix
    self.blobs: dict[str, bytes] = {}

  def _key(self, event_id: str, cluster_id: str) -> str:
    return f"{self._prefix}{event_id}/{cluster_id}.jpg"

  def put(self, event_id: str, cluster_id: str, jpeg: bytes) -> str:
    key = self._key(event_id, cluster_id)
    self.blobs[key] = jpeg
    return key

  def delete(self, event_id: str, cluster_id: str) -> None:
    self.blobs.pop(self._key(event_id, cluster_id), None)


if __name__ == "__main__":
  # 페이크의 키 규칙·덮어쓰기·멱등 삭제를 확인한다 (S3 구현은 AWS 없이 검증 불가 — embedding_store와 동일).
  store = InMemoryThumbnailStore()
  key = store.put("event-1", "person-A", b"jpeg-1")
  if key != "thumbnails/event-1/person-A.jpg" or store.blobs[key] != b"jpeg-1":
    raise SystemExit("실패: put 키 형식 또는 저장 내용")
  if store.put("event-1", "person-A", b"jpeg-2") != key or store.blobs[key] != b"jpeg-2":
    raise SystemExit("실패: 같은 키 덮어쓰기")
  store.delete("event-1", "person-A")
  store.delete("event-1", "person-A")  # 없는 키 삭제도 성공 (멱등)
  if store.blobs:
    raise SystemExit("실패: delete 후 잔존 blob")
  print("통과: InMemoryThumbnailStore 키 규칙·덮어쓰기·멱등 삭제")
