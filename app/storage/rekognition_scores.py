"""Rekognition 재판정 점수 캐시의 저장소 인터페이스 — S3 구현과 인메모리 페이크 (ADR-030).

event별 S3 JSON 객체 1개에 (source_face_id, rep_face_id) → Similarity(0~100) 맵을 보관한다.
재군집·보정마다 재판정이 다시 돌아도 같은 쌍은 캐시에서 읽어 CompareFaces 재과금이 없다.
face_id 기준 키라 삭제·append로 행 인덱스가 바뀌어도 안전하다. "crop에서 얼굴 미검출"은 -1.0
센티널로 캐싱해(영구 재과금 방지) 일시 장애와 구분한다 — 일시 장애는 캐싱하지 않는다.

핸들러는 이 Protocol만 알고 S3를 모른다 (embedding_store와 같은 구도). 캐시는 소모품이다:
손상 객체는 로그 후 빈 캐시로 취급하고(재측정하면 그만 — .npz의 StoreCorruptionError 보존
정책과 의도적으로 다르다), best-effort 격리(실패가 job을 죽이지 않게)는 handlers의 책임이다.
"""

import json
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# 캐시 객체 자체의 스키마 버전 — event .npz의 SCHEMA_VERSION과 무관한 독립 계약
_SCHEMA_VERSION = 1

ScorePair = tuple[str, str]  # (source_face_id, rep_face_id) — CompareFaces는 비대칭이라 방향 있는 쌍


def encode_scores(scores: dict[ScorePair, float]) -> bytes:
  """점수 맵을 JSON 바이트로 직렬화한다 — 키 정렬로 같은 내용이면 같은 바이트(결정성, 감사 diff 용이)."""
  rows = [[a, b, float(sim)] for (a, b), sim in sorted(scores.items())]
  return json.dumps({"schema_version": _SCHEMA_VERSION, "scores": rows}, ensure_ascii=False).encode("utf-8")


def decode_scores(blob: bytes) -> dict[ScorePair, float]:
  """JSON 바이트를 점수 맵으로 복원한다. 손상은 로그 후 빈 캐시 — 캐시는 소모품이라 재측정으로 자연 회복된다."""
  try:
    payload = json.loads(blob)
    if payload["schema_version"] != _SCHEMA_VERSION:
      raise ValueError(f"지원하지 않는 schema_version: {payload['schema_version']}")
    scores = {}
    for a, b, sim in payload["scores"]:
      if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("face_id 쌍은 문자열이어야 합니다")
      scores[(a, b)] = float(sim)
    return scores
  except (ValueError, KeyError, TypeError) as exc:
    logger.warning("Rekognition 점수 캐시 손상 — 빈 캐시로 취급: %s", exc)
    return {}


class RekognitionScoreStore(Protocol):
  """event_id 단위 (face_id, rep_face_id) → Similarity 점수 캐시 인터페이스."""

  def load(self, event_id: str) -> dict[ScorePair, float]:
    """저장된 캐시를 복원한다. 저장된 적 없거나 손상이면 빈 dict."""
    ...

  def save(self, event_id: str, scores: dict[ScorePair, float]) -> None:
    """캐시 전체를 덮어쓴다 — 죽은 face_id 프루닝은 호출자(handlers)의 책임이다."""
    ...

  def delete(self, event_id: str) -> None:
    """event 전체 삭제 시 캐시도 지운다(생체 파생 정보 위생). 없는 키 삭제도 성공(멱등)."""
    ...


class S3RekognitionScoreStore:
  """`s3://{bucket}/{prefix}{event_id}.json`을 읽고 쓰는 프로덕션 구현 (워커 소유 embeddings 버킷)."""

  def __init__(self, s3_client, bucket: str, prefix: str = "rekognition-scores/") -> None:
    self._client = s3_client
    self._bucket = bucket
    self._prefix = prefix

  def _key(self, event_id: str) -> str:
    return f"{self._prefix}{event_id}.json"

  def load(self, event_id: str) -> dict[ScorePair, float]:
    try:
      response = self._client.get_object(Bucket=self._bucket, Key=self._key(event_id))
    except self._client.exceptions.NoSuchKey:
      return {}
    return decode_scores(response["Body"].read())

  def save(self, event_id: str, scores: dict[ScorePair, float]) -> None:
    self._client.put_object(
      Bucket=self._bucket, Key=self._key(event_id), Body=encode_scores(scores), ContentType="application/json"
    )

  def delete(self, event_id: str) -> None:
    self._client.delete_object(Bucket=self._bucket, Key=self._key(event_id))


class InMemoryRekognitionScoreStore:
  """스모크/테스트용 페이크 — bytes로 저장해 직렬화 코덱까지 실제로 태운다 (InMemoryEmbeddingStore와 동일 구도)."""

  def __init__(self) -> None:
    self.blobs: dict[str, bytes] = {}

  def load(self, event_id: str) -> dict[ScorePair, float]:
    blob = self.blobs.get(event_id)
    if blob is None:
      return {}
    return decode_scores(blob)

  def save(self, event_id: str, scores: dict[ScorePair, float]) -> None:
    self.blobs[event_id] = encode_scores(scores)

  def delete(self, event_id: str) -> None:
    self.blobs.pop(event_id, None)


if __name__ == "__main__":
  # S3 없이 코덱·페이크 계약을 자가 검증한다 (embedding_store __main__과 같은 실행형 확인 패턴).
  # TODO(CHMO-165): pytest 도입 시 tests/storage/test_rekognition_scores.py로 승격
  logging.basicConfig(level="ERROR")  # 의도된 손상 케이스의 WARNING 로그 숨김
  store = InMemoryRekognitionScoreStore()
  if store.load("event-1") != {}:
    raise SystemExit("실패: 저장 전 load는 빈 dict여야 함")

  scores = {("face-1", "face-9"): 91.2, ("face-2", "face-9"): -1.0}  # -1.0 = 얼굴 미검출 센티널
  store.save("event-1", scores)
  if store.load("event-1") != scores:
    raise SystemExit("실패: 저장-로드 왕복이 원본과 달라짐")
  print("통과: 왕복 (코덱 경유, -1.0 센티널 포함)")

  if encode_scores(scores) != encode_scores(dict(reversed(list(scores.items())))):
    raise SystemExit("실패: 같은 내용이 삽입 순서에 따라 다른 바이트로 직렬화됨 (키 정렬 결정성)")
  print("통과: 직렬화 결정성 (키 정렬)")

  store.blobs["event-깨짐"] = b"{not json"
  store.blobs["event-버전"] = json.dumps({"schema_version": 99, "scores": []}).encode()
  store.blobs["event-형식"] = json.dumps({"schema_version": 1, "scores": [[1, 2, "x"]]}).encode()
  for event_id in ("event-깨짐", "event-버전", "event-형식"):
    if store.load(event_id) != {}:
      raise SystemExit(f"실패: 손상 캐시({event_id})는 빈 dict로 취급돼야 함")
  print("통과: 손상 캐시 3종 → 빈 캐시 (소모품 정책)")

  store.delete("event-1")
  store.delete("event-1")  # 없는 키 삭제도 성공 (멱등)
  if store.load("event-1") != {}:
    raise SystemExit("실패: delete 후 잔존 캐시")
  print("통과: 멱등 삭제")

  print("\n스모크 검증 전부 통과")
