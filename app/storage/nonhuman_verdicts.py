"""비인간 얼굴 판정 캐시의 저장소 인터페이스 — S3 구현과 인메모리 페이크 (ADR-032).

event별 S3 JSON 객체 1개에 face_id → 판정(NonhumanVerdictRecord) 맵을 보관한다. 강등 판정은
npz의 nonhuman_face_ids에 남지만 **통과 판정은 npz에 안 남는다** — 이 캐시가 없으면 매 재군집마다
같은 통과 얼굴을 DetectLabels로 재과금한다. face_id 키라 삭제·append로 행 인덱스가 바뀌어도 안전하다.

핸들러는 이 Protocol만 알고 S3를 모른다 (rekognition_scores와 같은 구도). 캐시는 소모품이다:
손상 객체는 로그 후 빈 캐시로 취급하고(재측정하면 그만), 재군집 결정성의 진실은 언제나 npz의
nonhuman_face_ids다. best-effort 격리(실패가 job을 죽이지 않게)는 handlers의 책임이다.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

# 캐시 객체 자체의 스키마 버전 — event .npz의 SCHEMA_VERSION과 무관한 독립 계약
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NonhumanVerdictRecord:
  """얼굴 1개의 비인간 판정 결과 — 재호출 없이 같은 판정을 재현하기 위한 전량 기록.

  n_faces는 DetectFaces의 FaceDetails 개수이며 -1은 "미호출"(규칙 B 강등·레이블 미달 통과는
  2콜째가 나가지 않는다)이다. labels는 관심 레이블만이 아니라 응답 전체를 남긴다 — 오판 사후
  분석(ADR-032 §롤아웃 감시)의 근거가 된다.
  """

  nonhuman: bool
  rule: str = ""  # "doll" | "sculpture" | "" (통과)
  labels: dict[str, float] = field(default_factory=dict)  # DetectLabels 응답 {이름: 신뢰도 0~100}
  n_faces: int = -1  # DetectFaces FaceDetails 개수, -1 = 미호출


def encode_verdicts(verdicts: dict[str, NonhumanVerdictRecord]) -> bytes:
  """판정 맵을 JSON 바이트로 직렬화한다 — 키 정렬로 같은 내용이면 같은 바이트(결정성, 감사 diff 용이)."""
  rows = [
    [
      face_id,
      {
        "nonhuman": record.nonhuman,
        "rule": record.rule,
        "labels": {name: float(confidence) for name, confidence in sorted(record.labels.items())},
        "n_faces": int(record.n_faces),
      },
    ]
    for face_id, record in sorted(verdicts.items())
  ]
  return json.dumps({"schema_version": _SCHEMA_VERSION, "verdicts": rows}, ensure_ascii=False).encode("utf-8")


def decode_verdicts(blob: bytes) -> dict[str, NonhumanVerdictRecord]:
  """JSON 바이트를 판정 맵으로 복원한다. 손상은 로그 후 빈 캐시 — 캐시는 소모품이라 재측정으로 자연 회복된다."""
  try:
    payload = json.loads(blob)
    if payload["schema_version"] != _SCHEMA_VERSION:
      raise ValueError(f"지원하지 않는 schema_version: {payload['schema_version']}")
    verdicts = {}
    for face_id, body in payload["verdicts"]:
      if not isinstance(face_id, str) or not isinstance(body["rule"], str):
        raise ValueError("face_id·rule은 문자열이어야 합니다")
      verdicts[face_id] = NonhumanVerdictRecord(
        nonhuman=bool(body["nonhuman"]),
        rule=body["rule"],
        labels={str(name): float(confidence) for name, confidence in body["labels"].items()},
        n_faces=int(body["n_faces"]),
      )
    return verdicts
  except (ValueError, KeyError, TypeError) as exc:
    logger.warning("비인간 판정 캐시 손상 — 빈 캐시로 취급: %s", exc)
    return {}


class NonhumanVerdictStore(Protocol):
  """event_id 단위 face_id → 비인간 판정 캐시 인터페이스."""

  def load(self, event_id: str) -> dict[str, NonhumanVerdictRecord]:
    """저장된 캐시를 복원한다. 저장된 적 없거나 손상이면 빈 dict."""
    ...

  def save(self, event_id: str, verdicts: dict[str, NonhumanVerdictRecord]) -> None:
    """캐시 전체를 덮어쓴다 — 죽은 face_id 프루닝은 호출자(handlers)의 책임이다."""
    ...

  def delete(self, event_id: str) -> None:
    """event 전체 삭제 시 캐시도 지운다(생체 파생 정보 위생). 없는 키 삭제도 성공(멱등)."""
    ...


class S3NonhumanVerdictStore:
  """`s3://{bucket}/{prefix}{event_id}.json`을 읽고 쓰는 프로덕션 구현 (워커 소유 embeddings 버킷)."""

  def __init__(self, s3_client, bucket: str, prefix: str = "nonhuman-verdicts/") -> None:
    self._client = s3_client
    self._bucket = bucket
    self._prefix = prefix

  def _key(self, event_id: str) -> str:
    return f"{self._prefix}{event_id}.json"

  def load(self, event_id: str) -> dict[str, NonhumanVerdictRecord]:
    try:
      response = self._client.get_object(Bucket=self._bucket, Key=self._key(event_id))
    except self._client.exceptions.NoSuchKey:
      return {}
    return decode_verdicts(response["Body"].read())

  def save(self, event_id: str, verdicts: dict[str, NonhumanVerdictRecord]) -> None:
    self._client.put_object(
      Bucket=self._bucket, Key=self._key(event_id), Body=encode_verdicts(verdicts), ContentType="application/json"
    )

  def delete(self, event_id: str) -> None:
    self._client.delete_object(Bucket=self._bucket, Key=self._key(event_id))


class InMemoryNonhumanVerdictStore:
  """스모크/테스트용 페이크 — bytes로 저장해 직렬화 코덱까지 실제로 태운다 (InMemoryRekognitionScoreStore와 동일 구도)."""

  def __init__(self) -> None:
    self.blobs: dict[str, bytes] = {}

  def load(self, event_id: str) -> dict[str, NonhumanVerdictRecord]:
    blob = self.blobs.get(event_id)
    if blob is None:
      return {}
    return decode_verdicts(blob)

  def save(self, event_id: str, verdicts: dict[str, NonhumanVerdictRecord]) -> None:
    self.blobs[event_id] = encode_verdicts(verdicts)

  def delete(self, event_id: str) -> None:
    self.blobs.pop(event_id, None)


if __name__ == "__main__":
  # S3 없이 코덱·페이크 계약을 자가 검증한다 (rekognition_scores __main__과 같은 실행형 확인 패턴).
  # TODO(CHMO-165): pytest 도입 시 tests/storage/test_nonhuman_verdicts.py로 승격
  logging.basicConfig(level="ERROR")  # 의도된 손상 케이스의 WARNING 로그 숨김
  store = InMemoryNonhumanVerdictStore()
  if store.load("event-1") != {}:
    raise SystemExit("실패: 저장 전 load는 빈 dict여야 함")

  verdicts = {
    "face-doll": NonhumanVerdictRecord(nonhuman=True, rule="doll", labels={"Doll": 99.6, "Toy": 99.6}, n_faces=0),
    "face-statue": NonhumanVerdictRecord(nonhuman=True, rule="sculpture", labels={"Sculpture": 77.1}),  # 2콜 미호출
    "face-human": NonhumanVerdictRecord(nonhuman=False, labels={"Art": 55.1, "Person": 95.0}),  # 통과도 캐싱
  }
  store.save("event-1", verdicts)
  if store.load("event-1") != verdicts:
    raise SystemExit("실패: 저장-로드 왕복이 원본과 달라짐")
  print("통과: 왕복 (코덱 경유, 통과 판정·n_faces=-1 미호출 센티널 포함)")

  reordered = dict(reversed(list(verdicts.items())))
  if encode_verdicts(verdicts) != encode_verdicts(reordered):
    raise SystemExit("실패: 같은 내용이 삽입 순서에 따라 다른 바이트로 직렬화됨 (키 정렬 결정성)")
  print("통과: 직렬화 결정성 (키 정렬)")

  store.blobs["event-깨짐"] = b"{not json"
  store.blobs["event-버전"] = json.dumps({"schema_version": 99, "verdicts": []}).encode()
  store.blobs["event-형식"] = json.dumps({"schema_version": 1, "verdicts": [[1, {"nonhuman": True}]]}).encode()
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
