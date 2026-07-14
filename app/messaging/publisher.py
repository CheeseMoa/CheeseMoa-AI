"""결과 큐 발행 인터페이스 — SQS 구현과 인메모리 페이크.

인바운드 3종(classify/feedback/delete)의 처리 결과는 전부 ClassifyResult 한 형식으로
이 인터페이스를 통해 발행된다 (feature-spec §6.2).
"""

import logging
from typing import Protocol

from app.schemas.messages import ClassifyResult

logger = logging.getLogger(__name__)

# SQS 메시지 크기 상한 (256KB). 클러스터 하나가 대표벡터 512개 float로 약 10KB라 인물 20명대 +
# 긴 image_id 목록이면 초과할 수 있다 — MVP 규모(event당 얼굴 56~300)에서는 여유가 있어 수용한다.
_SQS_BODY_LIMIT_BYTES = 256 * 1024


class ResultPublisher(Protocol):
  """classify-result 발행 인터페이스."""

  def publish(self, result: ClassifyResult) -> None: ...


class SqsPublisher:
  """결과 큐의 프로덕션 구현."""

  def __init__(self, sqs_client, queue_url: str) -> None:
    self._client = sqs_client
    self._queue_url = queue_url

  def publish(self, result: ClassifyResult) -> None:
    body = result.model_dump_json()
    body_bytes = len(body.encode("utf-8"))
    if body_bytes > _SQS_BODY_LIMIT_BYTES:
      # TODO(CHMO-165): 초과가 실제로 관측되면 payload-on-S3 포인터 패턴으로 전환
      logger.warning("결과 메시지가 SQS 상한을 초과합니다. job_id=%s, %d bytes", result.job_id, body_bytes)
    logger.info("결과 발행 job_id=%s %d bytes body=%s", result.job_id, body_bytes, body)
    kwargs = {"QueueUrl": self._queue_url, "MessageBody": body}
    if self._queue_url.endswith(".fifo"):
      # 결과 큐 유형은 미정 — FIFO로 확정될 경우를 대비한 분기. 그룹/중복제거 키는 job_id:
      # 같은 job의 재발행(저장 후 발행 전 크래시 → 재처리)이 5분 창 안에서 자연 억제된다.
      # 결과 메시지에는 event_id가 없어(계약 §6.2) event 단위 그룹핑은 불가능하다.
      kwargs["MessageGroupId"] = result.job_id
      kwargs["MessageDeduplicationId"] = result.job_id
    self._client.send_message(**kwargs)


class InMemoryPublisher:
  """스모크/테스트용 페이크 — 발행된 결과를 순서대로 쌓아둔다."""

  def __init__(self) -> None:
    self.published: list[ClassifyResult] = []

  def publish(self, result: ClassifyResult) -> None:
    self.published.append(result)


if __name__ == "__main__":
  publisher = InMemoryPublisher()
  publisher.publish(ClassifyResult(job_id="job-1", status="failed"))
  if not (len(publisher.published) == 1 and publisher.published[0].job_id == "job-1"):
    raise SystemExit("실패: InMemoryPublisher 기록")
  print("통과: InMemoryPublisher 기록")
