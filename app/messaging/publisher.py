"""발행 인터페이스 — SQS 구현과 인메모리 페이크.

두 종류를 발행한다:
  - 결과: 인바운드 3종(classify/feedback/delete)의 처리 결과는 전부 ClassifyResult 한 형식으로
    결과 큐에 발행된다 (feature-spec §6.2).
  - 진행률: classify 처리 도중 처리 장수를 별도 progress 큐에 여러 번 발행한다 (CHMO-274).
"""

import logging
from typing import Protocol

from app.schemas.messages import ClassifyResult, ProgressUpdate

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
    # 전문(body)은 event 사진 수에 비례해 자라(관측 68KB/job) CloudWatch 수집 요금을 지배한다 —
    # INFO에는 요약만 남기고, 전문이 필요하면 LOG_LEVEL=DEBUG로 잠깐 올려서 본다.
    logger.info(
      "결과 발행 job_id=%s status=%s %d bytes clusters=%d common=%d uncertain=%d eyes_closed=%d blurry=%d "
      "failed=%d retired=%d",
      result.job_id,
      result.status,
      body_bytes,
      len(result.clusters),
      len(result.common_album),
      len(result.uncertain),
      len(result.eyes_closed),
      len(result.blurry),
      len(result.failed_images),
      len(result.retired_cluster_ids),
    )
    logger.debug("결과 발행 body job_id=%s %s", result.job_id, body)
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


class ProgressPublisher(Protocol):
  """진행률(progress) 발행 인터페이스."""

  def publish(self, progress: ProgressUpdate) -> None: ...


class SqsProgressPublisher:
  """진행률 큐의 프로덕션 구현 — best-effort.

  진행률은 부수 신호일 뿐이라 발행 실패가 작업(classify) 자체를 죽여선 안 된다: 예외를 로그로만
  남기고 삼킨다. (결과 발행 SqsPublisher는 반대로 실패를 전파해 재전달에 맡긴다 — 그건 유실되면
  안 되는 진실이기 때문이다.)
  """

  def __init__(self, sqs_client, queue_url: str) -> None:
    self._client = sqs_client
    self._queue_url = queue_url

  def publish(self, progress: ProgressUpdate) -> None:
    kwargs = {"QueueUrl": self._queue_url, "MessageBody": progress.model_dump_json()}
    if self._queue_url.endswith(".fifo"):
      # 진행률 큐는 표준 큐를 상정하지만, FIFO로 확정될 경우를 대비한 분기. group은 job_id로 묶되
      # 중복제거 id는 job_id:processed로 메시지마다 유일해야 한다 — job_id만 쓰면 FIFO의 5분
      # 중복제거가 같은 job의 서로 다른 진행 메시지(30/300, 60/300 …)를 삭제해버린다.
      kwargs["MessageGroupId"] = progress.job_id
      kwargs["MessageDeduplicationId"] = f"{progress.job_id}:{progress.processed}"
    try:
      self._client.send_message(**kwargs)
    except Exception:
      logger.warning(
        "진행률 발행 실패 job_id=%s %d/%d — 무시하고 진행", progress.job_id, progress.processed, progress.total
      )
    else:
      # DEBUG 레벨: 3장마다 발행이라 job당 수십~수백 건이라 INFO에 섞으면 결과 로그를 묻는다.
      # 발행 여부를 확인할 땐 LOG_LEVEL=DEBUG로 잠깐 올려서 본다.
      logger.debug("진행률 발행 job_id=%s %d/%d", progress.job_id, progress.processed, progress.total)


class InMemoryProgressPublisher:
  """스모크/테스트용 페이크 — 발행된 진행률을 순서대로 쌓아둔다."""

  def __init__(self) -> None:
    self.published: list[ProgressUpdate] = []

  def publish(self, progress: ProgressUpdate) -> None:
    self.published.append(progress)


if __name__ == "__main__":
  publisher = InMemoryPublisher()
  publisher.publish(ClassifyResult(job_id="job-1", status="failed"))
  if not (len(publisher.published) == 1 and publisher.published[0].job_id == "job-1"):
    raise SystemExit("실패: InMemoryPublisher 기록")
  print("통과: InMemoryPublisher 기록")

  progress_publisher = InMemoryProgressPublisher()
  progress_publisher.publish(ProgressUpdate(job_id="job-1", event_id="event-1", processed=2, total=4))
  if not (len(progress_publisher.published) == 1 and progress_publisher.published[0].processed == 2):
    raise SystemExit("실패: InMemoryProgressPublisher 기록")
  print("통과: InMemoryProgressPublisher 기록")
