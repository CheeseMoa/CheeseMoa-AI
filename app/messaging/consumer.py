"""인바운드 큐 수신 인터페이스 — SQS 구현과 인메모리 페이크.

body 해석(parse_inbound_message)·처리·삭제 판단은 워커의 몫이고, 이 모듈은 수신·삭제라는
큐 연산만 안다. 큐 URL 등 실주소는 core.deps가 Settings에서 읽어 주입한다.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ReceivedMessage:
  """큐에서 수신한 메시지 1건 — 워커의 오류 정책 판단에 필요한 최소 정보만 담는다."""

  body: str
  receipt_handle: str  # 삭제(ack)용 핸들
  message_id: str
  receive_count: int  # 이번이 몇 번째 전달인지 (1부터) — "마지막 시도" 판단 근거


class MessageConsumer(Protocol):
  """인바운드 큐 수신/삭제 인터페이스."""

  def receive(self) -> list[ReceivedMessage]:
    """메시지를 기다렸다가 수신한다. 대기 시간 안에 없으면 빈 리스트."""
    ...

  def delete(self, message: ReceivedMessage) -> None:
    """처리 완료(ack) — 삭제하지 않은 메시지는 가시성 타임아웃 후 재전달된다."""
    ...


class SqsConsumer:
  """단일 FIFO 인바운드 큐의 프로덕션 구현."""

  def __init__(self, sqs_client, queue_url: str, wait_time_seconds: int = 20) -> None:
    self._client = sqs_client
    self._queue_url = queue_url
    self._wait_time_seconds = wait_time_seconds

  def receive(self) -> list[ReceivedMessage]:
    response = self._client.receive_message(
      QueueUrl=self._queue_url,
      # 1건 고정: FIFO는 같은 messageGroupId(event_id) 메시지를 순서대로 주는데, 배치로 여러 건을
      # 받아두면 앞 건 처리 중 크래시 시 뒤 건들이 가시성 타임아웃까지 통째로 잠긴다 — 직렬 워커에서는
      # 1건씩 받아야 수신-처리-삭제의 가시성 의미론이 단순하게 유지된다.
      MaxNumberOfMessages=1,
      WaitTimeSeconds=self._wait_time_seconds,
      AttributeNames=["ApproximateReceiveCount"],
    )
    return [
      ReceivedMessage(
        body=message["Body"],
        receipt_handle=message["ReceiptHandle"],
        message_id=message["MessageId"],
        receive_count=int(message.get("Attributes", {}).get("ApproximateReceiveCount", "1")),
      )
      for message in response.get("Messages", [])
    ]

  def delete(self, message: ReceivedMessage) -> None:
    self._client.delete_message(QueueUrl=self._queue_url, ReceiptHandle=message.receipt_handle)


@dataclass
class _QueuedMessage:
  body: str
  message_id: str
  receive_count: int = 0


class InMemoryConsumer:
  """스모크/테스트용 페이크 — SQS의 재전달·DLQ 의미론을 모사한다.

  delete되지 않은 메시지는 다음 receive에서 큐 뒤로 되돌아가고(가시성 타임아웃 만료 모사),
  max_receive_count만큼 전달된 뒤에도 삭제되지 않은 메시지는 dead_lettered로 빠진다(redrive 모사).
  """

  def __init__(self, bodies: Sequence[str], *, max_receive_count: int | None = None) -> None:
    self._queue: deque[_QueuedMessage] = deque(
      _QueuedMessage(body=body, message_id=f"msg-{index}") for index, body in enumerate(bodies)
    )
    self._max_receive_count = max_receive_count
    self._in_flight: _QueuedMessage | None = None
    self.acked: list[str] = []  # 삭제된 message_id (검증용)
    self.dead_lettered: list[str] = []  # DLQ로 빠진 message_id (검증용)

  def receive(self) -> list[ReceivedMessage]:
    if self._in_flight is not None:  # 직전 수신분이 미삭제 — 가시성 타임아웃 만료를 모사해 재큐잉
      self._queue.append(self._in_flight)
      self._in_flight = None
    while self._queue:
      entry = self._queue.popleft()
      if self._max_receive_count is not None and entry.receive_count >= self._max_receive_count:
        self.dead_lettered.append(entry.message_id)  # maxReceiveCount 초과 → redrive 모사
        continue
      entry.receive_count += 1
      self._in_flight = entry
      return [
        ReceivedMessage(
          body=entry.body,
          receipt_handle=f"handle-{entry.message_id}-{entry.receive_count}",
          message_id=entry.message_id,
          receive_count=entry.receive_count,
        )
      ]
    return []

  def delete(self, message: ReceivedMessage) -> None:
    if self._in_flight is not None and self._in_flight.message_id == message.message_id:
      self.acked.append(message.message_id)
      self._in_flight = None


if __name__ == "__main__":
  # 페이크의 재전달·DLQ 의미론을 자가 검증한다 (worker 스모크가 이 의미론에 의존한다).
  consumer = InMemoryConsumer(["본문-1", "본문-2"], max_receive_count=2)

  first = consumer.receive()[0]
  if not (first.body == "본문-1" and first.receive_count == 1):
    raise SystemExit("실패: 첫 수신")
  # 미삭제 상태로 다음 receive → 본문-2가 나오고 본문-1은 큐 뒤로
  second = consumer.receive()[0]
  if second.body != "본문-2":
    raise SystemExit("실패: 미삭제 메시지 재큐잉")
  consumer.delete(second)

  redelivered = consumer.receive()[0]
  if not (redelivered.body == "본문-1" and redelivered.receive_count == 2):
    raise SystemExit("실패: 재전달 수신 횟수 증가")
  # 2회 전달 후에도 미삭제 → 다음 receive에서 DLQ로
  if consumer.receive() != []:
    raise SystemExit("실패: maxReceiveCount 초과분은 전달되면 안 됨")
  if not (consumer.acked == ["msg-1"] and consumer.dead_lettered == ["msg-0"]):
    raise SystemExit("실패: ack/DLQ 기록")
  print("통과: InMemoryConsumer 재전달·DLQ 의미론")
