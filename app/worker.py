"""SQS consumer 워커 엔트리포인트 — `python -m app.worker`.

폴링 루프와 메시지 수준 오류 정책(feature-spec §9)만 담당하고 처리 로직은 handlers에 위임한다.
메시지 1건의 처리 순서는 항상 「handle(내부에서 event .npz 저장) → 결과 발행 → 메시지 삭제」다:
어느 지점에서 죽어도 SQS 재전달 + photo_id 멱등 append + 결정적 재군집이 같은 결과를
다시 만들므로 at-least-once로 안전하다.

오류 정책:
  - 계약 위반(ValidationError, 포이즌): 재시도해도 똑같이 실패하고 event FIFO 그룹만 막으므로,
    본문 전문을 로그로 남기고 (job_id 추출 가능하면 failed 발행 후) 삭제한다.
  - 작업 전체 실패(저장소 장애·손상 등): 삭제하지 않고 가시성 타임아웃 재전달에 맡긴다.
    마지막 시도(receive_count ≥ max)면 Spring을 풀어주기 위해 failed를 발행하되, 메시지는
    남겨 redrive policy가 DLQ(classify-dlq)로 옮기게 한다 — 증거 보존.

`python -m app.worker --smoke`: AWS·모델 없이 페이크 전체 배선으로 위 정책을 자가 검증한다.
"""

import json
import logging
import signal
import sys
import time

from pydantic import ValidationError

from app.messaging.consumer import MessageConsumer, ReceivedMessage
from app.messaging.publisher import ResultPublisher
from app.schemas.messages import ClassifyResult, parse_inbound_message

logger = logging.getLogger(__name__)

# 수신 루프 자체가 실패(네트워크 순단 등)했을 때 재시도 전 대기 — 핫루프로 로그를 채우지 않는다
_RECEIVE_RETRY_DELAY_SECONDS = 5.0


class Worker:
  """단일 스레드 폴링 워커. 처리 로직(handlers)·큐 연산(consumer/publisher)은 주입받는다."""

  def __init__(
    self,
    consumer: MessageConsumer,
    publisher: ResultPublisher,
    handlers,  # JobHandlers — 모듈 수준 임포트가 pipeline(무거운 체인)을 끌고 오지 않도록 타입만 느슨하게 둔다
    *,
    max_receive_count: int,
  ) -> None:
    self._consumer = consumer
    self._publisher = publisher
    self._handlers = handlers
    self._max_receive_count = max_receive_count
    self._stop_requested = False

  def request_stop(self, *_args) -> None:
    """종료 요청 플래그를 세운다 (시그널 핸들러 겸용) — 처리 중인 메시지는 완주한다."""
    if not self._stop_requested:
      logger.info("종료 요청 수신 — 현재 메시지 처리 후 폴링을 멈춥니다")
    self._stop_requested = True

  def poll_once(self) -> int:
    """1회 수신-처리 사이클. 처리를 시도한 메시지 수를 반환한다 (스모크/테스트의 루프 구동용)."""
    messages = self._consumer.receive()
    for message in messages:
      self._process(message)
    return len(messages)

  def run(self) -> None:
    """종료 요청까지 폴링을 반복한다. long poll이 블로킹 단위라 종료 지연은 최대 wait time만큼이다."""
    self._install_signal_handlers()
    logger.info("SQS 폴링 시작")
    while not self._stop_requested:
      try:
        self.poll_once()
      except KeyboardInterrupt:
        self.request_stop()
      except Exception:
        # 수신/발행의 일시 장애로 워커가 죽으면 안 된다 — 메시지는 미삭제라 재전달로 보전된다
        logger.exception("폴링 사이클 실패 — %.0f초 후 재시도", _RECEIVE_RETRY_DELAY_SECONDS)
        time.sleep(_RECEIVE_RETRY_DELAY_SECONDS)
    logger.info("폴링 루프 종료")

  def _install_signal_handlers(self) -> None:
    signal.signal(signal.SIGINT, self.request_stop)
    for name in ("SIGTERM", "SIGBREAK"):  # SIGTERM은 Windows에 전달되지 않을 수 있고 SIGBREAK는 Windows 전용
      if hasattr(signal, name):
        try:
          signal.signal(getattr(signal, name), self.request_stop)
        except (ValueError, OSError):  # 메인 스레드가 아니거나 플랫폼이 거부하는 경우
          pass

  def _process(self, message: ReceivedMessage) -> None:
    try:
      parsed = parse_inbound_message(message.body)
    except ValidationError:
      self._handle_poison(message)
      return

    try:
      result = self._handlers.handle(parsed)
    except Exception:
      logger.exception(
        "작업 처리 실패 job_id=%s (시도 %d/%d)", parsed.job_id, message.receive_count, self._max_receive_count
      )
      if message.receive_count >= self._max_receive_count:
        # 마지막 시도 — Spring이 job을 무한 대기하지 않도록 failed를 발행한다. 메시지는 삭제하지
        # 않아 redrive policy가 DLQ로 옮긴다 (이르게 발행하면 재시도 성공 시 failed→succeeded로 번복된다)
        self._publish_best_effort(ClassifyResult(job_id=parsed.job_id, status="failed"))
      return  # 미삭제 → 가시성 타임아웃 후 재전달

    self._publisher.publish(result)
    self._consumer.delete(message)
    logger.info("처리 완료 job_id=%s status=%s", result.job_id, result.status)

  def _handle_poison(self, message: ReceivedMessage) -> None:
    # 본문 전문이 유일한 증거다 — 삭제 전에 반드시 남긴다
    logger.error("메시지 계약 위반(포이즌) — 삭제합니다. message_id=%s body=%s", message.message_id, message.body)
    job_id = self._extract_job_id(message.body)
    if job_id:
      self._publish_best_effort(ClassifyResult(job_id=job_id, status="failed"))
    self._consumer.delete(message)

  def _publish_best_effort(self, result: ClassifyResult) -> None:
    try:
      self._publisher.publish(result)
    except Exception:
      logger.exception("failed 결과 발행 실패 job_id=%s — 발행 없이 진행", result.job_id)

  @staticmethod
  def _extract_job_id(body: str) -> str | None:
    """포이즌 본문에서 상관관계 키를 최선으로 회수한다 — 실패하면 None (발행 생략)."""
    try:
      job_id = json.loads(body).get("job_id")
    except (ValueError, AttributeError):
      return None
    return job_id if isinstance(job_id, str) and job_id else None


def main() -> None:
  # pydantic-settings도 .env를 읽지만, model_source의 os.getenv 오버라이드(YUNET_MODEL_PATH 등)에도
  # .env가 먹히려면 프로세스 환경에 올려야 한다 (기존 환경변수를 덮어쓰지는 않는다)
  from dotenv import load_dotenv

  load_dotenv()

  from app.core.config import Settings

  settings = Settings()  # 큐 URL·버킷명 미설정이면 누락 필드를 나열하며 여기서 즉시 실패
  logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

  # 무거운 임포트 체인(cv2·onnxruntime·boto3)은 설정 검증을 통과한 뒤에만 로드한다
  from app.core.deps import build_worker_deps, check_readiness

  deps = build_worker_deps(settings)  # 모델 적재 (레디니스 1단계)
  check_readiness(settings)  # SQS/S3 연결 확인 (레디니스 2단계)

  Worker(
    deps.consumer,
    deps.publisher,
    deps.handlers,
    max_receive_count=settings.sqs_max_receive_count,
  ).run()


def _run_smoke() -> None:
  """AWS·모델 없이 페이크 전체 배선으로 워커 루프와 오류 정책을 자가 검증한다."""
  import math

  import numpy as np

  from app.handlers import JobHandlers
  from app.messaging.consumer import InMemoryConsumer
  from app.messaging.publisher import InMemoryPublisher
  from app.storage.embedding_store import InMemoryEmbeddingStore
  from app.storage.event_embeddings import EMBED_DIM
  from app.storage.image_source import InMemoryImageSource

  logging.basicConfig(level="WARNING")  # 의도된 실패 케이스의 ERROR 로그만 보이게
  passed = 0

  def check(name: str, condition: bool) -> None:
    nonlocal passed
    if not condition:
      raise SystemExit(f"실패: {name}")
    passed += 1
    print(f"통과: {name}")

  def person_vector(person: int, step: int) -> np.ndarray:
    theta = math.radians(5.0 * step)
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[2 * person] = math.cos(theta)
    vector[2 * person + 1] = math.sin(theta)
    return vector

  def fake_image(faces: list[tuple[int, int]]) -> np.ndarray:
    image = np.zeros((2, 16, 3), dtype=np.uint8)
    image[0, 0, 0] = len(faces)
    for slot, (person, step) in enumerate(faces):
      image[0, slot + 1, 0] = person
      image[0, slot + 1, 1] = step
    return image

  def fake_extractor(image: np.ndarray) -> list[np.ndarray]:
    count = int(image[0, 0, 0])
    return [person_vector(int(image[0, slot + 1, 0]), int(image[0, slot + 1, 1])) for slot in range(count)]

  store = InMemoryEmbeddingStore()
  # "event-폭발"의 .npz를 손상시켜 두면 해당 classify가 StoreCorruptionError로 작업 전체 실패한다
  store.blobs["event-폭발"] = b"\x00\x01\x02"

  handlers = JobHandlers(
    store=store,
    images=InMemoryImageSource(
      {
        "a1.jpg": fake_image([(0, 0)]),
        "a2.jpg": fake_image([(0, 1)]),
        "b1.jpg": fake_image([(1, 0)]),
        "b2.jpg": fake_image([(1, 1)]),
      }
    ),
    extract_faces=fake_extractor,
  )

  ok_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-정상",
      "group_id": "group-1",
      "event_id": "event-1",
      "images": [
        {"image_id": "img-a1", "s3_key": "a1.jpg"},
        {"image_id": "img-a2", "s3_key": "a2.jpg"},
        {"image_id": "img-b1", "s3_key": "b1.jpg"},
        {"image_id": "img-b2", "s3_key": "b2.jpg"},
      ],
    }
  )
  poison_body = json.dumps({"type": "unknown_type", "job_id": "job-포이즌"})
  exploding_body = json.dumps(
    {
      "type": "classify_request",
      "job_id": "job-폭발",
      "group_id": "group-1",
      "event_id": "event-폭발",
      "images": [{"image_id": "img-x", "s3_key": "a1.jpg"}],
    }
  )

  max_receive_count = 2
  consumer = InMemoryConsumer([ok_body, poison_body, exploding_body], max_receive_count=max_receive_count)
  publisher = InMemoryPublisher()
  worker = Worker(consumer, publisher, handlers, max_receive_count=max_receive_count)

  while worker.poll_once():  # 큐가 마를 때까지 실제 루프 경로(poll_once)로 구동
    pass

  results = {result.job_id: result for result in publisher.published}
  check(
    "정상 메시지: succeeded 발행 + 삭제(ack)",
    results["job-정상"].status == "succeeded" and len(results["job-정상"].clusters) == 2 and "msg-0" in consumer.acked,
  )
  check(
    "포이즌 메시지: failed 발행 + 삭제 (재시도 없음)",
    results["job-포이즌"].status == "failed" and "msg-1" in consumer.acked,
  )
  check(
    "작업 전체 실패: 마지막 시도에 failed 발행, 미삭제 → DLQ",
    results["job-폭발"].status == "failed" and consumer.dead_lettered == ["msg-2"] and "msg-2" not in consumer.acked,
  )
  check(
    "failed 발행은 마지막 시도 1회뿐",
    sum(1 for result in publisher.published if result.job_id == "job-폭발") == 1,
  )
  check("손상 .npz는 덮어쓰이지 않고 보존", store.blobs["event-폭발"] == b"\x00\x01\x02")

  worker.request_stop()
  check("종료 플래그 후 run()은 즉시 반환", (worker.run() or True) and worker._stop_requested)

  print(f"\n스모크 검증 {passed}건 전부 통과")


if __name__ == "__main__":
  if "--smoke" in sys.argv[1:]:
    _run_smoke()
  else:
    main()
