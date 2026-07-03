"""프로덕션 의존성 조립 — 실제 AWS 클라이언트와 AI 모델을 각 계층 부품에 연결한다.

여기가 boto3와 detect/embed(무거운 모델 임포트 체인)를 아는 유일한 조립 지점이다.
handlers·messaging·storage는 전부 Protocol/콜러블 주입으로 설계되어 있어, 스모크/테스트는
이 모듈을 거치지 않고 페이크를 직접 조립한다 (프로덕션 배선에 테스트 분기를 두지 않는다).
"""

import logging
from dataclasses import dataclass

import boto3

from app.core.config import Settings
from app.handlers import FaceExtractor, JobHandlers
from app.messaging.consumer import MessageConsumer, SqsConsumer
from app.messaging.publisher import ResultPublisher, SqsPublisher
from app.pipeline.align import align_face
from app.pipeline.detect import FaceDetector
from app.pipeline.embed import FaceEmbedder
from app.storage.embedding_store import S3EmbeddingStore
from app.storage.image_source import S3ImageSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerDeps:
  """워커 실행에 필요한 조립 완료 부품 일체."""

  consumer: MessageConsumer
  publisher: ResultPublisher
  handlers: JobHandlers
  settings: Settings


def build_face_extractor(detector: FaceDetector, embedder: FaceEmbedder) -> FaceExtractor:
  """detect → align → embed를 합성해 handlers가 요구하는 FaceExtractor를 만든다.

  퇴화 얼굴(정렬 실패·비유한 임베딩)은 각 단계가 None으로 걸러내는 계약이라 여기서 제거한다.
  단일 스레드 워커 전제: FaceDetector는 스레드 안전하지 않지만 동시 호출이 없어 무해하다.
  """

  def extract_faces(image):
    detected = detector.detect(image)
    crops = [crop for crop in (align_face(image, face.landmarks) for face in detected) if crop is not None]
    return [embedding for embedding in embedder.embed_batch(crops) if embedding is not None]

  return extract_faces


def build_worker_deps(settings: Settings) -> WorkerDeps:
  """실제 AWS 클라이언트·모델로 워커 부품을 조립한다.

  FaceDetector/FaceEmbedder 생성 = 모델 파일 획득(필요 시 다운로드)과 메모리 적재 —
  레디니스의 1단계다 (feature-spec §8: 모델 적재 완료 후에만 폴링을 시작한다).
  """
  sqs_client = boto3.client("sqs", region_name=settings.aws_region)
  s3_client = boto3.client("s3", region_name=settings.aws_region)

  logger.info("AI 모델 로딩 중 (YuNet + AuraFace) — 최초 실행은 다운로드로 오래 걸릴 수 있습니다")
  detector = FaceDetector()
  embedder = FaceEmbedder()
  logger.info("AI 모델 로딩 완료")

  handlers = JobHandlers(
    store=S3EmbeddingStore(s3_client, settings.embeddings_bucket, settings.embeddings_prefix),
    images=S3ImageSource(s3_client, settings.images_bucket),
    extract_faces=build_face_extractor(detector, embedder),
    cluster_config=settings.to_cluster_config(),
  )
  return WorkerDeps(
    consumer=SqsConsumer(sqs_client, settings.inbound_queue_url, settings.sqs_wait_time_seconds),
    publisher=SqsPublisher(sqs_client, settings.result_queue_url),
    handlers=handlers,
    settings=settings,
  )


def check_readiness(settings: Settings) -> None:
  """SQS 큐 2종·S3 버킷 2종 연결을 확인한다 — 레디니스 2단계 (feature-spec §8).

  주소가 placeholder이거나 자격 증명이 없으면 폴링을 시작하기 전에 여기서 명확히 실패한다.
  boto3 클라이언트 생성은 네트워크 비용이 없어 검사용으로 새로 만들어도 무방하다.
  """
  sqs_client = boto3.client("sqs", region_name=settings.aws_region)
  s3_client = boto3.client("s3", region_name=settings.aws_region)
  for queue_url in (settings.inbound_queue_url, settings.result_queue_url):
    sqs_client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
  for bucket in (settings.embeddings_bucket, settings.images_bucket):
    s3_client.head_bucket(Bucket=bucket)
  logger.info("SQS/S3 연결 확인 완료 — 레디니스 통과")
