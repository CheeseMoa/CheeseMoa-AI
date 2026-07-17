"""프로덕션 의존성 조립 — 실제 AWS 클라이언트와 AI 모델을 각 계층 부품에 연결한다.

여기가 boto3와 detect/embed(무거운 모델 임포트 체인)를 아는 유일한 조립 지점이다.
handlers·messaging·storage는 전부 Protocol/콜러블 주입으로 설계되어 있어, 스모크/테스트는
이 모듈을 거치지 않고 페이크를 직접 조립한다 (프로덕션 배선에 테스트 분기를 두지 않는다).
"""

import logging
import os
from dataclasses import dataclass

import boto3

from app.core.config import Settings
from app.handlers import ExtractedFaces, FaceExtractor, JobHandlers, ProgressReporter
from app.messaging.consumer import MessageConsumer, SqsConsumer
from app.messaging.publisher import ResultPublisher, SqsProgressPublisher, SqsPublisher
from app.pipeline.align import align_face
from app.pipeline.detect import FaceDetector
from app.pipeline.embed import EmbedConfig, FaceEmbedder
from app.pipeline.quality import (
  EyeStateClassifier,
  QualityConfig,
  blur_variance,
  judge_faces,
  shake_confirmed,
  shake_signals,
)
from app.schemas.messages import ProgressUpdate
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


def build_face_extractor(
  detector: FaceDetector,
  embedder: FaceEmbedder,
  eye_classifier: EyeStateClassifier,
  quality_config: QualityConfig,
  align_antialias: bool = True,
) -> FaceExtractor:
  """detect → align → embed에 품질 판정(눈감음/흔들림)을 합성해 handlers의 FaceExtractor를 만든다.

  퇴화 얼굴(정렬 실패·비유한 임베딩)은 각 단계가 None으로 걸러내는 계약이라 임베딩에서 제거한다.
  품질 판정은 정렬 crop(눈감음)과 원본 bbox crop(흔들림)을 쓰며, 정렬 crop은 임베딩용과 공유해 중복 정렬이 없다.
  단일 스레드 워커 전제: FaceDetector는 스레드 안전하지 않지만 동시 호출이 없어 무해하다.
  """

  def extract_faces(image):
    detected = detector.detect(image)
    aligned_crops = [align_face(image, face.landmarks, antialias=align_antialias) for face in detected]
    # 얼굴별 (정렬 crop|None, 원본 bbox crop) 쌍 — judge_faces가 눈감음(정렬)·흔들림(bbox)을 판정한다
    face_pairs = []
    for face, aligned in zip(detected, aligned_crops):
      x, y, w, h = face.bbox
      face_pairs.append((aligned, image[y : y + h, x : x + w]))
    eyes_closed, blurry = judge_faces(face_pairs, eye_classifier, quality_config)

    # 흔들림 fallback: blurry=None = 판정 자격 얼굴이 없음(미검출이거나 전부 극소 얼굴) —
    # 완전 흔들린 사진은 얼굴 검출 자체가 실패하고, 검출됐어도 극소 얼굴은 variance를 신뢰할 수 없다.
    # 1차 신호는 전체 이미지 variance 폭락(선명 300+ → 흔들림 ~1), 2차 신호는 방향성 블러(ADR 014) —
    # variance는 텍스처 양을 재는 지표라 배경 무늬가 많은 흔들린 사진을 놓치는데, 손떨림은 모든
    # 에지가 한 방향으로 번져 그라디언트 방향 쏠림으로 잡힌다.
    # 한계: 앞사람만 모션블러이고 배경이 선명한 부분 블러, 장노출 빛궤적(에지 방향이 궤적을 따라
    # 다양함)은 여전히 잡지 못한다.
    if blurry is None:
      blurry = blur_variance(image) < quality_config.whole_image_blur_threshold
      if not blurry and quality_config.shake_coherence_threshold > 0:
        norm_var, coherence = shake_signals(image)
        blurry = (
          coherence >= quality_config.shake_coherence_threshold and norm_var < quality_config.shake_max_norm_variance
        )

    # 흔들림 재확인 게이트: variance 기반 blurry(얼굴 경로·fallback 공통)를 전체 이미지 방향 쏠림으로
    # 재확인한다 — 옛날 인화 재촬영처럼 원판이 소프트한 사진은 잔결이 없어도 손떨림이 아니다
    # (event 50 실측, shake_coherence_floor 주석). 2차 신호로 잡힌 사진은 쏠림이 이미 높아 통과한다.
    if blurry and not shake_confirmed(image, quality_config):
      blurry = False

    crops = [crop for crop in aligned_crops if crop is not None]
    embeddings = [embedding for embedding in embedder.embed_batch(crops) if embedding is not None]
    return ExtractedFaces(embeddings, eyes_closed, blurry)

  return extract_faces


def _build_progress_reporter(sqs_client, progress_queue_url: str | None) -> ProgressReporter | None:
  """진행률 발행 리포터를 만든다 (CHMO-274) — 큐 URL이 없으면 None(발행 비활성).

  핸들러는 (job_id, event_id, processed, total)만 넘긴다 — ProgressUpdate 조립·SQS 발행은 여기서
  가둔다(관심사 분리). 발행 자체는 SqsProgressPublisher가 best-effort로 삼킨다.
  """
  if not progress_queue_url:
    return None
  publisher = SqsProgressPublisher(sqs_client, progress_queue_url)

  def report(job_id: str, event_id: str, processed: int, total: int) -> None:
    publisher.publish(ProgressUpdate(job_id=job_id, event_id=event_id, processed=processed, total=total))

  return report


def _available_cores() -> int:
  """이 프로세스가 실제로 쓸 수 있는 코어 수.

  `os.cpu_count()`는 CPU 어피니티/cpuset을 무시하므로 리눅스에서는 `sched_getaffinity`를 우선한다
  (컨테이너에 코어를 제한해 띄운 경우 cpu_count는 호스트 전체를 세어 과다 산정한다).
  """
  try:
    return len(os.sched_getaffinity(0))  # 리눅스 전용
  except AttributeError:
    return os.cpu_count() or 1


def build_worker_deps(settings: Settings) -> WorkerDeps:
  """실제 AWS 클라이언트·모델로 워커 부품을 조립한다.

  FaceDetector/FaceEmbedder 생성 = 모델 파일 획득(필요 시 다운로드)과 메모리 적재 —
  레디니스의 1단계다 (feature-spec §8: 모델 적재 완료 후에만 폴링을 시작한다).
  """
  sqs_client = boto3.client("sqs", region_name=settings.aws_region)
  s3_client = boto3.client("s3", region_name=settings.aws_region)

  # 스레드 수를 코어 수에 맞춘다. EmbedConfig의 기본값 8은 PoC 배포 타깃(8vCPU) 기준이라,
  # 코어가 그보다 적은 호스트에서 그대로 쓰면 오버서브스크립션으로 급격히 느려진다
  # (t4g.small 2코어 실측: 8스레드 2860ms/장 vs 2스레드 476ms/장 — 6배).
  cores = _available_cores()
  threads = settings.ort_num_threads or cores

  logger.info("AI 모델 로딩 중 (YuNet + AuraFace + 눈감음 CNN) — 최초 실행은 다운로드로 오래 걸릴 수 있습니다")
  detector = FaceDetector(settings.to_detector_config())
  embedder = FaceEmbedder(EmbedConfig(intra_op_num_threads=threads, inter_op_num_threads=threads))
  eye_classifier = EyeStateClassifier()
  logger.info("AI 모델 로딩 완료 (추론 스레드=%d, 가용 코어=%d)", threads, cores)

  handlers = JobHandlers(
    store=S3EmbeddingStore(s3_client, settings.embeddings_bucket, settings.embeddings_prefix),
    images=S3ImageSource(s3_client, settings.images_bucket),
    extract_faces=build_face_extractor(
      detector, embedder, eye_classifier, settings.to_quality_config(), align_antialias=settings.align_antialias
    ),
    cluster_config=settings.to_cluster_config(),
    report_progress=_build_progress_reporter(sqs_client, settings.progress_queue_url),
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
  # progress 큐는 선택 — 미설정(비활성)이면 확인 대상에서 뺀다 (CHMO-274)
  queue_urls = [settings.inbound_queue_url, settings.result_queue_url]
  if settings.progress_queue_url:
    queue_urls.append(settings.progress_queue_url)
  for queue_url in queue_urls:
    sqs_client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
  for bucket in (settings.embeddings_bucket, settings.images_bucket):
    s3_client.head_bucket(Bucket=bucket)
  logger.info("SQS/S3 연결 확인 완료 — 레디니스 통과")
