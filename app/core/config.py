"""워커 실행 설정의 단일 진실 소스 — 환경변수(.env 포함)에서 읽는다.

SQS 큐 URL·S3 버킷명은 아직 미정(feature-spec §10 #7)이라 기본값 없는 필수 필드로 둔다:
값이 정해지면 .env(.env.example 참고)나 배포 환경변수에 채우면 되고, 누락 상태로 기동하면
pydantic이 누락 필드명을 그대로 나열하며 즉시 실패한다 — 자리만 있고 값이 없는 배포가
폴링을 시작한 뒤에야 죽는 것을 막는다.

모델 파일 경로(YUNET_MODEL_PATH 등)는 여기 두지 않는다 — `core.model_source`가 소유한다.
"""

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
  from app.pipeline.cluster import ClusterConfig
  from app.pipeline.quality import QualityConfig


class Settings(BaseSettings):
  # frozen: 메시지 스키마·파이프라인 config와 동일한 불변 계약 — 실행 중 설정 변형을 막는다
  # extra="ignore": dotenv 소스는 .env 파일의 "모든" 키를 입력으로 넘기므로, boto3가 직접 읽는
  #   AWS 자격증명(AWS_ACCESS_KEY_ID 등)처럼 Settings 필드가 아닌 키가 .env에 함께 있어도 거부하지
  #   않게 한다 (이게 없으면 .env에 자격증명을 둔 로컬/컨테이너 기동이 extra_forbidden으로 실패한다).
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", frozen=True, extra="ignore")

  # ── AWS 주소 (미정 — 확정 시 .env/배포 환경변수에 실값 주입) ──────────────────
  inbound_queue_url: str  # 단일 FIFO 인바운드 큐 (classify/feedback/delete, messageGroupId=event_id)
  result_queue_url: str  # classify-result 발행 큐
  embeddings_bucket: str  # event .npz 버킷 — 이 워커가 단일 writer (ADR-007)
  images_bucket: str  # 원본 이미지 버킷 — Spring 소유, 워커는 읽기 전용
  aws_region: str = "ap-northeast-2"
  embeddings_prefix: str = "embeddings/"  # .npz 키 = {prefix}{event_id}.npz

  # ── SQS 소비 파라미터 ──────────────────────────────────────────────────────
  sqs_wait_time_seconds: int = Field(default=20, ge=0, le=20)  # long poll 대기 (20 = SQS 상한)
  # 큐 redrive policy의 maxReceiveCount와 반드시 일치시킬 것 — 워커가 "마지막 시도"를 판단해
  # DLQ로 넘어가기 직전에 status="failed" 결과를 발행하는 기준이다 (worker._process).
  sqs_max_receive_count: int = Field(default=3, ge=1)

  # ── 클러스터링 임계값 (feature-spec §10 #3·#4 — 하드코딩 금지, 기본값은 PoC 레시피) ──
  cluster_min_cluster_size: int = 2
  cluster_min_samples: int = 2  # PoC 검증값 유지 (ADR-009: 3은 소규모 이벤트 회귀로 기각)
  cluster_selection_epsilon: float = 0.15
  cluster_min_match_jaccard: float = 0.0
  cluster_merge_centroid_similarity: float = 0.7
  cluster_rescue_similarity: float = 0.6
  cluster_min_membership_similarity: float = 0.4
  cluster_min_membership_margin: float = 0.05
  cluster_blob_promote_similarity: float = 0.45
  cluster_blob_promote_floor: float = 0.4

  # ── 품질 게이트 임계값 (눈감음/흔들림 — 하드코딩 금지, 기본값은 초기값이며 face-test 실측 보정) ──
  quality_blur_threshold: float = 100.0
  quality_whole_image_blur_threshold: float = 100.0  # 얼굴 미검출 시 전체 이미지 흔들림 fallback (별도 보정)
  quality_eye_closed_confidence: float = 0.85  # face-test 실측 보정 (약한 오탐 제거, feature-spec §10 #3)
  quality_eye_box_px: int = 24

  log_level: str = "INFO"

  def to_cluster_config(self) -> "ClusterConfig":
    """설정값을 recluster의 ClusterConfig로 변환한다 (값 검증은 ClusterConfig.__post_init__이 수행).

    core는 pipeline이 역으로 의존하는 계층(model_source)이라 모듈 수준 pipeline 임포트를
    피하고 여기서 지연 임포트한다.
    """
    from app.pipeline.cluster import ClusterConfig

    return ClusterConfig(
      min_cluster_size=self.cluster_min_cluster_size,
      min_samples=self.cluster_min_samples,
      cluster_selection_epsilon=self.cluster_selection_epsilon,
      min_match_jaccard=self.cluster_min_match_jaccard,
      merge_centroid_similarity=self.cluster_merge_centroid_similarity,
      rescue_similarity=self.cluster_rescue_similarity,
      min_membership_similarity=self.cluster_min_membership_similarity,
      min_membership_margin=self.cluster_min_membership_margin,
      blob_promote_similarity=self.cluster_blob_promote_similarity,
      blob_promote_floor=self.cluster_blob_promote_floor,
    )

  def to_quality_config(self) -> "QualityConfig":
    """설정값을 품질 게이트의 QualityConfig로 변환한다 (값 검증은 QualityConfig.__post_init__이 수행).

    to_cluster_config와 같은 이유로 pipeline 임포트를 지연시킨다 (core→pipeline 역의존 회피).
    """
    from app.pipeline.quality import QualityConfig

    return QualityConfig(
      blur_threshold=self.quality_blur_threshold,
      whole_image_blur_threshold=self.quality_whole_image_blur_threshold,
      eye_closed_confidence=self.quality_eye_closed_confidence,
      eye_box_px=self.quality_eye_box_px,
    )
