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
  from app.pipeline.detect import DetectorConfig
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
  # 진행률(progress) 발행 큐 (CHMO-274). 미설정이면 진행률 발행 비활성 — 큐 provisioning 전에
  # 코드가 배포돼도 안전하게 뜬다 ("0/미설정 = 비활성" 토글 관례). 값이 있으면 classify 처리 중
  # 처리 장수를 이 큐로 흘려보낸다.
  progress_queue_url: str | None = None
  embeddings_bucket: str  # event .npz 버킷 — 이 워커가 단일 writer (ADR-007)
  images_bucket: str  # 원본 이미지 버킷 — Spring 소유, 워커는 읽기 전용
  aws_region: str = "ap-northeast-2"
  embeddings_prefix: str = "embeddings/"  # .npz 키 = {prefix}{event_id}.npz

  # ── SQS 소비 파라미터 ──────────────────────────────────────────────────────
  sqs_wait_time_seconds: int = Field(default=20, ge=0, le=20)  # long poll 대기 (20 = SQS 상한)
  # 큐 redrive policy의 maxReceiveCount와 반드시 일치시킬 것 — 워커가 "마지막 시도"를 판단해
  # DLQ로 넘어가기 직전에 status="failed" 결과를 발행하는 기준이다 (worker._process).
  sqs_max_receive_count: int = Field(default=3, ge=1)

  # ── 추론 스레드 ────────────────────────────────────────────────────────────
  # onnxruntime intra/inter op 스레드. 0 = 가용 코어 수 자동 감지(권장).
  # 코어 수보다 크게 잡으면 오버서브스크립션으로 급격히 느려진다 — 2코어(t4g.small) 실측에서
  # 8스레드는 2스레드 대비 임베딩이 6배 느렸다(2860ms/장 vs 476ms/장).
  ort_num_threads: int = Field(default=0, ge=0)

  # ── 클러스터링 임계값 (feature-spec §10 #3·#4 — 하드코딩 금지, 기본값은 PoC 레시피) ──
  cluster_min_cluster_size: int = 2
  cluster_min_samples: int = 2  # PoC 검증값 유지 (ADR-009: 3은 소규모 이벤트 회귀로 기각)
  cluster_selection_epsilon: float = 0.15
  cluster_min_match_jaccard: float = 0.0
  cluster_merge_centroid_similarity: float = 0.55  # ADR-012: 분포 측정 기반 재보정 0.68 → 0.55
  cluster_merge_facepair_floor: float = 0.475  # ADR-016(+재보정): 파편병합 face-level 응집 바닥 (0 = 비활성)
  cluster_rescue_similarity: float = 0.6
  cluster_min_membership_similarity: float = 0.4
  cluster_min_membership_margin: float = 0.05
  cluster_blob_promote_similarity: float = 0.45
  cluster_blob_promote_floor: float = 0.4

  # ── 입력 품질 교정 (2026-07-14 리뷰: 정렬 안티에일리어싱 + 랜드마크 2단계 정제) ──────────
  # 같은 얼굴 임베딩이 촬영·설정마다 흔들리던 노이즈(최저 유사도 0.43)를 잡는 두 교정의 토글.
  # 운영에서 문제가 생기면 코드 수정 없이 .env로 끈다.
  align_antialias: bool = True  # 축소 warp 전 배율 기반 가우시안 프리블러
  detect_max_side: int = 2000  # 검출 전 긴 변 축소 상한 (기존 하드코딩 값을 노출)
  detect_refine_landmarks: bool = True  # 대형 얼굴 랜드마크를 정규 스케일 크롭 재검출로 정제
  detect_refine_norm_face_width: int = 224  # 재검출 크롭에서의 얼굴 목표 폭(px) — 스윕 확정값
  detect_refine_margin_ratio: float = 0.75  # bbox 대비 여유 크롭 비율 — 스윕 확정값
  # 배경 인물 필터 (ADR-013): bbox 폭이 이미지 긴 변의 이 비율 미만인 얼굴을 검출 단계에서
  # 버린다 — 멀리 배경에 찍힌 행인이 앨범을 만드는 것 방지 (분포 측정 확정값, 0 = 비활성).
  detect_min_face_rel_width: float = 0.025
  # 대형 오검출 결합 필터 (ADR-015): score가 이 값 미만 AND 종횡비(폭/높이)가 아래 값 미만이면
  # 검출 단계에서 버린다 — 팔짱 낀 팔·조형물 등 진짜 얼굴 크기의 오검출 제거 (크기 필터로는 못
  # 걸러진다). 두 축이 동시에 낮은 것은 오검출뿐이라 결합해야 진짜 얼굴 손실이 없다. 둘 중 하나라도
  # 0이면 (score·종횡비가 음수 불가라 조건이 항상 거짓) 필터 전체 비활성.
  detect_fp_score_threshold: float = 0.78
  detect_fp_aspect_threshold: float = 0.70
  # 대형 근접 얼굴 재검출 회복 (ADR-017): rel_w가 이 값 이상인 저score 얼굴을 정규 스케일로 재검출해
  # score가 아래 값 이상이면 되살린다 — YuNet이 초근접 대형 얼굴에 저score를 줘 공통첩으로 빠지던 문제.
  # 둘 중 하나라도 0이면 비활성(기존 검출 동작).
  detect_big_face_rel_width: float = 0.30
  detect_big_face_redetect_score: float = 0.80

  # ── 품질 게이트 임계값 (눈감음/흔들림 — 하드코딩 금지, 기본값은 초기값이며 face-test 실측 보정) ──
  quality_blur_threshold: float = 25.0  # 정규화 variance 기준 (test2 라벨셋 보정, QualityConfig 주석 참고)
  quality_min_blur_face_px: int = 64  # 이보다 작은 얼굴은 blur 판정 제외 (variance 신뢰 불가)
  quality_blur_main_face_ratio: float = 0.5  # 최대 얼굴 폭 대비 이 비율 미만은 배경 얼굴로 보고 blur 판정 제외
  quality_whole_image_blur_threshold: float = 100.0  # 판정 자격 얼굴 없을 때 전체 이미지 fallback (별도 보정)
  quality_shake_coherence_threshold: float = 0.40  # fallback 2차 신호 — 방향 쏠림 임계 (ADR 014). 0 = 비활성
  quality_shake_max_norm_variance: float = 60.0  # 쏠림이 높아도 정규화 variance가 이 값 이상이면 선명으로 본다
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
      merge_facepair_floor=self.cluster_merge_facepair_floor,
      rescue_similarity=self.cluster_rescue_similarity,
      min_membership_similarity=self.cluster_min_membership_similarity,
      min_membership_margin=self.cluster_min_membership_margin,
      blob_promote_similarity=self.cluster_blob_promote_similarity,
      blob_promote_floor=self.cluster_blob_promote_floor,
    )

  def to_detector_config(self) -> "DetectorConfig":
    """설정값을 FaceDetector의 DetectorConfig로 변환한다 (값 검증은 DetectorConfig.__post_init__이 수행).

    to_cluster_config와 같은 이유로 pipeline 임포트를 지연시킨다 (core→pipeline 역의존 회피).
    """
    from app.pipeline.detect import DetectorConfig

    return DetectorConfig(
      max_side=self.detect_max_side,
      refine_landmarks=self.detect_refine_landmarks,
      refine_norm_face_width=self.detect_refine_norm_face_width,
      refine_margin_ratio=self.detect_refine_margin_ratio,
      min_face_rel_width=self.detect_min_face_rel_width,
      fp_score_threshold=self.detect_fp_score_threshold,
      fp_aspect_threshold=self.detect_fp_aspect_threshold,
      big_face_rel_width=self.detect_big_face_rel_width,
      big_face_redetect_score=self.detect_big_face_redetect_score,
    )

  def to_quality_config(self) -> "QualityConfig":
    """설정값을 품질 게이트의 QualityConfig로 변환한다 (값 검증은 QualityConfig.__post_init__이 수행).

    to_cluster_config와 같은 이유로 pipeline 임포트를 지연시킨다 (core→pipeline 역의존 회피).
    """
    from app.pipeline.quality import QualityConfig

    return QualityConfig(
      blur_threshold=self.quality_blur_threshold,
      min_blur_face_px=self.quality_min_blur_face_px,
      blur_main_face_ratio=self.quality_blur_main_face_ratio,
      whole_image_blur_threshold=self.quality_whole_image_blur_threshold,
      shake_coherence_threshold=self.quality_shake_coherence_threshold,
      shake_max_norm_variance=self.quality_shake_max_norm_variance,
      eye_closed_confidence=self.quality_eye_closed_confidence,
      eye_box_px=self.quality_eye_box_px,
    )
