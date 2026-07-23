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
  from app.pipeline.thumbnail import ThumbnailConfig


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
  # ADR-024: 병합 승인을 컴포넌트 '현재 전체 멤버' 재평가로 판정 (false = 구 파편 스냅샷 완전 연결)
  cluster_merge_component_linkage: bool = True
  cluster_rescue_similarity: float = 0.6
  # margin 구제 (2026-07-23 실측) — 절대 유사도는 낮지만 2위 군집 대비 여유가 큰 노이즈(옆얼굴·역광)
  # 편입. 0 = 비활성(기본 — 실 이벤트 적대 검증 전 실험 전용). 근거·한계는 ClusterConfig 주석 참조.
  cluster_margin_rescue_floor: float = 0.0
  cluster_margin_rescue_ratio: float = 1.7
  cluster_min_membership_similarity: float = 0.4
  cluster_min_membership_margin: float = 0.05
  cluster_evict_gray_ceiling: float = 0.46  # ADR-020: LOO centroid 회색지대 상한 — face-pair 재확인 대상
  cluster_evict_facepair_floor: float = 0.45  # ADR-020: 회색지대 잔류 자격(최강 쌍) — 미만이면 남남 부착 축출
  cluster_blob_promote_similarity: float = 0.45
  cluster_blob_promote_floor: float = 0.4
  # 라우팅 정책: 주 인물 얼굴 2명+ 사진을 매칭 여부와 무관하게 공용 앨범에도 노출한다 (인물 앨범과 중복, feature-spec §6.2).
  # False면 구 정책(전원 미매칭 2+만 공용). Spring/앱이 새 common_album 의미를 감당할 때까지 끄는 롤아웃 스위치.
  cluster_group_photo_to_common: bool = True
  # 매칭 사진의 주 인물 미매칭 얼굴을 uncertain에도 노출 — 미등록 인물의 수동 편입 진입점 (feature-spec §6.2).
  # False면 구 정책(매칭 얼굴이 하나라도 있으면 uncertain 제외 — 인물 앨범 우선 배타)
  cluster_unmatched_main_to_uncertain: bool = True
  # 주 인물 자격 — 사진 최대 얼굴 폭 대비 이 비율 미만은 행인으로 보고 위 카운트에서 제외 (ADR 022 규칙). 0=전체 카운트
  cluster_common_main_face_ratio: float = 0.5
  # 실인물 자격 — 미배정 얼굴이 event 내 어떤 얼굴과도 유사도가 이 값 미만이면 오검출로 보고 카운트에서 제외 (ADR 025). 0=비활성
  cluster_common_face_min_similarity: float = 0.185
  # 이중 검출 붕괴 — 같은 사진의 두 얼굴 행이 이 값 이상 닮으면 한 얼굴의 두 박스로 보고 한 명으로 센다 (ADR 027). 0=비활성
  cluster_common_duplicate_face_similarity: float = 0.95
  # 근중복 행 붕괴 (ADR 029) — 재업로드·유령 행 복제(유사도 ≥ 이 값)를 재군집 전에 대표 1행으로 접어
  # 앨범이 쌍 단위로 와해되는 오염(실 event 8)을 차단한다. 0=비활성 (붕괴 없이 종전 동작)
  cluster_duplicate_collapse_similarity: float = 0.985
  # uncertain 품질 원인 판정 (CHMO-404) — 주 얼굴이 이 px 미만이면 small_faces, 그중 원본 긴 변이 아래 값 미만이면
  # low_resolution도 함께. 앱이 "분류가 어려워요" 화면에 설명·재업로드 안내를 띄우는 근거. small_face_px=0이면 기능 전체 비활성
  cluster_uncertain_small_face_px: float = 100.0
  cluster_uncertain_low_res_long_side: float = 2000.0  # 0이면 low_resolution 원인만 비활성(small_faces는 유지)

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
  # 둘 중 하나라도 0이면 비활성(기존 검출 동작). 0.30→0.20 재보정(ADR-017 §재보정): 화면을 덮는
  # 얼굴은 bbox 파편이 rel_w 0.28대로 나와 게이트 미달 — 스윕 실측 FP 0·손실 0.
  detect_big_face_rel_width: float = 0.20
  detect_big_face_redetect_score: float = 0.80
  # 크기 인지형 confident 게이트 (ADR-028): 대형 얼굴(rel_w ≥ BIG_FACE_REL_WIDTH)은 confident 게이트를
  # 이 값으로 올려 저score 구간을 회복 재검출로 재판정한다 — 손·종이 등 대형 오검출이 게이트 0.6을
  # 턱걸이로 통과해 유령 앨범을 만들던 문제(event 115). 대형은 재검출이 실얼굴/오검출을 가르므로 손실
  # 없이 FP만 떨어진다. 소형은 회복이 없어 미적용. 기본 게이트(0.6)와 같게 두면 비활성(구 동작).
  detect_big_face_confident_score: float = 0.70
  # 재검출 랜드마크 신뢰 임계 (survey_refine_shift.py 실측): 재검출 score가 이 값 이상이면 이동량
  # 가드를 무시하고 재검출 랜드마크를 채택 — 초대형 얼굴은 bbox가 파편이라 올바른 교정도 파편 폭
  # 기준 가드에 걸려 깨진 랜드마크가 유지되던 문제(event 73 공통첩 유출·유령 앨범). 0 = 비활성.
  detect_refine_trust_redetect_score: float = 0.80
  # confident 파편 디둡 (ADR-027): 초대형 얼굴의 파편 박스 여러 개가 전부 score 게이트를 통과하면 한
  # 사람이 두 명으로 세어진다(event 105 셀피 공용 노출). 정제 랜드마크 중심거리가 이 비율×얼굴폭 미만인
  # 대형(> refine_norm) confident 쌍은 score 최상 박스만 남긴다. 0 = 비활성.
  detect_confident_dedup_landmark_ratio: float = 0.10

  # ── 품질 게이트 임계값 (눈감음/흔들림 — 하드코딩 금지, 기본값은 초기값이며 face-test 실측 보정) ──
  quality_blur_threshold: float = 25.0  # 정규화 variance 기준 (test2 라벨셋 보정, QualityConfig 주석 참고)
  quality_min_blur_face_px: int = 64  # 이보다 작은 얼굴은 blur 판정 제외 (variance 신뢰 불가)
  quality_blur_main_face_ratio: float = 0.5  # 최대 얼굴 폭 대비 이 비율 미만은 배경 얼굴로 보고 blur 판정 제외
  quality_whole_image_blur_threshold: float = 100.0  # 판정 자격 얼굴 없을 때 전체 이미지 fallback (별도 보정)
  quality_shake_coherence_threshold: float = 0.35  # fallback 2차 신호 — 쏠림 임계 (ADR 014 §재보정). 0 = 비활성
  quality_shake_max_norm_variance: float = 60.0  # 쏠림이 높아도 정규화 variance가 이 값 이상이면 선명으로 본다
  quality_shake_coherence_floor: float = 0.35  # 흔들림 재확인 게이트 — 쏠림이 이 값 미만이면 해제. 0 = 비활성
  quality_whole_image_collapse_variance: float = 40.0  # fallback 한정 게이트 면제 — 붕괴는 흔들림 확정. 0 = 비활성
  quality_collapse_face_rel_width: float = 0.22  # 붕괴 면제 얼굴 경로 확장 — rel_w 하한 (ADR 018). 0 = 비활성
  quality_face_var_collapse_floor: float = (
    7.0  # 얼굴 face_var 붕괴 면제 — 미만이면 흔들림 확정 (ADR 018 §보강3). 0 = 비활성
  )
  quality_eye_closed_confidence: float = 0.85  # face-test 실측 보정 (약한 오탐 제거, feature-spec §10 #3)
  quality_eye_box_px: int = 24
  quality_min_eye_face_px: int = 64  # 이보다 작은 얼굴은 눈감음 판정 제외 (정보 부족, ADR 019). 0 = 비활성
  quality_eye_cheek_brightness_ceiling: float = (
    1.4  # 눈/볼 밝기 비 상한 — 초과면 가림으로 보고 미판정 (ADR 019). 0 = 비활성
  )
  # 눈감음 하이브리드 1차(blendshape, ADR 021): min(blinkL, blinkR)가 이 값 이상이면 눈감음.
  # 0 = blink 비활성(롤백 스위치) — litert·face_landmarker 로딩 자체를 건너뛰고 순수 CNN 경로로 복귀.
  quality_blink_threshold: float = 0.40
  quality_blink_presence_floor: float = 0.5  # 랜드마크 presence가 미만이면 blink를 버리고 CNN 폴백 (ADR 021)
  quality_eye_main_face_ratio: float = 0.5  # 최대 얼굴 폭 대비 이 비율 미만은 배경 얼굴로 보고 눈감음 판정 제외
  quality_eye_min_rel_width: float = (
    0.08  # bbox 폭이 이미지 긴 변의 이 비율 미만은 눈감음 판정 제외 (원거리 내려뜸 오탐, ADR 026). 0 = 비활성
  )

  # ── 인물 앨범 대표 얼굴 썸네일 (CHMO-335) ────────────────────────────────────
  # 재군집 후 클러스터마다 대표 얼굴을 crop해 embeddings_bucket의 {thumbnail_prefix}{event_id}/
  # {cluster_id}.jpg (= thumbnails/events/{event_id}/{cluster_id}.jpg)에 덮어쓰고, 결과 메시지에 그
  # 키를 싣는다 (Spring이 presigned URL 발급).
  thumbnail_max_side: int = Field(default=256, ge=0)  # 썸네일 긴 변 상한 px. 0 = 기능 전체 비활성 (롤백 스위치)
  thumbnail_jpeg_quality: int = Field(default=85, ge=1, le=100)
  thumbnail_bbox_scale: float = Field(default=1.4, gt=0)  # 얼굴 bbox 확장 배율 — 여백 포함 crop
  # 썸네일 키 = {prefix}{event_id}/{cluster_id}.jpg. 기본 prefix에 events/를 포함해 원본
  # (originals/events/{id}/)·사진 썸네일(thumbnails/events/{id}/)과 같은 계층에 둔다 (버킷 레이아웃 통일).
  thumbnail_prefix: str = "thumbnails/events/"

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
      merge_component_linkage=self.cluster_merge_component_linkage,
      rescue_similarity=self.cluster_rescue_similarity,
      margin_rescue_floor=self.cluster_margin_rescue_floor,
      margin_rescue_ratio=self.cluster_margin_rescue_ratio,
      min_membership_similarity=self.cluster_min_membership_similarity,
      min_membership_margin=self.cluster_min_membership_margin,
      evict_gray_ceiling=self.cluster_evict_gray_ceiling,
      evict_facepair_floor=self.cluster_evict_facepair_floor,
      blob_promote_similarity=self.cluster_blob_promote_similarity,
      blob_promote_floor=self.cluster_blob_promote_floor,
      group_photo_to_common=self.cluster_group_photo_to_common,
      unmatched_main_to_uncertain=self.cluster_unmatched_main_to_uncertain,
      common_main_face_ratio=self.cluster_common_main_face_ratio,
      common_face_min_similarity=self.cluster_common_face_min_similarity,
      uncertain_small_face_px=self.cluster_uncertain_small_face_px,
      uncertain_low_res_long_side=self.cluster_uncertain_low_res_long_side,
      common_duplicate_face_similarity=self.cluster_common_duplicate_face_similarity,
      duplicate_collapse_similarity=self.cluster_duplicate_collapse_similarity,
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
      big_face_confident_score=self.detect_big_face_confident_score,
      refine_trust_redetect_score=self.detect_refine_trust_redetect_score,
      confident_dedup_landmark_ratio=self.detect_confident_dedup_landmark_ratio,
    )

  def to_thumbnail_config(self) -> "ThumbnailConfig":
    """설정값을 썸네일 렌더의 ThumbnailConfig로 변환한다 (값 검증은 ThumbnailConfig.__post_init__이 수행).

    to_cluster_config와 같은 이유로 pipeline 임포트를 지연시킨다 (core→pipeline 역의존 회피).
    비활성 여부(thumbnail_max_side == 0)는 호출자(deps)가 먼저 판단한다 — 0은 여기서 검증 실패다.
    """
    from app.pipeline.thumbnail import ThumbnailConfig

    return ThumbnailConfig(
      bbox_scale=self.thumbnail_bbox_scale,
      max_side=self.thumbnail_max_side,
      jpeg_quality=self.thumbnail_jpeg_quality,
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
      shake_coherence_floor=self.quality_shake_coherence_floor,
      whole_image_collapse_variance=self.quality_whole_image_collapse_variance,
      collapse_face_rel_width=self.quality_collapse_face_rel_width,
      face_var_collapse_floor=self.quality_face_var_collapse_floor,
      eye_closed_confidence=self.quality_eye_closed_confidence,
      eye_box_px=self.quality_eye_box_px,
      min_eye_face_px=self.quality_min_eye_face_px,
      eye_cheek_brightness_ceiling=self.quality_eye_cheek_brightness_ceiling,
      blink_threshold=self.quality_blink_threshold,
      blink_presence_floor=self.quality_blink_presence_floor,
      eye_main_face_ratio=self.quality_eye_main_face_ratio,
      eye_min_rel_width=self.quality_eye_min_rel_width,
    )
