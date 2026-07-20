"""SQS 메시지 스키마 — Spring 백엔드와의 wire 계약 (feature-spec §6).

인바운드 3종(분류 classify_request · 보정 cluster_feedback · 하드 삭제 delete_request)은 단일
FIFO 큐(messageGroupId=event_id, ADR-007)로 수신하므로 body의 `type` 필드로 판별한다
(`parse_inbound_message`). 모든 인바운드는 `job_id`를 가진다 — 멱등 처리 키이자, 처리 결과가
같은 job_id의 classify-result로 발행되는 상관관계 키다.

wire 필드명은 snake_case 그대로다(Spring이 SNAKE_CASE 직렬화 전략을 설정하는 것이 계약).
별칭(alias) 기계를 두지 않아 Python 필드명 = wire 필드명이 항상 성립한다.

보정 메시지의 merge/split/reassign/confirm_distinct → must-link/cannot-link 제약 변환, image_id ↔
face_id ↔ 행 인덱스 번역은 워커의 책임이다 (cluster.Constraints 독스트링 참고). 이 모듈은 순수 wire
계약만 안다 — pipeline·numpy에 의존하지 않는다.
"""

from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

# cluster.EMBED_DIM과 같은 값 — schemas → pipeline 의존을 금지하므로 로컬 상수로 중복 선언한다
# (cluster.py가 embed.EMBED_DIM을 중복 선언하는 것과 같은 이유).
EMBED_DIM = 512

# 빈 문자열 id는 매핑(.npz id 배열, 결과 상관관계)을 조용히 오염시키므로 계약 경계에서 거부한다.
# UUID 타입이 아닌 str인 이유: 결과 메시지가 받은 id를 바이트 그대로 되돌려야 Spring 쪽 상관관계가
# 성립하는데, UUID 타입은 라운드트립에서 표기를 정규화(대소문자·하이픈)한다.
Id = Annotated[str, Field(min_length=1)]
# NaN/inf는 JSON 직렬화 시 null 또는 비표준 토큰이 되어 Spring 파싱을 조용히 깨뜨린다 (recluster의 비유한값 거부와 동일 철학).
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# "분류가 어려워요"(uncertain) 앨범의 예약 id. 인물 앨범과 달리 uncertain 얼굴은 아무 cluster_id에도
# 속하지 않아(.npz엔 None) 이대로면 reassign의 대상(from_cluster_id)이 될 수 없다. 그래서 uncertain
# 사진을 하나의 가상 앨범으로 묶어 이 예약 id를 부여한다 — 사용자가 이 사진을 인물 앨범으로 옮기면
# Spring이 reassign(from_cluster_id=이 값)으로 되돌려주고, 워커가 해당 사진의 미매칭 얼굴을 매칭한다.
# .npz에는 저장하지 않는 결과-계약 전용 값이라(uuid가 아닌 예약 리터럴), 실제 cluster_id와 충돌하지 않는다.
UNCERTAIN_ALBUM_ID = "__uncertain__"


class _MessageBase(BaseModel):
  # frozen: 파이프라인 도메인 모델(frozen dataclass)과 동일한 불변 계약 — 워커가 처리 중 메시지를 변형하지 못한다.
  # extra="forbid" 양방향: 미지의 키 = 계약 드리프트이므로 MVP 개발 중 즉시 표면화한다
  # (아웃바운드 생성 시 오타 키도 잡힌다). 롤링 배포 전방 호환이 필요해지면 인바운드만 "ignore"로 완화한다.
  model_config = ConfigDict(frozen=True, extra="forbid")


# ── 인바운드 ① 분류 요청 (classify_request) ─────────────────────────────────────


class ImageRef(_MessageBase):
  """분류 대상 이미지 1장 — S3 원본 객체 참조 (업로드·저장은 Spring 소유, 워커는 읽기만)."""

  image_id: Id
  s3_key: Id


class ClassifyOptions(_MessageBase):
  """업로드 화면의 품질 제외 토글 (feature-spec §6.1, 기본 ON).

  ON이면 해당 사진을 인물 앨범 대신 eyes_closed/blurry 앨범으로 라우팅하고, OFF면 분리하지 않는다.
  """

  exclude_eyes_closed: bool = True
  exclude_blurry: bool = True


class ClassifyRequest(_MessageBase):
  """분류 작업 요청 (feature-spec §6.1) — 워커는 항상 event 전체 임베딩을 재군집한다 (최초/증분 플래그 없음)."""

  type: Literal["classify_request"] = "classify_request"
  job_id: Id
  group_id: Id
  event_id: Id
  images: list[ImageRef] = Field(min_length=1)
  options: ClassifyOptions = Field(default_factory=ClassifyOptions)

  @field_validator("images")
  @classmethod
  def _reject_duplicate_image_ids(cls, images: list[ImageRef]) -> list[ImageRef]:
    # 중복 image_id는 .npz의 photo_id 기준 멱등 append를 깨뜨려 같은 얼굴이 두 번 들어간다 (ADR-007 재군집 흐름 3단계)
    duplicated = sorted(image_id for image_id, count in Counter(ref.image_id for ref in images).items() if count > 1)
    if duplicated:
      raise ValueError(f"images의 image_id가 중복되었습니다. 중복 id: {duplicated}")
    return images


# ── 인바운드 ② 사용자 보정 (cluster_feedback) ───────────────────────────────────
# wire 형태는 feature-spec §6.3 그대로(action + 해당 payload 키 하나)이되, 스키마는 action별
# 판별 유니온으로 모델링한다 — 잘못된 조합(merge인데 split payload)이 스키마 수준에서 실패하고,
# 워커는 isinstance/match로 타입 안전하게 분기한다. action과 무관한 payload 키는 생략 또는 null이
# 계약이다 (None 필드 선언은 Jackson 기본 직렬화가 미사용 키를 null로 내보내도 forbid에 걸리지 않게 한다).

_NonEmptyGroup = Annotated[list[Id], Field(min_length=1)]


class MergePayload(_MessageBase):
  """여러 인물 클러스터를 하나로 병합 — 워커가 must-link 제약으로 변환한다."""

  target_cluster_id: Id
  source_cluster_ids: list[Id] = Field(min_length=1)

  @model_validator(mode="after")
  def _reject_self_merge(self) -> "MergePayload":
    if self.target_cluster_id in self.source_cluster_ids:
      raise ValueError(
        f"target_cluster_id가 source_cluster_ids에 포함될 수 없습니다. 받은 값: {self.target_cluster_id}"
      )
    return self


class SplitPayload(_MessageBase):
  """한 인물 클러스터를 사용자 지정 그룹들로 분리 — 워커가 그룹 간 cannot-link 제약으로 변환한다."""

  cluster_id: Id
  groups: list[_NonEmptyGroup] = Field(min_length=2)

  @model_validator(mode="after")
  def _reject_overlapping_groups(self) -> "SplitPayload":
    # 같은 image_id가 두 그룹에 있으면 그 얼굴들에 must-link(같은 그룹)와 cannot-link(다른 그룹)가 동시에
    # 걸려 recluster가 난해한 인덱스 수준 모순 오류로 거부한다 — 계약 경계에서 원인 그대로 거부한다.
    counts = Counter(image_id for group in self.groups for image_id in set(group))
    overlapping = sorted(image_id for image_id, count in counts.items() if count > 1)
    if overlapping:
      raise ValueError(f"split.groups 간에 같은 image_id가 있을 수 없습니다. 겹친 id: {overlapping}")
    return self


class ReassignPayload(_MessageBase):
  """특정 사진을 다른 인물로 이동 — 워커가 to와 must-link, from과 cannot-link 제약으로 변환한다."""

  image_id: Id
  from_cluster_id: Id
  to_cluster_id: Id

  @model_validator(mode="after")
  def _reject_noop_move(self) -> "ReassignPayload":
    if self.from_cluster_id == self.to_cluster_id:
      raise ValueError(f"이동 전후 클러스터가 같을 수 없습니다. 받은 값: {self.from_cluster_id}")
    return self


class ConfirmDistinctPayload(_MessageBase):
  """기존 인물 클러스터 여러 개를 서로 다른 사람으로 확정 — 워커가 대표 얼굴 전 쌍 cannot-link로 변환한다.

  merge의 반대 방향 선언이다: merge가 여러 클러스터를 하나로 묶는다면, 이 액션은 여러 클러스터가
  앞으로도 하나로 재군집되지 않도록 고정한다. must-link는 "같이 있어야 한다"만 강제할 뿐 "떨어져
  있어야 한다"는 강제하지 못해, 두 확정 앨범 사이로 유사도가 애매한 신규 사진(다리 사진)이 들어오면
  전체 재군집이 둘을 하나로 오병합할 수 있다 — 이 액션은 그 경로를 cannot-link로 막는다.
  """

  cluster_ids: list[Id] = Field(min_length=2)

  @model_validator(mode="after")
  def _reject_duplicate_clusters(self) -> "ConfirmDistinctPayload":
    duplicated = sorted(cluster_id for cluster_id, count in Counter(self.cluster_ids).items() if count > 1)
    if duplicated:
      raise ValueError(f"cluster_ids에 중복이 있을 수 없습니다. 중복 id: {duplicated}")
    return self


class _FeedbackBase(_MessageBase):
  type: Literal["cluster_feedback"] = "cluster_feedback"
  job_id: Id
  event_id: Id


class MergeFeedback(_FeedbackBase):
  action: Literal["merge"] = "merge"
  merge: MergePayload
  split: None = None
  reassign: None = None
  confirm_distinct: None = None


class SplitFeedback(_FeedbackBase):
  action: Literal["split"] = "split"
  split: SplitPayload
  merge: None = None
  reassign: None = None
  confirm_distinct: None = None


class ReassignFeedback(_FeedbackBase):
  action: Literal["reassign"] = "reassign"
  reassign: ReassignPayload
  merge: None = None
  split: None = None
  confirm_distinct: None = None


class ConfirmDistinctFeedback(_FeedbackBase):
  action: Literal["confirm_distinct"] = "confirm_distinct"
  confirm_distinct: ConfirmDistinctPayload
  merge: None = None
  split: None = None
  reassign: None = None


ClusterFeedback = Annotated[
  MergeFeedback | SplitFeedback | ReassignFeedback | ConfirmDistinctFeedback, Field(discriminator="action")
]


# ── 인바운드 ③ 하드 삭제 (delete_request) ──────────────────────────────────────


class DeleteRequest(_MessageBase):
  """사진 하드 삭제 요청 (ADR-007) — 워커가 event .npz를 마스킹 rewrite해 물리 제거한다.

  image_ids 중복은 검증하지 않는다 — 삭제 마스킹(np.isin)은 중복에 멱등이라 실해가 없다.
  """

  type: Literal["delete_request"] = "delete_request"
  job_id: Id
  event_id: Id
  image_ids: list[Id] = Field(min_length=1)


# ── 인바운드 판별 유니온 + 파서 ─────────────────────────────────────────────────

InboundMessage = Annotated[ClassifyRequest | ClusterFeedback | DeleteRequest, Field(discriminator="type")]
_INBOUND_ADAPTER: TypeAdapter[
  ClassifyRequest | MergeFeedback | SplitFeedback | ReassignFeedback | ConfirmDistinctFeedback | DeleteRequest
] = TypeAdapter(InboundMessage)  # 검증기 빌드 비용이 있어 모듈 로드 시 1회만 생성한다


def parse_inbound_message(
  body: str | bytes,
) -> ClassifyRequest | MergeFeedback | SplitFeedback | ReassignFeedback | ConfirmDistinctFeedback | DeleteRequest:
  """SQS 메시지 body(JSON)를 `type` 필드로 판별해 해당 스키마로 파싱한다.

  계약 위반(미지 type·필드 누락·형식 오류)은 pydantic.ValidationError로 전파한다 —
  재시도/DLQ 처리 정책은 워커의 몫이다.
  """
  return _INBOUND_ADAPTER.validate_json(body)


# ── 아웃바운드 분류 결과 (classify-result) ──────────────────────────────────────
# 결과 큐 전용이라 type 판별 필드가 없다. 인바운드 3종 모두 처리 결과를 이 형식으로 발행한다.


class ResultCluster(_MessageBase):
  """인물 클러스터 1개 — 앱 person 앨범과 1:1 (cluster_id = 앱의 personId)."""

  cluster_id: Id
  is_new: bool
  image_ids: list[Id] = Field(min_length=1)
  # PersonCluster.centroid(멤버 임베딩 L2 정규화 평균)의 사본 — Spring 표시용 파생값 (ADR-007)
  representative_vector: list[FiniteFloat] = Field(min_length=EMBED_DIM, max_length=EMBED_DIM)
  # 대표 얼굴 썸네일 JPEG의 S3 키 (워커 embeddings 버킷, CHMO-335) — Spring이 presigned URL 발급에 쓴다.
  # None = 썸네일 없음: 기능 비활성(THUMBNAIL_MAX_SIDE=0) / 렌더·업로드 실패(best-effort) /
  # v2 이하 .npz 행만으로 구성된 클러스터(bbox·원본 키 미상 — 대표 후보 부재).
  thumbnail_s3_key: Id | None = None


class UncertainImage(_MessageBase):
  """인물에 자신 있게 붙이지 못한 사진 ("분류가 어려워요" 앨범, 뷰어 비노출).

  reason 매핑 (ReclusterResult 기준): ambiguous = 두 인물 사이 저신뢰(ambiguous_indices),
  unmatched = 얼굴은 검출됐으나 어느 인물과도 매칭되지 않음(noise_indices, 예: 행인).
  얼굴 미검출(인물 없는) 사진은 uncertain이 아니라 common_album으로 간다 (feature-spec §6.2).

  album_id: 이 사진이 속한 uncertain 앨범의 예약 id(UNCERTAIN_ALBUM_ID). 사용자가 이 사진을 인물
  앨범으로 옮길 때 Spring이 reassign의 from_cluster_id로 그대로 되돌려주면 워커가 편입 처리한다.
  """

  image_id: Id
  reason: Literal["ambiguous", "unmatched"]  # TBD #2: back·duplicate 추가 합의 시 Literal 확장
  album_id: Id = UNCERTAIN_ALBUM_ID  # reassign의 from_cluster_id로 되돌려줄 출처 앨범 id (예약 리터럴)


class FailedImage(_MessageBase):
  """기술적 분석 실패 (재시도 대상 — 품질·매칭 문제인 앨범들과 별개)."""

  image_id: Id
  reason: Id  # 실패 사유 enum 확정 전 자유 형식 (예: "timeout")


class ClassifyResult(_MessageBase):
  """분류 결과 (feature-spec §6.2) — 필드들은 앱의 앨범 5종과 1:1 대응한다.

  리스트 기본값이 전부 빈 리스트인 이유: status="failed" 결과를 job_id·status만으로 구성할 수 있어야 한다.
  """

  job_id: Id
  status: Literal["succeeded", "partial", "failed"]
  clusters: list[ResultCluster] = Field(default_factory=list)
  common_album: list[Id] = Field(default_factory=list)
  uncertain: list[UncertainImage] = Field(default_factory=list)
  eyes_closed: list[Id] = Field(default_factory=list)
  blurry: list[Id] = Field(default_factory=list)
  failed_images: list[FailedImage] = Field(default_factory=list)
  # 이번 재군집에서 승계되지 못해 은퇴한 기존 cluster_id (ReclusterResult.retired_cluster_ids) —
  # Spring이 해당 인물 앨범을 정리하는 데 쓴다
  retired_cluster_ids: list[Id] = Field(default_factory=list)


# ── 아웃바운드 진행률 (progress) ────────────────────────────────────────────────
# 결과 큐가 아니라 별도 progress 큐로 발행한다 (CHMO-274). classify_request 처리는 이미지 루프가
# job 비용의 사실상 전부라(임베딩 ≈476ms/장), 워커가 그 루프 도중 처리 장수를 여러 번 흘려보내
# 백엔드가 분류 진행바를 그린다. ClassifyResult와 큐를 나누는 이유: 결과 큐는 "type 판별 필드가
# 없다"가 전제(위 §아웃바운드 분류 결과 註)라 여기에 섞을 수 없어, 이 메시지는 type을 갖는다.


class ProgressUpdate(_MessageBase):
  """분류 작업 진행률 1건 — progress 큐 전용 (CHMO-274).

  processed는 이 job 안에서 단조 증가한다(0 → … → total). SQS는 표준 큐에서 순서를 보장하지 않고
  at-least-once라, 백엔드는 processed를 순서·중복·재전달 방어 키로 쓴다 — 마지막으로 본 값 이하의
  메시지는 버리면 진행바가 거꾸로 튀지 않는다(job 재시도로 0부터 다시 발행돼도 안전). 그래서 별도
  seq 필드를 두지 않는다 — processed가 곧 순서 키다.
  """

  type: Literal["progress"] = "progress"
  job_id: Id
  event_id: Id
  processed: Annotated[int, Field(ge=0)]  # 지금까지 처리(루프 통과)한 이미지 수
  total: Annotated[int, Field(ge=1)]  # 이 job의 전체 이미지 수 (진행률 분모)

  @model_validator(mode="after")
  def _reject_overrun(self) -> "ProgressUpdate":
    if self.processed > self.total:
      raise ValueError(f"processed는 total을 넘을 수 없습니다. processed={self.processed} total={self.total}")
    return self


if __name__ == "__main__":
  # SQS/워커 없이 wire 계약 자체를 자가 검증한다 (cluster.py __main__과 같은 실행형 확인 패턴).
  # 성공 픽스처는 docs/spec/message-examples.md의 예시와 동일 형태를 유지한다.
  # TODO(CHMO-XX): pytest 도입 시 tests/schemas/test_messages.py로 승격
  import json

  from pydantic import ValidationError

  passed = 0

  def check(name: str, condition: bool) -> None:
    global passed
    if not condition:
      raise SystemExit(f"실패: {name}")
    passed += 1
    print(f"통과: {name}")

  classify_body = {
    "type": "classify_request",
    "job_id": "job-1",
    "group_id": "group-1",
    "event_id": "event-1",
    "images": [
      {"image_id": "img-1", "s3_key": "groups/group-1/events/event-1/IMG_0001.jpg"},
      {"image_id": "img-2", "s3_key": "groups/group-1/events/event-1/IMG_0002.jpg"},
    ],
    "options": {"exclude_eyes_closed": True, "exclude_blurry": False},
  }
  message = parse_inbound_message(json.dumps(classify_body))
  check(
    "classify_request 파싱",
    isinstance(message, ClassifyRequest)
    and message.images[1].image_id == "img-2"
    and not message.options.exclude_blurry,
  )

  no_options = {key: value for key, value in classify_body.items() if key != "options"}
  message = parse_inbound_message(json.dumps(no_options))
  check(
    "options 생략 시 기본 ON",
    isinstance(message, ClassifyRequest) and message.options.exclude_eyes_closed and message.options.exclude_blurry,
  )

  merge_body = {
    "type": "cluster_feedback",
    "job_id": "job-2",
    "event_id": "event-1",
    "action": "merge",
    "merge": {"target_cluster_id": "person-A", "source_cluster_ids": ["person-B", "person-C"]},
    "split": None,  # Jackson 기본 직렬화 호환 — 미사용 키의 null 동봉 허용
    "reassign": None,
  }
  message = parse_inbound_message(json.dumps(merge_body))
  check(
    "merge 보정 파싱 (null 동봉)", isinstance(message, MergeFeedback) and message.merge.target_cluster_id == "person-A"
  )

  split_body = {
    "type": "cluster_feedback",
    "job_id": "job-3",
    "event_id": "event-1",
    "action": "split",
    "split": {"cluster_id": "person-A", "groups": [["img-1", "img-2"], ["img-3"]]},
  }
  message = parse_inbound_message(json.dumps(split_body))
  check("split 보정 파싱 (미사용 키 생략)", isinstance(message, SplitFeedback) and len(message.split.groups) == 2)

  reassign_body = {
    "type": "cluster_feedback",
    "job_id": "job-4",
    "event_id": "event-1",
    "action": "reassign",
    "reassign": {"image_id": "img-7", "from_cluster_id": "person-A", "to_cluster_id": "person-B"},
  }
  message = parse_inbound_message(json.dumps(reassign_body))
  check("reassign 보정 파싱", isinstance(message, ReassignFeedback) and message.reassign.to_cluster_id == "person-B")

  confirm_distinct_body = {
    "type": "cluster_feedback",
    "job_id": "job-4b",
    "event_id": "event-1",
    "action": "confirm_distinct",
    "confirm_distinct": {"cluster_ids": ["person-A", "person-B", "person-C"]},
  }
  message = parse_inbound_message(json.dumps(confirm_distinct_body))
  check(
    "confirm_distinct 보정 파싱",
    isinstance(message, ConfirmDistinctFeedback)
    and message.confirm_distinct.cluster_ids == ["person-A", "person-B", "person-C"],
  )

  delete_body = {"type": "delete_request", "job_id": "job-5", "event_id": "event-1", "image_ids": ["img-1", "img-1"]}
  message = parse_inbound_message(json.dumps(delete_body))
  check("delete_request 파싱 (중복 id 허용)", isinstance(message, DeleteRequest) and len(message.image_ids) == 2)

  result = ClassifyResult(
    job_id="job-1",
    status="succeeded",
    clusters=[
      ResultCluster(
        cluster_id="person-A",
        is_new=False,
        image_ids=["img-1", "img-2", "img-7"],
        representative_vector=[0.0] * (EMBED_DIM - 1) + [1.0],
        thumbnail_s3_key="thumbnails/event-1/person-A.jpg",
      )
    ],
    common_album=["img-9"],
    uncertain=[
      UncertainImage(image_id="img-5", reason="ambiguous"),
      UncertainImage(image_id="img-6", reason="unmatched"),
    ],
    eyes_closed=["img-3"],
    blurry=["img-4"],
    failed_images=[FailedImage(image_id="img-8", reason="timeout")],
    retired_cluster_ids=["person-C"],
  )
  check("classify-result 직렬화 라운드트립", ClassifyResult.model_validate_json(result.model_dump_json()) == result)
  check(
    "thumbnail_s3_key 생략 시 None (구버전 워커·비활성 결과 하위호환)",
    ResultCluster(
      cluster_id="person-B", is_new=True, image_ids=["img-1"], representative_vector=[0.0] * (EMBED_DIM - 1) + [1.0]
    ).thumbnail_s3_key
    is None,
  )
  check(
    "uncertain 항목은 예약 앨범 id 기본 부여 (reassign 출처)",
    all(uncertain.album_id == UNCERTAIN_ALBUM_ID for uncertain in result.uncertain)
    and '"album_id":"__uncertain__"' in result.model_dump_json().replace(" ", ""),
  )
  check("failed 결과 최소 구성", ClassifyResult(job_id="job-9", status="failed").clusters == [])

  progress = ProgressUpdate(job_id="job-1", event_id="event-1", processed=30, total=300)
  check(
    "progress 직렬화 라운드트립 + type 동봉",
    ProgressUpdate.model_validate_json(progress.model_dump_json()) == progress
    and '"type":"progress"' in progress.model_dump_json().replace(" ", ""),
  )
  check(
    "progress 경계값 (0/total, total/total)",
    (
      ProgressUpdate(job_id="j", event_id="e", processed=0, total=1).processed == 0
      and ProgressUpdate(job_id="j", event_id="e", processed=5, total=5).processed == 5
    ),
  )

  invalid_cases = [
    ("type 누락", {key: value for key, value in classify_body.items() if key != "type"}),
    ("미지 type", {**classify_body, "type": "unknown_request"}),
    ("미지 키 (계약 드리프트)", {**classify_body, "priority": "high"}),
    ("빈 images", {**classify_body, "images": []}),
    (
      "image_id 중복",
      {**classify_body, "images": [{"image_id": "img-1", "s3_key": "a"}, {"image_id": "img-1", "s3_key": "b"}]},
    ),
    ("빈 문자열 id", {**classify_body, "job_id": ""}),
    (
      "merge에 split payload 채움",
      {**merge_body, "split": {"cluster_id": "person-A", "groups": [["img-1"], ["img-2"]]}},
    ),
    (
      "자기 병합 (target ∈ sources)",
      {**merge_body, "merge": {"target_cluster_id": "person-A", "source_cluster_ids": ["person-A"]}},
    ),
    ("split 그룹 1개", {**split_body, "split": {"cluster_id": "person-A", "groups": [["img-1", "img-2"]]}}),
    ("split 빈 그룹", {**split_body, "split": {"cluster_id": "person-A", "groups": [["img-1"], []]}}),
    (
      "split 그룹 간 교집합",
      {**split_body, "split": {"cluster_id": "person-A", "groups": [["img-1", "img-2"], ["img-2"]]}},
    ),
    (
      "reassign 이동 전후 동일",
      {**reassign_body, "reassign": {"image_id": "img-7", "from_cluster_id": "person-A", "to_cluster_id": "person-A"}},
    ),
    (
      "confirm_distinct에 merge payload 채움",
      {**confirm_distinct_body, "merge": {"target_cluster_id": "person-A", "source_cluster_ids": ["person-B"]}},
    ),
    (
      "confirm_distinct 클러스터 1개",
      {**confirm_distinct_body, "confirm_distinct": {"cluster_ids": ["person-A"]}},
    ),
    (
      "confirm_distinct 중복 cluster_id",
      {**confirm_distinct_body, "confirm_distinct": {"cluster_ids": ["person-A", "person-A"]}},
    ),
    ("빈 image_ids 삭제", {**delete_body, "image_ids": []}),
  ]
  for case_name, body in invalid_cases:
    try:
      parse_inbound_message(json.dumps(body))
    except ValidationError:
      check(f"거부: {case_name}", True)
    else:
      raise SystemExit(f"실패: {case_name} — ValidationError가 발생해야 하는데 통과됨")

  try:
    ResultCluster(
      cluster_id="person-A", is_new=True, image_ids=["img-1"], representative_vector=[0.0] * (EMBED_DIM - 1)
    )
  except ValidationError:
    check("거부: 대표벡터 길이 511", True)
  else:
    raise SystemExit("실패: 대표벡터 길이 511 — ValidationError가 발생해야 하는데 통과됨")

  try:
    ResultCluster(
      cluster_id="person-A", is_new=True, image_ids=["img-1"], representative_vector=[float("nan")] * EMBED_DIM
    )
  except ValidationError:
    check("거부: 대표벡터 NaN", True)
  else:
    raise SystemExit("실패: 대표벡터 NaN — ValidationError가 발생해야 하는데 통과됨")

  for case_name, kwargs in [
    ("progress total=0", {"job_id": "j", "event_id": "e", "processed": 0, "total": 0}),
    ("progress processed>total", {"job_id": "j", "event_id": "e", "processed": 4, "total": 3}),
  ]:
    try:
      ProgressUpdate(**kwargs)
    except ValidationError:
      check(f"거부: {case_name}", True)
    else:
      raise SystemExit(f"실패: {case_name} — ValidationError가 발생해야 하는데 통과됨")

  print(f"\n스모크 검증 {passed}건 전부 통과")
