"""event 단위 임베딩 묶음의 도메인 타입과 .npz 직렬화 코덱 (ADR-007).

S3 객체 `embeddings/{event_id}.npz` 1개 = `EventEmbeddings` 1개. 순수 numpy 모듈로
boto3·pipeline·schemas를 모른다 — S3 입출력은 `embedding_store`, 재군집 호출은 워커(handlers)의
책임이다. ADR-007이 확정한 직렬화 세부(id 배열 np.str_, 양방향 allow_pickle=False,
미배정 cluster_id="" 인코딩, savez_compressed, 로드 시 불변식 검증)는 전부 이 모듈이 소유한다.
"""

import io
import zipfile
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

# cluster.EMBED_DIM과 같은 값 — storage → pipeline 의존을 금지하므로 로컬 상수로 중복 선언한다
# (messages.py가 같은 이유로 중복 선언하는 것과 동일).
EMBED_DIM = 512
# v2: face_widths(얼굴 bbox 폭 px) 열 추가 — 라우팅의 주 인물 판정용 (CHMO-330). v1 파일은 폭 0(미상)으로
# 로드된다: 한 사진의 행들은 같은 요청에서 함께 추가되므로 사진 내 세대 혼합이 없고, 전원 0이면 최대 폭도
# 0이라 주 인물 판정(≥ 최대×비율)이 전원 통과 — 종전(전체 얼굴 수) 동작과 정확히 일치하는 안전 폴백.
# v3: bboxes(x,y,w,h px)·s3_keys(원본 객체 키) 열 추가 — 대표 얼굴 썸네일 crop의 원천 (CHMO-335).
# v2 이하 파일은 bbox 전부 0·s3_key ""로 로드된다: 폭 0/키 없음 행은 썸네일 대표 후보 자격이 없어
# 해당 행만으로 구성된 클러스터는 썸네일 None — 종전(썸네일 없음) 동작과 일치하는 안전 폴백.
SCHEMA_VERSION = 3
# numpy 문자열 배열에는 null이 없어 "직전 배정 없음"을 빈 문자열로 인코딩한다 (ADR-007)
_NO_CLUSTER = ""
_ID_KEYS = ("face_ids", "photo_ids", "cluster_ids")
_PAIR_KEYS = ("must_link_pairs", "cannot_link_pairs")


class StoreCorruptionError(Exception):
  """저장된 .npz가 스키마 계약을 위반함 — 재시도로 해소되지 않는 결정적 실패.

  워커는 이 예외를 작업 전체 실패로 다루되, 손상 객체를 자동 삭제/덮어쓰기하지 않는다
  (생체 파생 데이터이자 장애 증거 — 수동 복구 대상).
  """


def _as_str_array(values: Sequence[str] | Sequence[tuple[str, str]], columns: int | None = None) -> np.ndarray:
  """id 시퀀스(또는 id 쌍 시퀀스)를 np.str_ 배열로 변환한다.

  빈 입력을 np.array()에 그대로 넣으면 dtype이 float64로 추론돼 로드 시 문자열 계약이 깨지므로
  (ADR-007이 지적한 함정), 빈 배열은 명시적 dtype·shape로 만든다.
  """
  if len(values) == 0:
    shape = (0,) if columns is None else (0, columns)
    return np.empty(shape, dtype="<U1")
  return np.array(values, dtype=np.str_)


@dataclass(frozen=True)
class EventEmbeddings:
  """한 event의 얼굴 임베딩 전체와 부속 데이터 — 재군집(전체 재군집이 진실, ADR-003)의 입력 원천.

  행(row) 하나 = 검출된 얼굴 하나. face_id는 워커가 임베딩 시점에 발급하는 내부 키로 Spring은
  모른다(메시지 계약은 image_id 기준). 한 image_id가 얼굴 수만큼 여러 행을 가질 수 있다.
  제약 쌍은 face_id 참조로 저장하며 배열 순서 = 시간순(append-only) — later-wins 충돌 해소의 근거다.
  """

  # ndarray 필드의 자동 __eq__는 진리값 모호성으로 예외를 던지므로 비교에서 제외한다 (PersonCluster와 동일)
  embeddings: np.ndarray = field(compare=False)  # shape (N, EMBED_DIM), float32, L2 정규화 단위벡터
  face_ids: tuple[str, ...]  # 길이 N, 유일
  photo_ids: tuple[str, ...]  # 길이 N (= 메시지의 image_id) — 삭제 마스킹·멱등 append의 키
  cluster_ids: tuple[str | None, ...]  # 길이 N — 직전 재군집 배정 (None = 신규/노이즈/ambiguous)
  must_link_pairs: tuple[tuple[str, str], ...]  # face_id 쌍, 시간순
  cannot_link_pairs: tuple[tuple[str, str], ...]  # face_id 쌍, 시간순
  face_widths: tuple[float, ...] = ()  # 길이 N — 얼굴 bbox 폭 px (0 = 미상, v1 파일 폴백). 주 인물 판정용
  bboxes: tuple[
    tuple[float, float, float, float], ...
  ] = ()  # 길이 N — (x, y, w, h) 원본 px (전부 0 = 미상, v2 이하 폴백)
  s3_keys: tuple[str, ...] = ()  # 길이 N — 원본 이미지 S3 키 ("" = 미상, v2 이하 폴백). 썸네일 재fetch용

  def __post_init__(self) -> None:
    n = len(self.face_ids)
    if self.embeddings.ndim != 2 or self.embeddings.shape != (n, EMBED_DIM):
      raise ValueError(f"embeddings는 shape ({n}, {EMBED_DIM})이어야 합니다. 받은 shape: {self.embeddings.shape}")
    if self.embeddings.dtype != np.float32:
      raise ValueError(f"embeddings는 dtype float32여야 합니다. 받은 dtype: {self.embeddings.dtype}")
    if len(self.photo_ids) != n or len(self.cluster_ids) != n:
      raise ValueError(
        f"id 배열 길이가 서로 다릅니다. face_ids={n}, photo_ids={len(self.photo_ids)}, cluster_ids={len(self.cluster_ids)}"
      )
    if not self.face_widths and n:
      object.__setattr__(self, "face_widths", (0.0,) * n)  # 생략 = 전부 미상 (v1 로드·기존 생성처 호환)
    if len(self.face_widths) != n:
      raise ValueError(f"face_widths 길이가 다릅니다. face_ids={n}, face_widths={len(self.face_widths)}")
    if any(not np.isfinite(width) or width < 0 for width in self.face_widths):
      raise ValueError("face_widths는 0 이상의 유한값이어야 합니다.")
    if not self.bboxes and n:
      object.__setattr__(
        self, "bboxes", ((0.0, 0.0, 0.0, 0.0),) * n
      )  # 생략 = 전부 미상 (v2 이하 로드·기존 생성처 호환)
    if len(self.bboxes) != n:
      raise ValueError(f"bboxes 길이가 다릅니다. face_ids={n}, bboxes={len(self.bboxes)}")
    if any(
      len(bbox) != 4 or any(not np.isfinite(v) for v in bbox) or bbox[2] < 0 or bbox[3] < 0 for bbox in self.bboxes
    ):
      raise ValueError("bboxes는 (x, y, w, h) 유한값이고 w·h는 0 이상이어야 합니다.")
    if not self.s3_keys and n:
      # face_ids/photo_ids와 달리 빈 문자열을 허용한다 — ""는 손상이 아니라 "원본 키 미상"(v2 이하 폴백)의 인코딩
      object.__setattr__(self, "s3_keys", ("",) * n)
    if len(self.s3_keys) != n:
      raise ValueError(f"s3_keys 길이가 다릅니다. face_ids={n}, s3_keys={len(self.s3_keys)}")
    if len(set(self.face_ids)) != n:
      raise ValueError("face_ids에 중복이 있습니다.")
    if any(not face_id for face_id in self.face_ids) or any(not photo_id for photo_id in self.photo_ids):
      # 빈 문자열 id는 .npz의 미배정 인코딩("")·행 매핑을 조용히 오염시킨다 (messages.Id와 동일 정책)
      raise ValueError("face_ids/photo_ids에 빈 문자열이 있습니다.")
    if any(cid == _NO_CLUSTER for cid in self.cluster_ids):
      # 미배정은 None으로만 표현한다 — ""가 섞이면 직렬화 왕복에서 None과 구분이 사라진다
      raise ValueError('cluster_ids의 미배정은 None이어야 합니다 (빈 문자열 "" 금지).')
    if self.embeddings.size:
      if not np.isfinite(self.embeddings).all():
        raise ValueError("embeddings에 비유한값(NaN/inf)이 있습니다.")
      norms = np.linalg.norm(self.embeddings, axis=1)
      if not np.allclose(norms, 1.0, atol=1e-3):
        # recluster와 동일한 단위벡터 계약을 저장 경계에서 먼저 강제해 손상을 조기에 표면화한다
        raise ValueError("embeddings는 L2 정규화 단위벡터여야 합니다.")
    known = set(self.face_ids)
    for kind, pairs in (("must-link", self.must_link_pairs), ("cannot-link", self.cannot_link_pairs)):
      for a, b in pairs:
        if a == b:
          raise ValueError(f"{kind} 쌍은 서로 다른 얼굴이어야 합니다. 받은 쌍: ({a}, {b})")
        if a not in known or b not in known:
          raise ValueError(f"{kind} 쌍이 존재하지 않는 face_id를 참조합니다. 받은 쌍: ({a}, {b})")
    self.embeddings.flags.writeable = False  # frozen dataclass 내용이 하류에서 변형되지 않도록 보호

  @classmethod
  def empty(cls) -> "EventEmbeddings":
    """행이 없는 event — 최초 분류(.npz 부재)와 전체 삭제 후 상태."""
    return cls(
      embeddings=np.empty((0, EMBED_DIM), dtype=np.float32),
      face_ids=(),
      photo_ids=(),
      cluster_ids=(),
      must_link_pairs=(),
      cannot_link_pairs=(),
    )

  def append_faces(
    self,
    rows: Sequence[
      tuple[str, str, np.ndarray]
      | tuple[str, str, np.ndarray, float]
      | tuple[str, str, np.ndarray, float, tuple[float, float, float, float]]
      | tuple[str, str, np.ndarray, float, tuple[float, float, float, float], str]
    ],
  ) -> "EventEmbeddings":
    """(face_id, photo_id, 임베딩[, 얼굴 폭 px[, bbox[, s3_key]]]) 행들을 cluster_id=None(신규)으로 뒤에 추가한 새 인스턴스를 만든다.

    photo_id 기준 멱등 스킵(이미 저장된 사진 제외)은 호출자(classify 핸들러)의 책임이다 —
    이 타입은 요청 메시지를 모르고 행 단위 정합성만 지킨다. 폭 생략 = 0(미상, 주 인물 취급),
    bbox·s3_key 생략 = 미상(썸네일 대표 후보 자격 없음).
    """
    if not rows:
      return self
    appended = np.stack([row[2] for row in rows]).astype(np.float32, copy=False)
    return replace(
      self,
      embeddings=np.vstack([self.embeddings, appended]),
      face_ids=self.face_ids + tuple(row[0] for row in rows),
      photo_ids=self.photo_ids + tuple(row[1] for row in rows),
      cluster_ids=self.cluster_ids + (None,) * len(rows),
      face_widths=self.face_widths + tuple(float(row[3]) if len(row) > 3 else 0.0 for row in rows),
      bboxes=self.bboxes
      + tuple(tuple(float(v) for v in row[4]) if len(row) > 4 else (0.0, 0.0, 0.0, 0.0) for row in rows),
      s3_keys=self.s3_keys + tuple(str(row[5]) if len(row) > 5 else "" for row in rows),
    )

  def masked_by_photo_ids(self, deleted: Collection[str]) -> "EventEmbeddings":
    """삭제된 photo_id의 행을 물리 제거한 새 인스턴스를 만든다 (ADR-007 삭제 마스킹, 복원 없음).

    제거된 face_id를 참조하는 제약 쌍도 함께 프루닝한다 — 남기면 로드 불변식(참조 존재)이 깨진다.
    """
    deleted_set = set(deleted)
    keep = ~np.isin(_as_str_array(self.photo_ids), _as_str_array(sorted(deleted_set)))
    if bool(keep.all()):
      return self
    kept_faces = {face_id for face_id, kept in zip(self.face_ids, keep) if kept}
    return replace(
      self,
      embeddings=self.embeddings[keep],
      face_ids=tuple(face_id for face_id, kept in zip(self.face_ids, keep) if kept),
      photo_ids=tuple(photo_id for photo_id, kept in zip(self.photo_ids, keep) if kept),
      cluster_ids=tuple(cluster_id for cluster_id, kept in zip(self.cluster_ids, keep) if kept),
      face_widths=tuple(width for width, kept in zip(self.face_widths, keep) if kept),
      bboxes=tuple(bbox for bbox, kept in zip(self.bboxes, keep) if kept),
      s3_keys=tuple(key for key, kept in zip(self.s3_keys, keep) if kept),
      must_link_pairs=tuple(pair for pair in self.must_link_pairs if pair[0] in kept_faces and pair[1] in kept_faces),
      cannot_link_pairs=tuple(
        pair for pair in self.cannot_link_pairs if pair[0] in kept_faces and pair[1] in kept_faces
      ),
    )

  def with_cluster_ids(self, cluster_ids: Sequence[str | None]) -> "EventEmbeddings":
    """재군집 결과의 새 배정을 반영한 새 인스턴스를 만든다 (길이 검증은 __post_init__)."""
    return replace(self, cluster_ids=tuple(cluster_ids))

  def with_constraints(
    self,
    must_link_pairs: Sequence[tuple[str, str]],
    cannot_link_pairs: Sequence[tuple[str, str]],
  ) -> "EventEmbeddings":
    """제약 셋 전체를 교체한 새 인스턴스를 만든다 (later-wins 조정 결과 반영용)."""
    return replace(
      self,
      must_link_pairs=tuple(must_link_pairs),
      cannot_link_pairs=tuple(cannot_link_pairs),
    )

  def row_index_of(self) -> dict[str, int]:
    """face_id → 행 인덱스 매핑 — 제약 쌍을 recluster의 인덱스 쌍으로 번역할 때 쓴다."""
    return {face_id: index for index, face_id in enumerate(self.face_ids)}

  def to_npz_bytes(self) -> bytes:
    """ADR-007 레이아웃의 .npz 바이트로 직렬화한다 (S3 put_object의 Body)."""
    buffer = io.BytesIO()
    np.savez_compressed(
      buffer,
      schema_version=np.int64(SCHEMA_VERSION),
      embeddings=self.embeddings,
      face_ids=_as_str_array(self.face_ids),
      photo_ids=_as_str_array(self.photo_ids),
      cluster_ids=_as_str_array([cid if cid is not None else _NO_CLUSTER for cid in self.cluster_ids]),
      must_link_pairs=_as_str_array(self.must_link_pairs, columns=2),
      cannot_link_pairs=_as_str_array(self.cannot_link_pairs, columns=2),
      face_widths=np.asarray(self.face_widths, dtype=np.float32),
      bboxes=np.asarray(self.bboxes, dtype=np.float32).reshape(-1, 4),  # 빈 입력도 shape (0, 4) 보장
      s3_keys=_as_str_array(self.s3_keys),
    )
    return buffer.getvalue()

  @classmethod
  def from_npz_bytes(cls, blob: bytes) -> "EventEmbeddings":
    """저장된 .npz 바이트를 복원한다. 계약 위반은 전부 StoreCorruptionError로 수렴한다.

    allow_pickle=False는 손상/악성 파일의 임의 코드 실행을 차단한다 (ADR-007).
    """
    try:
      with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        version = int(archive["schema_version"])
        if version not in (1, 2, SCHEMA_VERSION):
          raise StoreCorruptionError(
            f"지원하지 않는 schema_version입니다. 받은 값: {version}, 지원: 1~{SCHEMA_VERSION}"
          )
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        ids = {key: tuple(str(value) for value in archive[key]) for key in _ID_KEYS}
        n = len(ids["face_ids"])
        # v1에는 face_widths가 없다 — 폭 미상(0)으로 채우면 주 인물 판정이 전원 통과해 종전 동작과 같다
        face_widths = tuple(float(width) for width in archive["face_widths"]) if version >= 2 else (0.0,) * n
        # v2 이하에는 bboxes·s3_keys가 없다 — 미상 폴백 행은 썸네일 대표 후보에서 빠질 뿐 나머지 동작 동일
        if version >= 3:
          bbox_array = archive["bboxes"]
          if bbox_array.ndim != 2 or bbox_array.shape[1] != 4:
            raise StoreCorruptionError(f"bboxes는 shape (N, 4)여야 합니다. 받은 shape: {bbox_array.shape}")
          bboxes = tuple(tuple(float(v) for v in row) for row in bbox_array)
          s3_keys = tuple(str(key) for key in archive["s3_keys"])
        else:
          bboxes = ((0.0, 0.0, 0.0, 0.0),) * n
          s3_keys = ("",) * n
        pairs: dict[str, tuple[tuple[str, str], ...]] = {}
        for key in _PAIR_KEYS:
          array = archive[key]
          if array.ndim != 2 or array.shape[1] != 2:
            raise StoreCorruptionError(f"{key}는 shape (K, 2)여야 합니다. 받은 shape: {array.shape}")
          pairs[key] = tuple((str(a), str(b)) for a, b in array)
    except StoreCorruptionError:
      raise
    except (KeyError, ValueError, OSError, zipfile.BadZipFile) as exc:
      # np.load의 포맷 오류·키 누락 등 — 원인 유형과 무관하게 "손상"이라는 단일 계약으로 노출한다
      raise StoreCorruptionError(f".npz를 해석할 수 없습니다: {exc}") from exc
    try:
      return cls(
        embeddings=embeddings,
        face_ids=ids["face_ids"],
        photo_ids=ids["photo_ids"],
        cluster_ids=tuple(cid if cid != _NO_CLUSTER else None for cid in ids["cluster_ids"]),
        must_link_pairs=pairs["must_link_pairs"],
        cannot_link_pairs=pairs["cannot_link_pairs"],
        face_widths=face_widths,
        bboxes=bboxes,
        s3_keys=s3_keys,
      )
    except ValueError as exc:  # __post_init__ 불변식 위반 (길이 정합·face_id 유일·제약 참조 존재 등)
      raise StoreCorruptionError(f".npz 불변식 위반: {exc}") from exc


if __name__ == "__main__":
  # S3 없이 코덱·불변식 계약을 자가 검증한다 (messages.py __main__과 같은 실행형 확인 패턴).
  # TODO(CHMO-165): pytest 도입 시 tests/storage/test_event_embeddings.py로 승격
  passed = 0

  def check(name: str, condition: bool) -> None:
    global passed
    if not condition:
      raise SystemExit(f"실패: {name}")
    passed += 1
    print(f"통과: {name}")

  def unit(axis: int) -> np.ndarray:
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[axis] = 1.0
    return vector

  empty = EventEmbeddings.empty()
  check("빈 event 직렬화 왕복", EventEmbeddings.from_npz_bytes(empty.to_npz_bytes()) == empty)

  event = empty.append_faces(
    [
      ("face-1", "img-1", unit(0), 120.0, (10.0, 20.0, 120.0, 150.0), "photos/img-1.jpg"),
      ("face-2", "img-1", unit(1), 40.0, (300.0, 5.0, 40.0, 50.0)),  # 한 image_id에 얼굴 여러 개 (N:M), s3_key 생략
      ("face-3", "img-2", unit(2)),  # 폭·bbox·s3_key 생략 = 미상
    ]
  )
  check("append_faces 폭 기록 (생략 = 0)", event.face_widths == (120.0, 40.0, 0.0))
  check(
    "append_faces bbox·s3_key 기록 (생략 = 미상)",
    event.bboxes == ((10.0, 20.0, 120.0, 150.0), (300.0, 5.0, 40.0, 50.0), (0.0, 0.0, 0.0, 0.0))
    and event.s3_keys == ("photos/img-1.jpg", "", ""),
  )
  event = event.with_cluster_ids(["person-A", None, "person-B"])
  event = event.with_constraints([("face-1", "face-3")], [("face-2", "face-3")])
  restored = EventEmbeddings.from_npz_bytes(event.to_npz_bytes())
  check("직렬화 왕복 (id·제약 보존)", restored == event)
  check(
    '미배정 None ↔ "" 인코딩 왕복',
    restored.cluster_ids == ("person-A", None, "person-B"),
  )
  check("행 인덱스 매핑", event.row_index_of() == {"face-1": 0, "face-2": 1, "face-3": 2})
  check("빈 append는 동일 인스턴스", event.append_faces([]) is event)

  masked = event.masked_by_photo_ids(["img-1", "img-1"])  # 중복 삭제 id 멱등
  check(
    "삭제 마스킹 + 댕글링 제약 프루닝",
    masked.face_ids == ("face-3",)
    and masked.photo_ids == ("img-2",)
    and masked.cluster_ids == ("person-B",)
    and masked.must_link_pairs == ()
    and masked.cannot_link_pairs == (),
  )
  check("삭제 마스킹이 폭도 함께 마스킹", masked.face_widths == (0.0,))
  check(
    "삭제 마스킹이 bbox·s3_key도 함께 마스킹",
    masked.bboxes == ((0.0, 0.0, 0.0, 0.0),) and masked.s3_keys == ("",),
  )
  check("삭제 대상 없음이면 동일 인스턴스", event.masked_by_photo_ids(["img-없음"]) is event)
  check("전체 삭제 → 빈 event", event.masked_by_photo_ids(["img-1", "img-2"]) == empty)

  # v1 하위호환: face_widths 열이 없는 구버전 .npz는 폭 0(미상)으로 로드된다 — 주 인물 판정이
  # 전원 통과해 종전(전체 얼굴 수) 라우팅과 동일해지는 안전 폴백 (SCHEMA_VERSION 주석).
  v1_buffer = io.BytesIO()
  np.savez_compressed(
    v1_buffer,
    schema_version=np.int64(1),
    embeddings=event.embeddings,
    face_ids=_as_str_array(event.face_ids),
    photo_ids=_as_str_array(event.photo_ids),
    cluster_ids=_as_str_array([cid if cid is not None else _NO_CLUSTER for cid in event.cluster_ids]),
    must_link_pairs=_as_str_array(event.must_link_pairs, columns=2),
    cannot_link_pairs=_as_str_array(event.cannot_link_pairs, columns=2),
  )
  from_v1 = EventEmbeddings.from_npz_bytes(v1_buffer.getvalue())
  check(
    "v1 .npz 로드: 폭 미상(0) 폴백 + 나머지 필드 보존",
    from_v1.face_widths == (0.0, 0.0, 0.0) and from_v1.face_ids == event.face_ids,
  )

  # v2 하위호환: bboxes·s3_keys 열이 없는 .npz는 미상(전부 0·"")으로 로드된다 — 해당 행은 썸네일
  # 대표 후보 자격이 없을 뿐 폭·라우팅 동작은 보존된다 (SCHEMA_VERSION 주석).
  v2_buffer = io.BytesIO()
  np.savez_compressed(
    v2_buffer,
    schema_version=np.int64(2),
    embeddings=event.embeddings,
    face_ids=_as_str_array(event.face_ids),
    photo_ids=_as_str_array(event.photo_ids),
    cluster_ids=_as_str_array([cid if cid is not None else _NO_CLUSTER for cid in event.cluster_ids]),
    must_link_pairs=_as_str_array(event.must_link_pairs, columns=2),
    cannot_link_pairs=_as_str_array(event.cannot_link_pairs, columns=2),
    face_widths=np.asarray(event.face_widths, dtype=np.float32),
  )
  from_v2 = EventEmbeddings.from_npz_bytes(v2_buffer.getvalue())
  check(
    "v2 .npz 로드: bbox·s3_key 미상 폴백 + 폭 보존",
    from_v2.bboxes == ((0.0, 0.0, 0.0, 0.0),) * 3
    and from_v2.s3_keys == ("", "", "")
    and from_v2.face_widths == event.face_widths,
  )

  invalid_cases = [
    (
      "photo_ids 길이 불일치",
      dict(
        embeddings=np.stack([unit(0)]),
        face_ids=("face-1",),
        photo_ids=(),
        cluster_ids=(None,),
        must_link_pairs=(),
        cannot_link_pairs=(),
      ),
    ),
    (
      "face_id 중복",
      dict(
        embeddings=np.stack([unit(0), unit(1)]),
        face_ids=("face-1", "face-1"),
        photo_ids=("img-1", "img-2"),
        cluster_ids=(None, None),
        must_link_pairs=(),
        cannot_link_pairs=(),
      ),
    ),
    (
      "존재하지 않는 face_id 참조 제약",
      dict(
        embeddings=np.stack([unit(0)]),
        face_ids=("face-1",),
        photo_ids=("img-1",),
        cluster_ids=(None,),
        must_link_pairs=(("face-1", "face-유령"),),
        cannot_link_pairs=(),
      ),
    ),
    (
      "자기 자신 제약 쌍",
      dict(
        embeddings=np.stack([unit(0)]),
        face_ids=("face-1",),
        photo_ids=("img-1",),
        cluster_ids=(None,),
        must_link_pairs=(),
        cannot_link_pairs=(("face-1", "face-1"),),
      ),
    ),
    (
      '미배정 cluster_id "" 직접 주입',
      dict(
        embeddings=np.stack([unit(0)]),
        face_ids=("face-1",),
        photo_ids=("img-1",),
        cluster_ids=("",),
        must_link_pairs=(),
        cannot_link_pairs=(),
      ),
    ),
    (
      "비정규 임베딩",
      dict(
        embeddings=np.stack([unit(0) * 2.0]),
        face_ids=("face-1",),
        photo_ids=("img-1",),
        cluster_ids=(None,),
        must_link_pairs=(),
        cannot_link_pairs=(),
      ),
    ),
    (
      "bboxes 길이 불일치",
      dict(
        embeddings=np.stack([unit(0)]),
        face_ids=("face-1",),
        photo_ids=("img-1",),
        cluster_ids=(None,),
        must_link_pairs=(),
        cannot_link_pairs=(),
        bboxes=((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 5.0, 5.0)),
      ),
    ),
    (
      "bbox NaN",
      dict(
        embeddings=np.stack([unit(0)]),
        face_ids=("face-1",),
        photo_ids=("img-1",),
        cluster_ids=(None,),
        must_link_pairs=(),
        cannot_link_pairs=(),
        bboxes=((float("nan"), 0.0, 10.0, 10.0),),
      ),
    ),
  ]
  for case_name, kwargs in invalid_cases:
    try:
      EventEmbeddings(**kwargs)
    except ValueError:
      check(f"거부: {case_name}", True)
    else:
      raise SystemExit(f"실패: {case_name} — ValueError가 발생해야 하는데 통과됨")

  corrupt_buffer = io.BytesIO()
  np.savez_compressed(corrupt_buffer, schema_version=np.int64(99))
  for corrupt_name, corrupt_blob in [
    ("미지 schema_version", corrupt_buffer.getvalue()),
    ("zip 아님", b"\x00\x01\x02"),
  ]:
    try:
      EventEmbeddings.from_npz_bytes(corrupt_blob)
    except StoreCorruptionError:
      check(f"거부: {corrupt_name}", True)
    else:
      raise SystemExit(f"실패: {corrupt_name} — StoreCorruptionError가 발생해야 하는데 통과됨")

  print(f"\n스모크 검증 {passed}건 전부 통과")
