"""인물 앨범 대표 얼굴 썸네일 렌더 — bbox crop → 다운스케일 → JPEG 인코딩 (CHMO-335).

cv2 + numpy만 아는 순수 함수 모듈이다. 대표 얼굴 선정·S3 저장·best-effort 격리는 handlers/
storage의 책임 — 여기서는 렌더 실패를 ValueError로 던지고 삼키지 않는다.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ThumbnailConfig:
  """렌더 파라미터 — 전부 Settings(.env)에서 주입된다 (core.config.to_thumbnail_config)."""

  bbox_scale: float = 1.4  # 얼굴 bbox를 중심 기준 이 배율로 확장해 crop — 딱 맞는 crop은 답답해 여백을 준다
  max_side: int = 256  # 결과 긴 변 상한 px — 초과 시 다운스케일만 한다 (작은 얼굴 업스케일은 품질만 해침)
  jpeg_quality: int = 85

  def __post_init__(self) -> None:
    if self.bbox_scale <= 0:
      raise ValueError(f"bbox_scale은 양수여야 합니다. 받은 값: {self.bbox_scale}")
    if self.max_side <= 0:
      raise ValueError(f"max_side는 양수여야 합니다. 받은 값: {self.max_side}")
    if not 1 <= self.jpeg_quality <= 100:
      raise ValueError(f"jpeg_quality는 1~100이어야 합니다. 받은 값: {self.jpeg_quality}")


def render_face_thumbnail(
  image: np.ndarray,
  bbox: tuple[float, float, float, float],
  config: ThumbnailConfig | None = None,
) -> bytes:
  """BGR 원본에서 얼굴 bbox(x, y, w, h) 주변을 잘라 JPEG 바이트로 만든다.

  bbox가 이미지 경계를 벗어나면 경계로 클램프한다 — 회복 검출 경로(ADR-017)의 파편 bbox가
  경계에 걸칠 수 있다. 클램프 후 유효 영역이 없으면(완전 이탈) ValueError.
  """
  config = config if config is not None else ThumbnailConfig()
  x, y, w, h = bbox
  if w <= 0 or h <= 0:
    raise ValueError(f"bbox 폭·높이는 양수여야 합니다. 받은 bbox: {bbox}")
  height, width = image.shape[:2]
  # 중심 고정 확장 — 정사각형화는 하지 않는다 (원본 비율 유지, 표시 crop은 앱/백엔드 몫)
  cx, cy = x + w / 2.0, y + h / 2.0
  half_w, half_h = w * config.bbox_scale / 2.0, h * config.bbox_scale / 2.0
  left = max(0, int(round(cx - half_w)))
  top = max(0, int(round(cy - half_h)))
  right = min(width, int(round(cx + half_w)))
  bottom = min(height, int(round(cy + half_h)))
  if right <= left or bottom <= top:
    raise ValueError(f"bbox가 이미지 밖입니다. bbox: {bbox}, 이미지: {width}x{height}")
  crop = image[top:bottom, left:right]

  long_side = max(crop.shape[:2])
  if long_side > config.max_side:
    ratio = config.max_side / long_side
    new_size = (max(1, round(crop.shape[1] * ratio)), max(1, round(crop.shape[0] * ratio)))
    crop = cv2.resize(crop, new_size, interpolation=cv2.INTER_AREA)  # 축소는 AA 내장 보간 (align.py의 AA와 같은 취지)

  ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
  if not ok:
    raise ValueError(f"JPEG 인코딩에 실패했습니다. crop shape: {crop.shape}")
  return encoded.tobytes()


if __name__ == "__main__":
  # cv2·numpy만으로 렌더 계약을 자가 검증한다 (event_embeddings __main__과 같은 실행형 확인 패턴).
  # TODO(CHMO-165): pytest 도입 시 tests/pipeline/test_thumbnail.py로 승격
  passed = 0

  def check(name: str, condition: bool) -> None:
    global passed
    if not condition:
      raise SystemExit(f"실패: {name}")
    passed += 1
    print(f"통과: {name}")

  # 수평 그라디언트 합성 이미지 — 디코딩 결과에서 crop 위치를 픽셀 값으로 역추적할 수 있다
  gradient = np.zeros((800, 1200, 3), dtype=np.uint8)
  gradient[:, :, 0] = np.linspace(0, 255, 1200, dtype=np.uint8)[np.newaxis, :]

  jpeg = render_face_thumbnail(gradient, (500.0, 300.0, 200.0, 200.0))
  decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
  check(
    "정상 crop: JPEG 디코드 왕복 + 긴 변 ≤ max_side + 확장 비율 유지(정사각 bbox → 정사각 crop)",
    decoded is not None and max(decoded.shape[:2]) == 256 and decoded.shape[0] == decoded.shape[1],
  )
  # bbox 200×200 × 1.4 = 280×280 crop이 256으로 축소 — 중심(600px)의 그라디언트 값이 보존되는지
  center_value = float(decoded[decoded.shape[0] // 2, decoded.shape[1] // 2, 0])
  check("정상 crop: 중심 픽셀이 bbox 중심 근방 값", abs(center_value - (600 / 1200) * 255) < 15)

  small = render_face_thumbnail(gradient, (100.0, 100.0, 60.0, 80.0))
  small_decoded = cv2.imdecode(np.frombuffer(small, dtype=np.uint8), cv2.IMREAD_COLOR)
  check(
    "작은 얼굴: 업스케일 안 함 (crop 원 크기 유지)",
    small_decoded is not None and small_decoded.shape[:2] == (112, 84),  # 80·60 × 1.4
  )

  edge = render_face_thumbnail(gradient, (-50.0, -50.0, 200.0, 200.0))
  edge_decoded = cv2.imdecode(np.frombuffer(edge, dtype=np.uint8), cv2.IMREAD_COLOR)
  check("경계 밖 bbox: 이미지 안쪽으로 클램프", edge_decoded is not None and edge_decoded.size > 0)

  for case_name, bad_bbox in [
    ("완전 이탈 bbox", (5000.0, 5000.0, 100.0, 100.0)),
    ("퇴화 bbox (w=0)", (10.0, 10.0, 0.0, 50.0)),
  ]:
    try:
      render_face_thumbnail(gradient, bad_bbox)
    except ValueError:
      check(f"거부: {case_name}", True)
    else:
      raise SystemExit(f"실패: {case_name} — ValueError가 발생해야 하는데 통과됨")

  tight = cv2.imdecode(
    np.frombuffer(
      render_face_thumbnail(gradient, (500.0, 300.0, 100.0, 100.0), ThumbnailConfig(bbox_scale=1.0)), np.uint8
    ),
    cv2.IMREAD_COLOR,
  )
  check("배율 1.0: bbox 크기 그대로 crop", tight.shape[:2] == (100, 100))

  for config_case, kwargs in [
    ("bbox_scale 0", {"bbox_scale": 0.0}),
    ("max_side 0", {"max_side": 0}),
    ("jpeg_quality 0", {"jpeg_quality": 0}),
  ]:
    try:
      ThumbnailConfig(**kwargs)
    except ValueError:
      check(f"거부: {config_case}", True)
    else:
      raise SystemExit(f"실패: {config_case} — ValueError가 발생해야 하는데 통과됨")

  print(f"\n스모크 검증 {passed}건 전부 통과")
