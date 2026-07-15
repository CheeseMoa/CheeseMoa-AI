"""순수 파이프라인 단계로서의 얼굴 정렬 (Umeyama 유사변환 직접 구현).

YuNet이 검출한 5점 랜드마크를 ArcFace 표준 기준점에 맞춰 112x112 크롭으로 정렬한다.
입력·출력 모두 BGR이며 이 모듈은 색공간 변환을 하지 않는다 — RGB 변환은 embed 전처리의 책임이다.
`insightface.utils.face_align` 대비 변환행렬 np.allclose=True, 픽셀 차이 0으로 동등성이 검증된 이식 구현이다
(픽셀 동등성은 antialias=False 또는 확대(s ≥ 1) 경로 기준 — 축소 경로는 안티에일리어싱 프리블러가
의도적으로 추가된다, 2026-07-14 리뷰).
"""

import math

import cv2
import numpy as np

ALIGN_SIZE = 112  # _ARCFACE_DST가 112x112 전용 좌표이므로 크기는 고정하고 파라미터로 노출하지 않는다
_NUM_LANDMARKS = 5

# ArcFace 표준 정렬 기준점 (112x112). dtype은 float32를 유지한다: float64로 선언하면
# float32→float64 승격 값과 하위 비트가 달라져 insightface 파리티(픽셀 diff 0)가 깨질 수 있다.
# 내부 연산은 _umeyama에서 float64로 승격되므로 정밀도 손실은 없다.
_ARCFACE_DST = np.array(
  [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
  dtype=np.float32,
)  # shape (5, 2)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
  """cv2.warpAffine 입력 계약(C-연속 메모리의 3채널 BGR)에 맞게 정규화한다.

  detect._to_contiguous_bgr와 동일 계약이지만 import하지 않고 로컬로 구현한다: detect를 import하면
  model_source → huggingface_hub 임포트 체인이 순수 수학 모듈에 유입되기 때문이다.
  """
  if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # (H,W) / (H,W,1) 그레이스케일 → BGR
  elif image.ndim == 3 and image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # 알파 채널 제거
  return np.ascontiguousarray(image)  # 3채널 통과 경로 포함, 비연속 뷰(RGB→BGR 슬라이스 등)를 연속 버퍼로 확정


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
  """Umeyama 유사변환(scale 포함) 추정 — skimage SimilarityTransform.estimate 와 동일.

  src 점집합을 dst로 옮기는 (3, 3) float64 동차 변환행렬을 반환한다. rank 0 퇴화 시 NaN 행렬을 반환한다.
  """
  num, dim = src.shape  # (5, 2)
  # 두 점집합을 각자 중심으로 평행이동한 뒤, 교차 공분산 A의 SVD로 최적 회전을 닫힌형식으로 구한다
  src_mean = src.mean(axis=0)
  dst_mean = dst.mean(axis=0)
  src_demean = src - src_mean
  dst_demean = dst - dst_mean
  A = dst_demean.T @ src_demean / num
  # det(A) < 0이면 최소제곱 해에 반사가 섞이므로, d로 마지막 특이값 부호를 뒤집어 순수 회전을 강제한다
  d = np.ones((dim,), dtype=np.float64)
  if np.linalg.det(A) < 0:
    d[dim - 1] = -1
  T = np.eye(dim + 1, dtype=np.float64)
  U, S, V = np.linalg.svd(A)
  rank = np.linalg.matrix_rank(A)
  if rank == 0:
    return np.full((dim + 1, dim + 1), np.nan)  # 모든 랜드마크가 한 점에 겹친 퇴화 케이스
  elif rank == dim - 1:
    # rank 결손(랜드마크가 한 직선 위) 시 det(U)·det(V) 부호에 따라 반사 보정 적용 여부가 갈린다
    if np.linalg.det(U) * np.linalg.det(V) > 0:
      T[:dim, :dim] = U @ V
    else:
      s = d[dim - 1]
      d[dim - 1] = -1
      T[:dim, :dim] = U @ np.diag(d) @ V
      d[dim - 1] = s
  else:
    T[:dim, :dim] = U @ np.diag(d) @ V
  # 등방 스케일 = (부호 보정된 특이값 합) / (src 분산 합) — Umeyama 논문 (1991) eq. 41-43
  scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
  T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
  T[:dim, :dim] *= scale
  return T


def _prefilter_roi(image: np.ndarray, M: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray] | None:
  """축소 warp의 에일리어싱 방지용으로 얼굴 주변 ROI만 가우시안 프리블러한다.

  warpAffine의 INTER_LINEAR는 2x2 이웃만 샘플링해 큰 축소(s ≪ 1)에서 저역통과 없이 원본을
  서브샘플링한다 — 어느 픽셀이 걸리는지가 서브픽셀 위치에 좌우돼 같은 얼굴도 촬영마다 크롭이
  달라지고 임베딩이 흔들린다(2026-07-14 리뷰: 같은 얼굴 유사도 최저 0.43). σ = (1/s)/2 는 나이퀴스트
  기준 축소 배율의 절반 파장에 해당하는 저역통과다.

  블러를 이미지 전체가 아니라 warp가 실제로 샘플링하는 영역(112x112 목적지의 역사상 + 커널 마진)에만
  적용해, 얼굴이 여러 개인 이미지에서 얼굴마다 전체 이미지를 블러하는 낭비를 피한다. 마진이 커널
  반경을 덮으므로 픽셀 결과는 전체 블러와 동일하다. 원본 image는 절대 수정하지 않는다(호출자가 같은
  버퍼를 다른 얼굴 정렬·품질 판정에 재사용한다).

  반환: (블러된 ROI, ROI 좌표계로 보정한 아핀 행렬). ROI가 이미지 밖으로 퇴화하면 None(블러 생략).
  """
  sigma = (1.0 / s) / 2.0
  # 목적지 네 모서리를 원본 좌표로 역사상해 warp가 읽는 영역을 구한다
  corners = np.array([[0, 0], [ALIGN_SIZE, 0], [0, ALIGN_SIZE], [ALIGN_SIZE, ALIGN_SIZE]], dtype=np.float64)
  inv = cv2.invertAffineTransform(M)
  src_corners = corners @ inv[:, :2].T + inv[:, 2]
  # 마진 = 가우시안 커널 반경(ksize 자동 유도 시 ≈ 4σ) + INTER_LINEAR 2x2 이웃 — 이보다 작으면
  # ROI 경계 안쪽 픽셀이 borderValue로 오염돼 전체 블러와 결과가 달라진다
  margin = int(math.ceil(4.0 * sigma)) + 2
  h, w = image.shape[:2]
  x0 = max(0, int(math.floor(src_corners[:, 0].min())) - margin)
  y0 = max(0, int(math.floor(src_corners[:, 1].min())) - margin)
  x1 = min(w, int(math.ceil(src_corners[:, 0].max())) + margin)
  y1 = min(h, int(math.ceil(src_corners[:, 1].max())) + margin)
  if x1 <= x0 or y1 <= y0:
    return None  # 얼굴이 사실상 화면 밖 — 기존 동작(블러 없는 warp)으로 폴백
  blurred = cv2.GaussianBlur(image[y0:y1, x0:x1], (0, 0), sigma)
  # ROI 크롭은 src 좌표를 (x0, y0)만큼 평행이동한 것 — M을 같은 좌표계로 보정한다
  M_roi = M.copy()
  M_roi[:, 2] += M[:, :2] @ np.array([x0, y0], dtype=np.float64)
  return blurred, M_roi


def align_face(image: np.ndarray, landmarks: np.ndarray, antialias: bool = True) -> np.ndarray | None:
  """5개 랜드마크를 ArcFace 기준점에 맞춰 ALIGN_SIZE x ALIGN_SIZE BGR 크롭으로 정렬한다.

  변환행렬이 퇴화(rank 0 → NaN)하면 None을 반환한다 — 얼굴 단위 실패는 해당 얼굴만 건너뛰는
  파이프라인 정책으로, detect.py의 None 반환 패턴과 일관된다.

  antialias=True(기본)면 축소(s < 1) warp에 한해 배율 기반 가우시안 프리블러를 적용한다.
  확대 경로와 antialias=False는 기존(insightface 픽셀 동등) 경로 그대로다.
  """
  if landmarks.shape != (_NUM_LANDMARKS, 2):
    raise ValueError(f"landmarks는 shape (5, 2)여야 합니다. 받은 shape: {landmarks.shape}")
  image = _ensure_bgr(image)
  M = _umeyama(landmarks.astype(np.float64), _ARCFACE_DST)[0:2, :]  # shape (2, 3) 아핀 행렬
  if not np.isfinite(M).all():
    return None
  if antialias:
    # _umeyama는 d-보정으로 순수 회전을 강제하므로 M[:, :2] = s·R, det = s² — s를 det로 복원한다.
    # det ≤ 0은 스케일 퇴화(랜드마크 일직선 등)라 배율이 정의되지 않으므로 블러를 건너뛴다.
    det = float(np.linalg.det(M[:, :2]))
    if det > 0.0 and (s := math.sqrt(det)) < 1.0:
      roi = _prefilter_roi(image, M, s)
      if roi is not None:
        blurred, M_roi = roi
        return cv2.warpAffine(blurred, M_roi, (ALIGN_SIZE, ALIGN_SIZE), borderValue=0.0)
  return cv2.warpAffine(image, M, (ALIGN_SIZE, ALIGN_SIZE), borderValue=0.0)


if __name__ == "__main__":
  # SQS/S3 없이 PoC 레시피와의 파리티를 확인: 로컬 이미지에서 얼굴을 검출·정렬해 크롭을 저장한다.
  import sys
  import time
  from pathlib import Path

  # detect는 model_source → huggingface_hub 임포트 체인을 끌고 오므로 CLI 확인 블록에서만 지연 import한다
  from app.pipeline.detect import FaceDetector

  detector = FaceDetector()
  for path in sys.argv[1:]:
    image = cv2.imread(path)
    if image is None:
      print(f"{path}: 건너뜀 (이미지를 읽을 수 없음)")
      continue

    start = time.perf_counter()
    detected = detector.detect(image)
    aligned = [align_face(image, face.landmarks) for face in detected]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    stem = Path(path).stem
    saved = 0
    skipped = 0
    for i, crop in enumerate(aligned):
      if crop is None:
        skipped += 1  # 퇴화 랜드마크 얼굴은 저장하지 않고 개수만 집계
        continue
      # Windows에서 cv2.imwrite는 비ASCII(한글) 경로에 예외 없이 False만 반환하므로
      # imencode + tofile로 저장해 성공 여부를 확실히 가른다
      ok, buf = cv2.imencode(".png", crop)
      if ok:
        buf.tofile(f"{stem}_face{i}.png")
        saved += 1
      else:
        skipped += 1
    print(f"{path}: {len(detected)} face(s), 저장 {saved}, 건너뜀 {skipped} in {elapsed_ms:.1f} ms")
