"""순수 파이프라인 단계로서의 얼굴 정렬 (Umeyama 유사변환 직접 구현).

YuNet이 검출한 5점 랜드마크를 ArcFace 표준 기준점에 맞춰 112x112 크롭으로 정렬한다.
입력·출력 모두 BGR이며 이 모듈은 색공간 변환을 하지 않는다 — RGB 변환은 embed 전처리의 책임이다.
`insightface.utils.face_align` 대비 변환행렬 np.allclose=True, 픽셀 차이 0으로 동등성이 검증된 이식 구현이다.
"""

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


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
  """5개 랜드마크를 ArcFace 기준점에 맞춰 ALIGN_SIZE x ALIGN_SIZE BGR 크롭으로 정렬한다.

  변환행렬이 퇴화(rank 0 → NaN)하면 None을 반환한다 — 얼굴 단위 실패는 해당 얼굴만 건너뛰는
  파이프라인 정책으로, detect.py의 None 반환 패턴과 일관된다.
  """
  if landmarks.shape != (_NUM_LANDMARKS, 2):
    raise ValueError(f"landmarks는 shape (5, 2)여야 합니다. 받은 shape: {landmarks.shape}")
  image = _ensure_bgr(image)
  M = _umeyama(landmarks.astype(np.float64), _ARCFACE_DST)[0:2, :]  # shape (2, 3) 아핀 행렬
  if not np.isfinite(M).all():
    return None
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
