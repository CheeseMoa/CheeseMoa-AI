"""순수 파이프라인 단계로서의 AuraFace 얼굴 임베딩.

align이 생산한 112x112 BGR uint8 크롭을 받아 L2 정규화된 512차원 float32 벡터로 변환한다.
BGR→RGB 색공간 변환은 이 모듈의 책임이다 (align.py는 색공간을 바꾸지 않는 계약).
모델 로딩(다운로드 포함)은 `FaceEmbedder` 생성 시 1회만 일어난다 — 워커는 부트스트랩에서
임베더를 생성해 모델을 메모리에 적재 완료한 뒤에 SQS 폴링을 시작한다.
PoC(face-detection-PoC)의 검증 레시피(BGR→RGB, NCHW, (x-127.5)/127.5, L2 정규화)를 수치 동일하게 이식했다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from app.core.model_source import ModelSource, default_auraface_source
from app.pipeline.align import ALIGN_SIZE

EMBED_DIM = 512
DEFAULT_INTRA_OP_THREADS = 8  # PoC가 배포 타깃(8vCPU)에서 검증한 값. 0이면 ORT 자동 감지
DEFAULT_INTER_OP_THREADS = 8
_PIXEL_CENTER = 127.5  # uint8 [0,255] → [-1,1] 정규화의 중심이자 스케일 (PoC 레시피와 동일)


@dataclass(frozen=True)
class EmbedConfig:
  """`FaceEmbedder`의 튜닝 파라미터."""

  model_source: ModelSource | None = None
  # 워커가 스레드 병렬화를 도입하면 intra 8은 오버서브스크립션이 된다 — 그때는 워커 부트스트랩에서
  # 코어 수에 맞춘 config를 주입해 오버라이드한다 (detect.py의 cv2.setNumThreads 역할 분담과 동일).
  intra_op_num_threads: int = DEFAULT_INTRA_OP_THREADS
  inter_op_num_threads: int = DEFAULT_INTER_OP_THREADS

  def __post_init__(self) -> None:
    # 0은 "ORT 자동 감지"라는 유효한 의미가 있지만 음수는 ORT가 환경에 따라 모호하게 처리하므로
    # 생성 시점에 거부한다 (DetectorConfig.max_side 검증과 같은 정책).
    if self.intra_op_num_threads < 0 or self.inter_op_num_threads < 0:
      raise ValueError(
        f"스레드 수는 0(자동) 이상이어야 합니다. 받은 값: "
        f"intra={self.intra_op_num_threads}, inter={self.inter_op_num_threads}"
      )


def _preprocess(aligned: np.ndarray) -> np.ndarray:
  """정렬 크롭 1장을 (3, 112, 112) float32 [-1, 1] 블롭으로 변환한다.

  align_face는 항상 112x112x3 uint8 BGR을 생산하므로 다른 입력은 호출자의 프로그래밍 오류다.
  특히 float 이미지는 (x-127.5)/127.5 가 예외 없이 쓰레기 임베딩을 만들어 클러스터링 전체를
  오염시키므로, 변환해 주지 않고 거부한다 (detect.py의 dtype 무변환 원칙과 동일 철학).
  """
  if aligned.shape != (ALIGN_SIZE, ALIGN_SIZE, 3):
    raise ValueError(f"aligned는 shape ({ALIGN_SIZE}, {ALIGN_SIZE}, 3)이어야 합니다. 받은 shape: {aligned.shape}")
  if aligned.dtype != np.uint8:
    raise ValueError(f"aligned는 dtype uint8이어야 합니다. 받은 dtype: {aligned.dtype}")
  rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)  # 모델은 RGB 학습 — BGR 유지 계약은 여기서 끝난다
  blob = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
  return (blob - _PIXEL_CENTER) / _PIXEL_CENTER


def _l2_normalize(raw: np.ndarray) -> np.ndarray | None:
  """원시 임베딩을 단위벡터로 정규화한다. 퇴화(비유한값·영벡터) 시 None을 반환한다.

  PoC는 norm==0일 때 raw를 그대로 반환하지만, 영벡터·NaN 벡터는 코사인 거리가 정의되지 않아
  하류(HDBSCAN cosine, pgvector, L2 정규화 평균 대표벡터)를 깨뜨린다. 얼굴 단위 None 스킵이
  파이프라인 정책(align의 퇴화 None)과 정합하므로 의도적으로 이탈한다.
  """
  if not np.isfinite(raw).all():
    return None
  norm = float(np.linalg.norm(raw))
  if norm == 0.0:
    return None
  return (raw / norm).astype(np.float32)


class FaceEmbedder:
  """AuraFace(glintr100.onnx) 임베더.

  `ort.InferenceSession.run`은 스레드 안전하므로 `FaceDetector`(스레드당 1개 필요)와 달리
  인스턴스 하나를 여러 스레드가 공유해도 된다. 모델 파일 획득·세션 생성은 생성자에서 1회만 수행한다.
  """

  def __init__(self, config: EmbedConfig | None = None) -> None:
    resolved_config = config or EmbedConfig()
    source = resolved_config.model_source or default_auraface_source()
    model_path = source.resolve()  # 생성 시 모델을 1회만 로딩
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = resolved_config.intra_op_num_threads
    sess_opts.inter_op_num_threads = resolved_config.inter_op_num_threads
    # glintr100.onnx는 입력 배치 축은 동적('None')인데 출력 메타데이터는 {1, 512}로 고정 선언되어 있어,
    # N>1 배치 추론마다 ORT가 무해한 shape 경고를 찍는다 (배치 결과가 단건과 동일함은
    # 동등성 검증 스크립트로 확인). 메시지마다 로그가 오염되므로 ERROR 미만 로그를 끈다.
    sess_opts.log_severity_level = 3
    # GPU 빌드가 섞여 설치돼도 PoC 검증 환경(CPU)과 동일하게 동작하도록 provider를 명시 고정한다
    self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    model_input = self._session.get_inputs()[0]
    model_output = self._session.get_outputs()[0]
    self._input_name = model_input.name
    self._output_name = model_output.name

    # 잘못된 모델 파일 주입(예: AURAFACE_MODEL_PATH 오설정)을 첫 추론이 아니라 로딩 시점에 잡는다
    output_dim = model_output.shape[-1]
    if isinstance(output_dim, int) and output_dim != EMBED_DIM:
      raise ValueError(f"모델 출력 차원이 {EMBED_DIM}이 아닙니다. 받은 값: {output_dim} (경로: {model_path})")

    # 배치 축이 정수(고정 크기)가 아니라 심볼릭('None', 'batch' 등)이면 동적 배치를 지원한다.
    # 동적이면 N개 크롭을 (N,3,112,112) 블롭 1회 추론으로 처리하고, 고정이면 단건 루프로 폴백한다.
    batch_axis = model_input.shape[0]
    self._supports_batch = not isinstance(batch_axis, int)

  def embed(self, aligned: np.ndarray) -> np.ndarray | None:
    """정렬 크롭 1장을 L2 정규화된 (512,) float32 벡터로 변환한다. 퇴화 시 None."""
    return self.embed_batch([aligned])[0]

  def embed_batch(self, aligned_faces: Sequence[np.ndarray]) -> list[np.ndarray | None]:
    """정렬 크롭 여러 장을 입력 순서를 보존해 임베딩한다. 퇴화 얼굴만 해당 슬롯이 None이 된다."""
    if not aligned_faces:
      return []
    blobs = [_preprocess(face) for face in aligned_faces]  # 계약 위반은 추론 전에 ValueError로 전부 거른다
    if self._supports_batch:
      raw_batch = self._session.run([self._output_name], {self._input_name: np.stack(blobs)})[0]
    else:
      raw_batch = np.concatenate(
        [self._session.run([self._output_name], {self._input_name: blob[np.newaxis]})[0] for blob in blobs]
      )
    return [_l2_normalize(raw) for raw in raw_batch]


if __name__ == "__main__":
  # SQS/S3 없이 PoC 레시피와의 파리티를 확인: 로컬 이미지에서 검출→정렬→임베딩을 실행하고
  # 벡터 norm(전부 1.0이어야 정상)과 얼굴 쌍별 코사인 유사도를 출력한다.
  import sys
  import time

  # detect는 model_source → huggingface_hub 임포트 체인을 끌고 오므로 CLI 확인 블록에서만 지연 import한다
  from app.pipeline.detect import FaceDetector
  from app.pipeline.align import align_face

  detector = FaceDetector()
  embedder = FaceEmbedder()
  labeled_embeddings: list[tuple[str, np.ndarray]] = []
  for path in sys.argv[1:]:
    image = cv2.imread(path)
    if image is None:
      print(f"{path}: 건너뜀 (이미지를 읽을 수 없음)")
      continue

    start = time.perf_counter()
    detected = detector.detect(image)
    aligned = [align_face(image, face.landmarks) for face in detected]
    crops = [crop for crop in aligned if crop is not None]
    embeddings = embedder.embed_batch(crops)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    skipped = len(aligned) - len(crops) + sum(1 for e in embeddings if e is None)
    print(
      f"{path}: {len(detected)} face(s), 임베딩 {sum(1 for e in embeddings if e is not None)}, "
      f"건너뜀 {skipped} in {elapsed_ms:.1f} ms"
    )
    for i, emb in enumerate(embeddings):
      if emb is None:
        continue
      print(f"  face{i}: norm={float(np.linalg.norm(emb)):.6f}")
      labeled_embeddings.append((f"{path}#face{i}", emb))

  if len(labeled_embeddings) > 1:
    print("\n얼굴 쌍별 코사인 유사도 (동일 인물 ≳0.6, 타인 ≲0.3 기대):")
    for i in range(len(labeled_embeddings)):
      for j in range(i + 1, len(labeled_embeddings)):
        name_a, emb_a = labeled_embeddings[i]
        name_b, emb_b = labeled_embeddings[j]
        print(f"  {name_a} vs {name_b} : {float(np.dot(emb_a, emb_b)):.4f}")
