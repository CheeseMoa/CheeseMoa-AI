# 코드 리뷰 — 얼굴 감지 경로 (유지보수성 · 운영 관점)

- 일자: 2026-07-09
- 대상: `app/pipeline/detect.py` + 감지 경로의 운영 접점
  (`app/core/model_source.py`, `app/storage/image_source.py`, `app/core/deps.py`, `app/worker.py`)
- 관점: 유지보수성, 운영 환경(실 AWS 배포)에서 발생 가능한 문제
- 상태: **발견 기록 — 전 항목 미수정** (수정 시 각 항목에 반영 커밋을 기입할 것)

## 요약

코드 품질 자체는 높다 — 경계 방어(`_clamp_bbox`, `_to_contiguous_bgr`), 스레드 안전성 제약의
명시적 문서화, 이미지 단위 오류 격리 계약이 모두 잘 설계되어 있다. 실제 리스크는 세 가지다:
① 문서가 약속한 `cv2.setNumThreads()` 부트스트랩 호출이 실제로는 없음, ② 모델 revision 미고정으로
배포 간 감지 동작이 조용히 바뀔 수 있음, ③ `max_side=2000` 다운스케일이 핵심 유스케이스(단체 사진
원거리 얼굴)에서 미검출을 유발할 수 있는데 env로 조정 불가.

## 운영 환경 리스크

### 1. `cv2.setNumThreads()` — 문서와 코드의 불일치

`detect.py`의 `FaceDetector` 독스트링은 "전역 설정인 `cv2.setNumThreads()`는 여기가 아니라 워커
부트스트랩에서 호출한다"고 약속하지만, `worker.py`에는 cv2 관련 호출이 전혀 없다.

운영 영향: OpenCV DNN은 기본적으로 **호스트의 모든 코어**를 스레드풀로 쓴다. ECS/K8s에서 CPU
limit(예: 1 vCPU)을 건 컨테이너에 배포하면 cv2는 호스트 코어 수(예: 16)를 보고 스레드를 만들어
CPU 스로틀링 + 지연시간 널뛰기가 발생하는 전형적 패턴이다. onnxruntime(`intra_op_num_threads`)도
같은 문제를 공유한다. 워커가 단일 스레드 순차 처리 설계이므로 부트스트랩에서 명시적으로 제한하거나,
하지 않기로 했다면 독스트링을 고쳐야 한다. 문서가 존재하지 않는 코드를 가리키는 것 자체가
유지보수성 결함이다.

- 권장: 워커 부트스트랩에서 스레드 수 명시 설정 (한 줄 수정)

### 2. YuNet 모델 revision 미고정 — 재현 불가능한 배포

`model_source.py`의 `HuggingFaceModelSource`가 `hf_hub_download(repo_id, filename)`을 `revision`
없이 호출해 `opencv/face_detection_yunet` 레포의 **main 브랜치 최신**을 받는다. 업스트림이 모델
파일을 갱신하면:

- 새로 콜드스타트한 워커만 새 가중치를 받아 동일 이벤트를 워커마다 다르게 감지할 수 있고,
- "어제는 잡히던 얼굴이 오늘 안 잡힌다"류의 장애를 재현·역추적할 수 없다.

파일명에 `2023mar`가 있어 위험이 낮아 보이지만, 같은 파일명으로 내용이 갱신되는 것을 막지 못한다.
모델 프리베이크(배포 로드맵 항목) 전까지의 임시 방어로도 revision 고정은 한 줄이다.

- 권장: `revision="<commit hash>"` 고정 — `default_auraface_source`(AuraFace)도 동일 적용

### 3. `max_side=2000` 다운스케일 — 단체 사진 원거리 얼굴 소실 가능

현대 폰 기본 카메라가 12MP(4000×3000)~48MP이므로 사실상 모든 프로덕션 입력이 2분의 1 이하로 축소된
뒤 감지된다. YuNet은 대략 10px 미만 얼굴을 못 잡으므로, 원본 기준 ~20px 얼굴(단체 사진 뒷줄, 넓은
행사장 컷)이 감지 한계 아래로 내려간다. 미검출 얼굴은 임베딩 자체가 없어 그 사람 앨범에서 **조용히
누락**되고, 사용자 보정으로도 구제할 수 없다(reassign할 face가 없음). "정확도 최우선" 설계 원칙과
상충하는 지점이다.

- 권장: 실제 행사 사진 셋으로 `max_side` 2000 vs 3000 vs 무제한의 검출 수·지연시간을 실측해 값 확정.
  값 조정은 아래 6번(Settings 연결) 선행 필요

### 4. 감지 경로 관측성 부재 — 조용한 드롭

- `detect.py` `_to_detected_face`: non-finite bbox/landmark 행을 로그 없이 버린다.
- `deps.py` `extract_faces`: 정렬 실패(None), non-finite 임베딩도 조용히 걸러진다.
- detect 전체에 로그·타이밍·카운터가 하나도 없다.

운영에서 "특정 사진의 얼굴이 앨범에 안 들어왔다"는 문의가 오면, 현재 구조로는 미검출인지 /
non-finite 드롭인지 / 정렬 실패인지 / 품질 게이트 라우팅인지 구분할 방법이 없다.

- 권장: 이미지당 `검출 N → 정렬성공 M → 임베딩 K` 한 줄 로그(image_id 포함). non-finite 드롭은 최소
  warning — 발생 자체가 모델/입력 회귀 신호다. 추후 CloudWatch 지표(배포 로드맵 항목)의 기반이 된다.

### 5. 이미지 크기 상한 부재 — OOM 리스크

`image_source.py`의 `S3ImageSource.fetch`가 크기 확인 없이 `read()` → `imdecode` 한다. OpenCV의
내장 픽셀 한도(기본 2³⁰ 픽셀) 덕에 무한정은 아니지만, 한도 내 최대 이미지도 BGR 전개 시 3GB다.
작은 컨테이너면 OOM kill → 워커 사망 → 같은 메시지 재전달 → 반복 사망(poison message)으로 이어질
수 있다. `handlers.py`의 이미지 단위 `except Exception` 격리는 잘 되어 있어 `cv2.error`류는
`failed_images`로 수렴하지만, **OOM은 프로세스가 죽어 이 격리가 무력화**된다.

- 권장: `ContentLength` 상한 검사(예: 50MB 초과 시 `ImageFetchError`) — 프로세스 수준 리스크를
  이미지 수준 실패로 격하. `.env.example`에 메모된 redrive policy(DLQ)가 마지막 방어선이니 실 배포
  시 반드시 함께 설정할 것

## 유지보수성

### 6. `DetectorConfig`만 Settings에 미연결

`QualityConfig`·`ClusterConfig`는 `settings.to_quality_config()`/`to_cluster_config()`로 env 조정이
가능한데, 감지 파라미터(`score_threshold`, `max_side` 등)만 `deps.py`에서 `FaceDetector()` 기본값으로
하드코딩되어 있다. 위 3번의 `max_side` 튜닝이나 운영 중 threshold 조정이 전부 코드 수정 + 재배포를
요구한다.

- 권장: `Settings.to_detector_config()` 추가로 다른 config와 대칭

### 7. 소소한 것들

- `DetectorConfig.__post_init__`이 `max_side`는 검증하면서 `score_threshold`/`nms_threshold`의
  [0,1] 범위는 검증하지 않는다. env 연결(6번) 시 오타 값이 조용히 들어갈 통로가 되므로 함께 추가.
- `requirements.txt`의 `opencv-python-headless>=4.8.0`은 상한이 없어 설치 시점에 따라 메이저
  동작(디코더, DNN)이 달라질 수 있다. `<5` 상한 또는 정확한 핀 권장 — 아래 EXIF 검증도 4.11 기준.
- `detect.py` `__main__` 스모크의 pytest 승격은 이미 로드맵(pytest 도입)에 있으므로 생략.

## 확인해서 문제 없었던 것 (오탐 방지 기록)

- **EXIF 회전**: 폰 사진의 EXIF orientation을 `cv2.imdecode`가 무시하면 회전된 얼굴이 미검출될 수
  있어 직접 검증했다 — orientation=6을 심은 JPEG로 테스트한 결과 **OpenCV 4.11의
  `imdecode(IMREAD_COLOR)`는 EXIF 회전을 적용**한다. 문제 없음 (단, 7번의 버전 핀과 엮이는 근거 —
  구버전/차기 메이저에서 동작이 다를 수 있다).
- `_clamp_bbox`의 4변 축소 처리, frozen dataclass + ndarray의 eq/hash 처리, `UrlModelSource`의
  임시파일 + 원자적 rename, 스레드 안전성 제약의 명시적 문서화, `handlers.py`의 이미지 단위 오류
  격리 계약은 모두 잘 설계되어 있다.

## 우선순위

| 순위 | 항목 | 비용 |
|------|------|------|
| 즉시 | 1 (setNumThreads), 2 (revision 핀) | 각 한 줄 수정 |
| 실측 후 | 3 (max_side) | 실사진 셋 벤치마크 필요 |
| 실 AWS 통합 검증 전 | 4 (관측성), 5 (크기 상한), 6 (DetectorConfig↔Settings) | 소규모 |
