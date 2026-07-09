# 코드 리뷰 — 랜드마크 추출·정렬 경로 (유지보수성 · 운영 관점)

- 일자: 2026-07-09
- 대상: `app/pipeline/detect.py`(5점 랜드마크 추출) + `app/pipeline/align.py`(Umeyama 정렬)
  및 이 둘의 조립 지점 `app/core/deps.py`
- 관점: 유지보수성, 운영 환경(실 AWS 배포)에서 발생 가능한 문제
- 상태: **발견 기록 — 전 항목 미수정** (수정 시 각 항목에 반영 커밋을 기입할 것)
- 관련: [2026-07-09-face-detection-review.md](./2026-07-09-face-detection-review.md) —
  감지 경로 공통 발견(모델 revision 미고정, `cv2.setNumThreads` 불일치, 관측성 부재,
  `DetectorConfig`↔Settings 미연결)은 그 문서 1·2·4·6번을 따르고 여기서는 중복 기술하지 않는다.
  이 문서는 **정렬(align.py)과 랜드마크 계약**에 고유한 발견만 다룬다.

## 요약

정렬 코드 품질은 높다 — insightface 파리티 검증(변환행렬 allclose, 픽셀 diff 0), float32 dst
유지 근거의 문서화, 퇴화 케이스의 None 격리가 모두 잘 되어 있다. 고유 리스크는 두 가지다:
① 정렬 정확성이 "YuNet 랜드마크 순서 = `_ARCFACE_DST` 행 순서"라는 **주석으로만 유지되는 암묵
계약**에 전적으로 의존 — 검출기 교체 시 예외 없이 조용히 망가지는 최악의 회귀 경로,
② `_ensure_bgr` 중복의 명시된 근거가 사실과 달라(지연 import를 즉시 import로 오인) 잘못된
주석이 "건드리지 마라" 신호로 작동 중.

## 유지보수성

### 1. YuNet 랜드마크 순서 ↔ `_ARCFACE_DST` 순서 결합이 주석으로만 유지됨

정렬의 정확성은 "YuNet 출력 순서(우안·좌안·코·우구각·좌구각)가 `_ARCFACE_DST`의 행 순서와
일치한다"는 암묵 계약에 전적으로 의존한다. 이 계약이 `detect.py`의 `DetectedFace.landmarks`
주석("YuNet 원본 순서, 재배열 금지")과 `align.py`의 `_ARCFACE_DST` 좌표 리터럴로 **흩어져 있고
서로를 참조하지 않는다**.

운영 영향: 검출기를 교체하면(예: SCRFD — 랜드마크 순서가 다를 수 있음) 정렬이 **예외 없이
조용히** 뒤틀리고 임베딩 품질만 떨어진다. Umeyama는 순서가 뒤섞인 대응점에도 유효한 변환행렬을
반환하므로 런타임에서 잡을 방법이 없다 — 증상은 "같은 인물이 자꾸 갈라진다"류의 군집 품질
저하로만 나타나 역추적이 극히 어렵다.

- 권장: ① 두 주석이 서로를 명시 참조(detect → "align의 `_ARCFACE_DST` 순서와 짝" / align →
  "detect의 YuNet 원본 순서 전제"), ② pytest 도입(로드맵 항목) 시 insightface 파리티 검증을
  골든 테스트로 고정해 순서가 깨지면 즉시 실패하게 할 것

### 2. `_ensure_bgr` 중복의 명시된 근거가 사실과 다름

`align.py`의 `_ensure_bgr` 독스트링은 "detect를 import하면 model_source → huggingface_hub
임포트 체인이 순수 수학 모듈에 유입되기 때문"이라며 `detect._to_contiguous_bgr`와의 중복을
정당화한다. 그러나 실제로는 `huggingface_hub`가 `HuggingFaceModelSource.resolve()` **내부의
지연 import**라서 detect를 import해도 유입되지 않는다 — 근거가 무효다.

유지보수 영향: 동일 계약의 사본 2개는 한쪽만 수정되는 drift의 전형적 통로인데, 잘못된 근거가
"이 중복은 의도된 것이니 건드리지 마라"는 신호로 작동해 통합 시도를 차단한다. 문서가 존재하지
않는 제약을 가리키는 것 자체가 결함이다(감지 리뷰 1번의 setNumThreads와 같은 유형).

- 권장 (둘 중 하나): ① 공용 헬퍼를 의존성 없는 모듈(예: `app/pipeline/image_ops.py`)로 추출해
  중복 제거, ② 중복을 유지하려면 진짜 이유(순수 수학 모듈의 독립성 유지 등)로 주석 교체

### 3. `align_face`의 얼굴당 전체 이미지 재정규화

`deps.py`의 `extract_faces`가 `[align_face(image, face.landmarks) for face in detected]`로
호출하므로, 얼굴 N개면 `_ensure_bgr(image)`가 전체 이미지에 대해 N회 실행된다. 정상 경로
(`imdecode`의 uint8 연속 BGR)는 `np.ascontiguousarray`가 no-op이라 무해하지만, 그레이스케일/
알파/비연속 입력이 오면 **전체 이미지 변환·복사가 얼굴 수만큼** 반복된다. 단체 사진(얼굴
수십 개)이 이 서비스의 주 워크로드라는 점에서 잠재 비용이 0은 아니다.

- 권장: 방어적 정규화를 호출 체인에서 1회로 일원화 — detect가 이미 내부에서 정규화하므로,
  extract_faces가 정규화된 이미지를 받아 align에 넘기는 구조로 정리하거나 align_face의 방어를
  "이미 정규화된 입력 전제"로 격하 (우선순위 낮음, 구조 정리 시 함께)

## 운영 환경 리스크

### 4. 정렬 실패의 조용한 드롭 — 감지 리뷰 4번(관측성)의 정렬 측 세부

`align_face`가 퇴화 변환(rank 0 → NaN 행렬)에서 None을 반환하고, `deps.py`가 이를 로그 없이
걸러낸다. 감지 리뷰 4번이 권장한 이미지당 `검출 N → 정렬성공 M → 임베딩 K` 로그가 도입되면
함께 해소된다 — 별도 조치 불요, 도입 시 정렬 단계 카운트가 포함되는지만 확인할 것.

### 5. rank 1(랜드마크 일직선) 케이스는 유효 행렬로 통과

`_umeyama`는 rank 0(전 랜드마크 한 점)만 NaN → None으로 거르고, rank 1(일직선)은 반사 보정을
거친 **유효한 행렬을 반환**한다. 이 경우 기하학적으로 무의미한 크롭이 임베딩까지 흘러간다.
실제 YuNet 출력에서 5점이 정확히 일직선이 되는 경우는 극단적이고, 품질 게이트·HDBSCAN 노이즈
처리가 흡수할 가능성이 높아 우선순위는 낮다.

- 권장: 현재 동작(노이즈로 흡수 기대)이 의도라면 `align_face` 독스트링에 한 줄 명시.
  skimage `SimilarityTransform.estimate`와 동작 파리티를 유지하려는 의도라면 그것도 명시

### 6. 소소한 것들

- `detect.py` 모듈 상수 네이밍 비일관: `DEFAULT_SCORE_THRESHOLD`는 `DEFAULT_` 접두사가 있는데
  `NMS_THRESHOLD`/`TOP_K`는 없다. 셋 다 `DetectorConfig` 기본값이므로 통일 권장.
- `DetectorConfig` 검증 비대칭(threshold 범위 미검증)은 감지 리뷰 7번에 이미 기록 — 생략.

## 확인해서 문제 없었던 것 (오탐 방지 기록)

- **NaN 랜드마크 유입**: detect가 non-finite 랜드마크를 생성 시점에 거르고, 뚫고 들어와도
  `_umeyama`의 NaN 전파 → `align_face`의 `np.isfinite(M).all()` 검사 → None으로 이중 방어된다.
- **`_ARCFACE_DST`의 float32 유지**: float64 선언 시 insightface 파리티(픽셀 diff 0)가 깨질 수
  있다는 근거가 주석으로 명시되어 있고, 내부 연산은 `_umeyama`에서 float64 승격이라 정밀도
  손실도 없다 — 잘 설계된 결정.
- **`ALIGN_SIZE` 고정(파라미터 미노출)**: `_ARCFACE_DST`가 112×112 전용 좌표이므로 크기를
  열어두면 오히려 오용 통로가 된다 — 근거 명시된 올바른 결정.
- **오류 전략 혼용(ValueError vs None)**: landmarks shape 오류는 프로그래밍 오류라 raise,
  퇴화 변환은 데이터 문제라 None — 구분 기준이 독스트링에 명시되어 있고 detect의 None 패턴과
  일관된다.
- **`landmarks.flags.writeable = False`**: frozen dataclass의 ndarray 필드가 하류에서 변형되는
  것을 막는 방어 — 새로 생성한 배열에만 적용하므로 부작용 없음.

## 우선순위

| 순위 | 항목 | 비용 |
|------|------|------|
| 즉시 | 2 (`_ensure_bgr` 주석 정정 또는 헬퍼 추출) | 주석 교체는 한 줄 |
| pytest 도입 시 | 1 (랜드마크 순서 골든 테스트), 5 (rank 1 의도 명시) | 파리티 테스트 승격에 편승 |
| 구조 정리 시 | 3 (정규화 1회 일원화), 6 (네이밍) | 소규모 |

감지 경로 공통 항목(revision 핀·setNumThreads·관측성·Settings 연결)의 우선순위는
[감지 리뷰](./2026-07-09-face-detection-review.md) 우선순위 표를 따른다.
