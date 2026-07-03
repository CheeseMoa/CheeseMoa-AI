# CheeseMoa-AI 코드 컨벤션

CheeseMoa-AI 워커 서버(`app/`)의 코드 작성 규칙을 정의한다.
자동 강제 가능한 항목은 아래 파일로 설정돼 있어 저장 시 자동 적용된다.

- `ruff.toml` (또는 `pyproject.toml`) — 린트·포맷 규칙
- `.vscode/settings.json` — format on save (Ruff)
- `.pre-commit-config.yaml` — 커밋 전 ruff --fix + ruff-format 자동 실행

## Formatter

- 포맷터: **Ruff** (`ruff format`)
- 들여쓰기: **스페이스 2칸** (tab 문자 사용 금지)
- 최대 줄 길이: **120자**
- 저장 시 자동 포맷 활성화 (VSCode Ruff 확장 필요)

```sh
ruff format .
ruff check . --fix
```

## 파일 인코딩

- 모든 소스 파일은 UTF-8, 줄바꿈은 LF
- 파일명은 `snake_case.py`
- 한글 파일명은 문서 파일을 제외하고 사용하지 않는다

## 네이밍

### 클래스

- `PascalCase` 사용
- 추상 클래스에 `Abstract` 또는 `Base` prefix를 붙이지 않는다 (필요 시만)

### 함수 / 변수

- `snake_case` 사용
- boolean은 `is_`, `has_`, `can_` prefix 권장
- private은 `_` prefix

### 상수

- `UPPER_SNAKE_CASE` 사용 (예: `ARCFACE_DST`, `MODEL_PATH`)

## Python 언어 규칙

- Type hint를 명시한다. `Any`는 꼭 필요할 때만 사용한다.
- public 함수는 return type을 명시한다.
- 재할당하지 않는 모듈 레벨 변수는 상수로 취급한다.
- `Optional[X]` 대신 `X | None`을 사용한다 (Python 3.10+).

```python
def detect_faces(image: np.ndarray) -> list[FaceResult]:
    ...
```

## Import 규칙

import 순서:

1. 표준 라이브러리 (`os`, `pathlib` 등)
2. 서드파티 패키지 (`boto3`, `numpy`, `cv2` 등)
3. 프로젝트 내부 모듈 (`app.pipeline`, `app.schemas` 등)

같은 그룹 안에서는 알파벳순. 사용하지 않는 import는 남기지 않는다 (ruff 자동 정리).

```python
import os
from pathlib import Path

import boto3
import cv2
import numpy as np

from app.pipeline.detect import detect_faces
from app.schemas.messages import ClassifyRequest
```

## 함수 규칙

- 함수는 하나의 역할만 담당한다. 이름은 동사로 시작한다.
- parameter가 많아지면 Pydantic 모델 또는 dataclass로 묶는다.
- early return으로 중첩을 줄인다.

## 워커 / CPU 바운드 규칙

- 워커는 SQS 메시지를 폴링해 메시지 단위로 파이프라인을 실행한다.
- AI 추론은 CPU/GPU 바운드다. I/O(SQS 폴링·S3)와 추론 단계를 분리해 블로킹을 최소화한다.
- 동시 처리가 필요하면 스레드/프로세스 풀로 병렬화한다. 한 메시지 실패가 다른 메시지를 죽이지 않게 격리한다.
- 함수의 return type을 명시한다.

```python
def handle_message(message: ClassifyRequest) -> ClassifyResult:
    faces = run_pipeline(message.images)
    return build_result(message.job_id, faces)
```

## 주석 규칙

- 코드가 "무엇"을 하는지 설명하는 주석은 피한다.
- 코드가 "왜" 필요한지 설명하는 주석만 작성한다.
- `_umeyama()` 같은 수학 로직은 예외적으로 설명 주석 허용.
- TODO: 이슈 번호와 함께 작성.

```python
# TODO(CHMO-XX): 재조정 overlap 임계값 설정값으로 분리
```

## AI 파이프라인 코드 특이사항

- 모델 로딩은 워커 시작 시 1회만 수행한다 (모듈 레벨 또는 워커 부트스트랩).
- numpy 배열 shape은 주석이나 변수명으로 명확히 한다 (예: `face_112x112`).
- magic number는 상수로 분리한다 (예: `EMBED_DIM = 512`, `ALIGN_SIZE = 112`).
