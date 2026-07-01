# CheeseMoa-AI

치즈모아(CheeseMoa)의 AI 추론 서버다. Spring 백엔드가 AWS SQS에 발행한 작업을 consumer로 받아
얼굴 감지 → 정렬 → 임베딩 → 클러스터링 파이프라인을 실행하고 결과를 결과 큐에 발행한다.

자세한 파이프라인 설계는 [docs/architecture/pipeline-overview.md](docs/architecture/pipeline-overview.md) 참고.

## 구성

```text
CheeseMoa-AI/
  app/     # Python 워커 애플리케이션 (SQS consumer)
  docs/    # 설계·결정·컨벤션 문서
```

## 워커 실행

```sh
# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m app.worker
```

이 서버는 HTTP를 제공하지 않는다. SQS 메시지를 폴링하는 워커 프로세스로 동작한다.

## 개발 규칙

- 코드 스타일: [docs/conventions/code-style.md](docs/conventions/code-style.md)
- 폴더 구조: [docs/conventions/project-structure.md](docs/conventions/project-structure.md)

VSCode에서 Ruff 확장을 설치하면 저장 시 자동으로 포맷 + 린트가 적용된다.
커밋 전 pre-commit 훅도 동일하게 동작한다 (`pre-commit install`로 등록).

## 검증

```sh
ruff format --check .
ruff check .
```
