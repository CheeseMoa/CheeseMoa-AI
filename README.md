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

## 개발 환경 세팅

### 1. 가상환경 생성 & 활성화

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

`.venv`는 커밋되지 않는다. 활성화되면 프롬프트에 `(.venv)`가 붙는다. 빠져나올 땐 `deactivate`.

### 2. 의존성 설치

```sh
pip install -r requirements.txt
```

### 3. 환경변수 준비

`.env.example`을 `.env`로 복사해 실값을 채운다 (`.env`는 커밋 금지). 필수 항목은 SQS 큐 URL·S3
버킷명·리전이며, 미설정 시 워커가 기동 시점에 누락 필드를 나열하며 실패한다.

```sh
cp .env.example .env      # Windows: copy .env.example .env
```

AWS 자격증명(`AWS_ACCESS_KEY_ID` 등)은 `.env`에 넣지 않는다 — 로컬은 AWS SSO 프로필, 배포 환경은
IAM 롤이 대신한다. SSO 설정은 아래 [Docker로 실행](#docker로-실행) 참고.

### 4. (선택) pre-commit 훅 등록

```sh
pip install pre-commit
pre-commit install
```

## 워커 실행

```sh
python -m app.worker
```

AWS·모델 없이 전체 배선만 자가 검증하려면:

```sh
python -m app.worker --smoke
```

이 서버는 HTTP를 제공하지 않는다. SQS 메시지를 폴링하는 워커 프로세스로 동작한다.

## Docker로 실행

실 AWS(SQS/S3)에 붙으므로 먼저 SSO 로그인이 되어 있어야 한다. AWS CLI v2 설치와 `cheesemoa`
프로필 등록은 1회만 하면 되고(→
[E2E 테스트 가이드 §1–2](docs/guides/local-docker-e2e-testing.md)), 이후로는 세션이 만료될 때마다
아래 한 줄만 다시 실행한다.

```sh
aws sso login --profile cheesemoa
```

```sh
docker build -t cheesemoa-worker .
```

PowerShell:
```powershell
docker run --rm `
  -v "$PWD\.env:/app/.env:ro" `
  -v "$HOME\.aws:/root/.aws:ro" `
  -e AWS_PROFILE=cheesemoa `
  -v "$HOME\.cache:/root/.cache" `
  -e HF_HUB_OFFLINE=1 `
  cheesemoa-worker
```

bash (macOS/Linux):
```bash
docker run --rm \
  -v "$PWD/.env:/app/.env:ro" \
  -v "$HOME/.aws:/root/.aws:ro" \
  -e AWS_PROFILE=cheesemoa \
  -v "$HOME/.cache:/root/.cache" \
  -e HF_HUB_OFFLINE=1 \
  cheesemoa-worker
```

- `.env`·`~/.aws`(AWS SSO 캐시)·`~/.cache`(모델 캐시)를 읽기 전용으로 마운트한다 —
  자격증명·모델을 이미지에 굽지 않고 런타임에 주입한다.
- **`~/.aws` 마운트는 필수다.** 컨테이너는 호스트의 SSO 캐시를 읽어 호출 시점마다 단기 자격증명을
  갱신한다. 이 마운트가 없으면 `AWS_PROFILE=cheesemoa`를 찾지 못해 `ProfileNotFound`로 죽는다.
  bash에서 `$HOME\.aws`처럼 백슬래시를 쓰면 잘못된 경로가 조용히 빈 디렉터리로 마운트되어 같은
  증상이 나므로, 반드시 슬래시로 통일한다.
- 호스트에서 `aws sso login`으로 세션을 갱신한 뒤엔 **실행 중이던 컨테이너를 재시작**해야 새
  자격증명을 집어간다. `.env`나 프로필 설정은 다시 만질 필요 없다.
- 최초 설정, 세션 만료 시 대응, 테스트 메시지 발송, 결과 큐 확인까지 전체 절차는
  [docs/guides/local-docker-e2e-testing.md](docs/guides/local-docker-e2e-testing.md) 참고.

## 개발 규칙

- 코드 스타일: [docs/conventions/code-style.md](docs/conventions/code-style.md)
- 폴더 구조: [docs/conventions/project-structure.md](docs/conventions/project-structure.md)

VSCode에서 Ruff 확장을 설치하면 저장 시 자동으로 포맷 + 린트가 적용된다.
커밋 전 pre-commit 훅도 동일하게 동작한다 (등록은 [개발 환경 세팅](#개발-환경-세팅) 4단계).

## 검증

```sh
ruff format --check .
ruff check .
```
