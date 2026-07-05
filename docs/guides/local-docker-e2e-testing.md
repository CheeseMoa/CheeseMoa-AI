# 로컬 Docker + 실 AWS End-to-End 테스트 가이드

로컬 PC에서 Docker로 띄운 워커를 **실제 SQS/S3**(AWS 콘솔 확정값, 계정 `889918307386` / 서울)에 붙여
전체 파이프라인(폴링 → 얼굴 감지 → 정렬 → 임베딩 → 품질 게이트 → 클러스터링 → 결과 발행)을
검증하는 절차다. 배포(EC2 등) 방식과는 무관하다 — 배포 시엔 IAM 롤이 자격증명을 대신하므로
아래 SSO 설정 자체가 필요 없다 ([.env.example](../../.env.example) 참고).

## 왜 SSO 프로필인가

포털에서 임시 자격증명 3개 값(`AWS_ACCESS_KEY_ID`/`SECRET`/`SESSION_TOKEN`)을 복사해 `.env`에
붙여넣는 방식은 단기 세션이라 자주 만료돼 매번 다시 복사해야 한다. **AWS SSO 프로필**을 한 번
등록해두면, SSO 세션이 살아있는 동안은 `aws sso login`만 다시 실행하면 되고 `.env`를 건드릴
필요가 없다 — boto3가 호출 시점마다 캐시에서 알아서 새 단기 자격증명을 받아온다.

## 1. 사전 준비 — AWS CLI v2 설치 (팀원별 1회)

```powershell
winget install -e --id Amazon.AWSCLI --silent --accept-package-agreements --accept-source-agreements
```

설치 후 **새 터미널**을 열어야 `aws` 명령이 PATH에 잡힌다 (설치 전에 열어둔 터미널은 안 됨).

## 2. AWS SSO 프로필 등록 (팀원별 1회)

```powershell
aws configure sso
```

| 프롬프트 | 입력값 |
|---|---|
| SSO session name (Recommended) | **비워두고 엔터** (세션명을 넣으면 아래 "알려진 이슈" 참고) |
| SSO start URL | `https://d-9b675698c6.awsapps.com/start` |
| SSO region | `ap-northeast-2` |
| (브라우저 로그인 후) 계정 | `889918307386` |
| (브라우저 로그인 후) 역할 | 부여받은 역할 (예: `myisb_IsbUsersPS`) |
| CLI default client Region | `ap-northeast-2` |
| CLI default output format | `json` |
| CLI profile name | **`cheesemoa`** (아래 모든 명령이 이 이름을 전제로 함) |

확인:
```powershell
aws sts get-caller-identity --profile cheesemoa
```
계정 ID·역할 ARN이 출력되면 성공.

## 3. SSO 세션이 만료되면? (핵심)

증상: 워커 컨테이너가 레디니스 단계에서 자격증명 오류로 죽거나, `aws sts get-caller-identity
--profile cheesemoa`가 만료 에러를 낸다.

**해결: 딱 한 줄이면 된다.**
```powershell
aws sso login --profile cheesemoa
```
브라우저가 열리고 로그인하면 끝. `.env`도, 프로필 설정도 다시 만질 필요 없다 — 컨테이너를
재시작하기만 하면 새 자격증명을 자동으로 집어간다(실행 중이던 컨테이너는 재시작 필요).

SSO 세션 자체의 만료 주기는 조직 Identity Center 설정값이라 여기 적힌 값이 없다 — 실사용하면서
얼마나 자주 재로그인이 필요한지 팀 내에서 공유해두면 좋다.

## 4. `.env` 준비

`AWS_ACCESS_KEY_ID`/`SECRET`/`SESSION_TOKEN` 3줄은 **넣지 않는다** (SSO 프로필이 대신함). 나머지
큐 URL·버킷명·리전은 [.env.example](../../.env.example)을 복사해 채운다. `.env`는 git에 커밋되지 않는다.

## 5. 빌드 & 실행

```powershell
docker build -t cheesemoa-worker .
```

**워커 실행** (터미널 1 — 로그를 보기 위해 포그라운드 권장):
```powershell
docker run --rm `
  -v "$PWD\.env:/app/.env:ro" `
  -v "$HOME\.aws:/root/.aws:ro" `
  -e AWS_PROFILE=cheesemoa `
  -v "$HOME\.cache:/root/.cache" `
  -e HF_HUB_OFFLINE=1 `
  cheesemoa-worker
```

- `~/.aws` 마운트로 SSO 캐시를 컨테이너에 전달 (자격증명 자동 갱신).
- `~/.cache` 마운트 + `HF_HUB_OFFLINE=1`로 호스트에 이미 받아둔 모델(YuNet·AuraFace·눈감음 CNN)을
  재사용 — 회사 TLS 프록시 때문에 컨테이너 안에서 직접 다운로드가 막힐 수 있어 이 방식을 쓴다.
  호스트에 모델이 없으면 먼저 `python -m app.worker`를 로컬(venv)에서 한 번 실행해 캐시를 채워둔다.

다음 로그가 순서대로 뜨면 정상이다:
```
AI 모델 로딩 완료
SQS/S3 연결 확인 완료 — 레디니스 통과
SQS 폴링 시작
```

> **bash/git-bash 사용 시 주의**: `$HOME\.aws`처럼 백슬래시를 그대로 이어붙이면
> `/c/Users/xxx\.aws`라는 잘못된 경로가 되어 마운트가 조용히 실패한다(빈 디렉터리 마운트 →
> `ProfileNotFound`). bash에서는 반드시 슬래시로 통일: `"$HOME/.aws:/root/.aws:ro"`.

## 6. 테스트 메시지 발송

`scripts/`는 저장소 관례상 커밋되지 않는다(`.gitignore`의 "로컬 검증 스크립트 디렉토리"). 아래
내용을 각자 로컬에 `scripts/send_classify.py`로 저장해서 쓴다 — 인바운드 큐로 발행만 하고 빠지는
러너다. 실제 수신·처리·결과 발행은 컨테이너 안 워커가 한다(같은 프로세스에서 발송+수신을 다 하는
`scripts/classify_check.py`를 쓰면 컨테이너와 메시지를 서로 뺏으니 도커 테스트엔 쓰지 않는다).

```python
"""도커로 띄운 워커에게 classify_request 1건을 발송만 하는 러너 (수신은 안 함).

    python scripts/send_classify.py [<s3_key> ...]

로컬 AWS SSO 프로필을 쓰면 실행 전에 해당 프로필을 활성화한다:
    PowerShell: $env:AWS_PROFILE = "cheesemoa"
    bash:       export AWS_PROFILE=cheesemoa
"""

import json
import os
import sys
import uuid

# 인자 미지정 시 기본 대상 (버킷에 올려둔 테스트 이미지 — karina1~5.jpg)
_DEFAULT_KEYS = ["karina1.jpg", "karina2.jpg", "karina3.jpg", "karina4.jpg", "karina5.jpg"]


def main(argv: list[str]) -> None:
  from dotenv import load_dotenv

  load_dotenv()  # 저장소 루트에서 실행한다고 가정 (다른 스크립트와 동일한 관례)

  import boto3

  region = os.getenv("AWS_REGION", "ap-northeast-2")
  inbound_url = os.environ["INBOUND_QUEUE_URL"]

  image_keys = argv or _DEFAULT_KEYS
  run_id = uuid.uuid4().hex[:8]
  event_id = f"event-dockertest-{run_id}"  # 매 실행 새 event → 깨끗한 .npz (append 누적 방지)
  body = {
    "type": "classify_request",
    "job_id": f"dockertest-{run_id}",
    "group_id": "group-dockertest",
    "event_id": event_id,
    "images": [{"image_id": f"img-{i}", "s3_key": key} for i, key in enumerate(image_keys)],
  }

  sqs = boto3.client("sqs", region_name=region)
  sqs.send_message(
    QueueUrl=inbound_url,
    MessageBody=json.dumps(body),
    MessageGroupId=event_id,  # FIFO 필수 (=event_id)
    MessageDeduplicationId=uuid.uuid4().hex,  # 재실행마다 새 메시지 보장
  )
  print(f"발송 완료  job_id={body['job_id']}  event_id={event_id}  images={image_keys}")
  print("→ 컨테이너 로그에서 '처리 완료 job_id=... status=...' 를 확인하세요.")


if __name__ == "__main__":
  main(sys.argv[1:])
```

실행:
```powershell
$env:AWS_PROFILE = "cheesemoa"
python scripts/send_classify.py karina1.jpg karina2.jpg karina3.jpg karina4.jpg karina5.jpg
```
인자 없이 실행하면 버킷의 기본 테스트 이미지(`karina1~5.jpg`)로 발송한다.

## 7. 결과 큐 확인

```powershell
aws sqs receive-message `
  --queue-url "https://sqs.ap-northeast-2.amazonaws.com/889918307386/CheeseMoa-cluster-response.fifo" `
  --max-number-of-messages 10
```
`--queue-url`은 `.env`의 `RESULT_QUEUE_URL`과 동일한 값. `Body`가 이스케이프된 JSON 문자열이라
필요하면 `| ConvertFrom-Json` 또는 파이썬 `json.loads`로 펼쳐서 본다.

`receive-message`만으로는 큐에서 **삭제되지 않는다** — visibility timeout이 지나면 다시 보인다.
확실히 지우려면 응답의 `ReceiptHandle`로 `aws sqs delete-message`를 호출한다. 결과 스키마 필드
설명은 [message-examples.md §④](../spec/message-examples.md#④-분류-결과--classify-result-ai--spring-결과-큐)
참고.

## 8. 워커 중단

```powershell
docker stop <container-id>   # docker ps 로 확인
```
`SIGTERM` → `request_stop()`이 호출돼 **처리 중인 메시지를 완주한 뒤** 폴링을 멈춘다(worker.py의
종료 정책).

## 알려진 이슈

- **`RegisterClient` 단계에서 `InvalidRequestException`**: `aws configure sso`에서 "SSO session
  name"에 값을 넣으면 발생할 수 있다(조직 Identity Center 설정에 따라 신규 sso-session 방식 미지원).
  세션명을 비워두고(legacy 방식) 재시도하면 해결.
- **`StartDeviceAuthorization` 단계에서 같은 에러**: SSO region을 잘못 입력한 경우가 많다. 이
  조직은 `ap-northeast-2`가 정답 — 다른 값(`us-east-1` 등)으로 시도했다면 region만 바꿔 재시도.
- **컨테이너가 뜨자마자 낯선 메시지를 처리함**: 인바운드 큐에 이전 테스트/Spring 연동 시도로 남은
  메시지가 있을 수 있다. `s3_key`가 실제로 없는 메시지면 `status=partial`로 처리되고 정상 삭제(ack)
  되니 워커 자체 문제는 아니다 — 다만 예상 못 한 메시지라면 누가 보낸 건지 확인해볼 것.
