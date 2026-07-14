# CloudWatch 로그 연동

- 적용: 2026-07-14
- 대상: EC2에서 도는 AI 워커 컨테이너(`cheesemoa-ai`)의 stdout/stderr
- 배포 구성 전반은 [ec2-deployment.md](ec2-deployment.md) — 이 문서는 **로그 경로**만 다룬다.

## 구성 요약

워커 로그는 Docker **`awslogs` 로그 드라이버**가 CloudWatch Logs로 직송한다. 별도 에이전트
(CloudWatch Agent 등) 설치 없이 Docker 데몬이 인스턴스 롤 자격증명으로 직접 쓴다.

| | 값 |
|---|---|
| 로그 그룹 | `/cheesemoa/ai-worker` (ap-northeast-2, 보존 30일 — Spring `/cheesemoa/app`과 동일 정책) |
| 로그 스트림 | `cheesemoa-ai` (컨테이너 재시작 시에도 같은 스트림에 이어 씀) |
| IAM | `cheesemoa-ec2-role`의 인라인 정책 `cheesemoa-cloudwatch-logs` — 두 로그 그룹에 `logs:CreateLogStream` · `logs:PutLogEvents`만 허용 |
| 컨테이너 옵션 | `--log-driver=awslogs` + `awslogs-region/group/stream` + `cache-max-size=10m` · `cache-max-file=3` ([deploy.yml](../../.github/workflows/deploy.yml)과 [ec2-deployment.md §2](ec2-deployment.md)에 반영) |

`cache-max-*`는 dual logging(Docker 20.10+) 로컬 캐시의 상한이다 — 원격 드라이버를 써도 호스트에서
`docker logs`가 그대로 동작하는 이유이며, 배포 워크플로의 기동 로그 검증(`SQS 폴링 시작` grep)도
이 캐시를 읽는다. json-file 시절의 `max-size/max-file` 디스크 상한을 승계한 값이다.

## 로그 보는 법

CloudWatch 콘솔 → 로그 그룹 `/cheesemoa/ai-worker`, 또는 EC2 접속 없이 로컬 CLI:

```bash
aws logs tail /cheesemoa/ai-worker --follow --profile cheesemoa --region ap-northeast-2

# 특정 job 추적
aws logs filter-log-events --log-group-name /cheesemoa/ai-worker \
  --filter-pattern '"job_id=<JOB_ID>"' --profile cheesemoa --region ap-northeast-2
```

## 무엇이 찍히나

- 기동 시퀀스: 모델 로딩 → 레디니스 → `SQS 폴링 시작`
- 처리 완료 요약: `처리 완료 job_id=... status=...`
- **결과 발행 본문 전체**: `결과 발행 job_id=... <N> bytes body={...}` (`app/messaging/publisher.py`) —
  Spring으로 넘어간 SQS 메시지의 JSON 원문이 그대로 남는다. Spring 실계약 통합검증 단계에서 유용해
  INFO로 상시 기록하기로 했다. 대형 이벤트의 결과가 수십 KB라 수집량이 늘 수 있으나 현 트래픽에선
  비용이 미미하다 — 트래픽이 붙은 뒤 부담되면 DEBUG로 강등한다.
- 인바운드 원문은 계약 위반(포이즌) 메시지일 때만 body 전체가 남는다 (`app/worker.py`)

## 함정

1. **awslogs 권한이 없으면 컨테이너 기동 자체가 실패한다.** 드라이버가 컨테이너 생성 시점에 로그
   스트림을 만들기 때문. 로그 유실보다 명시적 실패가 안전하다는 판단이지만, 뒤집어 말하면
   `cheesemoa-cloudwatch-logs` 정책이나 로그 그룹을 지우면 **배포가 깨진다**. 롤/정책을 정리할 일이
   있으면 여기부터 확인할 것.
2. **`max-size`/`max-file`은 awslogs와 함께 못 쓴다** (json-file 전용 옵션 — `docker run`이 거부).
   로컬 캐시 상한은 `cache-max-size`/`cache-max-file`로 지정한다.
3. 수동 기동(ec2-deployment.md §2)할 때도 awslogs 옵션을 빼먹지 말 것 — 빼먹으면 그 시점부터
   CloudWatch에 로그가 끊긴다.

## 남은 것

- **지표(metrics) 연동은 미완** — CPU 크레딧·메모리·처리 지연 등. CLAUDE.md "다음 구현 목표" 1번.
- 로그 기반 경보(예: `ERROR` 필터 → SNS) 미구성.
