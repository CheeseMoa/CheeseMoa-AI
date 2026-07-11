# EC2 배포 가이드 (ECR + Docker)

- 최초 배포: 2026-07-11
- 대상: AI 워커(`app/worker.py`)를 EC2에 컨테이너로 상시 실행
- 로컬 개발용 Docker 실행은 [local-docker-e2e-testing.md](local-docker-e2e-testing.md) 참고 — 이 문서는 **배포**다.

## 현재 배포 구성

| | 값 |
|---|---|
| EC2 | `i-09037b55ec32ab0ea` (`cheesemoa-be`) · **t4g.small · arm64(Graviton2) · 2코어 · RAM 1846MB · swap 2GB** |
| 동거 프로세스 | Spring(`cheesemoa-app-1`) · Postgres(pgvector) · Grafana · Prometheus |
| 이미지 | ECR `889918307386.dkr.ecr.ap-northeast-2.amazonaws.com/cheesemoa-ai:latest` |
| 컨테이너 | `cheesemoa-ai` (`--restart unless-stopped`, 메모리 상한 900MB) |
| 권한 | 인스턴스 롤 `cheesemoa-ec2-role` + 정책 `CheeseMoaAiWorkerPolicy` |
| 설정 | `/home/ec2-user/cheesemoa-ai/.env` |

> **워커는 Spring과 같은 호스트에 산다.** 전용 인스턴스가 아니다 — 아래 "리스크" 참고.

## 함정 3가지 (먼저 읽을 것)

1. **arm64로 빌드해야 한다.** t4g는 Graviton이다. 맥(Apple Silicon)에서는 `--platform` 없이 네이티브
   빌드하면 되고, x86 머신·GitHub Actions 기본 러너에서 빌드하면 `--platform linux/amd64`가 되어
   EC2에서 `exec format error`로 죽는다. CI를 붙일 때 반드시 arm64 러너(`ubuntu-24.04-arm`)를 쓸 것.
2. **`.env`에 `AWS_PROFILE`을 넣지 말 것.** 로컬은 SSO 프로필을 쓰지만 EC2는 **인스턴스 롤**이 자격증명을
   공급한다. 프로필이 남아 있으면 컨테이너 안에 `~/.aws/config`가 없어 `ProfileNotFound`로 죽는다.
3. **컨테이너별 IAM 롤 분리는 불가능하다.** 이 계정은 AWS Innovation Sandbox라 조직 SCP가
   `sts:AssumeRole`을 막는다(정책을 정확히 짜도 implicit deny). 맨 EC2에서는 롤이 인스턴스 단위이므로,
   워커 권한은 Spring이 쓰는 `cheesemoa-ec2-role`에 직접 붙일 수밖에 없다. 진짜 격리가 필요하면
   인스턴스를 분리해야 한다.

## 1. 이미지 빌드 & ECR 푸시 (로컬 맥)

```bash
docker build -t cheesemoa-worker .          # arm64 네이티브 (--platform 금지)
docker tag cheesemoa-worker:latest 889918307386.dkr.ecr.ap-northeast-2.amazonaws.com/cheesemoa-ai:latest

aws ecr get-login-password --region ap-northeast-2 --profile cheesemoa \
  | docker login --username AWS --password-stdin 889918307386.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 889918307386.dkr.ecr.ap-northeast-2.amazonaws.com/cheesemoa-ai:latest
```

푸시 전 검증(네트워크를 끊고 모델 적재·스모크가 되는지 = 프리베이크가 살아있는지):

```bash
docker run --rm --network none cheesemoa-worker python -m app.worker --smoke
```

**모델 프리베이크**: Dockerfile이 빌드 시점에 모델 3종(~264MB)을 이미지에 굽는다. 덕분에 기동 시 모델
적재가 0.7초다. 이게 없으면 컨테이너가 재시작될 때마다 AuraFace 261MB를 다시 받고, 런타임이 HF Hub·
OpenVINO 네트워크에 의존하게 된다.

ECR 리포지토리에는 수명주기 정책이 걸려 있다 — 태그 없는 이미지는 1일 후, 태그 이미지는 최근 5개만 보관.

## 2. EC2 실행

EC2 접속은 SSH가 아니라 **SSM**을 쓴다(포트 개방·키 불필요). 인스턴스는 이미 SSM 관리형이다.

```bash
aws ssm start-session --target i-09037b55ec32ab0ea --profile cheesemoa --region ap-northeast-2
```

컨테이너 기동:

```bash
ECR=889918307386.dkr.ecr.ap-northeast-2.amazonaws.com
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin $ECR
docker pull $ECR/cheesemoa-ai:latest

docker rm -f cheesemoa-ai 2>/dev/null || true
docker run -d --name cheesemoa-ai --restart unless-stopped \
  --memory=900m --memory-swap=1400m \
  --log-opt max-size=10m --log-opt max-file=3 \
  --env-file /home/ec2-user/cheesemoa-ai/.env \
  $ECR/cheesemoa-ai:latest

docker logs -f cheesemoa-ai
```

정상 기동 로그:

```
Found credentials from IAM Role: cheesemoa-ec2-role
AI 모델 로딩 완료 (추론 스레드=2, 가용 코어=2)
SQS/S3 연결 확인 완료 — 레디니스 통과
SQS 폴링 시작
```

**`추론 스레드`가 코어 수와 같은지 반드시 확인한다** — 다르면 성능이 몇 배로 무너진다
([worker-scaling-and-performance.md §7](worker-scaling-and-performance.md)).

**`--memory=900m`은 안전장치다.** 워커 피크 RSS가 608MB이고 호스트 가용 메모리가 얇아서, 상한이 없으면
커널 OOM killer가 **Spring을 골라 죽일 수 있다.** 상한이 있으면 워커 컨테이너만 죽고 재시작되며,
처리 중이던 메시지는 SQS가 돌려주므로 유실되지 않는다.

## 3. IAM

인스턴스 롤 `cheesemoa-ec2-role`에 `CheeseMoaAiWorkerPolicy`가 붙어 있다. 워커가 실제로 호출하는 API만 담았다:

- 인바운드 큐(`CheeseMoa-cluster-request.fifo`): `ReceiveMessage` · `DeleteMessage` · `GetQueueAttributes`
- 결과 큐(`CheeseMoa-cluster-response.fifo`): `SendMessage`
- `cheesemoa-dev`(이미지): `GetObject`
- `cheesemoa-test-...-an`(임베딩): `GetObject` · `PutObject` (`embeddings/` 프리픽스)
- 두 버킷: `ListBucket` — 레디니스의 `head_bucket`이 이 권한을 요구한다
- ECR pull (인스턴스 롤엔 `AmazonEC2ContainerRegistryReadOnly`도 이미 있음)

## 4. 운영

```bash
docker logs --tail 50 cheesemoa-ai              # 로그
docker stats --no-stream cheesemoa-ai           # 메모리·CPU
docker inspect cheesemoa-ai --format '{{.RestartCount}} {{.State.OOMKilled}}'   # 재시작·OOM 여부
docker restart cheesemoa-ai                     # 재기동 (SIGTERM → 처리 중 메시지 완주 후 종료)
```

재배포는 main에 머지하면 자동이다(5번 절). 수동으로 하려면 `docker pull` 후 `docker rm -f` +
`docker run`(2번 절)을 다시 하면 된다.

## 5. 자동 배포 (GitHub Actions)

main에 push(=PR 머지)하면 [.github/workflows/deploy.yml](../../.github/workflows/deploy.yml)이
자동으로 배포한다. `docs/**`·`**.md`·`image/**`만 바뀐 푸시는 건너뛴다. 수동 트리거는 GitHub
Actions 탭의 `workflow_dispatch`.

파이프라인: **arm64 네이티브 빌드**(`ubuntu-24.04-arm` 러너 — 함정 1 참고) → **오프라인 스모크**
(`--network none`에서 `--smoke`, 프리베이크·배선 검증) → **ECR 푸시**(`:latest` + `:커밋sha`) →
**SSM `AWS-RunShellScript`**로 EC2에서 pull + 컨테이너 교체(2번 절과 동일 파라미터, 커밋 sha 태그로
기동) → **`SQS 폴링 시작` 로그 확인**까지 통과해야 성공. 교체 시 `docker stop -t 120`으로 SIGTERM을
보내 처리 중 메시지를 완주시킨다(완주 못 해도 SQS가 돌려준다).

자격증명은 GitHub OIDC다 — 시크릿에 액세스 키를 넣지 않는다. 전용 롤
`cheesemoa-github-actions-ai`(인라인 정책 `cheesemoa-ai-deploy`)가 `repo:CheeseMoa/CheeseMoa-AI:*`를
신뢰하고, ECR `cheesemoa-ai` 푸시와 이 인스턴스로의 `ssm:SendCommand`만 허용한다. 조직 SCP가
`sts:AssumeRole`을 막지만 OIDC의 `sts:AssumeRoleWithWebIdentity`는 막지 않는다 — BE 리포의
`cheesemoa-github-actions` 롤로 이미 검증된 패턴이다.

배포가 실패하면(스모크·기동 로그 검증 실패) 워크플로가 빨간불이 되고 SSM 실행 로그가 잡 출력에
찍힌다. 기동 검증 실패 시점에는 이전 컨테이너가 이미 내려간 상태이므로, 2번 절 수동 절차로 이전
태그(`aws ecr describe-images`로 확인)를 기동해 롤백한다.

## 리스크 (미해결)

**CPU 크레딧 — 가장 큰 리스크.** t4g는 버스터블이다. 얼굴 감지·임베딩은 CPU를 지속적으로 쓰는 작업이라
실트래픽이 붙으면 크레딧을 소진하고, 바닥나면 인스턴스 **전체**가 baseline으로 스로틀된다 — 같은 호스트의
**Spring API도 함께 느려진다**. 메모리 상한으로는 막을 수 없다. 배포 시점(2026-07-11) 측정에서는 트래픽이
거의 없어 크레딧이 만점(576)이었으나, 이는 안전하다는 뜻이 아니라 **아직 부하가 없었다는 뜻**이다.

**메모리 여유가 얇다.** 워커 ~500MB + Spring ~320MB + Grafana/Prometheus/Postgres에 시스템 가용은
500MB~1GB 수준이다. swap 2GB가 완충하지만 스왑으로 흘러가면 추론이 느려진다.

**대응**: 실트래픽이 붙기 전에 **AI 전용 인스턴스로 분리**하는 것이 정공법이다. 롤도 그때 함께 분리하면
위 "함정 3"의 권한 공유도 해소된다. 워커 프로세스를 늘리는 것은 도움이 안 된다 — SQS FIFO가 이벤트 단위로
직렬화하므로 같은 이벤트는 여전히 한 워커가 처리한다(worker-scaling-and-performance.md §4.1).
