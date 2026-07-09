# 코드 리뷰 — SQS 메시징·S3 접근 경로 (정확성 · 운영 관점)

- 일자: 2026-07-09
- 대상: `app/messaging/consumer.py`·`publisher.py`(SQS 수신·발행),
  `app/storage/embedding_store.py`·`image_source.py`·`event_embeddings.py`(S3 접근 + `.npz` 코덱),
  그리고 오류 정책의 실 소유자인 `app/worker.py` 폴링 루프와 `app/core/config.py`·`deps.py`(레디니스)
- 관점: 정확성(오류 유무), 운영 환경(실 AWS 배포)에서 조심해야 할 것
- 상태: **발견 기록 — 전 항목 미수정** (수정 시 각 항목에 반영 커밋을 기입할 것)
- 관련: [감지](./2026-07-09-face-detection-review.md) · [정렬](./2026-07-09-face-alignment-review.md) ·
  [임베딩](./2026-07-09-face-embedding-review.md) · [클러스터링](./2026-07-09-clustering-review.md) ·
  [사용자 보정](./2026-07-09-user-feedback-review.md) — 이미지 크기 상한 부재(감지 5번),
  O(N²) 재군집 시간(클러스터링 1번)은 각 문서를 따르고, 여기서는 **메시징·저장 계층 고유
  발견**만 다룬다.

## 요약

at-least-once 의미론 설계(handle→발행→삭제 순서 + photo_id 멱등 append + 결정적 재군집)와
`.npz` 코덱의 방어(`allow_pickle=False`, 스키마 버전, 손상 객체 보존)는 잘 되어 있다. 실질
버그는 하나다: **발행(publish)이 워커 오류 정책 밖에 있어, 256KB 초과 결과 등 결정적 발행
실패가 "재처리 반복 → 무통보 DLQ"로 떨어진다** — `.npz`엔 앨범이 계산돼 있는데 Spring은
succeeded도 failed도 영영 받지 못한다. 나머지는 실 AWS 배포 시 반드시 알아야 할 함정들이다:
IAM `s3:ListBucket` 없으면 최초 분류 전멸, 단일 writer 전제의 visibility timeout 조건부성,
인프라 장애가 이미지 단위 실패로 위장되는 문제.

## 정확성 버그

### 1. 발행 실패가 오류 정책 밖 — 256KB 초과 결과는 무통보 DLQ로 감

`worker.py`의 `_process`에서 `publisher.publish(result)`는 try 블록 **밖**에 있다. 발행 예외는
`run()`의 폴링 catch-all까지 올라가고 메시지는 미삭제 재전달된다. 문제는 **결정적으로 실패하는
발행**이다:

- `publisher.py`는 256KB 초과 시 **경고만 찍고 그대로 발행을 시도**하는데, SQS `send_message`는
  256KiB 초과에서 예외를 던진다. 대형 이벤트(주석 스스로 "인물 20명대 + 긴 image_id 목록이면
  초과 가능"이라 인정)의 결과는 매번: handle 성공(재군집 + `.npz` 저장 완료) → 발행 예외 →
  재전달 → 전체 재처리 → 또 발행 예외 → … maxReceiveCount 소진 후 DLQ.
- "마지막 시도에 failed 발행" 로직은 **handle 예외에만 반응**하므로 이 경로에서 절대 실행되지
  않는다. 결과: Spring은 succeeded도 failed도 받지 못하고 job이 무한 대기로 남는다. 잘못된 결과
  큐 URL·발행 권한 누락 같은 다른 지속 실패도 같은 경로를 탄다.

- 권장: ① 발행을 오류 정책 안으로 — 발행 실패도 시도 횟수 기반으로 다루고 마지막 시도엔 failed
  발행을 시도(그것마저 초과하면 최소 요약 형태로), ② 256KB 검사를 경고가 아니라 **결정적
  처리**로 — TODO의 payload-on-S3 포인터 전환 또는 그 전까지 초과분 축약 + 부분 실패 표시.
  "MVP 규모에선 여유"라는 판단은 맞지만, 초과의 결말이 품질 저하가 아니라 무통보 유실인 현
  구조는 규모 판단과 무관하게 고칠 가치가 있다.

## 운영 환경 리스크

### 2. S3 IAM 함정 — `s3:ListBucket` 없으면 최초 분류가 전부 실패

`embedding_store.py`의 `load`는 `NoSuchKey`를 "저장된 적 없음(최초 분류) → None"으로 해석한다.
그런데 S3는 **`s3:ListBucket` 권한이 없으면 미존재 키에 404(NoSuchKey)가 아니라
403(AccessDenied)을 반환**한다. GetObject/PutObject만 부여한 최소 권한 IAM으로 배포하면 모든
신규 event의 최초 classify가 AccessDenied → 작업 전체 실패 → 재시도 소진 → DLQ가 된다.
`check_readiness`의 `head_bucket`도 같은 권한 계열이라 기동 시점에 잡힐 가능성이 높지만,
로드맵의 "IAM 자격증명" 작업에서 embeddings 버킷의 `s3:ListBucket`이 **필수**임을 명시할 것.

### 3. "단일 writer" 전제는 visibility timeout까지만 유효

`embedding_store.py`의 save 계약 주석은 "FIFO(messageGroupId=event_id)가 동시 쓰기를
직렬화하므로 단일 writer"라고 선언하는데, 이 직렬화는 **처리 시간 < visibility timeout일
때만** 성립한다. timeout이 지나면 FIFO 그룹 락이 풀려 같은 메시지가 다른 워커(또는 재기동
워커)에 재전달되고, 원래 워커가 살아있으면 같은 event `.npz`에 두 writer가 생겨 lost update가
난다. 클러스터링 리뷰 1번(이벤트가 클수록 재군집 시간 제곱 증가)과 결합하면 대형 이벤트에서
이 전제가 먼저 깨진다.

- 권장: ① visibility timeout을 최악 처리 시간에 맞추는 실측(클러스터링 리뷰 1번과 같은 작업),
  ② 장기 작업 대비 `ChangeMessageVisibility` heartbeat 도입 검토, ③ save 계약 주석에
  "전제 조건: 처리 시간 < visibility timeout" 명시

### 4. 인프라 장애가 이미지 단위 실패로 위장되어 메시지가 소비됨

`image_source.py`의 `fetch`는 원인 무관 `except Exception` → `ImageFetchError`로 수렴하고,
핸들러는 이미지 단위 격리로 `failed_images`에 담는다. 그 결과 **자격증명 만료·이미지 버킷 권한
오설정 같은 인프라 장애**도 "전 이미지 실패 + partial 발행 + 메시지 삭제(소비 완료)"가 된다.
주석의 "재시도 대상이 된다"에서 재시도 주체는 SQS가 아니라 Spring이다 — 워커 관점에선 정상
처리로 끝나 DLQ에도 남지 않는다. 진짜 이미지 단위 문제(잘못된 s3_key, 손상 파일)와 인프라
장애가 같은 정책일 이유가 없다.

- 권장: ClientError 코드로 분기 — `NoSuchKey`·디코드 실패는 이미지 단위 실패 유지,
  `ExpiredToken`·`AccessDenied`·5xx는 작업 전체 실패로 올려 SQS 재시도·DLQ 경로를 태울 것

### 5. `sqs_max_receive_count` ↔ 큐 redrive `maxReceiveCount` 드리프트를 코드가 못 잡음

`config.py` 주석이 "반드시 일치시킬 것"이라 경고하는 값인데, 어긋나면 (워커 값이 크면) failed
발행 없이 DLQ로 가고, (작으면) DLQ 전에 failed를 여러 번 발행한다. `check_readiness`가 이미
`get_queue_attributes`를 호출하므로 `RedrivePolicy` 속성을 추가로 읽어 **기동 시점에
maxReceiveCount 일치를 검증**하면 사람 규율이 코드 검증이 된다 — 비용은 몇 줄.

### 6. 소소한 것들

- **결과 중복의 최종 방어는 Spring**: 발행 후 삭제 전 크래시 → 재처리 → 재발행은 at-least-once
  설계상 정상인데, FIFO dedup(job_id)은 5분 창 안에서만 막는다. 5분을 넘는 재전달이면 중복
  결과가 Spring에 도달하므로 Spring 쪽 job_id 멱등 처리가 계약임을 통합 검증 때 명시할 것.
  결과 큐가 표준 큐로 확정되면 dedup 자체가 없어 이 요구가 더 중요해진다.
- **발행 실패 로그 라벨**: publish 예외가 run()에서 "폴링 사이클 실패"로 찍힌다 — 트레이스백으로
  구분되지만 장애 조사 시 오독 여지가 있는 라벨이다. 1번 수정 시 자연 해소.
- **graceful shutdown ↔ 컨테이너 stopTimeout**: 종료 요청 후 long poll 최대 20초 + 처리 중 메시지
  완주(대형 job이면 분 단위)를 기다린다. ECS 기본 stopTimeout(30초)이면 SIGKILL로 끊긴다 —
  멱등성 덕에 안전하나 재전달 낭비가 생기므로 배포 시 stopTimeout을 처리 시간에 맞출 것.
- `receive_message`의 `AttributeNames` 파라미터는 최신 boto3에서 `MessageSystemAttributeNames`로
  대체 예고됨(동작 유지). 의존성 핀 논의(감지 리뷰 7번)와 함께 볼 항목.

## 확인해서 문제 없었던 것 (오탐 방지 기록)

- **처리 순서의 at-least-once 정합**: handle(내부 `.npz` 저장) → 발행 → 삭제 순서와 "photo_id
  멱등 append + 결정적 재군집"의 조합은 크래시 지점별로 검토해도 상태 일관성이 유지된다
  (1번의 발행 실패 경로만 예외).
- **`MaxNumberOfMessages=1` 고정**: FIFO + 직렬 워커에서 배치 수신이 만드는 가시성 잠금 문제를
  정확히 이해한 결정이고 근거가 주석에 있다.
- **포이즌 정책**: 본문 전문 로그(증거 보존) → job_id 최선 회수 → failed 발행 → 삭제(FIFO 그룹
  언블록) — 순서·근거 모두 올바르다.
- **`.npz` 코덱**: `allow_pickle=False`(임의 코드 실행 차단), 스키마 버전 검사, 빈 배열 dtype
  함정 방어, 미배정 `""` 인코딩의 양방향 불변식, 로드 시 전 불변식 재검증 + `StoreCorruptionError`
  단일 수렴, 손상 객체 자동 삭제 금지(생체 파생 데이터·증거 보존) — 저장 경계 설계가 탄탄하다.
- **Settings**: 필수 필드 기본값 없음(placeholder 배포가 폴링 시작 전에 죽음), `extra="ignore"`의
  근거(.env의 AWS 자격증명 키 공존), frozen — 모두 명시적이다.
- **페이크의 의미론 충실도**: `InMemoryConsumer`가 재전달·receive_count·DLQ까지 모사해
  `--smoke`가 오류 정책을 실제로 검증한다 — 페이크가 해피패스만 흉내 내는 흔한 함정을 피했다.
- **수신 루프 백오프**: 네트워크 순단 시 5초 대기로 핫루프·로그 폭주를 막고, 미삭제 메시지는
  재전달로 보전된다.

## 우선순위

| 순위 | 항목 | 비용 |
|------|------|------|
| 즉시 | 1 (발행을 오류 정책 안으로 + 256KB 결정적 처리) | 중간 — 정책 결정 필요 |
| 실 AWS 통합 검증 전 | 2 (IAM ListBucket 문서화), 5 (RedrivePolicy 기동 검증) | 소규모 |
| 배포 설계 시 | 3 (visibility timeout 정합·heartbeat), 4 (인프라 오류 분기), 6 (stopTimeout 등) | 소규모~중간 |

다른 경로 공통 항목의 우선순위는 각 리뷰 문서의 우선순위 표를 따른다.
