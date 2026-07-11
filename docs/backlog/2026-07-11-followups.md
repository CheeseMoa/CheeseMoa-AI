# 후속 작업 목록 (2026-07-11 논의)

실 AWS 운영 중 발견한 결함과, 구조 재검토 논의에서 나온 할 일을 우선순위별로 정리한다.
각 항목의 근거는 링크된 문서에 있다.

---

## P0 — 데이터 오염 (실 데이터가 이미 깨져 있음)

### 1. 중복 임베딩 방어 (앨범 쌍 단위 와해)

동일 사진이 다른 `image_id`로 재업로드되면 유사도 1.00인 임베딩 쌍이 생기고, HDBSCAN이 인물
군집 대신 **복제 쌍 5개를 각각 앨범으로** 선택한다. 실 event 8에서 발생·재현 완료.

- 원인·재현: [2026-07-11-duplicate-embedding-split.md](../reviews/2026-07-11-duplicate-embedding-split.md)
- 대응: 재군집 직전 유사도 ≈1.0 행을 대표 1행으로 접고(collapse) 결과 조립 시 펼치기
- 미결: 임계값, 적용 위치(`recluster()` 입력 전처리 vs `handlers` 계층)

### 2. `delete_request` 미도달 — 유령 행

앱에서 지운 사진이 `.npz`에 남아 계속 재군집에 참여한다 (앨범엔 2장인데 앱엔 1장만 보이는 것이
증거). 재업로드와 **독립적인** 결함이며 PIPA(생체정보 삭제) 관점 문제이기도 하다.

- Spring이 사진 삭제 시 `delete_request`를 실제로 발행하는지 확인 — **Spring 쪽 작업**
- ADR-007 §3의 "삭제 메시지 자체가 트리거라 사각지대 없음" 전제가 지켜지지 않고 있음

### 3. event 8 복구

유령 행 제거 후 재군집 → 원래 앨범 구성 복원. (`delete_request` 발행 또는 수동 `.npz` 정리)

---

## P1 — 저장소·배포 보강

### 4. S3 버킷 버저닝 활성화

현재 **비활성**이다. `S3EmbeddingStore.save()`는 `put_object` 통짜 덮어쓰기라, 한 번 잘못 쓰면
**이전 상태가 영구 소실**된다. event 8이 오염됐을 때도 버저닝이 있었다면 롤백으로 끝났다.
객체가 수십 KB라 비용은 사실상 0.

> 이것이 "S3 백업"의 올바른 형태다 — 별도 백업 저장소를 만드는 게 아니라 **원본 저장소의 버전
> 이력을 켜는 것**.

### 5. (선택) S3 조건부 쓰기 — `If-Match` ETag

현재 단일 writer 보장은 **오직 SQS FIFO + `messageGroupId=event_id`** 하나에 걸려 있다. 큐가
FIFO인 것은 확인됨(`...cluster-request.fifo`)이나, **Spring이 `messageGroupId`를 `event_id`로
넣고 있는지는 미확인**. 어긋나면 같은 event를 두 워커가 동시 처리해 lost update가 나고 **삭제된
얼굴이 되살아난다**(ADR-007 §4가 지목한 PIPA 위험). ETag 조건부 쓰기는 이를 조용한 오염 대신
재시도 가능한 실패로 바꾼다.

### 6. `requirements.txt`에서 웹 스택 제거

FastAPI는 **코드만 제거됐고 의존성은 그대로 남아 있다**. Dockerfile이 이를 그대로 설치하므로
프로덕션 이미지에 안 쓰는 웹 프레임워크가 통째로 들어간다 — 이미지 크기·콜드스타트·취약점 표면.

- 제거 후보: `fastapi`, `uvicorn`, `starlette`, `h11`, `httptools`, `websockets`, `watchfiles`,
  `anyio`, `click`, `colorama`, `PyYAML`
- 실제 사용: `pydantic`, `pydantic-settings`, `python-dotenv`, `numpy`, `opencv-python-headless`,
  `onnxruntime`, `boto3`, `huggingface_hub`

### 7. 워커 헬스체크 — HTTP가 아니라 큐 지표로

`check_readiness`는 **기동 시 1회만** 돈다. 이후 폴링 루프는 모든 예외를 삼키고 5초 뒤 재시도하므로
(`worker.py`), 자격증명 만료·권한 오류로 영원히 실패만 반복해도 프로세스는 "살아 있고" 아무도 모른다.

- `/health` 엔드포인트는 **해답이 아니다** — 200을 주면서 큐를 하나도 못 비울 수 있다
- CloudWatch 알람: `ApproximateAgeOfOldestMessage`(적체), DLQ 메시지 수 > 0
- 배포는 죽으면 재시작하는 supervisor 아래에서 (ECS 서비스 / systemd / `--restart unless-stopped`)
  — 워커는 설정 누락·레디니스 실패 시 의도적으로 즉시 종료한다

---

## P2 — 계약 개편 (제품 합의 필요)

### 8. 상태 기반 보정 계약 (ADR-010 후보)

사용자 보정을 액션 4종(merge/split/reassign/confirm_distinct)이 아니라 **검토완료 앨범의 멤버십
선언**으로 받는 안. `handlers.py`의 later-wins 조정 계열이 통째로 사라지는 단순화 이득이 있으나,
"검토완료"의 의미에 대한 제품 합의가 선행돼야 한다.

- 상세: [state-based-feedback-contract.md](./state-based-feedback-contract.md)
- **P0 데이터 오염을 먼저 잡은 뒤에 착수한다** — 계약을 갈아엎기 전에 데이터를 안정시킨다

### 9. `confirm_distinct` 트리거 정책 (기존 미결)

즉시 발행(안전) vs 공유 시점 일괄 발행(발행 전 새 업로드가 끼면 그 사이 재군집은 보호 공백).
8번을 채택하면 이 항목은 **자동 해소**된다(멤버십 선언이 곧 이격 선언).

---

## 검토했으나 유지 — 다시 논의하지 않기 위한 기록

### 저장소를 EC2 로컬 디스크로 옮기는 안 → **기각**

- **워커가 stateless가 아니게 된다**: EBS는 한 번에 한 인스턴스에만 붙는다(공유 드라이브가 아님).
  워커를 2대 이상 띄우면 event 파일이 있는 머신으로 메시지를 보낼 방법이 없다(SQS는 라우팅 불가).
  워커 1대로 제한하면 수평 확장 불가 + 단일 장애점.
- **재생성 불가능한 데이터다**: `.npz`엔 임베딩뿐 아니라 `cluster_ids`(앨범 번호 연속성)와
  사용자 보정 제약이 들어 있다. 임베딩은 사진에서 다시 뽑으면 되지만 **사용자 결정은 복원 불가**.
  EBS는 인스턴스 교체(재배포·오토스케일링·스팟 회수) 시 기본 설정으로 함께 삭제된다.
- **성능 이득이 없다**: `.npz`는 수십~수백 KB, 동일 리전 S3 왕복은 수십 ms. 사진 1장 추론이 수 초라
  전체의 0.1%도 안 된다.
- EFS는 공유되지만 상시 요금 + 락 직접 구현 → S3가 하는 일을 더 비싸게 다시 하는 셈.

> **한 EC2에 워커 프로세스 여러 개를 띄우는 것은 별개 문제이며, 권장 방식이다**
> ([worker-scaling-and-performance.md](../guides/worker-scaling-and-performance.md)). 저장소를
> 디스크로 바꿀 필요가 없다 — 프로세스 다중화와 S3 저장소는 서로 독립이다.

### FastAPI 복원 → **불필요**

이 워크로드는 서비스가 아니라 **큐 워커**다. HTTP로 만들었다면 비동기 잡 API + 재시도 + 백프레셔 +
DLQ를 직접 구현해야 했고, SQS가 그걸 전부 제공한다(`worker.py`의 오류 정책이 이를 활용 중).
HTTP가 필요한 유일한 자리는 헬스체크인데, 워커에겐 큐 지표가 더 정확한 신호다(7번).
