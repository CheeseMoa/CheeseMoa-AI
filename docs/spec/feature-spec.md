# 치즈모아 AI 서버 기능명세서

> **한 줄 요약** — 이 서버는 AWS SQS로 받은 사진 묶음을 얼굴 인식 파이프라인으로 처리해, 인물별 클러스터 결과를 결과 큐로 돌려주는 **비동기 AI 워커**다.

| 항목 | 내용 |
|------|------|
| 레포 | CheeseMoa-AI (Python 워커 기반 AI 추론 서버) |
| 연동 | AWS SQS 비동기 단일 경로 (Spring producer → AI consumer → 결과 큐) |
| 실행 구조 | Python 워커 프로세스(SQS consumer) 단일 실행. HTTP 미제공 |
| 핵심 알고리즘 | 전체 재군집(HDBSCAN) + 기존 `cluster_id` 재조정 |
| 저장소 | S3 (event 단위 `.npz`: 개별 임베딩 + 직전 `cluster_id` + 보정 제약) — [ADR 007](../decisions/007-embedding-storage-s3.md) |

---

## 1. 서버의 책임 범위

| 한다 ✅ | 하지 않는다 ❌ |
|---------|----------------|
| 얼굴 감지·정렬·임베딩·클러스터링 | 사용자 인증 (Spring) |
| 전체 재군집 + `cluster_id` 재조정 | 이미지 업로드·저장 (S3는 Spring) |
| 개별 얼굴 임베딩 저장·조회 (S3 `.npz`) | 앨범 메타데이터·공개 상태 관리 (PostgreSQL은 Spring) |
| 사용자 보정(병합/분리/이동)을 대표벡터에 반영 | 사람의 검토(미검토/검토완료) 상태 관리 |
| 분류 결과 발행 (SQS 결과 큐) | 결제·권한·모임 운영 |

---

## 2. 연동 구조 (AWS SQS)

```text
Spring (producer)
   │  ① 분류 작업·사용자 보정(병합/분리/이동)·하드 삭제를 입력 큐에 발행
   ▼
[입력 큐 FIFO]  classify_request · cluster_feedback · delete_request
   │            (messageGroupId = event_id, body `type` 필드로 판별)
   ▼
치즈모아 AI 워커 (consumer)  ── 처리 실패(redrive) ──▶ [DLQ] classify-dlq
   │  ② AI 파이프라인 실행 / 대표벡터 갱신
   │  ③ 결과 큐에 발행
   ▼
[결과 큐] classify-result
   │
   ▼
Spring (consumer)  ④ 결과 구독 → DB 반영
```

- 분류 요청은 **HTTP로 받지 않는다.** 오직 SQS로만 수신한다.
- AI 워커는 입력 메시지를 소비한다: 분류 작업(`classify-request`)·사용자 보정(`cluster-feedback`)·하드 삭제. 셋 다 Spring이 발행하고, 이 워커가 이벤트 `.npz`의 **단일 writer**로 처리한다(별도 Lambda 없음).
- 입력 큐는 **SQS FIFO**(`messageGroupId = event_id`)로 구성해 같은 이벤트의 쓰기를 직렬화한다 — 동시 rewrite로 인한 lost update(삭제가 되살아나는 PIPA 재발)를 막는다([ADR 007](../decisions/007-embedding-storage-s3.md)). 삭제 메시지 자체가 트리거라, 재업로드가 없어도 상시 폴링 워커가 처리한다(사각지대 없음).
- 이 서버는 HTTP를 제공하지 않는다. 운영 헬스체크는 프로세스 liveness로 확인한다(8장).
- DLQ는 SQS redrive policy로 구성한다(수신 횟수 초과 시 `classify-dlq`로 이동).

---

## 3. 실행 구조

| 컴포넌트 | 역할 | 진입점 |
|----------|------|--------|
| **AI 워커** | SQS consumer. 파이프라인 실행의 본체 (단일 실행 단위) | `app/worker.py` |
| **공통 파이프라인** | 워커가 사용하는 AI 로직 | `app/pipeline/*` |

> AI 워커는 SQS 메시지를 폴링해 파이프라인을 실행하는 **단일 Python 프로세스**다.
> HTTP 서버(FastAPI 등)는 두지 않는다. AI 추론은 CPU/GPU 바운드라 워커 프로세스로 격리해 블로킹·장애를 관리한다.

---

## 4. 핵심 알고리즘 — 전체 재군집 + ID 재조정

**군집의 진실은 항상 그 이벤트의 전체 임베딩(기존 + 신규)에 대한 HDBSCAN 재군집이다.**
증분 매칭이 아니라, 개별 임베딩을 event 단위 S3 `.npz`에 전부 보관해 매 트리거마다 전체를 다시 군집화한다.
정확도가 최우선이며, 클러스터링 연산은 전체 파이프라인 비용의 0.1% 미만이라 재군집 비용은 부담이 없다.
인물 앨범은 모임 안의 이벤트 단위로 만들어지므로 재군집 격리 단위는 event다([ADR 007](../decisions/007-embedding-storage-s3.md)).

```text
① 신규 사진 임베딩 생성        detect → align → embed (512-dim) → S3 `.npz` 저장
        │
② 전체 임베딩 로드             event_id의 기존 + 신규 임베딩 전부 (개별 벡터)
        │
③ 전체 재군집                  HDBSCAN (전체 벡터, cosine)
        │   → 이번 실행의 파티션 P_new (사용자 보정은 제약으로 강제, 아래 참고)
        ▼
④ 기존 ID에 재조정(reconcile)  P_new ↔ 기존 클러스터를 멤버 overlap 최대 매칭으로 연결
        │   ├─ 대응되는 기존 클러스터 → 그 `cluster_id` 승계 (연속성 유지)
        │   ├─ 대응 없는 새 군집       → 신규 `cluster_id` 부여 (`is_new: true`)
        │   └─ 멤버가 사라진 기존 군집  → 은퇴
        ▼
⑤ 대표벡터 계산·결과 발행       각 클러스터 대표벡터 = 멤버 임베딩 L2 정규화 평균(파생값) → 결과 메시지
   (재군집 후 새 cluster_id·삭제 반영을 event `.npz`에 rewrite)
```

**설계 원칙**
- **재군집이 원천, 대표벡터는 파생 캐시.** 대표벡터(centroid)는 빠른 근사 조회·Spring 표시용일 뿐 군집 판단의 근거가 아니다. 판단은 전체 임베딩 밀도(HDBSCAN)로 한다.
- **연속성은 재조정으로 확보.** 전체 재군집은 파티션이 실행마다 바뀔 수 있으므로, overlap 매칭(Jaccard / 헝가리안)으로 기존 `cluster_id`를 승계해 업로드 간 같은 사람을 같은 번호로 유지한다.
- **사용자 보정은 제약으로.** merge/split/reassign/confirm_distinct(6.3)은 must-link / cannot-link 제약(또는 재군집 후 강제 후처리)으로 반영해, 재군집이 사람의 결정을 절대 뒤집지 않게 한다.
- **대표벡터 정의**: 멤버 임베딩의 L2 정규화 평균. (이상치 강건성이 필요하면 medoid 검토 — 현재 미채택)

> **규모 탈출구**: 재군집 격리 단위가 event(이벤트당 수백 벡터)라 규모 부담은 사실상 없다. 극단적으로 한 이벤트가 수만+로 커지는 경우에만, 증분 근사 매칭을 fast-path로 쓰고 **주기적 전체 재군집(§10 #1)**으로 드리프트를 교정하는 탈출구가 있다. 이 경우에도 군집의 정답 기준은 전체 재군집이다.

---

## 5. 저장소 책임 분리

```text
AI 서버 (S3 .npz, event 단위)          Spring (PostgreSQL)
────────────────────────────          ───────────────────
• 개별 얼굴 임베딩 전부           ↔    • 대표벡터 사본 (앱 조회용)
• 직전 cluster_id + 보정 제약           • 앨범·인물·공개상태 등 메타데이터
• 전체 재군집·재조정·대표벡터 계산       • 결과 메시지로 대표벡터 수신·저장
```

- **AI 서버 S3 `.npz`** = 군집 계산의 원천. 이벤트당 한 객체(`{eventId}.npz`)에 임베딩·id·직전 `cluster_id`·보정 제약을 함께 담는다. **개별 임베딩 전부를 보관하는 것이 전체 재군집의 전제**다(요약본만으론 재군집 불가). 레이아웃·삭제·동시성 상세: [ADR 007](../decisions/007-embedding-storage-s3.md).
- 대표벡터는 재군집 때마다 재계산되는 **파생값**이라 AI가 굳이 저장하지 않는다 — 결과 메시지로 Spring에 보내 표시용으로만 쓴다.
- **Spring Postgres** = 앱이 쓰는 대표벡터 + 비즈니스 메타데이터. 결과 큐로 받아 저장.
- **인물 식별의 연속성은 AI 몫**: AI가 전체 재군집 후 **overlap 재조정으로 기존 `cluster_id`를 승계**해 같은 사람을 (이벤트 범위 내) 업로드 간 같은 번호로 유지해 보낸다. Spring은 `cluster_id`(앱의 `personId`) ↔ 이름(예: "민준") 매핑만 담당하며 **AI는 이름을 모른다**. 즉 사진을 사람별로 묶는 일은 AI가 끝내고, Spring은 번호표에 이름표만 붙인다(Spring이 다시 군집하지 않음).
- **저장소 선택 근거**: 매 트리거마다 이벤트 전체를 통짜 로드해 재군집하므로 **ANN 검색(pgvector의 핵심)을 쓰지 않고**, 이벤트당 수백 규모라 벡터 인덱스 이점이 없다. 상시 요금 없는 S3 blob이 버스트 워크로드에 적합하다([ADR 007](../decisions/007-embedding-storage-s3.md)). 저장소 접근은 인터페이스로 추상화한다.

---

## 6. 메시지 스키마

> 인바운드 3종(분류·보정·삭제)은 단일 FIFO 큐로 수신하며 body의 **`type` 필드**로 판별한다
> (`classify_request` | `cluster_feedback` | `delete_request`). 모든 인바운드는 **`job_id`**를
> 가진다 — 멱등 처리 키이자, 처리 결과가 같은 job_id의 classify-result로 발행되는 상관관계 키.
> 필드명은 snake_case. 스키마 구현: [`app/schemas/messages.py`](../../app/schemas/messages.py),
> 팀 공유용 예시 모음: [message-examples.md](message-examples.md).

### 6.1 요청 메시지 (classify-request)

```jsonc
{
  "type": "classify_request",
  "job_id": "uuid",          // 작업 식별자 (멱등 처리 키)
  "group_id": "uuid",        // 모임 ID (상위 컨테이너: 멤버십·공유·접근)
  "event_id": "uuid",        // 이벤트 ID (재군집 격리 단위 = 저장 .npz 키)
  "images": [
    { "image_id": "uuid", "s3_key": "string" }
  ],
  "options": {                    // 업로드 화면의 품질 제외 토글 (기본 ON)
    "exclude_eyes_closed": true,  // 눈감은 사진 → eyes_closed 앨범으로 분리
    "exclude_blurry": true        // 흔들린 사진 → blurry 앨범으로 분리
  }
}
```

> 증분/최초 분석을 별도 플래그로 구분하지 않는다. 서버는 항상 `event_id`의 전체 임베딩
> (기존 + 신규)을 로드해 **전체 재군집**한다. 기존 클러스터가 없으면 최초 군집, 있으면
> 재군집 후 `cluster_id` 재조정으로 이어진다(4장).
>
> `options`는 업로드 화면의 "눈감은 사진 제외 / 흔들린 사진 제외" 토글(기본 ON)에 대응한다.
> ON이면 해당 사진을 인물 앨범 대신 `eyes_closed`/`blurry` 앨범으로 라우팅하고, OFF면
> 품질 사유로 분리하지 않고 인물 군집에 남긴다(6.2).

### 6.2 결과 메시지 (classify-result)

```jsonc
{
  "job_id": "uuid",
  "status": "succeeded",     // "succeeded" | "partial" | "failed"
  "clusters": [
    {
      "cluster_id": "uuid",      // 대표벡터와 1:1 (신규/기존 모두). 앱 person 앨범의 personId
      "is_new": true,            // 이번에 새로 생긴 인물인지
      "image_ids": ["uuid"],     // 한 image_id가 여러 클러스터에 속할 수 있음(사진↔앨범 N:M)
      "representative_vector": [0.01]  // 512-dim 대표벡터 (§4 ⑤, L2 정규화 평균) — Spring 표시용 사본
    }
  ],
  "common_album": ["uuid"],     // common 앨범 — 단체 사진(얼굴 2명+)·배경·얼굴 미검출. 뷰어 노출
  "uncertain": [                // uncertain("분류가 어려워요") — 뷰어 비노출
    // album_id: 인물 앨범 편입 시 reassign의 from_cluster_id로 되돌려줄 예약 앨범 id ("__uncertain__")
    { "image_id": "uuid", "reason": "ambiguous", "album_id": "__uncertain__" }  // "ambiguous"(저신뢰) | "unmatched"
  ],
  "eyes_closed": ["uuid"],      // eyes_closed 앨범 — exclude_eyes_closed=ON일 때만. 뷰어 비노출
  "blurry": ["uuid"],           // blurry 앨범 — exclude_blurry=ON일 때만. 뷰어 비노출
  "failed_images": [            // 기술적 분석 실패 (재시도 대상, 위 앨범들과 별개)
    { "image_id": "uuid", "reason": "timeout" }
  ],
  "retired_cluster_ids": ["uuid"]  // 이번 재군집에서 승계되지 못해 은퇴한 기존 cluster_id (4장 ④)
}
```

- 결과 필드는 앱의 **앨범 5종**과 1:1 대응한다: `clusters`→`person`, `common_album`→`common`, `uncertain`→"분류가 어려워요", `eyes_closed`→"눈감은 사진", `blurry`→"흔들린 사진". 이 중 **`person`·`common`만 뷰어에 노출**되고 나머지는 내부 검수용이다.
- `clusters`: 자신 있게 인물로 묶인 결과(앱 `person` 앨범, `cluster_id` = `personId`). **AI는 신뢰도 등급(`tier`)을 부여하지 않는다** — 사람의 `미검토/검토완료`는 Spring/앱의 상태이지 AI 출력이 아니다. 한 사진에 여러 인물이 있으면 각 인물 클러스터의 `image_ids`에 모두 넣는다(다대다).
  - **단일 사진 클러스터 강등**(결정 2026-07-03): 한 장의 사진 안 얼굴들로만 구성된 군집은 인물로 승격하지
    않는다 — 같은 사진에 같은 인물이 두 번 나올 수 없으므로 우연히 닮은 타인들이다(대형 단체 사진 실측에서
    재현). 강등된 얼굴은 미매칭으로 처리한다. 사용자 보정(제약) 당사자가 포함된 군집은 사람의 결정이므로 예외.
- `uncertain`: 인물에 자신 있게 못 붙인 사진("분류가 어려워요", 내부 검수용). `reason`은
  `ambiguous`(두 인물 사이 저신뢰)와 `unmatched`(얼굴은 검출됐으나 어느 인물과도 매칭되지 않음 — 예: 행인).
  **얼굴 미검출(인물 없는) 사진은 uncertain이 아니라 `common_album`으로 확정**(결정 2026-07-03).
  - **단체 사진 = 공용 앨범**(결정 2026-07-20, `CLUSTER_GROUP_PHOTO_TO_COMMON=true` 기본): 검출 얼굴이
    **2개 이상인 사진은 매칭 여부와 무관하게 `common_album`에 노출**한다 — 단체 사진은 그 자리에 함께
    있던 모두의 사진이라는 제품 결정. 얼굴이 인물에 매칭되면 해당 인물 앨범에도 함께 들어간다(다대다,
    공용과 인물 앨범 **중복 노출**). 얼굴 1개(행인·미등록 1인) 미매칭 사진만 `uncertain(unmatched)`으로 보낸다.
    - **롤아웃 스위치**: `common_album`이 이제 단체 사진 전부를 포함하도록 의미가 넓어졌다(종전엔 전원
      미매칭만) — Spring/앱이 이를 감당할 준비가 될 때까지 `CLUSTER_GROUP_PHOTO_TO_COMMON=false`로 배포하면
      구 정책(전원 미매칭인 2+ 사진만 공용, 매칭되면 인물 앨범만)으로 되돌아간다.
    - **Spring 이중 커버리지(의도됨, 지우지 말 것)**: Spring은 별도로 "여러 인물 앨범에 중복으로 뜨는
      사진(=둘 이상 매칭된 단체 사진)을 `common`에도 복제"하는 로직을 갖고 있다. AI 정책과 겹치지만
      공용 앨범 멤버십은 집합이라 멱등해 무해하다. 두 경로의 커버리지는 정확히 같지 않다 — AI는 얼굴
      개수를 알아 **한 명만 매칭된 단체 사진**까지 공용에 넣지만, Spring의 "앨범 중복" 추론은 이 케이스를
      (앨범 하나에만 떠서) 못 잡는다. 어느 한쪽을 "중복"으로 보고 제거하면 이 케이스가 공용에서 누락된다.
  - **인물 앨범 편입**(계약 확장): 각 uncertain 항목은 예약 앨범 id `album_id`(`"__uncertain__"`)를 함께 싣는다.
    사용자가 이 사진을 인물 앨범으로 옮기면 Spring이 `reassign(from_cluster_id=이 값)`으로 되돌려주고(6.3),
    워커가 해당 사진의 미매칭 얼굴을 must-link해 편입한다 — uncertain 얼굴은 실 `cluster_id`가 없어(`.npz`엔
    `None`) 일반 reassign 대상이 못 되므로 이 가상 앨범을 출처로 인정한다.
  - (TBD: `back`(뒷모습)·`duplicate`(중복)를 `uncertain` 사유로 추가할지는 백엔드와 합의)
- `eyes_closed` / `blurry`: **눈감음·흔들림은 "분류가 어려워요"와 별개의 독립 앨범**이다(제품 명세 정정 반영). 업로드 토글(6.1 `options`)이 ON일 때만 인물 앨범 대신 이 앨범으로 라우팅하고, OFF면 분리하지 않는다.
- `failed_images`: 타임아웃 등 **기술적 실패**. 화질·매칭 문제인 위 앨범들과 구분한다(재시도 대상).
- `representative_vector`·`retired_cluster_ids`: 대표벡터는 표시용 파생값 사본(AI는 저장하지 않음, [ADR 007](../decisions/007-embedding-storage-s3.md)), 은퇴 id는 Spring이 해당 인물 앨범을 정리하는 데 쓴다.

### 6.3 보정 피드백 메시지 (cluster-feedback)

사용자가 앱에서 인물을 병합·분리하거나 사진을 다른 인물로 옮기면, Spring이 그 사실을 발행한다.
AI 워커는 이를 받아 **must-link / cannot-link 제약으로 저장**(해당 이벤트 `.npz`, face_id 참조)해, 다음 전체 재군집이 이 결정을 유지하도록 한다. 저장된 제약은 매 재군집마다 현재 행 인덱스로 번역돼 강제된다(4장·[ADR 007](../decisions/007-embedding-storage-s3.md)).

```jsonc
{
  "type": "cluster_feedback",
  "job_id": "uuid",    // 멱등 키 — 처리 결과는 이 job_id의 classify-result로 발행
  "event_id": "uuid",  // 보정 대상 이벤트 (클러스터가 속한 재군집 단위)
  "action": "merge",   // "merge" | "split" | "reassign" | "confirm_distinct"

  // action="merge": 여러 클러스터를 하나로
  "merge":    { "target_cluster_id": "uuid", "source_cluster_ids": ["uuid"] },
  // action="split": 한 클러스터를 사용자 지정 그룹으로 분리
  "split":    { "cluster_id": "uuid", "groups": [["image_id"], ["image_id"]] },
  // action="reassign": 특정 사진을 다른 인물로 이동. from_cluster_id에 예약 id "__uncertain__"을 주면
  //                     "분류가 어려워요"(uncertain)에 있던 사진을 인물 앨범으로 편입한다 (6.2 결과 필드 참조)
  "reassign": { "image_id": "uuid", "from_cluster_id": "uuid", "to_cluster_id": "uuid" },
  // action="confirm_distinct": 이미 분리된 클러스터 여러 개를 서로 다른 사람으로 확정 (계약 확장, 아래 註)
  "confirm_distinct": { "cluster_ids": ["uuid", "uuid"] }
}
```

> 처리 결과(갱신된 대표벡터/클러스터)는 동일하게 `classify-result` 형식으로 결과 큐에 발행해 Spring이 동기화한다.
> `action`과 무관한 payload 키(예: merge 메시지의 `split`·`reassign`)는 생략하거나 `null`이어야 한다
> (Jackson 기본 직렬화의 null 동봉 허용).

> **註 — `confirm_distinct` (계약 확장)**: merge의 반대 방향 선언이다. must-link는 "같이 있어야 한다"만
> 강제할 뿐 "떨어져 있어야 한다"는 강제하지 못해, 사용자가 서로 다른 사람으로 검토·확정한 두 인물 앨범
> 사이로 유사도가 애매한 신규 사진(다리 사진)이 들어오면 전체 재군집(4장)이 둘을 하나로 오병합할 수
> 있다. `confirm_distinct`는 `cluster_ids`(2개 이상)의 대표 얼굴 전 쌍에 cannot-link를 걸어, 이후 어떤
> 전체 재군집에서도 이 클러스터들이 하나로 합쳐지지 않게 한다. 대표 한 쌍만으로 충분한 이유는 재군집의
> cannot-link 강제(4장)가 위반 라벨을 쪼갠 뒤 제약 없는 나머지 멤버를 최근접 앵커로 재배정하기
> 때문이다(split의 그룹 간 cannot-link와 동일 원리). `cluster_ids` 중 이미 사라진 id는 경고 후
> 무시하며, 유효 id가 2개 미만이면 전체를 무시한다(다른 action의 stale 처리와 동일 정책).

### 6.4 하드 삭제 메시지 (delete-request)

사용자가 사진을 하드 삭제하면 Spring이 발행한다. **삭제 메시지 자체가 트리거**이므로 재업로드를
기다리지 않는다 — 워커(이벤트 `.npz`의 단일 writer)가 마스킹 rewrite로 해당 임베딩과 이를 참조하는
보정 제약을 물리 제거하고, 갱신된 결과를 `classify-result`로 발행한다
([ADR 007](../decisions/007-embedding-storage-s3.md)).

```jsonc
{
  "type": "delete_request",
  "job_id": "uuid",
  "event_id": "uuid",
  "image_ids": ["uuid"]   // 삭제 대상 사진
}
```

---

## 7. 정책 반영 (치즈모아 정책 → 서버 동작)

이 서버의 출력은 치즈모아 서비스 정책을 따른다. **정책 위반 출력은 백엔드가 거부**하므로 필수.

| 정책 | 서버 동작 |
|------|-----------|
| 저신뢰 매칭 사진 | 인물에 자신 있게 못 붙인 사진은 `uncertain`("분류가 어려워요")에 `reason: ambiguous`로. **매칭 임계값은 설정값(`core/config.py`), 하드코딩 금지** |
| 눈감음·흔들림 제외 (업로드 토글) | `options.exclude_eyes_closed`/`exclude_blurry`(기본 ON)면 해당 사진을 인물 앨범 대신 `eyes_closed`/`blurry` **별도 앨범**으로 분리. OFF면 분리 안 함 (판정 방식·한계는 아래 註) |
| 자동삭제 금지 | `uncertain`·`eyes_closed`·`blurry`는 분류·표시만 — 삭제는 사용자/백엔드 몫 |
| 단체·배경·인물 없는 사진 = 공통 사진첩 | 얼굴 2명+ 단체 사진(매칭돼도)·배경·얼굴 미검출은 `common_album`으로 (6.2, 롤아웃 스위치) |
| 분석 중 비노출 | `status: succeeded`일 때만 결과 발행. 부분 결과 노출 금지 |
| 새 인물 = 임시 라벨 | 서버는 `cluster_id`+`is_new`만 부여. 이름 지정은 사용자 |
| 병합/분리/이동 (사용자 보정) | `cluster-feedback`로 받아 must-link/cannot-link 제약으로 반영 → 재군집이 사람 결정을 뒤집지 않음 (4·6.3장) |
| 분석 실패 / 타임아웃 | 이미지 단위 처리, 실패분은 `failed_images`로. 메시지 단위 재처리는 SQS 재시도 → 수신 횟수 초과 시 DLQ |
| 원본 무손실 | 서버는 원본을 변형·삭제하지 않음. 산출물만 생성 |
| 전체 재군집 | 매 트리거마다 event 전체 임베딩 재군집(HDBSCAN) + 기존 `cluster_id` 재조정 (4장) |
| 새벽 배치 (전역 재파라미터·재군집) | 재군집 격리 단위가 event(수백 규모)라 **거의 불필요**. 극단적 대규모 이벤트에서만 fast-path 병용 검토 (§10 #1) |
| 얼굴 임베딩 = 생체정보 | 임베딩·대표벡터 S3 저장 시 암호화·격리 전제. 하드 삭제·보존기간·파기는 [ADR 007](../decisions/007-embedding-storage-s3.md)·TBD |
| 전원 동의 전제 | 미동의자 차단은 백엔드/계약에서 처리 → **얼굴 마스킹 로직 MVP 불필요** |

> **註 — 품질 판정 방식과 한계** (`app/pipeline/quality.py`)
> - **눈감음** ([ADR 021](../decisions/021-blink-blendshape-litert.md)): Face Landmarker blendshape
>   (litert 이식, `app/pipeline/blink.py`) — YuNet 5점 RoI에서 478 랜드마크를 추정해
>   min(eyeBlinkL/R) ≥ `blink_threshold`(0.40)면 그 얼굴이 눈감음. 얼굴 presence < 0.5는 미판정.
>   눈 패치 CNN의 도메인 실패(유아 오탐·보정 이미지 미탐·수면 미탐)를 871 얼굴 A/B로 해소.
>   `QUALITY_BLINK_THRESHOLD=0` 롤백 시에만 종전 CNN(open-closed-eye-0001, 양눈 모두
>   `eye_closed_confidence` 이상 + [ADR 019](../decisions/019-eye-judgment-eligibility-gate.md)
>   자격 게이트) 경로를 쓴다. *한계*: presence 미달 ~9% 미판정(실측 감음 손실 0 — 코퍼스 한정),
>   blendshape는 눈 감김을 문자 그대로 재서 웃으며 감은 캔디드도 잡힌다(임계로 조정 가능).
> - **흔들림**: 검출된 얼굴 bbox 크롭의 Laplacian variance가 `blur_threshold` 미만이면 흔들림. **얼굴이 하나도
>   검출되지 않으면**(완전 흔들려 검출 실패) 전체 이미지 variance(`whole_image_blur_threshold`)로 fallback 판정한다.
>   *한계*: **앞사람만 모션블러이고 배경이 선명한 부분 블러**는 얼굴 미검출 + 전역 variance 높음으로 어느 쪽도
>   못 잡는 사각지대다 (variance 방식의 근본 한계).
> - 이미지 단위 집계는 "얼굴 1개라도 해당하면 분리". 임계값은 전부 `core/config.py` 설정값이며 실측 보정 대상(§10 #3).

---

## 8. 운영 (헬스체크)

HTTP 엔드포인트를 제공하지 않는다. 워커는 프로세스 liveness와 준비 상태로 건강도를 판단한다.

| 항목 | 방식 |
|------|------|
| 프로세스 생존 (liveness) | 컨테이너/오케스트레이터가 워커 프로세스 실행 여부로 판단 |
| 준비 완료 (readiness) | 모델(YuNet·AuraFace) 로딩 + SQS·S3 연결 확인 후 폴링 시작 |

운영 지표(처리량·지연·실패율)는 로그와 CloudWatch로 관측한다.

---

## 9. 예외 처리

- **멱등성**: 동일 `job_id` 재수신 시 중복 분석 방지.
- **재시도**: 메시지 처리 실패는 SQS 가시성 타임아웃 후 재수신, 수신 횟수 초과 시 redrive policy로 **DLQ(`classify-dlq`)**에 격리.
- **부분 실패**: 일부 이미지만 실패하면 작업 전체를 죽이지 않고 `failed_images` + `status: partial`.
- **외부 의존 장애**(S3·모델): 원본 무손실 우선. 임베딩·산출물은 재생성 가능 자원으로 취급.

---

## 10. 미정 항목 (TBD)

| # | 항목 | 비고 |
|---|------|------|
| 1 | **새벽 배치 (대규모 fast-path 드리프트 교정)** | event 단위(수백)라 거의 불필요. 극단적 대규모 이벤트에서만. 여부·주기·방식 미정 |
| 2 | `uncertain` 사유 태그 범위 | 눈감음·흔들림은 별도 앨범, 얼굴 미검출은 `common_album`, 인물 미매칭 노이즈는 `unmatched`로 확정(6.2장). 뒷모습·중복 추가 여부만 잔존 |
| 3 | 저신뢰(애매) 판정 임계값 | 인물 매칭을 `ambiguous`로 떨굴 기준 |
| 4 | HDBSCAN 파라미터 · 재조정 임계 | `min_cluster_size`·`cluster_selection_epsilon` 등 재군집 파라미터, 신·구 클러스터 overlap 매칭 기준(최소 Jaccard), 연결 성분 부분 승격 임계(`blob_promote_similarity`·`blob_promote_floor` — [ADR 008](../decisions/008-blob-promotion-connected-components.md)) |
| 5 | `split` 보정 처리 방식 | 지정 그룹대로 cannot-link 제약 vs 해당 클러스터만 부분 재군집 |
| 6 | 얼굴 임베딩 보존기간·파기 시점 | 생체정보 정책. 하드 삭제 처리(워커가 삭제 메시지 처리, Lambda 불필요)·복원 없음(MVP)은 [ADR 007](../decisions/007-embedding-storage-s3.md) 결정. 보존기간·즉시성 재확인만 남음 |
| 7 | SQS 큐 네이밍 | 메시지 종별은 body `type` 필드로 확정(6장: `classify_request`·`cluster_feedback`·`delete_request`). 큐 리소스 이름만 TBD |

---

## 관련 문서

- 시스템 구조: [architecture/system-overview.md](../architecture/system-overview.md)
- 파이프라인: [architecture/pipeline-overview.md](../architecture/pipeline-overview.md)
- 폴더 구조: [conventions/project-structure.md](../conventions/project-structure.md)
