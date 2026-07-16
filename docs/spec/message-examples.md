# SQS 메시지 예시 모음 (팀 공유용)

> Spring ↔ AI 워커 사이의 wire 계약 예시. 형식 정의의 원천은 [feature-spec.md §6](feature-spec.md#6-메시지-스키마),
> 스키마 구현은 [`app/schemas/messages.py`](../../app/schemas/messages.py)다.
> 아래 예시는 스키마의 `__main__` 스모크 픽스처와 동일 형태로 유지한다 (`python -m app.schemas.messages`로 검증).

**공통 규칙**

- 필드명은 **snake_case** (Spring은 SNAKE_CASE 직렬화 전략 설정 필요).
- 인바운드 3종(분류·보정·삭제)은 **단일 SQS FIFO 큐**로 수신하며 `messageGroupId = event_id`로 같은
  이벤트의 쓰기를 직렬화한다 ([ADR 007](../decisions/007-embedding-storage-s3.md)).
- 메시지 종별은 body의 **`type` 필드**로 판별한다: `classify_request` | `cluster_feedback` | `delete_request`.
- 모든 인바운드는 **`job_id`**를 가진다 — 멱등 처리 키이자, 처리 결과가 같은 `job_id`의
  classify-result로 발행되는 상관관계 키.
- 미지의 필드는 거부된다(`extra="forbid"`) — 계약 변경은 반드시 양쪽 합의 후 반영.

---

## ① 분류 요청 — `classify_request` (Spring → AI)

```jsonc
{
  "type": "classify_request",
  "job_id": "3f2b9c1e-8d4a-4f6b-9a01-5c7e2d8b4a10",     // 멱등 키 — 재수신 시 중복 분석 방지
  "group_id": "7a1d4e92-3b5f-4c8a-b6d2-090f1e3a5c77",   // 모임 (멤버십·공유·접근 컨테이너)
  "event_id": "b8e2f014-6c3d-4a9b-8e57-2d1f0a9c6b34",   // 이벤트 = 재군집 격리 단위 = FIFO messageGroupId
  "images": [
    { "image_id": "c1a2b3d4-0001-4000-8000-000000000001", "s3_key": "groups/7a1d.../events/b8e2.../IMG_0001.jpg" },
    { "image_id": "c1a2b3d4-0002-4000-8000-000000000002", "s3_key": "groups/7a1d.../events/b8e2.../IMG_0002.jpg" }
  ],
  "options": {                        // 업로드 화면의 품질 제외 토글 — 생략 시 둘 다 true (기본 ON)
    "exclude_eyes_closed": true,      // 눈감은 사진 → eyes_closed 앨범으로 분리
    "exclude_blurry": true            // 흔들린 사진 → blurry 앨범으로 분리
  }
}
```

- `s3_key`는 URL이 아니라 **S3 객체 key** — 원본 업로드·저장은 Spring 소유, 워커는 읽기만 한다.
- 최초/증분 구분 플래그는 없다 — 워커는 항상 `event_id`의 전체 임베딩(기존+신규)을 재군집한다.
- `images` 안의 `image_id` 중복은 거부된다 (event `.npz` 멱등 append 보호).

## ② 사용자 보정 — `cluster_feedback` (Spring → AI, action 4종)

사용자가 앱에서 인물을 병합·분리하거나 사진을 옮기면 발행한다. 워커는 must-link/cannot-link
제약으로 저장해 이후 재군집이 사람의 결정을 뒤집지 않게 한다.

```jsonc
// action="merge": 여러 인물(B·C)을 하나(A)로 병합
{
  "type": "cluster_feedback",
  "job_id": "9d0e1f2a-3b4c-4d5e-8f60-000000000001",
  "event_id": "b8e2f014-6c3d-4a9b-8e57-2d1f0a9c6b34",
  "action": "merge",
  "merge": { "target_cluster_id": "person-A", "source_cluster_ids": ["person-B", "person-C"] },
  "split": null,        // action과 무관한 payload 키는 생략하거나 null (Jackson 기본 직렬화 호환)
  "reassign": null
}
```

```jsonc
// action="split": 한 인물 앨범을 사용자 지정 그룹들로 분리 (그룹 2개 이상, 그룹 간 image_id 중복 금지)
{
  "type": "cluster_feedback",
  "job_id": "9d0e1f2a-3b4c-4d5e-8f60-000000000002",
  "event_id": "b8e2f014-6c3d-4a9b-8e57-2d1f0a9c6b34",
  "action": "split",
  "split": { "cluster_id": "person-A", "groups": [["img-1", "img-2"], ["img-3"]] }
}
```

```jsonc
// action="reassign": 사진 1장을 다른 인물로 이동 (from ≠ to)
{
  "type": "cluster_feedback",
  "job_id": "9d0e1f2a-3b4c-4d5e-8f60-000000000003",
  "event_id": "b8e2f014-6c3d-4a9b-8e57-2d1f0a9c6b34",
  "action": "reassign",
  "reassign": { "image_id": "img-7", "from_cluster_id": "person-A", "to_cluster_id": "person-B" }
}
```

> **uncertain 사진을 인물 앨범으로 편입**: `reassign`의 `from_cluster_id`에 인물 `cluster_id` 대신 예약 id
> **`"__uncertain__"`**(결과의 `uncertain[].album_id` 값)을 넣으면, "분류가 어려워요"에 있던 사진을 인물 앨범으로
> 옮긴다. uncertain 얼굴은 실 `cluster_id`가 없어(`.npz`엔 `None`) 일반 reassign 대상이 못 되므로, 이 가상 앨범을
> 출처로 인정한다. 워커는 해당 사진의 미매칭 얼굴을 `to_cluster_id`와 must-link해 재군집이 이 결정을 유지하게 한다.
>
> ```jsonc
> { ..., "action": "reassign",
>   "reassign": { "image_id": "img-6", "from_cluster_id": "__uncertain__", "to_cluster_id": "person-A" } }
> ```

```jsonc
// action="confirm_distinct": 이미 분리된 인물 앨범 여러 개를 서로 다른 사람으로 확정 (2개 이상, 계약 확장)
{
  "type": "cluster_feedback",
  "job_id": "9d0e1f2a-3b4c-4d5e-8f60-000000000004",
  "event_id": "b8e2f014-6c3d-4a9b-8e57-2d1f0a9c6b34",
  "action": "confirm_distinct",
  "confirm_distinct": { "cluster_ids": ["person-A", "person-B"] }
}
```

> **왜 필요한가**: merge의 반대 방향 선언이다. must-link는 "같이 있어야 한다"만 강제할 뿐 "떨어져 있어야
> 한다"는 강제하지 못한다 — 그래서 사용자가 서로 다른 사람으로 검토·확정한 두 인물 앨범 사이로 유사도가
> 애매한 신규 사진(다리 사진)이 들어오면, 다음 전체 재군집이 둘을 하나로 오병합할 위험이 있다.
> `confirm_distinct`는 `cluster_ids`의 대표 얼굴 전 쌍에 cannot-link를 걸어 이후 어떤 재군집에서도 이
> 클러스터들이 합쳐지지 않게 고정한다(대표 한 쌍만으로 충분 — cannot-link 강제가 나머지 멤버를 최근접
> 앵커로 재배정한다). `cluster_ids` 중 이미 사라진 id는 경고 후 무시, 유효 id가 2개 미만이면 전체 무시.

## ③ 하드 삭제 — `delete_request` (Spring → AI)

사진 하드 삭제 요청. 삭제 메시지 자체가 트리거라 재업로드를 기다리지 않는다 — 워커(이벤트 `.npz`의
단일 writer)가 마스킹 rewrite로 임베딩을 물리 제거한다 ([ADR 007](../decisions/007-embedding-storage-s3.md)).

```jsonc
{
  "type": "delete_request",
  "job_id": "5e6f7a8b-9c0d-4e1f-8a2b-000000000001",
  "event_id": "b8e2f014-6c3d-4a9b-8e57-2d1f0a9c6b34",
  "image_ids": ["c1a2b3d4-0001-4000-8000-000000000001", "c1a2b3d4-0002-4000-8000-000000000002"]
}
```

## ④ 분류 결과 — `classify-result` (AI → Spring, 결과 큐)

인바운드 3종(분류·보정·삭제) 모두 처리 결과를 이 형식으로 발행한다. `job_id`는 요청과 동일한 값이다.
결과 필드는 앱의 앨범 5종과 1:1 대응하며, 이 중 person(`clusters`)·common(`common_album`)만 뷰어에 노출된다.

```jsonc
{
  "job_id": "3f2b9c1e-8d4a-4f6b-9a01-5c7e2d8b4a10",
  "status": "succeeded",                          // "succeeded" | "partial" | "failed"
  "clusters": [                                   // person 앨범 — cluster_id = 앱의 personId
    {
      "cluster_id": "person-A",                   // 기존 번호 승계 또는 신규 발급 (AI는 이름을 모른다)
      "is_new": false,                            // 이번에 새로 생긴 인물인지
      "image_ids": ["img-1", "img-2", "img-7"],   // 한 사진이 여러 인물에 속할 수 있음 (N:M)
      "representative_vector": [0.0123, -0.0456 /* …총 512개 float (L2 정규화 평균, 표시용 파생값) */]
    }
  ],
  "common_album": ["img-9"],                      // 인물 귀속 불가 (단체·배경·얼굴 미검출) — 뷰어 노출
  "uncertain": [                                  // "분류가 어려워요" — 뷰어 비노출
    // album_id: 이 사진을 인물 앨범으로 옮길 때 reassign의 from_cluster_id로 되돌려줄 예약 앨범 id
    { "image_id": "img-5", "reason": "ambiguous", "album_id": "__uncertain__" },  // 두 인물 사이 저신뢰
    { "image_id": "img-6", "reason": "unmatched", "album_id": "__uncertain__" }   // 얼굴은 있으나 인물 미매칭 (예: 행인)
  ],
  "eyes_closed": ["img-3"],                       // exclude_eyes_closed=ON일 때만 — 뷰어 비노출
  "blurry": ["img-4"],                            // exclude_blurry=ON일 때만 — 뷰어 비노출
  "failed_images": [                              // 기술적 실패 (재시도 대상, 앨범 아님)
    { "image_id": "img-8", "reason": "timeout" }
  ],
  "retired_cluster_ids": ["person-C"]             // 이번 재군집에서 은퇴한 인물 번호 — 앨범 정리용
}
```

- `status: "failed"`일 때는 `job_id`·`status` 외 전 필드가 빈 리스트일 수 있다.
- `representative_vector`는 항상 512-dim, NaN/inf 없음.
- 실패 케이스를 포함한 계약 검증 전체는 `python -m app.schemas.messages`로 실행할 수 있다.

## ⑤ 분류 진행률 — `progress` (AI → Spring, progress 큐, CHMO-274)

`classify_request` 처리 중 워커가 이미지 루프를 도는 **도중** 처리 장수를 이 progress 큐로 여러 번
발행한다(결과 큐가 아니다). 백엔드는 이 값을 메모리에 들고 있다가 FE 폴링에 분류 진행바로 응답한다.
progress 큐가 설정되지 않은 배포에서는 이 메시지가 발행되지 않는다(기능 비활성).

```jsonc
{
  "type": "progress",                              // 결과 메시지와 큐가 다르지만 판별 필드를 갖는다
  "job_id": "3f2b9c1e-8d4a-4f6b-9a01-5c7e2d8b4a10", // 요청·결과와 동일한 상관관계 키
  "event_id": "event-1",
  "processed": 30,                                 // 지금까지 처리한 이미지 수
  "total": 300                                     // 이 job의 전체 이미지 수 → 진행률 = processed/total
}
```

- 한 job에서 `processed`는 `0 → … → total`로 **단조 증가**한다(루프 진입 시 `0/total` 1회 + 처리
  3장마다 + 마지막 `total/total`). SQS 표준 큐는 순서 보장이 없고 at-least-once이므로, 백엔드는 `processed`를 순서·중복
  방어 키로 쓴다 — **마지막으로 본 값 이하의 메시지는 버린다**(job 재시도로 0부터 다시 와도 안전).
- 진행률은 유실돼도 되는 부수 신호라 발행은 best-effort다(발행 실패가 classify를 죽이지 않는다).
- `보정(cluster_feedback)`·`삭제(delete_request)`는 이미지 루프가 없어 progress를 발행하지 않는다.
