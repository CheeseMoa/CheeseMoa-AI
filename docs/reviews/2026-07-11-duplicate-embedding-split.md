# 결함 기록: 동일 사진 재업로드가 만든 중복 임베딩이 인물 앨범을 쌍 단위로 쪼갠다 (2026-07-11)

- 상태: **원인 특정 완료, 수정 미착수** (대응 후보는 [backlog](../backlog/2026-07-11-followups.md) P0)
- 증상: 같은 인물 사진만 넣었는데 앨범이 사진 1장씩 여러 개로 쪼개짐 (실 event 8)
- 관련: [ADR 007](../decisions/007-embedding-storage-s3.md)(event `.npz` 단일 writer) ·
  [ADR 009](../decisions/009-clustering-parameter-tuning.md)(클러스터링 파라미터)

---

## 1. 증상

앱의 이벤트 앨범 화면에서 동일 인물(카리나) 사진들이 **인물 앨범 5개로 쪼개져** 각각 "1장"으로
표시됐다. 어제까지는 하나의 앨범으로 정상 묶였다.

## 2. 근인 — `.npz`에 유사도 1.00인 임베딩 쌍이 5개

`s3://{embeddings-bucket}/embeddings/8.npz`(2026-07-11 15:24 저장분)를 디코딩하면 19행 중
**5쌍이 완전히 동일한 임베딩**이다 (코사인 유사도 정확히 1.00):

```
row 0 ≡ row 12,  row 1 ≡ row 13,  row 2 ≡ row 14,  row 3 ≡ row 15,  row 4 ≡ row 16
```

`photo_id`는 쌍마다 서로 다르다 — 즉 **같은 사진이 서로 다른 image_id로 두 번 임베딩됐다.**

원본 버킷의 S3 ETag(콘텐츠 해시)로 확정:

| ETag | 어제 업로드 | 오늘 업로드 |
|---|---|---|
| `b4dfe022…` | 07-10 18:29 | 07-11 15:24 |
| `ede2f0e8…` | 07-10 18:29 | 07-11 15:24 |
| `40b0c4cc…` | 07-10 18:29 | 07-11 15:24 |
| `07b93f3a…` | 07-10 18:29 | 07-11 15:24 |
| `b97892f6…` | 07-10 18:29 | 07-11 15:24 |

오늘 올라온 9장 중 **5장이 어제 파일과 바이트 단위로 동일**하다. 워커의 멱등 스킵은
`image_id` 기준(`handlers.py` `_handle_classify`)이라, Spring이 같은 파일에 새 `image_id`를
부여해 발행하면 그대로 새 얼굴 행이 append된다.

## 3. 왜 쪼개지는가 — HDBSCAN의 밀도 선택

동일 인물 5장의 상호 유사도는 **0.39~0.67**(포즈·조명 변화). 반면 복제 쌍은 **거리 0**이다.
HDBSCAN은 밀도 기반이라 "거리 0의 완벽한 쌍"을 느슨한 인물 덩어리보다 훨씬 안정적인 클러스터로
선택한다. `cluster_selection_epsilon=0.15`로는 0.33 이상 떨어진 쌍들을 다시 붙이지 못한다.

재현 (`recluster()` 직접 호출):

| 입력 | 결과 |
|---|---|
| 원본 5행만 (어제 상태) | **앨범 1개** (+ 노이즈 1) — 정상 |
| 원본 5행 + 복제 5행 (오늘 상태) | **2장짜리 쌍 앨범 5개** — 증상 재현 |

쌍이 서로 다른 `photo_id` 2개에 걸쳐 있어 **단일 사진 클러스터 강등**(`handlers.py`
`_recluster_and_save`)도 통과해 정식 인물 앨범으로 승격된다.

> **부분 복제는 증상이 안 드러난다.** 같은 event의 다른 인물 군집(6행)에도 복제 쌍이 1개 섞여
> 있으나(row 5 ≡ row 11, 어제 이미 발생), 6장 중 1쌍뿐이라 군집이 유지됐다. **전 사진이 복제될
> 때만 군집이 쌍 단위로 와해된다** — 그래서 어제는 멀쩡해 보였고 오늘 전량 재업로드에서 터졌다.

## 4. 어제의 `cluster.py` 수정과는 무관

[2026-07-10 reassign 동률 수정](./2026-07-10-reassign-mustlink-tiebreak.md)(36627d6)은 must/cannot-link
제약이 있을 때만 경로가 갈린다. 이 event의 저장된 제약은 **must-link 0개, cannot-link 0개**이므로
해당 코드 경로를 아예 타지 않는다. 용의선상에서 제외된다.

## 5. 두 번째 원인 — 삭제가 워커에 전달되지 않았다 (유령 행)

워커가 만든 각 쌍 앨범은 `image_id`를 **2개씩** 담고 있는데, 앱은 앨범마다 **1장만** 보여준다.
즉 **Spring DB에는 어제 사진 레코드가 없다.** 앱에서 사진(또는 이벤트)을 지웠으나
`delete_request`가 워커까지 도달하지 않아, `.npz`에 어제 행이 유령으로 남았다.

이는 재업로드와 독립적인 결함이다 — 재업로드가 없었어도 유령 행은 계속 재군집에 참여한다.
ADR-007 §3이 "삭제 메시지 자체가 트리거라 사각지대 없음"이라고 전제한 지점이 실제로는 지켜지지
않고 있다는 뜻이며, PIPA(생체정보 삭제) 관점의 문제이기도 하다.

## 6. 대응 후보

1. **워커 방어 — 중복 임베딩 붕괴(collapse)**: 재군집 직전 유사도 ≈1.0인 행을 대표 1행으로 접고,
   결과 조립 시 다시 펼친다. 재업로드·유령 행 어느 쪽이 원인이든 앨범이 쪼개지지 않는다.
   (임계·구현 위치는 미결 — `recluster()` 입력 전처리 vs `handlers` 계층)
2. **Spring 근본 해결**: (a) 업로드 시 콘텐츠 해시(ETag) 중복 검사, (b) 사진 삭제 시
   `delete_request` 발행이 실제로 되는지 통합 검증.
3. **S3 버킷 버저닝**: 현재 **비활성**. `put_object` 통짜 덮어쓰기라 오염된 순간 이전 상태가
   영구 소실된다. 버저닝이 켜져 있었다면 어제 버전으로 롤백해 즉시 복구 가능했다.
4. **event 8 복구**: 유령 행 제거(Spring의 `delete_request` 발행 또는 수동 `.npz` 정리) 후 재군집.

## 7. 재현 절차

```sh
aws s3 cp s3://{embeddings-bucket}/embeddings/8.npz ./8.npz
```

```python
from app.storage.event_embeddings import EventEmbeddings
from app.pipeline.cluster import recluster, Constraints

ev = EventEmbeddings.from_npz_bytes(open("8.npz", "rb").read())
sim = ev.embeddings @ ev.embeddings.T          # 비대각 1.00 쌍 = 중복
recluster(ev.embeddings[[0, 1, 2, 3, 4]], [None] * 5, Constraints())          # 앨범 1개
recluster(ev.embeddings[[0, 1, 2, 3, 4, 12, 13, 14, 15, 16]], [None] * 10, Constraints())  # 쌍 앨범 5개
```
