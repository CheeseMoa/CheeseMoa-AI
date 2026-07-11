# 사용자 보정의 동작 — 앨범 수동 수정이 제약으로 반영되는 과정

사용자가 앱에서 앨범을 수동으로 수정하면(병합·분리·이동·구분 확정), 워커는 그 결정을 **must-link /
cannot-link 제약**으로 번역해 event `.npz`에 영속화하고 **전체 재군집을 다시 돌린다.** 배정을 직접
바꾸지 않는 이유는 하나다 — 군집의 진실은 항상 event 전체 임베딩에 대한 재군집이므로([ADR 003](../decisions/003-full-reclustering.md)),
사람의 결정도 재군집이 뒤집을 수 없는 형태로 남겨야 한다.

> **핵심 계약**: 재군집은 사람의 결정을 뒤집지 않는다. 단, 이 계약에는 §7에 기록된 구멍이 있다.

- 구현: `app/handlers.py`(번역·조정) + `app/pipeline/cluster.py`(강제)
- 계약: [feature-spec](../spec/feature-spec.md) §6.3, [message-examples](../spec/message-examples.md)
- 저장: [ADR 007](../decisions/007-embedding-storage-s3.md) — event 단위 `.npz`

---

## 1. 두 가지 제약

| 제약 | 의미 | 전이성 |
|------|------|--------|
| **must-link** `(a, b)` | 두 얼굴은 **같은 사람** | O — (a,b),(b,c)면 a,c도 한 컴포넌트 |
| **cannot-link** `(a, b)` | 두 얼굴은 **다른 사람** | X — 그 쌍에만 적용 |

둘 다 **face_id 쌍**으로 저장한다(`.npz`의 `must_link_pairs`·`cannot_link_pairs`, shape `(K, 2)`).
face_id는 워커가 임베딩 시점에 발급하는 내부 키로 Spring은 모른다 — 메시지 계약은 image_id·cluster_id
기준이므로 번역이 필요하다.

**must-link는 "붙어라"만 말하고 "떨어져라"는 못 한다.** 이 비대칭이 `confirm_distinct`(§3.4)가 존재하는
이유이자, reassign 동률 결함(§7.2)의 뿌리다.

---

## 2. 보정 1건의 처리 순서

```
Spring → SQS(cluster_feedback) → 워커
  │
  ① 로드      store.load(event_id)          .npz 전체(임베딩·배정·기존 제약)
  │
  ② 번역      _translate_feedback()          image_id·cluster_id → face_id 제약 쌍
  │                                          (스테일 참조는 경고 후 건너뜀)
  ③ 조정      _reconcile_constraints()       기존 제약과의 모순을 later-wins로 해소
  │           (+ reassign superseded 규칙)
  ④ 저장      with_constraints() → store.save()
  │
  ⑤ 재군집    recluster()                    HDBSCAN → 제약 강제 → 후처리 → ID 승계
  │
  └→ ClassifyResult (clusters + retired_cluster_ids + uncertain + common)
                                             발행·삭제는 worker.py의 몫
```

②~③이 `handlers.py`, ⑤가 `cluster.py`다. 이 분리의 계약은 "**cluster.py에는 자체 모순이 없는 제약 셋만
전달한다**"이다 — 시간순 충돌 해소는 보정 이력을 아는 핸들러가 끝낸다.

---

## 3. 액션 4종 → 제약 번역

앨범 A = `{a1, a2, a3}`, 앨범 B = `{b1, b2}`를 예로 든다(얼굴 = face_id, 대표 얼굴은 **행 순서상 첫
얼굴**로 고정 — `.npz` 행 순서는 append-only라 메시지 재전달에도 같은 대표가 뽑힌다).

| 액션 | 만드는 must-link | 만드는 cannot-link | 부수 효과 |
|------|------------------|--------------------|-----------|
| `merge` | 전원 체인 | 없음 | — |
| `split` | 그룹 내부 체인 | 그룹 대표 간 | — |
| `reassign` | (이동 얼굴, 목적지 대표) | (이동 얼굴, 출처 잔류 대표) | 이동 얼굴의 **기존 must-link 전부 폐기** |
| `confirm_distinct` | 없음 | 대표 얼굴 전 쌍(클리크) | — |

### 3.1 merge — A와 B를 합친다

```json
{ "action": "merge", "merge": { "target_cluster_id": "A", "source_cluster_ids": ["B"] } }
```

```
must   = (a1,a2), (a2,a3), (a3,b1), (b1,b2)      ← 체인
cannot = 없음
```

전 쌍(클리크)이 아니라 **체인**인 이유는 must-link가 전이적이라 체인만으로 전원이 한 컴포넌트가 되기
때문이다. 대가는 §7.1 — 중간 얼굴 하나가 사라지면 체인이 끊긴다.

> `target_cluster_id`는 **살아남을 id를 보장하지 않는다** (§6, §7.4).

### 3.2 split — 한 앨범을 그룹 2개 이상으로 가른다

```json
{ "action": "split", "split": { "cluster_id": "A", "groups": [["img-1","img-2"], ["img-3"]] } }
```

```
must   = 그룹1 내부 체인, 그룹2 내부 체인, …      ← "이 그룹은 확실히 같은 사람"
cannot = (그룹1 대표, 그룹2 대표)  …전 그룹 쌍     ← "그룹끼리는 다른 사람"
```

그룹 **내부**가 must-link로 이미 한 덩어리이므로, 그룹 **사이**는 대표 1쌍만 갈라도 전체가 갈라진다.
그룹을 가로지르던 기존 must-link는 새 cannot-link와 충돌해 later-wins가 폐기한다(§4).

### 3.3 reassign — 사진 한 장을 A → B로 옮긴다

```json
{ "action": "reassign", "reassign": { "image_id": "img-3", "from_cluster_id": "A", "to_cluster_id": "B" } }
```

```
must   = (a3, b1)     ← "B에 있어야 한다"   (목적지가 비었으면 생략)
cannot = (a3, a1)     ← "A에 있으면 안 된다" (출처에 잔류 얼굴이 없으면 생략)
```

**superseded 규칙** — 이동 얼굴 `a3`가 낀 **기존 must-link는 전부 폐기**한다. 남기면 체인 전이성으로
옛 동료(`a2`, 나아가 `a1`)가 `a3`를 따라 앨범 B로 끌려간다. 어느 쌍이 "낡은 결정"인지 저장 구조상
특정할 수 없으므로 보수적으로 버린다. 기존 cannot-link는 남긴다 — 새 must와 충돌하면 later-wins가 판단.

**uncertain 사진의 편입** (계약 확장, feature-spec §6.2·§6.3) — uncertain 사진은 실 cluster_id가 없어
(`.npz`엔 `None`) 일반 reassign의 "from_cluster_id 일치" 조건에 걸리지 않는다. 예약 앨범 id
`"__uncertain__"`(`UNCERTAIN_ALBUM_ID`)을 `from_cluster_id`로 보내면 **cluster_id가 `None`인 얼굴**을
대상으로 삼는다. 출처에 잔류 얼굴이 없으므로 cannot-link 없이 must-link만 생긴다.

### 3.4 confirm_distinct — A와 B는 서로 다른 사람이 맞다

```json
{ "action": "confirm_distinct", "confirm_distinct": { "cluster_ids": ["A", "B"] } }
```

```
must   = 없음
cannot = (a1, b1)     ← 대표 얼굴 전 쌍. 앨범 3개면 3쌍(클리크)
```

merge의 정확한 반대 방향 선언이다. **왜 필요한가**: must-link는 응집만 강제하고 이격은 못 한다. 확정된
두 앨범 사이로 유사도가 애매한 신규 사진(다리 사진)이 들어오면 전체 재군집이 둘을 하나로 오병합할 수
있다 — 실측 기하로 재현됨(`handlers.py` 자가검증 ⑬: 인물 40·41이 cos≈0.64로 확실히 다른 사람인데,
두 대표와 cos≈0.906인 다리 사진 하나가 들어오면 오병합).

**대표 하나로 충분한 이유**: ① `_merge_fragments`는 두 집합 사이에 cannot-link가 **하나라도** 걸치면
병합을 포기하고(`_sets_blocked`), ② 재군집이 애초에 한 라벨로 묶어버린 경우 `_enforce_cannot_link`가
대표를 앵커로 라벨을 쪼갠 뒤 나머지 멤버를 **코사인 최근접 앵커**로 재배정한다.

---

## 4. later-wins — 보정끼리 모순될 때

`_reconcile_constraints` (handlers.py). 저장된 셋은 매번 이 함수를 통과하므로 **항상 자체 일관**이다.
따라서 모순은 "새 쌍 vs 기존 쌍" 사이에서만 생긴다.

```
1. 새 must / 새 cannot 은 무조건 수용         ← 최신 결정이 이긴다
2. 기존 must 를 최신 것부터(배열 뒤부터) 검사
     → 이 쌍을 살리면 새 cannot 이 깨지는가?  (would_connect)
        예 → 폐기 (오래된 결정)
        아니오 → 수용
3. 기존 cannot 중 새 must 로 연결돼버린 쌍은 폐기
```

배열 순서 = 시간순(append-only)이 근거다. 다만 must 배열과 cannot 배열 **사이에는** 상대 시간순이
없어, 동률에서는 must(병합 결정)를 먼저 수용한다 — 임의 규칙이 아니라 저장 구조의 한계를 문서화한
명시적 결정이다.

**대표 시나리오**: merge(A+B) → split(A|B). merge가 남긴 그룹 간 must-link `(a3,b1)`이 split의 새
cannot-link `(a1,b1)`과 모순되므로 폐기되고 split이 이긴다 (자가검증 ④).

**자체 모순은 버그로 취급** — 한 메시지의 번역 결과가 스스로 모순이면(must로 연결된 쌍에 cannot)
later-wins로 감추지 않고 `ValueError`로 즉시 거부한다. 스키마도 파싱 시점에 자기 병합(target ∈ sources),
split 그룹 겹침, no-op 이동(from == to), confirm_distinct 중복 id를 거부한다.

---

## 5. 재군집에서 제약이 강제되는 순서

`recluster()` (cluster.py). **순서가 곧 정확성**이다.

```
① HDBSCAN 전체 재군집          제약을 모르는 순수 밀도 군집. 익명 라벨 0,1,2… 산출
   └ 클러스터 0개면 연결 성분 부분 승격 (ADR-008)

② _enforce_must_link           컴포넌트 전원을 한 라벨로 통일
   - 대상 라벨 = 컴포넌트 내 비노이즈 다수결
   - cannot-link 가 금지한 라벨은 후보에서 제외  ← §7.2 수정
   - 전원 노이즈면 새 라벨 발급 (밀도와 무관하게 승격 — 사람이 확정했으므로)

③ _enforce_cannot_link         같은 라벨에 남은 위반 쌍을 분리
   - 이동 단위 = must-link 컴포넌트 (컴포넌트는 쪼개지지 않는다)
   - 위반 앵커들을 greedy 컬러링으로 최소한만 가름
   - 가장 큰 앵커가 원 라벨 유지, 나머지 색은 새 라벨
   - 제약 없는 나머지 멤버는 코사인 최근접 앵커를 따라감

④ _merge_fragments             cannot-link 가 걸친 클러스터 쌍은 병합하지 않음
⑤ _rescue_noise                cannot-link 상대가 있는 클러스터는 건너뜀
⑥ _evict_ambiguous             제약 당사자는 축출 면제(protected)
                               cannot-link 쌍 클러스터는 마진 비교에서 서로 제외
⑦ _match_cluster_ids           Jaccard 승계 / 신규 발급 / 은퇴
```

**②가 ③보다 먼저인 이유**: 병합을 다 끝낸 최종 상태에서 cannot-link 위반을 봐야 분리가 정확하다.

**⑥의 두 예외가 필요한 이유**: split로 생긴 소형 클러스터는 구성상 내부 유사도가 낮을 수 있어, 절대
바닥 축출이 클러스터를 통째로 비워 사용자 분리 결정과 cluster_id를 지우는 것이 리뷰에서 재현됐다.
또 사용자가 갈라둔 동일 인물 양쪽에 가까운 것은 당연하므로, cannot-link 쌍끼리는 마진 비교에서 뺀다.

**단일 사진 클러스터 강등도 면제** (handlers.py) — 한 장의 사진 얼굴로만 된 군집은 보통 인물로 승격하지
않지만(우연히 닮은 타인), 보정 당사자가 포함된 군집은 사람의 결정이므로 유지한다.

---

## 6. ID 승계 — 앨범 번호는 어떻게 이어지는가

재군집은 **익명의 그룹**을 낸다. 라벨 번호는 내부 계산 순서일 뿐이라 매 실행 달라질 수 있다. 승계가
없으면 사진 한 장 올릴 때마다 앨범이 전부 삭제·재생성된다.

`_match_cluster_ids`는 새 그룹과 기존 앨범의 **멤버 겹침(Jaccard = 교집합/합집합)**을 재고, 강한 짝부터
greedy 1:1로 배정한다.

```
merge 후 새 그룹 {f1..f5} vs 기존 A {f1,f2,f3} → 3/5 = 0.6   ← 승계
                            vs 기존 B {f4,f5}   → 2/5 = 0.4   ← 은퇴
```

세 가지 결말:

- **승계** — 겹치는 기존 앨범의 id를 물려받음 (`is_new = false`)
- **신규 발급** — 어떤 기존 앨범과도 안 겹침 = 처음 등장한 인물 (`is_new = true`)
- **은퇴** — 어떤 새 그룹에게도 승계되지 못한 기존 id → `retired_cluster_ids`로 통보, **Spring이 앨범 삭제**

액션별로 보면 — merge는 "큰 쪽 승계 + 작은 쪽 은퇴", split은 "겹침이 강한 쪽 승계 + 다른 쪽 **신규
발급**"이다(split의 진 그룹은 은퇴가 아니라 탄생). 은퇴는 얼굴 데이터 삭제가 아니다 — 얼굴은 합쳐진
앨범에 살아 있고, id만 소멸한다. 은퇴한 id는 되살아나지 않는다.

---

## 7. 알려진 한계 (핵심 계약의 구멍)

### 7.1 보정의 내구성이 앵커 얼굴의 생존에 의존한다

앨범 수준의 결정이 **특정 얼굴 쌍**으로 인코딩되므로, 그 얼굴이 사라지면 결정도 조용히 사라진다.
통보도 로그도 없다. ([상세](../reviews/2026-07-09-user-feedback-review.md) §1)

- **merge**: 체인 중간 얼굴 하나만 삭제돼도 체인이 두 컴포넌트로 단절 → 병합이 부분 상실
- **confirm_distinct**: 대표 얼굴의 사진이 삭제되면 `masked_by_photo_ids`가 댕글링 쌍을 프루닝(구조상
  필연 — 남기면 로드 불변식이 깨진다) → 오병합 방지가 소멸
- **reassign**: 목적지 대표 `b1`을 나중에 다른 곳으로 reassign하면, superseded 규칙이 `b1`의 기존
  must-link를 전부 폐기하므로 **과거 편입 결정까지 소실**
- **A·C 병합 후 다리 얼굴 이동**: merge 체인 `…(a3,c1)…`에서 `a3`를 reassign하면 superseded가
  `(a2,a3)`·`(a3,c1)`을 함께 버려 **{a1,a2}와 {c1,c2}의 연결이 끊긴다.** 이후 판정은 기하로 넘어가므로,
  손으로 merge해야 했던(기하로는 안 붙는) 앨범이라면 도로 갈라진다

개별 규칙은 각자 정당하다(프루닝은 필연, superseded는 옛 동료 견인 방지). 문제는 **합성 효과**다.
개선 방향은 **재앵커링**(프루닝·폐기 시 같은 클러스터의 다른 생존 얼굴로 쌍을 재연결)이며, 최소 조치로
폐기 발생 시 warning 로그 표면화가 우선순위 "즉시"에 있다.

### 7.2 reassign의 must-link 동률 — 수정됨, 잔존 구멍 있음

must-link 쌍에는 방향이 없다. reassign의 컴포넌트는 `{이동 얼굴, 목적지 대표}` 2명뿐이라 다수결이
**항상 1:1 동률**인데, 옛 규칙("작은 라벨")이 우연히 출처 라벨을 고르면 **목적지 대표가 자기 앨범에서
끌려나오고**, 뒤이은 cannot-link 강제가 그 쌍을 통째로 떼어내 목적지 앨범이 쪼개지고 신규 id가 발급됐다.

수정(커밋 `36627d6`): cannot-link("출처에 있으면 안 됨")가 **방향을 보존**하고 있으므로, 다수결 후보에서
금지 라벨을 제외해 방향을 복원한다. 자가검증으로 회귀 고정.
([재현 기록](../reviews/2026-07-10-reassign-mustlink-tiebreak.md))

**잔존 구멍 — 제3앨범 동률**: 이동 사진이 재군집에서 출처도 목적지도 아닌 **제3앨범**에 흡수되면,
제3앨범 1표 vs 목적지 1표 동률이 되고 제3앨범을 금지하는 cannot-link는 없다. 약 90개 기하 조합 스캔에서
발현시키지 못했으나 구조적 배제도 불가하다. 해법 후보는 동률을 "작은 라벨" 대신 **컴포넌트 평균 벡터와
후보 centroid의 유사도**로 깨는 것.

### 7.3 스테일 보정이 `succeeded`로 보고된다

이미 사라진 cluster_id·사진을 가리키는 보정은 경고 로그만 남기고 건너뛴다. 유효 대상이 2개 미만이면
보정 전체를 무시하지만, **재군집·결과 발행은 진행**하고 결과는 `succeeded`다(Spring이 현재 상태로
수렴하게 하려는 의도적 설계). 결과 계약에 "보정이 적용되었는가" 신호가 없어, 사용자 관점에서는 "병합
버튼을 눌렀는데 아무 일도 없고 오류도 없는" UX가 된다. → Spring과 계약 확장 협의 필요.

### 7.4 `target_cluster_id`가 승계를 보장하지 않는다

must-link 쌍은 대칭이라 target/source 구분을 담지 못하고, 최종 id는 §6의 Jaccard(멤버 수)가 정한다.
사용자가 "B로 합쳐줘"라고 해도 A가 크면 **A의 id가 남는다.** 앨범 이름·커버가 id에 매달린 Spring
쪽에서는 어긋남이 보일 수 있다. 이름은 Spring의 도메인이므로 **Spring이 이름을 이전**하는 것이
권장이나(워커의 승계 규칙은 의미 없는 순수 규칙으로 유지), 계약상 미합의 상태다.

### 7.5 기타

- **`to_cluster_id = "__uncertain__"`**: 예약 id는 `from_cluster_id` 쪽만 계약인데 목적지에 와도
  거부되지 않고, 우연히 "소속 해제" 유사 동작이 성립한다 — 계약에 없는 우연이라 명시 거부 또는 공식
  액션 승격이 필요하다
- **유령 event 보정**: 저장된 적 없는 event의 보정은 재시도로 해소되지 않는 결정적 이상이므로 `failed`를
  반환한다(발행 후 삭제 → FIFO 그룹 비차단). delete의 멱등 `succeeded`와 구분은 의도적이다

---

## 8. 자가검증 매핑

`python -m app.handlers` — AWS·모델 없이 전 시나리오가 돈다. 본 문서를 코드로 따라갈 때의 진입점이다.

| 검증 | 시나리오 | 본 문서 |
|------|----------|---------|
| ③ | merge → 클러스터 1개, 나머지 id 은퇴 | §3.1, §6 |
| ④ | merge 후 split → later-wins가 그룹 간 옛 must-link 폐기 | §3.2, §4 |
| ⑤ | reassign → 기하를 이기는 사용자 결정 | §3.3 |
| ⑥ | 제약 당사자 삭제 → 댕글링 프루닝 | §7.1 |
| ⑫ | `__uncertain__` reassign으로 uncertain 편입 | §3.3 |
| ⑬ | confirm_distinct → 다리 사진 오병합 차단 (실측 기하) | §3.4 |

`python -m app.pipeline.cluster` — 합성 임베딩으로 재군집 후처리·제약 강제를 검증한다(자가검증 (i)가
§7.2 동률 결함의 회귀 고정).

> ⑤의 단언은 "앵커 쌍이 함께 움직이는가"만 보므로 §7.2 결함을 통과시켰다. 목적지 id 승계·나머지 멤버
> 유지·신규 앨범 부재 단언 보강이 미해결 과제로 남아 있다.

---

## 참고

- [feature-spec](../spec/feature-spec.md) §4(재군집)·§6.2(라우팅)·§6.3(보정 계약)
- [ADR 003](../decisions/003-full-reclustering.md) — 전체 재군집이 군집의 진실
- [ADR 007](../decisions/007-embedding-storage-s3.md) — event 단위 `.npz`, 재군집 격리 단위
- [사용자 보정 리뷰](../reviews/2026-07-09-user-feedback-review.md) — 운영 리스크 4건
- [reassign 동률 결함 재현 기록](../reviews/2026-07-10-reassign-mustlink-tiebreak.md)
