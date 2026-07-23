# 완료된 목표 이력

CLAUDE.md에서 이동한 완료 목표 아카이브 (2026-07-22, 컨텍스트 절감). 새 목표를 완료하면 이 파일
**맨 위에** 같은 형식(문제 → 원인 → 해법 → 실측 검증 → 롤백 스위치)으로 추가한다 — CLAUDE.md에는 넣지 않는다.

- **Rekognition uncertain 재판정 — 하드케이스 자동 편입·제안 (P0 목표 0 구현분)** (2026-07-24, CHMO-420,
  [ADR 030](decisions/030-rekognition-uncertain-rejudge.md),
  [실측 리뷰](reviews/2026-07-23-rekognition-uncertain-ab.md)) — 옆얼굴·가림·나이차 하드케이스는
  AuraFace 코사인으로 원리적으로 못 가른다(동일인 0.18~0.48 vs 타인 0.20~0.36 완전 겹침, 로컬 임계
  불가는 2026-07-23 두 리뷰로 확정)는 문제에, 실측으로 근거 확보된 CompareFaces(동일인 83~100 vs
  타인 0~16, 실서버 미배정 24.5% 회수 전수 정답)를 보조 신호로 도입. 해법: 재군집 후 uncertain 확정
  직전 핸들러 opt-in 훅(`pipeline/rejudge.py` + `deps._build_rejudger`) — 미배정 얼굴 × AuraFace
  top-3 앨범 폭최대 대표를 **전부 호출 후 argmax**(조기 종료 없음 — AuraFace 순위를 못 믿는 대역이라)
  로 ≥90 자동 편입(must-link 기록 + 재군집 2차 패스 — 모든 후처리 불변식 자동 재적용·재전달 안전성
  유지·이후 uncertain 재진입 차단), [85,90) 제안(`uncertain[].suggestions` 계약 확장 — face_bbox 값
  결속, Spring 합의 필요·OFF 동안 빈 배열 하위 호환), ≥95 복수 매칭 파편 힌트(로그). 편입 전 3중
  검사(사용자 cannot-link 존중·같은사진 공존·사진당 argmax 가드)로 사용자 결정·같은사진 불변식 보호.
  (face,대표) 점수는 event별 S3 JSON(`rekognition-scores/`)에 캐싱해 재군집·보정 재과금 방지(얼굴
  미검출 -1.0 센티널, 삭제 시 캐시 동반 삭제), crop은 실측 파리티(bbox+0.25 여백·q92·무축소).
  장애는 best-effort(전면 장애 = 비활성과 동일 결과 자가검증 고정), job당 호출 상한 150. 검증:
  rejudge 24건·scores 코덱·handlers 종단 10건(편입 지속·재실행 0호출·cannot-link 존중·웜 캐시)·
  스모크 발행 왕복. 롤백: `REJUDGE_ENABLED=false`(기본은 CHMO-420에서 ON 전환 — 운영 전제 조건
  미충족 환경은 false 배포, ADR 030 §활성화 게이트). 잔여: AWS AI 학습 opt-out 확인·워커 IAM
  `rekognition:CompareFaces`·Spring `suggestions` 소비 합의·파편 힌트 wire 계약.
- **재군집 전 근중복 행 붕괴 — 재업로드·유령 행 오염의 앨범 와해 방어(P0 워커 방어층)** (2026-07-23,
  CHMO-419, [ADR 029](decisions/029-duplicate-embedding-collapse.md)) — 같은 사진 재업로드(멱등 스킵이
  `image_id` 기준이라 새 id면 재임베딩)·`delete_request` 미도달 유령 행이 만든 유사도 ≈1.0 복제
  쌍을 HDBSCAN이 실제 인물 덩어리(쌍 0.39~0.67)보다 우선 선택해 동일 인물 앨범이 2장짜리 쌍
  앨범들로 와해되던 문제(실 event 8: 카리나 5장→앨범 5개,
  [원인·재현](reviews/2026-07-11-duplicate-embedding-split.md)). ADR 012 임계 재보정은 고유사도
  대역만 우연히 구제 — 합성 스윕에서 0.36~0.65 대역 앨범 2개, 0.25~0.55 대역 4~5개로 여전히
  갈라짐을 확인 후 착수. 해법: `recluster()` 입력 전처리(⓪)로 쌍 유사도 ≥0.985(ADR 005 burst
  관례 = 2026-07-22 실서버 "같은 사진" 판정값, 이중 검출 0.978~0.979와 새-세션 최고 0.934는
  아래) 그룹을 대표 1행으로 접고 결과에서 펼친다 — 제약 인덱스 리매핑 + 사람 결정 우선(그룹 내
  사용자 cannot-link면 미붕괴, 모순 유발 시 전체 폴백) + 고아 그룹 승격은 클러스터 0개
  이벤트에만(버스트 계약 보존, 흡수 실패 복제 그룹은 uncertain — 쌍 앨범 부활 차단) + 은퇴
  목록 원 행 기준 재계산(흡수된 쌍 앨범 id를 Spring이 정리하게). 실측 검증: 오염 스윕(시드 6 ×
  3대역) 전 조합에서 오염 후 군집이 오염 전과 동일 구성으로 복원, 자가검증 cluster 35건(신규
  (n) 6건)·handlers 59건(⑬ 지터 0.5°→12° 재배치 — 종전 값은 cos 0.99996으로 사실상 재업로드
  기하)·스모크 9건 통과. 롤백: `CLUSTER_DUPLICATE_COLLAPSE_SIMILARITY=0`(비활성). 잔여(별건):
  Spring ETag 재업로드 검사·`delete_request` 발행 검증(PIPA)·S3 버저닝 — backlog P0 §2~4.
- **uncertain 얼굴 bbox 단일 → 배열 계약 교체 `face_bboxes` — 주 인물 얼굴 전부 동봉** (2026-07-22,
  CHMO-407) — 백엔드가 BE#107(CHMO-393)에서 "분류를 어렵게 한 얼굴들"을 배열 `uncertain[].face_bboxes`로
  받도록 바꿨는데(관용 파싱, 빈 배열이면 API 응답에서 필드 생략) 워커는 단일 `face_bbox`(CHMO-388)만 보내
  BE가 bbox를 하나도 저장 못 하던 짝 불일치. 해법: 단일 필드를 배열로 **대체**(BE는 배열만 소비, null 개념
  소멸) — 그 사진의 uncertain 얼굴 중 **주 인물 자격(counted — ADR 025·027 AND 폭 게이트 — ADR 022) 통과
  얼굴 전부**를 폭 내림차순(동률은 event 행 순)으로 싣는다. 행인·오검출·파편은 제외 — 자격 얼굴이 없으면
  (오검출 전용 사진, v2 이하 .npz 행) **빈 배열**(동작 변화: 종전엔 비자격 최대폭 얼굴 bbox라도 보냈으나
  이제 오검출에 박스를 그리지 않는다). 요소 2개+가 나오는 유일한 경로는 매칭 사진의 미매칭 주 인물
  복수(unmatched_main_to_uncertain — 비매칭 사진은 주 인물 2명+이면 단체 규칙으로 공용행). 토글 없음
  (계약 필드 교체, 롤백은 배포 롤백). 자가검증 handlers 59건(⑭⑮⑳ 교체 + 신규 ㉒ 2건 — 주 인물 2명
  배열·동률 정렬)·schemas 35건·스모크 통과. 워커 단독 배포로 동작(BE 머지 완료), 노션 스키마 문서 갱신 필요.
- **uncertain 품질 원인 `causes` 계약 확장 — "분류가 어려워요" 설명글 근거** (2026-07-22, CHMO-404) —
  저해상도·초소형 얼굴(유아 단체사진 등) 때문에 대량 미배정→uncertain/공용으로 흩어진 결과를 사용자가
  "왜 이렇게 됐지?" 물을 때, 앱이 그 자리에서 이유를 설명하고 재업로드를 넛지할 근거가 없던 문제. 진단으로
  근본 원인 확정: 스택(워커·Spring·웹)은 사진을 축소하지 않으며(워커 `detect_max_side=2000`>업로드 크기,
  Spring은 presigned 직PUT, 웹은 원본 File PUT), 저해상도는 사용자가 이미 압축된 사진(메신저 전달본 등)을
  올려서 생긴 업로드 이전 손실 — 워커가 복구 불가. 해법: `UncertainImage`에 `causes: list[Literal[
  "low_resolution","small_faces"]]` 추가(계약 확장) — 워커는 **코드만** 내고 문구·톤은 앱 소유, Spring은
  relay. 코드 3종: `low_resolution`(주 얼굴이 저해상도로 작게 잡힘 — "원본으로 다시"가 유효한 유일
  actionable, 항상 `small_faces` 동반)·`small_faces`(고해상도인데 멀리·작게 — 재업로드 무효, 참고용)·
  `single_appearance`(선명한 대형 얼굴인데 이 인물이 이벤트에 한 번만 등장 → 묶을 짝이 없어 앨범 미생성,
  "더 나오면 자동 앨범/직접 지정" 안내, counted 실인물만). 직교 분리로 멀쩡한 사진 오안내 차단. 빈
  배열=품질·데이터 문제 아님(예: 두 인물 사이 ambiguous)→"직접 지정"만 안내. **공용엔 안 싣고
  uncertain에만** (공용은 실패가 아니라 정상 목적지). 판정은 결과 조립(`_assemble_result`)에서 저장 데이터로
  전 경로 일관 계산 — 이를 위해 **.npz 스키마 v4**(image_long_sides 열, v3 이하는 긴 변 0=미상 폴백으로
  low_resolution만 빠지고 small_faces는 face_widths로 유지). 임계는 실측 근거(얼굴폭 매칭 무릎 ~100px,
  단체사진 얼굴 rel_w ~5%라 100px엔 긴 변 ~2000px): `CLUSTER_UNCERTAIN_SMALL_FACE_PX`(100, 0=기능 전체
  비활성)·`CLUSTER_UNCERTAIN_LOW_RES_LONG_SIDE`(2000, 0=low_resolution만 비활성). 자가검증 handlers 57건
  (신규 ㉑ 5건 — 저해상도·4분기·비활성·single_appearance 종단)·schemas 35건(신규 1)·storage 27건(신규 v4
  왕복·하위호환 3)·스모크 통과. Spring·FE엔 필드 하나 추가 통지만 필요.
- **크기 인지형 confident 게이트 — 대형 오검출 유령 앨범 해소** (2026-07-22, CHMO-403,
  [ADR 028](decisions/028-size-aware-confident-score-gate.md)) — event 115에서 사람은 2명인데
  앨범이 3개 생기고 3번째 앨범 썸네일이 "손"으로 뜨던 문제. YuNet이 손 브이(V)포즈를 얼굴로 약하게
  오검출(score 0.640)했는데 게이트 0.6을 턱걸이로 넘었고, 근중복 재업로드로 두 번 잡혀 서로 닮은(0.590)
  두 오검출이 min_cluster_size=2를 채워 별도 인물 앨범으로 승격됐다. 기존 방어막(ADR 013/015/025/027)
  전부 미발동. 해법: confident 게이트를 크기로 나눈다 — 대형(rel_w≥0.20)은 0.70으로 올려 저score
  [0.6,0.70) 구간을 ADR 017 회복(정규 스케일 재검출)으로 재판정, 소형은 0.6 유지. 게이트 상향이 과거
  막혔던 이유(YuNet이 초근접 대형 실얼굴에도 저score)는 ADR 017이 회복으로 풀어, 대형은 재검출이
  실얼굴(≥0.86)/오검출(재검출 실패)을 깨끗이 가른다 — 회복은 대형만이라 소형은 백스톱 없고 실얼굴이
  같은 score 구간에 겹쳐(유아 0.605 vs 그림 0.686) 상향 불가. 실측(코퍼스 1,181장): 대형 [0.6,0.70)
  실얼굴 41개 전원 회복, 오검출 2개(event 115 손·event 5 손글씨 장부)만 폐기. 검증(구동작 0.6 vs 신규
  0.70 전 코퍼스 detect diff): 대형 실얼굴 손실 0·소형 폐기 0·추가 0, event 115 손 앨범 해소(대형 손
  폐기로 min_cluster_size 미달). `DETECT_BIG_FACE_CONFIDENT_SCORE`(0.6=비활성), 도구
  `scripts/sweep_score_gate.py`·`scripts/verify_size_gate.py`. 한계: 소형 손(rel_w 19.6%)은 잔존하나
  단독이라 앨범 미형성; 저장 이벤트는 재검출 트리거 전까지 미치유(ADR 027 잔여와 동류).
- **매칭 사진의 미매칭 주 인물 얼굴 uncertain 동시 노출 — 미등록 인물 수동 구제 진입점** (2026-07-21,
  feature-spec §6.2 결정) — 2명이 인식된 사진에서 한 명만 매칭되면 사진이 인물 앨범(+공용)에만 실려,
  미매칭 인물을 "분류가 어려워요 → 인물 앨범 편입"(`__uncertain__` reassign)으로 수동 구제할 진입점이
  없던 문제. 백엔드 합의로 계약 확장: `_assemble_result` 라우팅의 "매칭 얼굴 있으면 uncertain 제외"를
  주 인물 미매칭 얼굴에 한해 해제(인물·공용·uncertain 중복 노출 허용). 행인(크기 게이트)·오검출
  (ADR 025)·파편(ADR 027)은 종전대로 숨김 — 주 인물 자격(counted+크기)이 그대로 노이즈 방어를 겸한다.
  face_bbox는 미매칭 주 얼굴(CHMO-388 crop_face_of 재사용), 편입 reassign은 원래 `cluster_id=None`
  얼굴만 must-link라 수정 없이 그대로 동작. `CLUSTER_UNMATCHED_MAIN_TO_UNCERTAIN`(false=구 정책 롤백),
  자가검증 51건(신규 ⑳ 3건 — 동시 노출·행인 숨김·토글 OFF 재현).
- **uncertain 주 얼굴 face_bbox 계약 확장 — 상세 화면 얼굴 crop** (2026-07-21, CHMO-388) —
  "분류가 어려워요" 사진 상세 화면에서 어느 얼굴이 분류가 어려웠는지 보여주기 위해
  `uncertain[].face_bbox`(`FaceBox` — 원본 픽셀 x·y 좌상단, w·h 폭·높이, 정수)를 결과 계약에 추가.
  워커 crop→S3 업로드(인물 앨범 썸네일 방식)는 uncertain 목록이 매 재군집 event 전체 스냅샷이라
  원본 재fetch·디코드 반복 + 고아 썸네일 정리가 필요해 기각 — 상세 화면엔 원본이 이미 있어 앱이
  bbox로 직접 오린다(.npz v3 bboxes가 원본 px라 좌표계 일치). crop 대상은 그 사진 uncertain 얼굴 중
  머릿수 자격(ADR 025·027 통과) 우선 → 최대 폭 순(행인·파편·오검출 배제, `_uncertain_face_box`).
  bbox 미상(v2 이하 .npz 행)은 null — crop 없이 사진만 표시. 자가검증 48건(신규 2)·스키마 34건
  (신규 1)·스모크 통과. Spring은 값 저장·전달만(표시 측 경계 클램프 필요), 상세는 feature-spec §6.2·
  message-examples §④·노션 "SQS 메시지 스키마" 갱신 완료.
- **초대형 얼굴 파편 이중 검출 디둡 — 1인 사진 공용 앨범 오노출 해소** (2026-07-21,
  [ADR 027](decisions/027-duplicate-face-fragment-dedup.md)) — group 37 / event 105에서 혼자
  찍힌 셀피가 공용 앨범에도 노출되던 문제. YuNet이 초대형 얼굴에 그린 파편 박스 2개가 둘 다 score
  게이트를 통과(0.769/0.638)했고, YuNet NMS(쌍 IoU 0.297<0.3)·ADR-017 디둡(회복 경로 전용)·이중
  검출 안전판 0.95(cannot-link 면제 전용이라 인물 앨범은 무사)를 전부 비껴가 머릿수만 2명으로
  세어졌다. 두 층 수정: ① 검출 — confident 얼굴도 정제 후 디둡(양쪽 폭>224px AND 랜드마크 중심거리
  <0.1×얼굴폭이면 score 최상만, 원본 1,081장 실측 파편 쌍 0.019·0.024 vs 실제 타인 겹침 최저 0.436,
  최초 후보 0.5는 실제 타인 쌍을 삼켜 기각), ② 라우팅 — 머릿수에 이중 검출 붕괴 이식(같은사진 근중복
  ≥0.95 그룹은 폭 최대 행만 카운트, 같은사진 쌍 758개 실측 이중 검출 0.978~0.979 vs 타인 최고 0.756
  — ①만으로는 .npz에 저장된 파편 행이 안 고쳐진다) + ADR-025 최근접에서 같은사진 근중복 제외(파편
  쌍끼리 바닥을 뚫는 구멍, 전면 제외는 그 사진에만 있는 낯선 단체를 오판해 기각). 검증: 검출 diff
  1,081장 중 파편 15장(유니크 2)만 정리, 라우팅 diff 24개 이벤트 중 이중 검출 6장만 공용→인물앨범,
  자가검증 46건(신규 ⑲)·스모크 통과. 의미 변화: 미배정 근중복만 있는 사진은 공용이 아니라 uncertain.
  `DETECT_CONFIDENT_DEDUP_LANDMARK_RATIO`·`CLUSTER_COMMON_DUPLICATE_FACE_SIMILARITY`(각 0=비활성),
  도구 `scripts/survey_confident_dup.py`·`scripts/verify_dup_dedup.py`. 잔여: .npz 파편 행 자체는
  남음(P0 중복 정리와 동류), 저장 이벤트 치유는 다음 재군집 트리거 때(즉시 치유는 85·90·91·104·105
  재트리거).
- **눈감음 상대 크기 게이트 — 원거리 "아래 쳐다봄" 오탐 제외** (2026-07-21,
  [ADR 026](decisions/026-eye-closed-relative-size-gate.md)) — group 35 / event 99에서 아래를
  쳐다보는 얼굴이 눈감음으로 오탐되던 문제. blink blendshape는 "윗눈꺼풀 내려옴"을 재서 내려뜬 뜬
  눈과 감은 눈을 못 가른다 — threshold 불가(정탐 0.516~0.637 한가운데 오탐 박힘, event 61 정탐과
  event 99 오탐이 같은 0.625), 절대 px 불가(정탐 함성 182px가 오탐 58·188px 사이 중첩). 판별축은
  **이미지 대비 상대 크기**: 고해상도 사진의 원거리 얼굴은 절대 188~210px여도 프레임에선 4.7~5.2%
  작은 피사체, 진짜 감음은 14~17% 주 피사체. 전 이벤트 961장 실측에서 내려뜸 오탐 전부 ≤5.2% vs
  진짜 감음 전부 ≥11.3%, 빈 구간에서 0.08 채택. `eye_main_face_ratio`(최대 얼굴 대비)와 달리
  denominator가 이미지라 솔로 사진의 원거리 얼굴을 잡는다(ADR 013·022 "주 피사체 vs 배경" 원리).
  검증: 전 이벤트 959장 before/after diff에서 37장 해제(전부 오탐), 진짜 감음 미탐 0, event 99 오탐
  4장 해제·정탐 2장 유지, 스모크 통과. `QUALITY_EYE_MIN_REL_WIDTH`(0=비활성), 도구
  `scripts/survey_eye_rel_width.py`. 한계: 프레임에서 큰 내려뜸(event 103 rel 32.8%)은 못 거름,
  가림 오탐(선글라스·안경)은 별개 문제, 원거리 진짜 감음은 미판정(ADR 013 원리상 허용).
- **공통(단체) 판정 머릿수 실인물 자격 게이트 — 오검출 얼굴 제외** (2026-07-20,
  [ADR 025](decisions/025-common-headcount-facesim-gate.md)) — event 93에서 퍼 후드 털 뭉치
  오검출(score 0.652·종횡비 0.96·rel_w 16.8%로 기존 검출 필터 전부 통과)이 주 인물로 세어져 1인
  셀피가 "주 인물 2명 단체"로 오판, 공용 앨범에 노출되던 문제. 판별 신호는 임베딩에 남는다 —
  오검출은 event 내 어떤 얼굴과도 안 닮는다(max-sim 0.183 vs 실인물 미배정 최저 0.191, 배정 최저
  0.407). 미배정 얼굴이 event 내 최근접 유사도 < 0.185(빈 구간 하단, 실인물 보호 쪽)면 머릿수에서
  제외(라우팅 전용, 군집 불변). 검증: 실 이벤트 13개 diff에서 클러스터 전 이벤트 불변, 변화는 공용
  3건 전부 의도된 수정(93 퍼 후드 + 84·87 w=0 레거시 행이 미러 이벤트의 크기 게이트와 정렬), 진짜
  2인 사진(0.191·0.239) 공용 유지. `CLUSTER_COMMON_FACE_MIN_SIMILARITY`(0=비활성). 한계: 보정 표본
  오검출 1건·빈 구간 0.008로 얇음 — 리포트 축적 시 검출 score 병행 재설계 (ADR 025 §한계).
- **인물 앨범 대표 얼굴 썸네일 — 워커 crop·S3 업로드 + 계약 확장** (2026-07-20, CHMO-335) — 앱의
  인물 앨범 목록용 썸네일. 대표 선정 신호(LOO centroid 유사도)·디코딩 원본·bbox가 전부 워커 안에
  있으므로 워커가 재군집 직후 클러스터마다 대표 얼굴을 crop(bbox 1.4배 여백)→다운스케일(긴 변 256px)
  →JPEG→S3 업로드하고 결과에 키만 싣는다(`ResultCluster.thumbnail_s3_key`, null 가능 — Spring은
  presigned URL 매 조회 발급으로 서빙만). Lambda·백엔드 크롭은 원본 재디코딩 중복으로 기각.
  `.npz` **스키마 v3**(bboxes·s3_keys 열 — v2 이하는 미상 폴백으로 해당 행만 대표 후보 제외),
  `pipeline/thumbnail.py`(순수 렌더)·`storage/thumbnail_store.py`(Protocol+S3+페이크) 신설,
  키는 `thumbnails/{event_id}/{cluster_id}.jpg` 고정·덮어쓰기, 은퇴 클러스터 썸네일은 best-effort
  삭제. 대표 원본은 클러스터당 1장만 재fetch(공유 t4g RAM 제약 — 요청 전체 이미지 캐시 금지),
  썸네일 실패는 경고 로그 + 해당 키 null로 격리(job 정상 진행). `THUMBNAIL_MAX_SIDE=0`이 롤백
  스위치(기능 전체 비활성). 잔여: 워커 IAM에 embeddings 버킷 `s3:DeleteObject` 권한 확인,
  Spring과 presigned URL 캐시 정책(매 조회 발급) 합의.
- **파편병합 승인을 컴포넌트 전체 재평가로 — 같은 인물 앨범 분리 해소** (2026-07-20,
  [ADR 024](decisions/024-merge-component-linkage.md)) — event 90(group 35)에서 주 인물의
  2얼굴 파편 앨범이 병합 게이트(0.632/0.479)를 통과하고도 별도 앨범으로 남던 문제. 구 완전 연결
  검사가 "병합 전 파편 스냅샷 쌍" 전부에 게이트를 요구해, 먼저 합류한 2얼굴 파편과의 노이즈 낀
  스냅샷 쌍(0.508/0.458)이 합류를 차단 — 15얼굴 컴포넌트의 안정된 증거(0.641/0.476)는 반영되지
  않는 구조. 임계 조정은 2D 그리드 스윕으로 기각(치유하는 어느 조합도 ADR-016이 잡은 아동
  오병합을 이벤트 7~15개에서 재점화). 대신 승인 검사를 컴포넌트 '현재 전체 멤버' 재평가(재계산
  centroid + 전체 얼굴 교차 face-pair 평균, 같은 임계 재사용)로 교체 — 다리(bridge) 융합은 남남
  쌍이 전체 평균을 끌어내려 여전히 차단(자가검증 (m) 합성 기하). 검증: 이벤트 52개 중 event 90만
  치유([15,9,8,2,2]→[17,9,8,2]), 코퍼스·나머지 51개·자가검증 전부 무변화.
  `CLUSTER_MERGE_COMPONENT_LINKAGE`(false=구 동작). 잔여 한계: face-pair 0.420짜리 파편은 동일인
  증거 부재로 의도적 미병합(사용자 merge 보정 영역), 거대 컴포넌트의 이론상 다리 리스크는 아동
  코퍼스·이벤트 무변화로 실측상 안전(리포트 축적 시 구성 쌍 veto 재보정 — ADR 024 §한계).
- **재검출 랜드마크 신뢰 임계 — 초대형 얼굴 파편 bbox의 가드 오판 해소** (2026-07-17,
  [ADR 023](decisions/023-refine-trust-redetect-landmarks.md)) — event 73(group 27)에서 얼굴이
  크게 나온 1인 사진이 공통 사진첩으로 빠지던 문제. YuNet이 초대형 얼굴에 파편 bbox(score 0.640,
  게이트 0.6을 살짝 통과해 ADR-017 회복 경로 미적용)를 주고, 랜드마크 정제의 재검출이 올바른
  랜드마크(score 0.860)를 찾고도 이동량 가드(0.5 × 파편 bbox 폭)에 걸려 폐기 → 깨진 랜드마크의
  쓰레기 임베딩(동일인과도 0.24)이 노이즈로 유출. 같은 뿌리로 recover 경로의 "가드 걸리면 원
  랜드마크 유지" 폴백이 offset 파편 박스 3개를 쓰레기 임베딩으로 살려 유령 인물 앨범까지 생성.
  실측(34개 이벤트 520장, 가드 발동 33건): 좋은 교정은 재검출 score 전부 ≥0.86(NN 0.32→0.98 등),
  무익한 후보는 전부 ≤0.39 — 빈 구간에서 회복 임계와 같은 0.80을 신뢰 임계로 채택, 이상이면
  가드 무시하고 재검출 랜드마크 채택(refine·recover 공통). 교정된 중심이 본 얼굴을 가리켜 파편
  박스는 디둡으로 제거 → 유령 앨범 뿌리 차단. 검증: 31/34 이벤트 불변, 변경 3개 전부 의도된
  수정(리포트 사진 앨범 편입 + 유령 앨범 소멸), 라벨 코퍼스 child ARI 0.573→0.794(순개선)·나머지
  불변. `DETECT_REFINE_TRUST_REDETECT_SCORE`(0=비활성), 도구 `scripts/survey_refine_shift.py`.
- **눈감음 판정 blendshape 교체 — Face Landmarker litert 이식** (2026-07-17,
  [ADR 021](decisions/021-blink-blendshape-litert.md)) — 눈 패치 CNN의 도메인 실패(유아 오탐·
  보정 스톡 미탐·수면 미탐, event 61)를 A/B 실측(871 얼굴, [리뷰](reviews/2026-07-17-mediapipe-blink-ab.md))
  으로 검증된 eyeBlink blendshape로 교체. mediapipe pip은 linux aarch64 휠이 없어 EC2(t4g) 배포
  불가 → face_landmarker.task(Apache 2.0) 내부 tflite 2개만 ai-edge-litert(aarch64 휠 있음)로
  실행하는 이식본 `blink.py` 신설(HDBSCAN·face_align 이식 패턴). YuNet 5점 RoI(배율 3.0/시프트
  −0.05 — 참조 파리티 스윕 확정: |Δ| med 0.008, 판정 뒤집힘 0, 감음 6/6)라 mediapipe 자체 검출
  대비 판정 가능률 69%→91%(누운 수면 옆얼굴 회복). presence<0.5는 미판정 — CNN 폴백은 실측
  정탐 기여 0에 유아 오탐만 재생산해 제거, CNN+ADR-019는 `QUALITY_BLINK_THRESHOLD=0` 롤백
  경로로만 유지. 비용 +35MB RSS·3ms/얼굴, aarch64 컨테이너 검증 완료, Dockerfile 프리베이크
  4모델. event 61: 감음 5장 전부 eyes_closed(미탐 2 회복), 오탐 0.
- **저신뢰 분리 회색지대 face-pair 재확인 — 남남 부착 축출** (2026-07-17,
  [ADR 020](decisions/020-evict-facepair-gray-gate.md)) — event 61에서 남남 얼굴이 LOO centroid
  0.425로 저신뢰 바닥(0.4)을 통과해 인물 앨범에 남던 문제. 전역 바닥 상향은 스윕으로 기각(0.42에서
  이미 단일인물 코퍼스 회귀 + blob 불변식 커플링, 동일인/남남 LOO가 [0.40,0.46)에서 원리적으로 겹침).
  판별 신호는 ADR-016처럼 개별 쌍에 남는다: 남남 부착 top쌍 ≤0.440 vs 진짜 멤버 ≥0.469의 빈 구간에서
  facepair floor 0.45(blob 승격 간선과 동일값 — 승격 즉시 해체 churn 차단 불변식), 회색지대 ceiling
  0.46(남남 관측 최고 0.456 직상, 코퍼스 진짜 멤버 LOO 최저 0.502 아래). 재검측: 코퍼스 회귀 0,
  실 이벤트 45개 중 13개에서 20얼굴 강등(전부 타인 정합, 동일인 증거 보유 0, 의도 밖 변화 0),
  나머지 32개 불변, event 61 해소. 도구: `scripts/tune_membership_floor.py`.
- **눈감음 판정 자격 게이트 — 가림·초소형 얼굴 오탐 해소** (2026-07-17,
  [ADR 019](decisions/019-eye-judgment-eligibility-gate.md)) — 웃음 캔디드 오탐을 실측하려던
  조사([survey](reviews/2026-07-17-smile-eyes-geometry-survey.md), 859 얼굴)에서 웃음 오탐은
  0건이고 실제 오탐 축은 다른 것으로 판명: eyes_closed flagged 11건 중 정탐 1건, 오탐은 초소형
  그림·옆얼굴(32~52px) 4건 + 고글·마스크(눈 뜸) 2건 + 눈 뜬 아기 1건. 가림 검출용 신호 가설
  (눈 어두움·렌즈 매끈함)은 실측 기각 — 선글라스는 반사로 오히려 고텍스처. 대신 판정 자격 게이트
  2개가 오탐을 가른다: **bbox 짧은 변 ≥64px**(min_blur_face_px와 같은 근거) AND **눈/볼 밝기 비
  ≤1.4**(감은 눈꺼풀은 피부 — 초과면 고글 반사·마스크 가림, 빈 구간 [1.21, 1.73]). 코퍼스
  eyes_closed 이미지 11→5장(해제 전부 오탐, 정탐 유지, 신규 0). `quality.py`
  `_eye_judgment_eligible`+`eye_cheek_ratio`, `QualityConfig`/`.env` 설정 2개(0=비활성). 남은 한계:
  대형 실물 선글라스(무표본)·유아 눈(CNN 도메인). 도구: `scripts/survey_smile_eyes.py`·
  `scripts/survey_eye_occlusion.py`(신호 실측 + 크롭 육안 분류).
- **흔들림 재확인 게이트 — 옛날 사진 blurry 오탐 해소** (2026-07-17,
  [ADR 018](decisions/018-shake-coherence-floor.md)) — event 50(앨범 205)에서 옛날 인화 사진
  재촬영본 8장이 전부 blurry로 오분류되던 문제. variance는 잔결의 양만 재서 "원판이 소프트한 사진"과
  "흔들린 사진"을 구분 못한다(임계 조정 불가 — 옛날 사진 분포가 연속). 판별축은 전체 이미지 방향
  쏠림(손떨림은 이방성, 소프트 원판은 등방): 오탐 최고 0.268 vs 흔들림 최저 0.444(얼굴 경로)의 빈
  구간에서 0.35를 바닥으로 채택, variance 판정 말미에 게이트로 적용(`shake_confirmed`,
  `QUALITY_SHAKE_COHERENCE_FLOOR`, 0=비활성). 얼굴 crop 쏠림은 판별력 없음(겹침 실측). event 50
  오탐 8장 전부 해제 + 나머지 51장 무변경, test2 라벨셋 무회귀. **보강(당일)**: event 55에서
  고스팅형 손떨림(겹침 번짐 — 쏠림 0.306으로 낮음)이 게이트에 오해제되어 공통첩으로 유출 →
  fallback 한정 variance 붕괴 면제(whole_var<40이면 쏠림 무관 흔들림 확정, 흔들림 13.5 vs 무얼굴
  옛날 사진 98.9) 추가. **보강 2(당일, §보강 2)**: 같은 고스팅 셀피(재업로드)가 event 64에서 대형
  얼굴 회복 재보정(rel_w 0.20, 98a093c)으로 이번엔 얼굴 검출되어 얼굴 경로로 빠짐 → 게이트 오해제 →
  쓰레기 임베딩이 노이즈로 uncertain 앨범 유출. 실측(코퍼스 499장)에서 단일 축은 전부 겹치나
  (whole_var·face_var·rel_w·쏠림·자기상관 피크·타일 쏠림 각각 기각) **결합 규칙이 성립**: 붕괴 면제를
  대형 blurry 얼굴(rel_w ≥ 0.22, 빈 구간 [0.172, 0.280] 기하 중앙)에 한해 얼굴 경로로 확장 —
  옛날 인화 오탐은 얼굴이 작거나(붕괴 6장 rel_w≤0.172) whole_var 미붕괴(113.1)로 걸러진다.
  검증: 고스팅 blurry 복원, event 50 재점화 0, 라벨셋·event 64 나머지 무회귀, 스윕 발동 고스팅뿐.
  `quality.face_collapse_exempt` + `QUALITY_COLLAPSE_FACE_REL_WIDTH`(0=비활성),
  도구 `scripts/survey_face_collapse.py`. **보강 3(2026-07-21, CHMO-380, §보강 3)**: 소형 얼굴
  등방성 손떨림(test9 dcb66942 단체샷)이 소형 얼굴 검출 하나로 fallback 붕괴 면제를 비켜가 미탐 →
  blurry 얼굴 최저 face_var<7(빈 구간 [5.4, 10.0]) AND whole_var<40이면 게이트 면제·흔들림 확정.
  로컬 142장 중 대상 1장만 전환·회귀 0, 실 이벤트 58개 diff에서 정탐 8장(dcb 재업로드)만 전환.
  whole_var 결합은 실 이벤트 검증에서 추가 — face_var 단독은 선명 사진의 배경 얼굴을 오탐(event 51
  배경 사진기자 face_var 4.1·whole_var 1133). `quality.face_var_collapse_exempt` +
  `QUALITY_FACE_VAR_COLLAPSE_FLOOR`(0=비활성), 도구 `scripts/sim_facevar_floor.py`. 남은 한계: 소형
  얼굴 face_var 10~25 구간 등방성 블러(옛날 인화와 겹침), 어두운 선명 사진(event 16 암실 전시 —
  whole_var도 어둠으로 붕괴, 밝기 게이트는 저조도 흔들림 놓쳐 미도입·수용), 정탐 표본 유니크 1장.
- **대형 근접 얼굴 재검출 회복 — 초근접 얼굴 미검출 해소** (2026-07-16,
  [ADR 017](decisions/017-size-aware-detection-score-threshold.md)) — event 36에서 얼굴이 크게 나온
  아이(0010=`6acd1055`)가 공통 사진첩으로 빠지던 문제. 원인은 YuNet(WIDER FACE 학습)이 초근접 대형
  얼굴에 저score(0.55)를 줘 score 게이트 0.6에 탈락 → 검출 0. **단순 크기 인지형 score 임계는 오검출
  0 불가**(실측: 대형 저score 실얼굴 37 vs 오검출 41이 score·선명도 모두 겹침). 대신 **정규 스케일
  재검출**이 깨끗한 판별축 — 실얼굴은 "너무 커서" 저score였을 뿐이라 정규 크기 재검출 시 score가
  오르고(재검출≥0.80에서 실얼굴 회복·오검출 0/41), 진짜 FP(블러 블롭)는 어느 스케일에서도 낮다.
  `detect.py`에 재검출 회복(`_recover_large_face`, refine와 코어 공유) + 랜드마크 중심 디둡(같은사진
  cannot-link 분열 방지), `DetectorConfig` 설정 2개(0=비활성). 실측: child·event35·36 각 +7 실얼굴
  회복(0010 포함), 성인 event13·27 무회귀. 도구: `scripts/survey_bigface.py`(임베딩 매칭 조사) +
  vision 크롭 분류 워크플로우. **rel_w 하한 재보정 0.30→0.20** (2026-07-17, ADR-017 §재보정) —
  event 60에서 화면을 다 덮는 얼굴이 검출 0으로 공통첩에 빠짐. YuNet은 초대형 얼굴일수록 bbox를
  파편으로 작게 그려(실제 폭 ~1,100px에 박스 500~600px) rel_w 0.280·0.295로 게이트 미달 — 크기
  게이트가 가장 극단적인 얼굴에서 무너지는 구조. 게이트 스윕 실측(전 이벤트 783장 + child, 0.30/
  0.25/0.20 최종 출력 diff)에서 0.20이 기존 검출 손실 0·오검출 통과 0·실얼굴 2장(유일 사진) 회복.
  FP 방어는 rel_w가 아니라 재검출 score(≥0.80)가 담당. 도구: `scripts/sweep_bigface_gate.py`.
- **분류 진행률 SQS 발행 — job 내부 진행바** (2026-07-16, CHMO-274) — classify가 결과 1건만 끝에
  발행해 백엔드·앱이 처리 중 진행도를 알 수 없던 문제. `_handle_classify`의 이미지 루프(job 비용의
  사실상 전부)에서 처리 장수를 별도 progress 큐로 흘려보낸다: 루프 진입 시 `0/total` 1회 + 이후
  3장마다(`_PROGRESS_REPORT_EVERY`) + 마지막 `total/total`. `processed`가 단조 증가해 백엔드가 순서·중복·재전달을
  방어(마지막 본 값 이하 버림). best-effort(발행 실패가 job을 안 죽임), progress 큐 URL 미설정 시
  비활성. `ProgressUpdate`(messages.py)·`SqsProgressPublisher`(publisher.py)·`report_progress` 콜백
  주입(handlers·deps). Spring의 큐 소비·메모리 보관·FE 폴링은 이 레포 밖(백엔드 담당).
- **파편병합 face-level 응집 게이트 — 아동 교차연령 오병합 해소** (2026-07-16, CHMO-269,
  [ADR 016](decisions/016-merge-facepair-cohesion-gate.md)) — event 35에서 서로 다른 아이 30장이
  한 앨범으로 뭉치던 문제. 원인은 검출·해상도가 아니라 파편병합이 **centroid**로 판정하는데 아동
  얼굴은 평균이 "아기 얼굴" 영역으로 수렴해 타인도 0.55~0.63으로 붙는 것(나이대 효과). 판별 신호는
  개별 얼굴 쌍에 남아(같은인물 face평균 0.65 vs 다른아이 ≤0.50), 병합 조건에 **파편 간 face-pair 평균
  바닥**(`merge_facepair_floor`)을 centroid와 AND로 추가. 라벨 코퍼스 child 8인 ARI 0.245→0.788
  (성인·단일인물 무회귀), 실 S3 이벤트 분해는 적대적 검증 전부 GOOD(다른 인물 분리). event 35는
  30장 blob→7명 분리, 최대 클러스터 내부 median 0.403→0.592. ADR-012가 남긴 아동 미검증 리스크 해소.
  `cluster.py`에 게이트 + `ClusterConfig`/`.env` 설정(0=비활성). 도구: `scripts/tune_merge_facepair.py`
  (floor 스윕)·`scripts/verify_split.py`(분해 fix/regression 판별)·`scripts/diagnose_event.py`(층별 진단).
  **재보정 0.45→0.475** (당일, ADR-016 §재보정) — event 43·38(아동)에서 0.45로도 최대 앨범 오병합
  잔존(내부 타인쌍 37%). 재스윕에서 0.475가 코퍼스 사실상 무회귀(meanARI −0.002)로 실 이벤트를
  0.50과 동일하게 완전 분해(43 타인쌍 6%, 35는 0%), 갈라진 조각 전 쌍 cross 0.40~0.47(타인)로 검증.
  교훈: 라벨 코퍼스는 교차연령이라 조임에 민감하지만 실 이벤트는 단일 세션이라 더 조여도 안전 —
  재보정 시 실 이벤트 오병합 지표(최대 앨범 내부 타인쌍 비율)를 함께 볼 것.
- **대형 오검출 결합 필터 — score<0.78 AND 종횡비<0.70** (2026-07-16,
  [ADR 015](decisions/015-detection-false-positive-combined-filter.md)) — 팔짱 낀 팔·조형물 등 진짜
  얼굴 크기의 오검출이 ADR-013 크기 필터·품질 게이트를 통과해 사진을 blurry 오분류하던 문제. 단일
  필터는 불가(album 얼굴도 score·종횡비 각각 최저까지 내려감), 두 축이 동시에 낮은 것은 오검출뿐이라
  결합 규칙이 album 손실 0/114로 오검출 6종 전부 제거. `detect.py`에 `DetectorConfig` 설정 2개
  (+.env, 둘 중 0이면 비활성)로 구현.
- **배경 인물(행인) 앨범 방지 — 얼굴 크기 필터** (2026-07-15,
  [ADR 013](decisions/013-background-face-size-filter.md)) — event 27에서 배경에 멀리 찍힌
  행인(이미지 긴 변의 0.8%)이 사진 2장에 반복 등장해 앨범이 생긴 문제. 실사진 16개 이벤트 210
  얼굴 분포 측정에서 행인 최대 0.82% vs 앨범 얼굴 최소 3.29%의 빈 구간을 확인, 검출 단계에
  rel_w(bbox 폭/긴 변) 2.5% 하한을 추가(`DETECT_MIN_FACE_REL_WIDTH`, 0=비활성). 앨범 손실 0 +
  노이즈 얼굴 52% 제거 실측. 절대 px 기준은 저해상도 업로드를 잘라 기각. 앨범 최소값(3.29%)
  쪽 마진이 1.3배로 얇으니 앨범 사진 누락 리포트 시 이 값부터 하향 검토.
- **병합 임계 재보정 0.68 → 0.55 — 파편화 주 원인 제거** (2026-07-15,
  [ADR 012](decisions/012-merge-threshold-recalibration.md),
  [분포 측정 리뷰](reviews/2026-07-15-distance-distribution-verdict.md)) — 교정 후 라벨 코퍼스
  (5인, 동일인 103쌍·타인 781쌍)에서 타인 최고 0.4584 vs 동일인 쌍 77%가 0.68 미달을 실측. ARI 스윕
  0.45~0.60 고원(5인 완벽 분리) 중앙 0.55 채택, test4가 기본값만으로 앨범 정확히 2개. 유료 라이선스
  검토는 근거 상실로 보류. 미검증 리스크(아동 교차연령)는 다음 구현 목표 0.5.
- **입력 품질 교정 — 정렬 AA + 랜드마크 2단계 정제** (2026-07-15,
  [review §구현 결과](reviews/2026-07-14-input-quality-alignment-landmark.md)) — `align.py`에 ROI 한정
  가우시안 프리블러(σ=(1/s)/2, 확대 경로는 픽셀 동일 유지), `detect.py`에 대형 얼굴 정규 스케일 재검출
  (실패 시 원 랜드마크 폴백, 파라미터는 스윕으로 **224/0.75** 확정 — 리뷰 초기값 160/0.5보다 우수).
  같은 얼굴 임베딩 유사도(max_side 스윕) 평균 0.9133→**0.9596**, 최저 0.3254→**0.6881**, 랜드마크 지터
  최대 26.5%→11.0%. 토글 5종(`ALIGN_ANTIALIAS`·`DETECT_REFINE_*` 등)으로 .env 롤백 가능.
- **EC2 배포 + ORT 스레드 정합** (2026-07-11) — Docker 이미지(arm64, 모델 프리베이크)를 ECR `cheesemoa-ai`로
  올려 EC2에서 상시 실행. 배포 직후 임베딩이 로컬 대비 45배 느렸는데, 원인은 CPU 크레딧 스로틀링이 아니라
  ORT 스레드 오버서브스크립션(2코어 호스트에 기본값 8스레드)이었다 — 코어 수 정합으로 **6배 개선**
  (2860ms/장 → 476ms/장). 실측: [worker-scaling-and-performance.md §7](guides/worker-scaling-and-performance.md)
- **confirm_distinct — 확정 앨범 간 오병합 방지 (계약 확장)** (2026-07-06) — must-link는 응집만 강제하고
  이격은 못해 다리 사진이 확정된 두 앨범을 오병합할 위험을 `cluster_feedback`의 4번째 action
  `confirm_distinct`(`cluster_ids` 대표 얼굴 전 쌍 cannot-link)로 방지. 실측 오병합 기하로 `handlers.py`
  자가검증(⑬)에 회귀 고정 (feature-spec §6.3).
- **uncertain 사진의 인물 앨범 편입 (계약 확장)** (2026-07-04) — 실 `cluster_id`가 없어 일반 reassign
  대상이 못 되던 uncertain 얼굴을, 예약 앨범 id `"__uncertain__"`(`uncertain[].album_id`)로 해결. 실 AWS
  end-to-end 검증 완료.
- **눈감음/흔들림 품질 게이트** (2026-07-04, CHMO-172) — `quality.py` 신설, CNN+Laplacian 판정으로
  `eyes_closed`/`blurry` 라우팅. `eye_closed_confidence=0.85` face-test 보정.
- **클러스터링 파라미터 ARI 스윕** (2026-07-04, [ADR 009](decisions/009-clustering-parameter-tuning.md)) —
  현행 `ClusterConfig` 값이 최적 근방·안전임을 확정(개선 후보 전부 회귀/과적합으로 기각).
- **소규모 단일 인물 이벤트 앨범 미생성 개선** (2026-07-04,
  [ADR 008](decisions/008-blob-promotion-connected-components.md)) — 연결 성분 부분 승격 + peel로
  재설계, 병합 임계 0.7 유지.
