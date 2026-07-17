# ADR 021: 눈감음 판정을 blendshape로 교체 — Face Landmarker litert 이식

## Status

Accepted (2026-07-17). 눈감음 판정(CHMO-172)의 1차 신호를 눈 패치 CNN(open-closed-eye-0001)에서
MediaPipe Face Landmarker의 eyeBlink blendshape로 교체. CNN 경로는 롤백 스위치
(`QUALITY_BLINK_THRESHOLD=0`)로만 남는다. ADR 019(자격 게이트)는 CNN 롤백 경로에서 유지.

## Context

눈 패치 CNN의 남은 실패는 도메인 문제였다 — 운전자 모니터링 학습이라 유아 눈 오탐, 보정 스톡
이미지 미탐(event 61), 수면 사진 미탐. 임계 조정 불가(출력 양극화), 원본 해상도 눈 crop도 기각
([리뷰](../reviews/2026-07-17-native-eyecrop-verdict.md) — 오히려 소프트한 입력이 학습 분포 정합).
A/B 실측([리뷰](../reviews/2026-07-17-mediapipe-blink-ab.md), 871 얼굴)에서 Face Landmarker
blendshape가 실패 3종을 전부 해결하고 585 자격 얼굴 오탐 0을 확인했다.

**배포 제약이 구현을 결정했다**: mediapipe pip 패키지는 linux aarch64 휠이 없어 EC2(t4g) 배포
불가. `ai-edge-litert`(Google 공식 TFLite 런타임)는 cp312 manylinux aarch64 휠이 있으므로,
face_landmarker.task(zip, Apache 2.0) 내부의 tflite 2개(랜드마크 2.5MB + blendshape 0.9MB)만
litert로 직접 실행하고 그래프 글루를 numpy/cv2로 이식했다(`app/pipeline/blink.py`) —
HDBSCAN·face_align 이식과 같은 레포 패턴. 검출은 mediapipe 자체 검출기 대신 기존 YuNet을 쓴다.

## Decision

`FaceBlinkScorer`(blink.py): YuNet 5점 → 회전 정사각 RoI → 256×256 crop → 랜드마크 모델(478점 +
presence) → 146점 서브셋을 blendshape 모델에 입력 → eyeBlinkLeft/Right. `judge_faces`는
presence ≥ `blink_presence_floor`(0.5)인 얼굴만 `min(blinkL, blinkR) ≥ blink_threshold`(0.40,
A/B 빈 구간 [0.36, 0.42])로 판정한다.

**presence 미달은 미판정 — CNN 폴백 없음.** 실측(871 얼굴): 폴백 CNN이 flagged하는 얼굴은
이중 검출 잔재 RoI의 눈 뜬 아기(4중복) 하나뿐 — 정탐 기여 0, 오탐만 재생산. 랜드마커가 얼굴이
아니라는 RoI에서 눈 패치 CNN의 확신은 신뢰 근거가 없다.

### 글루 파리티 (macOS mediapipe 0.10.35 참조 대비, 604 얼굴)

RoI 기하는 원본(BlazeFace rect × 1.5)과 달리 YuNet 5점의 회전좌표계 bbox 기준이라 배율·시프트를
스윕으로 재보정: **3.0 / −0.05** — |Δmin_blink| 중앙값 0.0082·p95 0.0577, t=0.40 판정 뒤집힘 0,
알려진 감음 6/6 재현(3.5는 안경 감은 눈이 무너짐). 커버리지는 오히려 개선 — mediapipe 자체
검출기의 판정 가능률 69% → YuNet RoI 91% (극단 회전·누운 옆얼굴을 정렬된 RoI로 회복, 기존
미검출이던 누운 수면 감음 사진이 새로 정탐).

## Consequences

- 코퍼스 재검증: 알려진 실패 3종 해소(스톡 여성 blink 0.73, 수면 0.43~0.75, 아기 미판정) +
  기존 정탐 전부 유지(기도·안경·event 61 감음 3장) + 고글·윙크 정상. flagged 신규분은 전부
  육안으로 "실제 감았거나 거의 감은" 얼굴(햇빛 찡그림·웃으며 감음) — 눈 감김을 문자 그대로
  재므로 웃으며 감은 캔디드도 잡힌다(의미론 변화, 임계로 조정 가능).
- event 61 end-to-end: 감음 5장 전부 eyes_closed (기존 3 + 미탐 2 회복), 나머지 6장 무변경.
- 비용: 모델 +3.7MB(프리베이크), RSS +35MB, 얼굴당 +3.0ms (임베딩 476ms 대비 무시 가능).
  aarch64 리눅스 컨테이너에서 litert 설치·추론 검증 완료(추가 apt 의존 없음).
- 롤백: `QUALITY_BLINK_THRESHOLD=0` — litert 임포트·모델 로딩을 통째로 건너뛰고 CNN+ADR-019
  경로(종전 동작)로 복귀한다.
- 한계: presence 미달 ~9%는 눈감음 미판정(실측상 감음 손실 0이나 코퍼스 한정), blendshape
  임계 0.40은 라벨셋 부재 상태의 A/B 보정값 — 실서비스 리포트 축적 시 재보정(다음 목표 3).
  모델 버전은 URL 고정(/1/) — 갱신 시 파리티 재검증 필수.
