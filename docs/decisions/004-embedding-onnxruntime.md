# ADR 004: AuraFace 추론 런타임으로 onnxruntime을 채택한다

## Status

Accepted

## Context

AuraFace 임베딩 모델(`glintr100.onnx`)은 ONNX 포맷이다. YuNet 얼굴 감지는 OpenCV DNN
(`cv2.FaceDetectorYN`)으로 실행하고 있으므로, 임베딩도 `cv2.dnn.readNetFromONNX`로
런타임을 통일하는 방법과 onnxruntime을 새 의존성으로 추가하는 방법 중 선택해야 했다.

## Decision

onnxruntime(CPUExecutionProvider)을 채택한다.

## Rationale

- **PoC 파리티**: PoC(face-detection-PoC)가 onnxruntime으로 정확도·성능을 검증했다. 같은
  런타임을 쓰면 검증된 레시피를 그대로 이식할 수 있다 (동등성 검증: PoC 인라인 레시피 대비
  최대 절대 오차 0).
- **연산자 커버리지**: glintr100은 ResNet-100 급 대형 ArcFace 계열 그래프로, cv2.dnn의 ONNX
  연산자 커버리지는 이런 대형 그래프에서 보장이 불확실하다. onnxruntime은 ONNX 표준
  레퍼런스 런타임이다.
- **세션 단위 스레드 제어**: `SessionOptions`로 intra/inter op 스레드를 세션에 캡슐화해
  제어할 수 있어 8vCPU 배포 환경 튜닝이 명시적이다 (`cv2.setNumThreads`는 프로세스 전역).
- **동적 배치**: 입력 배치 축이 동적(`'None'`)이라 한 이미지의 여러 얼굴을 (N, 3, 112, 112)
  블롭 1회 추론으로 처리할 수 있다.

## Consequences

- `requirements.txt`에 `onnxruntime>=1.26,<2.0`을 추가한다 (PoC 검증 버전 1.26.0을 하한).
- YuNet(cv2 DNN)과 AuraFace(onnxruntime)의 추론 런타임이 이원화된다 — 단계별로 검증된
  레시피를 유지하는 비용으로 수용한다.
- `glintr100.onnx`의 출력 메타데이터가 배치 축을 1로 고정 선언해 N>1 배치 추론 시 무해한
  shape 경고가 발생한다. 세션 로그 레벨을 ERROR로 제한해 억제한다 (`embed.py` 주석 참조).
