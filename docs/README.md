# CheeseMoa-AI 문서

치즈모아(CheeseMoa) AI 서버 저장소의 문서 인덱스다.

## architecture — 설계

- [pipeline-overview.md](./architecture/pipeline-overview.md) — AI 파이프라인 전체 흐름과 각 단계 설명
- [system-overview.md](./architecture/system-overview.md) — 전체 시스템 구조와 Spring 백엔드 연동

## spec — 기능 명세

- [feature-spec.md](./spec/feature-spec.md) — 치즈모아 AI 서버 기능명세서 (SQS 연동·전체 재군집·정책 반영)

## decisions — 결정 기록 (ADR)

- [001-face-align-custom.md](./decisions/001-face-align-custom.md) — face_align 직접 구현 결정
- [002-hdbscan-sklearn.md](./decisions/002-hdbscan-sklearn.md) — HDBSCAN sklearn 라이브러리 유지 결정 (구현체 선택은 005로 대체)
- [003-full-reclustering.md](./decisions/003-full-reclustering.md) — 전체 재군집을 군집의 원천으로 삼는 결정
- [004-embedding-onnxruntime.md](./decisions/004-embedding-onnxruntime.md) — AuraFace 추론 런타임으로 onnxruntime을 채택한 결정
- [005-hdbscan-standalone-port.md](./decisions/005-hdbscan-standalone-port.md) — HDBSCAN을 PoC numpy 이식본으로 쓰는 결정
- [006-postprocessing-accuracy-hardening.md](./decisions/006-postprocessing-accuracy-hardening.md) — 재군집 후처리 정확도 보강 설계 (코드리뷰 반영 선택 기록)

## conventions — 개발 규칙

- [code-style.md](./conventions/code-style.md) — Python 워커 코드 컨벤션
- [project-structure.md](./conventions/project-structure.md) — 폴더 구조와 책임
