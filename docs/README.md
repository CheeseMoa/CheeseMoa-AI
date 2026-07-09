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
- [007-embedding-storage-s3.md](./decisions/007-embedding-storage-s3.md) — 임베딩 저장소를 S3 blob(event 단위 `.npz`)으로 결정 (pgvector 대체·재군집 격리 단위 event)
- [008-blob-promotion-connected-components.md](./decisions/008-blob-promotion-connected-components.md) — 연결 성분 부분 승격 + peel 재설계 (소규모 단일 인물 이벤트 앨범 미생성 개선)
- [009-clustering-parameter-tuning.md](./decisions/009-clustering-parameter-tuning.md) — 클러스터링 파라미터 ARI 스윕 (현행 값 유지 확정)

## reviews — 코드 리뷰 기록

- [2026-07-09-face-detection-review.md](./reviews/2026-07-09-face-detection-review.md) — 얼굴 감지
  경로 리뷰 (유지보수성·운영 관점, 발견 7건 + 오탐 방지 기록)
- [2026-07-09-face-alignment-review.md](./reviews/2026-07-09-face-alignment-review.md) — 랜드마크
  추출·정렬 경로 리뷰 (감지 리뷰와 상보 — 랜드마크 순서 암묵 계약, `_ensure_bgr` 중복 근거 무효 등
  정렬 고유 발견 6건 + 오탐 방지 기록)
- [2026-07-09-face-embedding-review.md](./reviews/2026-07-09-face-embedding-review.md) — 정렬 크롭
  → 임베딩 경로 리뷰 (임베딩 고유 발견 — `.npz` 모델 provenance 미기록으로 인한 임베딩 공간 혼합
  리스크, ORT 스레드 하드코딩 등 4건 + 오탐 방지 기록)
- [2026-07-09-clustering-review.md](./reviews/2026-07-09-clustering-review.md) — 재군집·클러스터링
  경로 리뷰 (클러스터링 고유 발견 — O(N²) 재군집의 이벤트 규모 가드 부재, 재군집 요약 로그 부재
  등 6건 + 오탐 방지 기록)
- [2026-07-09-user-feedback-review.md](./reviews/2026-07-09-user-feedback-review.md) — 사용자
  보정(앨범 수동 수정) 경로 리뷰 (동작 개요 포함 — 보정 내구성의 앵커 얼굴 의존, 스테일 보정의
  succeeded 보고 등 4건 + later-wins 알고리즘 검증 기록)
- [2026-07-09-sqs-s3-review.md](./reviews/2026-07-09-sqs-s3-review.md) — SQS 메시징·S3 접근 경로
  리뷰 (발행 실패가 오류 정책 밖이라 256KB 초과 결과가 무통보 DLQ로 가는 버그, IAM
  `s3:ListBucket` 함정 등 6건 + 오탐 방지 기록)

## conventions — 개발 규칙

- [code-style.md](./conventions/code-style.md) — Python 워커 코드 컨벤션
- [project-structure.md](./conventions/project-structure.md) — 폴더 구조와 책임

## guides — 실행 가이드

- [local-docker-e2e-testing.md](./guides/local-docker-e2e-testing.md) — 로컬 Docker + 실 AWS(SQS/S3)
  end-to-end 테스트 절차 (AWS SSO 프로필 설정·세션 만료 대응 포함)
