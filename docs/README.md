# CheeseMoa-AI 문서

치즈모아(CheeseMoa) AI 서버 저장소의 문서 인덱스다.

## architecture — 설계

- [pipeline-overview.md](./architecture/pipeline-overview.md) — AI 파이프라인 전체 흐름과 각 단계 설명
- [system-overview.md](./architecture/system-overview.md) — 전체 시스템 구조와 Spring 백엔드 연동
- [user-feedback-constraints.md](./architecture/user-feedback-constraints.md) — 사용자 보정(앨범 수동
  수정)이 must-link/cannot-link 제약으로 반영되는 과정 (액션 4종의 제약 번역, later-wins 조정, 재군집
  강제 순서, ID 승계·은퇴, 알려진 한계)

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
- [2026-07-10-reassign-mustlink-tiebreak.md](./reviews/2026-07-10-reassign-mustlink-tiebreak.md) —
  결함 재현 기록: reassign의 must-link 1대1 동률이 목적지 대표를 출처로 끌고 가 목적지 앨범을
  쪼개고 신규 id를 발급한다 (합성 임베딩 재현·원인 특정·수정 후보 검증 + **재현의 한계** 명시,
  사용자 보정 리뷰의 "문제 없음" 판단 2건 정정)
- [2026-07-11-duplicate-embedding-split.md](./reviews/2026-07-11-duplicate-embedding-split.md) —
  결함 기록(**수정 미착수**): 동일 사진이 다른 image_id로 재업로드돼 생긴 유사도 1.00 임베딩 쌍을
  HDBSCAN이 인물 군집 대신 선택해 앨범이 쌍 단위로 쪼개진다 (실 event 8 · S3 ETag로 재업로드 확정 ·
  재현 절차 포함). 부수 발견: `delete_request` 미도달로 인한 유령 행

## backlog — 예정 작업

- [2026-07-11-followups.md](./backlog/2026-07-11-followups.md) — 후속 작업 목록 (P0 데이터 오염 ·
  P1 저장소/배포 보강 · P2 계약 개편 + **검토 후 유지 결정** 기록: 로컬 디스크 저장소 기각,
  FastAPI 복원 불필요)
- [state-based-feedback-contract.md](./backlog/state-based-feedback-contract.md) — 제안(ADR-010 후보):
  사용자 보정을 액션 4종이 아니라 **검토완료 앨범의 멤버십 선언**으로 받는 계약 개편 (later-wins
  조정 로직 제거 · 단일 진실을 Spring으로 · 선결 조건은 "검토완료"의 의미에 대한 제품 합의)

## conventions — 개발 규칙

- [code-style.md](./conventions/code-style.md) — Python 워커 코드 컨벤션
- [project-structure.md](./conventions/project-structure.md) — 폴더 구조와 책임

## guides — 실행 가이드

- [ec2-deployment.md](./guides/ec2-deployment.md) — EC2 배포 (ECR + Docker). arm64 빌드·모델 프리베이크·
  IAM·SSM 운영 명령. 함정 3가지(아키텍처 불일치, `.env`의 `AWS_PROFILE`, SCP로 인한 롤 분리 불가)와
  미해결 리스크(Spring과 호스트 공유 → CPU 크레딧 스로틀)
- [local-docker-e2e-testing.md](./guides/local-docker-e2e-testing.md) — 로컬 Docker + 실 AWS(SQS/S3)
  end-to-end 테스트 절차 (AWS SSO 프로필 설정·세션 만료 대응 포함)
- [worker-scaling-and-performance.md](./guides/worker-scaling-and-performance.md) — 워커 동시성 모델과
  성능 실측 (프로세스·스레드 층위, 메모리/시간 병목 분리, 워커 수·인스턴스 사이징 판단 기준,
  개선 우선순위 — 클러스터링 리뷰의 O(N²) 추정치 실측 정정 포함)
