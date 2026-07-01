# 시스템 구조

CheeseMoa-AI는 전체 치즈모아(CheeseMoa) 시스템에서 AI 추론만 담당하는 워커 서버다.

## 전체 구조

```text
Flutter App
  │
  ▼
Spring API Server ─────────── PostgreSQL (메타데이터)
  │                                 ▲
  │ Presigned URL 발급               │ 분류 결과 저장
  ▼                                 │
AWS S3 (원본 이미지 저장)             │
  │                                 │
  ▼                                 │
Message Queue (AWS SQS) ───────────────┘
  │
  ▼
[이 서버] CheeseMoa-AI (Python 워커)
  │
  ├── YuNet (얼굴 감지)
  ├── face_align (Umeyama 유사변환)
  ├── AuraFace (임베딩)
  └── HDBSCAN (클러스터링)
       │
       ▼
  pgvector (임베딩 저장소)
```

## Spring 백엔드와의 연동

Spring 백엔드와 이 서버는 **AWS SQS 메시지 큐로만** 연동한다(비동기 단일 경로).

- Spring이 **요청 큐**에 분류 작업 메시지를 발행한다(producer).
- 이 서버가 **consumer**로 메시지를 소비해 파이프라인을 실행한다.
- 완료 후 **결과 큐**에 분류 결과를 발행하고, Spring이 이를 구독한다.

이 서버는 HTTP 엔드포인트를 제공하지 않는다. Python 워커 프로세스로 실행되며, 분류 요청은 오직 SQS로만 수신한다.

## 데이터 흐름

1. Spring이 S3에 원본 이미지를 업로드하고 S3 경로(또는 Presigned URL)를 SQS 요청 큐에 발행한다.
2. 이 서버의 consumer가 메시지를 수신해 AI 파이프라인을 실행한다.
3. 파이프라인 완료 후 클러스터 결과(인물 클러스터 → 이미지 목록 매핑)를 SQS 결과 큐에 발행한다.
4. 개별 임베딩과 클러스터 멤버십·대표벡터를 pgvector에 저장·갱신해 향후 전체 재군집에 재사용한다.

## AI 서버의 역할 범위

이 서버가 하는 것:

- 얼굴 감지 / 정렬 / 임베딩 / 클러스터링
- 임베딩 벡터 저장 (pgvector)

이 서버가 하지 않는 것:

- 사용자 인증
- 이미지 업로드 / 저장 (S3는 Spring이 관리)
- 앨범 메타데이터 관리 (PostgreSQL은 Spring이 관리)

## 관련 문서

- AI 파이프라인 상세: [pipeline-overview.md](./pipeline-overview.md)
