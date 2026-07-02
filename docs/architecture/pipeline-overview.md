# AI 파이프라인 개요

CheeseMoa-AI의 핵심은 얼굴 인식 파이프라인이다. S3에서 이미지를 읽어 인물별로
클러스터링한 결과를 SQS 결과 큐로 발행한다.

## 파이프라인 흐름

```text
S3에서 이미지 읽기
  │
  ▼
YuNet (얼굴 감지 + 5점 랜드마크 추출)
  │
  ▼
face_align (Umeyama 유사변환 — 112×112 ArcFace 기준점으로 정렬)
  │
  ▼
AuraFace (512-dim 임베딩 벡터 생성)
  │
  ▼
HDBSCAN (인물별 클러스터링, cosine distance)
  │
  ▼
클러스터 결과 → SQS 결과 큐 발행 / pgvector 저장
```

## 각 단계

### 1. 얼굴 감지 — YuNet

OpenCV DNN 기반 경량 얼굴 감지 모델이다. 얼굴 바운딩 박스와 5개 랜드마크
(양쪽 눈, 코, 양쪽 입꼬리)를 추출한다.

구현 위치: `app/pipeline/detect.py`

### 2. 얼굴 정렬 — face_align

5점 랜드마크를 ArcFace 기준 좌표(`_ARCFACE_DST`)에 맞춰 Umeyama 유사변환으로
정렬한다. 출력은 항상 112×112 BGR 이미지다. RGB 변환은 embed 전처리에서 수행한다.

외부 의존성(insightface, skimage) 없이 OpenCV + NumPy만으로 직접 구현했다.
결정 근거: [decisions/001-face-align-custom.md](../decisions/001-face-align-custom.md)

구현 위치: `app/pipeline/align.py`

### 3. 임베딩 생성 — AuraFace

정렬된 얼굴 이미지를 512차원 벡터로 변환한다. 같은 인물의 얼굴은 코사인 공간에서
가깝게, 다른 인물은 멀게 위치한다.

구현 위치: `app/pipeline/embed.py`

### 4. 클러스터링 — HDBSCAN

512-dim 임베딩 벡터들을 코사인 거리 기반으로 군집화해 인물별 클러스터를 만든다.
sklearn의 HDBSCAN을 사용한다 (`eps=0.15`, `metric='cosine'`).

증분 매칭이 아니라 **매 트리거마다 group 전체 임베딩(기존+신규)을 재군집**하고, 그 결과를
기존 `cluster_id`에 재조정해 인물 번호의 연속성을 유지한다(정확도 최우선). 전략 상세:
[decisions/003-full-reclustering.md](../decisions/003-full-reclustering.md) 및 [spec/feature-spec.md](../spec/feature-spec.md) §4.

노이즈 레이블(`-1`)은 어느 인물에도 속하지 않는 얼굴이다.

결정 근거: [decisions/002-hdbscan-sklearn.md](../decisions/002-hdbscan-sklearn.md)

구현 위치: `app/pipeline/cluster.py`

## 관련 문서

- 전체 시스템 구조: [system-overview.md](./system-overview.md)
- 폴더 구조: [conventions/project-structure.md](../conventions/project-structure.md)
