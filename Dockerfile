# CheeseMoa AI 워커 로컬 실행용 이미지 (SQS consumer)
FROM python:3.12-slim

# 런타임 시스템 의존성:
#  - libgomp1: onnxruntime(OpenMP)
#  - libglib2.0-0: opencv-python-headless(libgthread-2.0.so.0)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성을 먼저 설치해 코드 변경 시 레이어 캐시를 재활용한다
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드만 복사한다 (.env·자격증명은 이미지에 넣지 않고 런타임에 주입)
COPY app/ ./app/

# SQS consumer 워커 — 모델 적재 + SQS/S3 레디니스 통과 후 폴링 시작
CMD ["python", "-m", "app.worker"]
