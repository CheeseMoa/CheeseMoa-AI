# CheeseMoa AI 워커 이미지 (SQS consumer) — 로컬 실행 및 EC2 배포 공용
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

# 모델 캐시 위치를 이미지 안으로 고정한다. 빌드·런타임이 같은 경로를 보므로 런타임은 캐시 히트만
# 하고 다운로드하지 않는다 (HF_HOME은 huggingface_hub, CHEESEMOA_CACHE_DIR은 UrlModelSource가 읽는다).
ENV HF_HOME=/opt/models/hf \
    CHEESEMOA_CACHE_DIR=/opt/models/cheesemoa

# 모델 프리베이크 — 3개 모델(YuNet·AuraFace·눈감음 CNN, ~264MB)을 빌드 시점에 받아 이미지에 굽는다.
# 이게 없으면 컨테이너가 재시작될 때마다 매번 다시 내려받아 콜드스타트가 길고 런타임이 외부
# 네트워크(HF Hub·OpenVINO)에 의존하게 된다.
#
# model_source.py는 표준 라이브러리만 임포트하는 자기완결 모듈이라 앱 코드 전체 없이 단독 실행된다.
# 이 파일만 먼저 복사하는 이유는 레이어 캐시다 — 파이프라인 코드를 고쳐도 모델 레이어는 재사용된다.
COPY app/core/model_source.py /tmp/prebake/model_source.py
RUN python -c "\
import sys; sys.path.insert(0, '/tmp/prebake'); \
import model_source as m; \
[print('prebaked:', f().resolve()) for f in (m.default_yunet_source, m.default_auraface_source, m.default_eye_source)]" \
 && rm -rf /tmp/prebake

# 프리베이크된 캐시만 쓰고 런타임에 HF Hub로 나가지 않는다 (캐시 미스면 조용히 받지 않고 즉시 실패).
ENV HF_HUB_OFFLINE=1

# 애플리케이션 코드만 복사한다 (.env·자격증명은 이미지에 넣지 않고 런타임에 주입)
COPY app/ ./app/

# SQS consumer 워커 — 모델 적재 + SQS/S3 레디니스 통과 후 폴링 시작
CMD ["python", "-m", "app.worker"]
