FROM nvidia/cuda:12.1-runtime-ubuntu20.04

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 소스 코드 복사
COPY . .

# 기본 명령어
CMD ["python", "src/train.py"]
