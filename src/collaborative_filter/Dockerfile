FROM python:3.12-slim-bookworm

LABEL maintainer="mich910116@gmail.com"
LABEL version="1.0.0"
LABEL description="FastAPI service for Content-Based Recommendations"

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]