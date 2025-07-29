FROM python:3.12-slim-bookworm
LABEL maintainer="mich910116@gmail.com"
LABEL version="1.0.0"
LABEL description="FastAPI service for Popularity-Based Recommendations"
WORKDIR /app
COPY requirements.txt .
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip # Clean up pip cache to minimize image size
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]