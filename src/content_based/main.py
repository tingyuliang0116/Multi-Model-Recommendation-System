import os
import pandas as pd
from pydantic import BaseModel
import logging
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from prometheus_client import generate_latest, Counter, Histogram, Gauge
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from starlette.responses import Response
from functools import wraps  # For decorator
from time import perf_counter  # For decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Content-Based Recommendation Engine")


LOADED_MODEL = None
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
REQUEST_COUNT = Counter(
    'http_requests_total', 'Total HTTP requests', [
        'app_name', 'method', 'endpoint', 'status_code']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 'HTTP request latency', [
        'app_name', 'endpoint']
)
MODEL_LOAD_TIME = Gauge(
    'model_load_duration_seconds', 'Time taken to load model at startup', [
        'app_name', 'model_name']
)
RECOMMENDATION_COUNT = Counter(
    'recommendations_total', 'Total recommendations generated', [
        'app_name', 'model_type']
)
MODEL_LOAD_STATUS = Gauge(
    'model_load_status', 'Status of model loading (1=success, 0=failure)', [
        'app_name', 'model_name']
)


def measure_request_latency(app_name, endpoint_name):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = perf_counter()
            try:
                response = await func(*args, **kwargs)
                return response
            finally:
                end_time = perf_counter()
                duration = end_time - start_time
                REQUEST_LATENCY.labels(
                    app_name=app_name, endpoint=endpoint_name).observe(duration)
        return wrapper
    return decorator


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    product_id: str
    top_n: int = 10


class ProductRecommendation(BaseModel):
    product_id: str
    title: str
    brand: str
    similarity_score: float


@app.on_event("startup")
async def startup_event():
    global LOADED_MODEL
    model_load_start_time = perf_counter()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
    os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY
    LOADED_MODEL = mlflow.pyfunc.load_model(
        model_uri=f"models:/ContentBasedModel@champion")
    MODEL_LOAD_TIME.labels(app_name="content_based",
                           model_name="all-MiniLM-L6-v2").set(perf_counter() - model_load_start_time)
    MODEL_LOAD_STATUS.labels(
        # 1 for success
        app_name="content_based", model_name="all-MiniLM-L6-v2").set(1)


@app.post("/recommend/content_based", response_model=list[ProductRecommendation])
# <--- ADD THIS DECORATOR
@measure_request_latency(app_name="content_based", endpoint_name="/recommend/content_based")
async def get_popularity_recommendations(request: RecommendationRequest):
    model_input_df = pd.DataFrame(
        {'top_n': [request.top_n], 'product_id': [request.product_id]})
    recommendations_df = LOADED_MODEL.predict(pd.DataFrame(model_input_df))
    RECOMMENDATION_COUNT.labels(
        # <--- ADD THIS COUNTER
        app_name="content_based", model_type="all-MiniLM-L6-v2").inc()
    REQUEST_COUNT.labels(
        app_name="content_based", method="POST", endpoint="/recommend/content_based", status_code=200
    ).inc()
    if recommendations_df.empty:
        return []

    return [
        ProductRecommendation(
            product_id=row['product_id'],
            title=row['title'],
            similarity_score=round(row['similarity_score'], 2),
            brand=row['brand'],
            category=row['categories']
        ) for _, row in recommendations_df.iterrows()
    ]


@app.get("/health")
# <--- ADD THIS DECORATOR
@measure_request_latency(app_name="content_based", endpoint_name="/health")
async def health_check():
    # Ensure your model_loaded check is still here
    REQUEST_COUNT.labels(
        app_name="pcontent_based", method="GET", endpoint="/health", status_code=200
    ).inc()
    # Or LOADED_MODEL_CF for CF
    return {"status": "ok", "model_loaded": LOADED_MODEL is not None}


# --- NEW: Prometheus Metrics Endpoint ---
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    # For local testing, ensure MinIO and MLflow are port-forwarded to localhost
    # MinIO: kubectl port-forward svc/minio-service 9000:9000 -n minio
    # MLflow: kubectl port-forward svc/mlflow-service 5000:5000 -n mlflow
    uvicorn.run(app, host="0.0.0.0", port=8000)
