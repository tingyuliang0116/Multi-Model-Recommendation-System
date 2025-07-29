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
from functools import wraps
from time import perf_counter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Collaborative Filtering Recommendation Engine")


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
    user_id: str
    product_id: str | None = None
    top_n: int = 10


class ProductRecommendation(BaseModel):
    product_id: str
    title: str
    brand: str
    pred_rating: float


@app.on_event("startup")
async def startup_event():
    global LOADED_MODEL
    model_load_start_time = perf_counter()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
    os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY
    LOADED_MODEL = mlflow.pyfunc.load_model(
        model_uri=f"models:/CollaborativeFilteringdModel@champion")
    MODEL_LOAD_TIME.labels(app_name="collaborative_filtering", model_name="SVD").set(
        perf_counter() - model_load_start_time)
    MODEL_LOAD_STATUS.labels(
        app_name="collaborative_filtering", model_name="SVD").set(1)


@app.post("/recommend/collaborative_filtering", response_model=list[ProductRecommendation])
@measure_request_latency(app_name="collaborative_filtering", endpoint_name="/recommend/collaborative_filtering")
async def get_recommendations(request: RecommendationRequest):
    model_input_df = pd.DataFrame(
        {'top_n': [request.top_n], 'user_id': [request.user_id]})
    recommendations_df = LOADED_MODEL.predict(pd.DataFrame(model_input_df))

    RECOMMENDATION_COUNT.labels(app_name="collaborative_filtering",
                                model_type="collaborative_filtering").inc()
    REQUEST_COUNT.labels(
        app_name="collaborative_filtering", method="POST", endpoint="/recommend/collaborative_filtering", status_code=200
    ).inc()
    if recommendations_df.empty:
        return []

    return [
        ProductRecommendation(
            product_id=row['product_id'],
            title=row['title'],
            pred_rating=round(row['pred_rating'], 5),
            brand=row['brand'],
            category=row['categories']
        ) for _, row in recommendations_df.iterrows()
    ]


@app.get("/health")
@measure_request_latency(app_name="collaborative_filtering", endpoint_name="/health")
async def health_check():
    REQUEST_COUNT.labels(
        app_name="collaborative_filtering", method="GET", endpoint="/health", status_code=200
    ).inc()
    return {"status": "ok", "model_loaded": LOADED_MODEL is not None}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
