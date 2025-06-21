import os
import pandas as pd
from pydantic import BaseModel
import logging
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Content-Based Recommendation Engine")


LOADED_MODEL = None
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000") # Default to localhost for local Docker
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
    os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY
    LOADED_MODEL = mlflow.pyfunc.load_model(
        model_uri=f"models:/ContentBasedModel@champion")


@app.post("/recommend/content_based", response_model=list[ProductRecommendation])
async def get_popularity_recommendations(request: RecommendationRequest):
    model_input_df = pd.DataFrame({'top_n': [request.top_n],'product_id': [request.product_id]})
    recommendations_df = LOADED_MODEL.predict(pd.DataFrame(model_input_df))

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


if __name__ == "__main__":
    import uvicorn
    # For local testing, ensure MinIO and MLflow are port-forwarded to localhost
    # MinIO: kubectl port-forward svc/minio-service 9000:9000 -n minio
    # MLflow: kubectl port-forward svc/mlflow-service 5000:5000 -n mlflow
    uvicorn.run(app, host="0.0.0.0", port=8000)
