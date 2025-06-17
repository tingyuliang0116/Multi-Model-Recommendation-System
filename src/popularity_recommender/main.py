import os
import pandas as pd
from pydantic import BaseModel
import logging
from config.environment_config import Config
import mlflow
import mlflow.pyfunc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()
app = FastAPI(title="Popularity-Based Recommendation Engine")


LOADED_MODEL = None


class RecommendationRequest(BaseModel):
    user_id: str | None = None  
    top_n: int = 10  


class ProductRecommendation(BaseModel):
    product_id: str
    title: str
    brand: str
    avg_rating: float
    review_count: int
    popularity_score: float


@app.on_event("startup")
async def startup_event():
    global LOADED_MODEL

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    MINIO_HOST = config.MINIO_BIND["endpoint"]
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
    os.environ['AWS_ACCESS_KEY_ID'] = config.MINIO_BIND["access_key"]
    os.environ['AWS_SECRET_ACCESS_KEY'] = config.MINIO_BIND["secret_key"]
    LOADED_MODEL = mlflow.pyfunc.load_model(
        model_uri=f"models:/PopularityRecommenderModel@champion")


@app.post("/recommend/popularity", response_model=list[ProductRecommendation])
async def get_popularity_recommendations(request: RecommendationRequest):
    model_input_df = pd.DataFrame({'top_n': [request.top_n]})
    recommendations_df = LOADED_MODEL.predict(pd.DataFrame(model_input_df))

    if recommendations_df.empty:
        return []

    return [
        ProductRecommendation(
            product_id=row['product_id'],
            title=row['title'],
            avg_rating=round(row['avg_rating'], 2),
            review_count=int(row['review_count']),
            popularity_score=round(row['popularity_score'], 2),
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
