import os
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import logging
from config.environment_config import Config
import mlflow
import mlflow.pyfunc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()
app = FastAPI(title="Popularity-Based Recommendation Engine")

# --- MLflow and MinIO Configuration (from Environment Variables) ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") # Default to localhost for local Docker
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000") # Default to localhost for local Docker
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
# Note: MINIO_SECURE is often "False" for local MinIO. MLflow's S3 client handles this via endpoint_url.

MODEL_NAME = "Amazon_Popularity_Recommender"
MODEL_STAGE = "Production" # Or "Staging", "Latest"
LOADED_MODEL = None # Global variable to hold the loaded MLflow model

# --- Pydantic Models for Request/Response ---
class RecommendationRequest(BaseModel):
    user_id: str | None = None # User ID is optional for popularity-based
    limit: int = 10 # Number of recommendations to return

class ProductRecommendation(BaseModel):
    product_id: str
    product_title: str | None
    average_rating: float
    review_count: int

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Load the MLflow model when the FastAPI application starts up.
    """
    global LOADED_MODEL
    logger.info("Recommendation engine starting up...")

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        # Configure S3 client for MLflow artifact store (MinIO)
        MINIO_HOST = config.MINIO_BIND["endpoint"]
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
        os.environ['AWS_ACCESS_KEY_ID'] = config.MINIO_BIND["access_key"]
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.MINIO_BIND["secret_key"]
        
        
        
    
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        # In a real app, you might want to gracefully shut down or retry
        raise RuntimeError(f"Application startup failed: Could not load ML model. Error: {e}")

# --- Recommendation Endpoint ---
@app.post("/recommend/popularity", response_model=list[ProductRecommendation])
async def get_popularity_recommendations(request: RecommendationRequest):
    """
    Returns popularity-based product recommendations using the loaded MLflow model.
    """
    if LOADED_MODEL is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not yet loaded.")

    try:
        # Prepare input for the MLflow model's predict method
        # The model's predict method expects a pandas DataFrame as input.
        # Even if just passing a 'limit', it should be in DataFrame format.
        model_input_df = pd.DataFrame({'limit': [request.limit]})
        
        # Call the loaded MLflow model's predict method
        recommendations_df = LOADED_MODEL.predict(model_input_df)
        
        if recommendations_df.empty:
            return []

        # Convert DataFrame to list of Pydantic models
        return [
            ProductRecommendation(
                product_id=row['product_id'],
                product_title=row['product_title'],
                average_rating=round(row['average_rating'], 2),
                review_count=int(row['review_count'])
            ) for index, row in recommendations_df.iterrows()
        ]

    except Exception as e:
        logger.error(f"Error during recommendation inference: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error during recommendation generation.")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Reports 'ok' if the application is running and the model is loaded.
    """
    return {"status": "ok", "model_loaded": LOADED_MODEL is not None}

if __name__ == "__main__":
    import uvicorn
    # For local testing, ensure MinIO and MLflow are port-forwarded to localhost
    # MinIO: kubectl port-forward svc/minio-service 9000:9000 -n minio
    # MLflow: kubectl port-forward svc/mlflow-service 5000:5000 -n mlflow
    uvicorn.run(app, host="0.0.0.0", port=8000)