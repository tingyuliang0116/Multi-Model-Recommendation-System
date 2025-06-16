import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minio import Minio
from minio.error import S3Error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Popularity-Based Recommendation Engine")

# --- MinIO Configuration ---
# These will be read from environment variables when the Docker container runs
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000") # Default to localhost for local Docker
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true" # "False" by default for local

PROCESSED_DATA_BUCKET = "processed-amazon-data"
PROCESSED_REVIEWS_FILE = "processed_amazon_reviews.csv"
PROCESSED_PRODUCTS_FILE = "processed_amazon_products.csv"

# --- Data Storage for Recommendations ---
# In a real system, this might be a database or feature store for faster lookup
processed_reviews_df = pd.DataFrame()
processed_products_df = pd.DataFrame()
popular_products = []

# --- Pydantic Models for Request/Response ---
class RecommendationRequest(BaseModel):
    user_id: str | None = None # User ID is optional for popularity-based
    limit: int = 10 # Number of recommendations to return

class ProductRecommendation(BaseModel):
    product_id: str
    product_title: str | None
    average_rating: float
    review_count: int

# --- Utility Functions ---
def get_minio_client():
    """Initializes and returns a MinIO client."""
    return Minio(
        MINIO_HOST,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

async def load_and_preprocess_data():
    """Loads processed data from MinIO and computes popularity."""
    global processed_reviews_df, processed_products_df, popular_products
    minio_client = get_minio_client()

    try:
        logger.info(f"Loading {PROCESSED_REVIEWS_FILE} from {PROCESSED_DATA_BUCKET}...")
        # Use s3fs for pandas to directly read from MinIO (S3-compatible)
        # s3fs requires fsspec, aiobotocore, botocore, etc.
        reviews_s3_path = f"s3://{PROCESSED_DATA_BUCKET}/{PROCESSED_REVIEWS_FILE}"
        processed_reviews_df = pd.read_csv(
            reviews_s3_path,
            storage_options={
                "key": MINIO_ACCESS_KEY,
                "secret": MINIO_SECRET_KEY,
                "client_kwargs": {"endpoint_url": f"http://{MINIO_HOST}"} # Must be http://
            }
        )
        logger.info(f"Loaded reviews: {processed_reviews_df.shape[0]} rows")

        logger.info(f"Loading {PROCESSED_PRODUCTS_FILE} from {PROCESSED_DATA_BUCKET}...")
        products_s3_path = f"s3://{PROCESSED_DATA_BUCKET}/{PROCESSED_PRODUCTS_FILE}"
        processed_products_df = pd.read_csv(
            products_s3_path,
            storage_options={
                "key": MINIO_ACCESS_KEY,
                "secret": MINIO_SECRET_KEY,
                "client_kwargs": {"endpoint_url": f"http://{MINIO_HOST}"}
            }
        )
        logger.info(f"Loaded products: {processed_products_df.shape[0]} rows")

        # Calculate popularity: average rating and review count
        if not processed_reviews_df.empty:
            product_ratings = processed_reviews_df.groupby('product_id')['star_rating'].agg(
                average_rating='mean',
                review_count='size'
            ).reset_index()

            # Merge with product titles
            product_ratings = product_ratings.merge(
                processed_products_df[['product_id', 'title']].drop_duplicates(subset=['product_id']),
                on='product_id',
                how='left'
            )

            # Sort by a combination (e.g., average rating, then review count)
            # You might use a weighted average here for more robustness
            popular_products_df = product_ratings.sort_values(
                by=['average_rating', 'review_count'],
                ascending=[False, False]
            ).fillna({'title': 'Unknown Product'}) # Fill NA titles

            popular_products = popular_products_df.to_dict(orient='records')
            logger.info(f"Computed {len(popular_products)} popular products.")
        else:
            logger.warning("No review data loaded to compute popularity.")


    except S3Error as e:
        logger.error(f"MinIO S3 Error during data loading: {e}")
        raise HTTPException(status_code=500, detail="Could not load data from MinIO.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading or popularity computation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during data loading.")

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    logger.info("Recommendation engine starting up...")
    content_model = PopularityRecommender()
    content_model.train(train_reviews, products_clean)
    logger.info("Recommendation engine startup complete.")

# --- Recommendation Endpoint ---
@app.post("/recommend/popularity", response_model=list[ProductRecommendation])
async def get_popularity_recommendations(request: RecommendationRequest):
    """
    Returns popularity-based product recommendations.
    """
    if not popular_products:
        raise HTTPException(status_code=503, detail="Popularity data not yet loaded or available.")

    # Return top N popular products
    recommendations = popular_products[:request.limit]

    # Convert to Pydantic models
    return [
        ProductRecommendation(
            product_id=p['product_id'],
            product_title=p['title'],
            average_rating=round(p['average_rating'], 2),
            review_count=int(p['review_count'])
        ) for p in recommendations
    ]

@app.get("/health")
async def health_check():
    return {"status": "ok", "data_loaded": not processed_reviews_df.empty}

if __name__ == "__main__":
    import uvicorn
    # For local testing, ensure MinIO is port-forwarded to localhost:9000
    # kubectl port-forward pod/minio 9000:9000 -n minio-dev
    uvicorn.run(app, host="0.0.0.0", port=8000)