import pandas as pd
import io
import os
import tempfile
import pickle
import mlflow
import mlflow.pyfunc
import logging
from mlflow.tracking import MlflowClient
from utility.minio.minio_utility import MinIOUtility
from config.environment_config import Config

minio_utility = MinIOUtility()
config = Config()


class PopularityModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load the trained model artifacts"""
        pop_path = context.artifacts.get("popularity_recommender_model")

        with open(pop_path, "rb") as f:
            model_data = pickle.load(f)

        self.popularity_scores = model_data['popularity_scores']
        self.product_stats = model_data['product_stats']

    def predict(self, context, model_input=None):
        """Returns top-N popular products with full metadata"""
        top_n = 10
        if model_input is not None and 'top_n' in model_input.columns:
            top_n = int(model_input['top_n'].iloc[0])

        # Get top N products
        recommendations = self.popularity_scores.head(top_n).copy()

        # Merge with product stats to get full metadata
        result = recommendations.merge(
            self.product_stats,
            on="product_id",
            how="left"
        )

        # Return the required columns
        return result[[
            'product_id', 'title', 'brand', 'categories',
            'avg_rating', 'review_count', 'popularity_score'
        ]]

    def get_popular_products(self, top_n=10):
        """Method to get popular products directly (for client usage)"""
        recommendations = self.popularity_scores.head(top_n).copy()
        result = recommendations.merge(
            self.product_stats,
            on="product_id",
            how="left"
        )
        return result[[
            'product_id', 'title', 'brand', 'categories',
            'avg_rating', 'review_count', 'popularity_score'
        ]]


class PopularityRecommender:
    def __init__(self, category='fashion'):
        self.popularity_scores = None
        self.product_stats = None
        self.category = category
        self.minio_client = minio_utility.get_minio_client()

        # Set up MLflow S3 configuration for MinIO
        MINIO_HOST = config.MINIO_BIND["endpoint"]
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
        os.environ['AWS_ACCESS_KEY_ID'] = config.MINIO_BIND["access_key"]
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.MINIO_BIND["secret_key"]

    def load_data(self):
        """Load reviews and products data from MinIO"""
        # Load reviews
        reviews = self.minio_client.get_object(
            self.category, f"processed_{self.category}_reviews.parquet")
        reviews_data = reviews.read()
        reviews.close()
        reviews_df = pd.read_parquet(io.BytesIO(reviews_data))

        # Load products
        products = self.minio_client.get_object(
            self.category, f"processed_{self.category}_products.parquet")
        products_data = products.read()
        products.close()
        products_df = pd.read_parquet(io.BytesIO(products_data))

        return reviews_df, products_df

    def calculate_weighted_rating(self, reviews: pd.DataFrame):
        """Calculate weighted ratings using IMDB formula"""
        # Group by product_id and calculate statistics
        stats = reviews.groupby('product_id').agg({
            'rating': ['mean', 'count'],
            'helpful_votes': 'sum'
        }).round(4)

        # Flatten column names
        stats.columns = ['avg_rating', 'review_count', 'total_helpful_votes']
        stats = stats.reset_index()

        # Filter products with at least 5 reviews
        stats = stats[stats['review_count'] >= 5]

        # Calculate weighted rating using IMDB formula
        C = stats['avg_rating'].mean()  # Average rating across all products
        # Minimum reviews threshold (80th percentile)
        m = stats['review_count'].quantile(0.8)

        # Weighted rating formula: (v / (v + m)) * R + (m / (v + m)) * C
        # where v = review_count, R = avg_rating, m = minimum reviews, C = mean rating
        stats['weighted_rating'] = (
            (stats['review_count'] / (stats['review_count'] + m)) * stats['avg_rating'] +
            (m / (stats['review_count'] + m)) * C
        )

        return stats

    def prepare_data(self, reviews: pd.DataFrame, products: pd.DataFrame):
        """Prepare final dataset with product information"""
        # Calculate weighted ratings
        stats = self.calculate_weighted_rating(reviews)

        # Get product information
        product_info = products[['product_id',
                                 'title', 'categories', 'brand', 'price']]

        # Merge stats with product info
        final_data = stats.merge(product_info, on='product_id', how='left')

        return final_data

    def train_and_register(self):
        """Train the popularity model and register it with MLflow"""
        with mlflow.start_run(run_name="popularity_recommender_training"):
            # Log training parameters
            mlflow.log_param("recommendation_type", "popularity")
            mlflow.log_param("category", self.category)

            # Load and prepare data
            reviews, products = self.load_data()
            self.product_stats = self.prepare_data(reviews, products)

            # Create popularity scores (sorted by weighted rating)
            self.popularity_scores = self.product_stats[[
                'product_id', 'weighted_rating']].copy()
            self.popularity_scores.columns = ['product_id', 'popularity_score']

            # Sort by popularity score (descending)
            self.popularity_scores = self.popularity_scores.sort_values(
                'popularity_score', ascending=False
            ).reset_index(drop=True)

            # Log metrics
            mlflow.log_metric("total_products", len(self.popularity_scores))
            mlflow.log_metric("avg_popularity_score",
                              self.popularity_scores['popularity_score'].mean())
            mlflow.log_metric("max_popularity_score",
                              self.popularity_scores['popularity_score'].max())
            mlflow.log_metric("min_popularity_score",
                              self.popularity_scores['popularity_score'].min())

            model_data = {
                'popularity_scores': self.popularity_scores,
                'product_stats': self.product_stats
            }
            # Save model artifacts
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifacts_path = os.path.join(
                    tmp_dir, "popularity_recommender_model.pkl")
                with open(artifacts_path, "wb") as f:
                    pickle.dump(model_data, f)

                artifacts = {
                    "popularity_recommender_model": artifacts_path
                }

                mlflow.pyfunc.log_model(
                    artifact_path="popularity_recommender_model",
                    python_model=PopularityModelWrapper(),
                    artifacts=artifacts,
                )

                # Register model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/popularity_recommender_model"
                result = mlflow.register_model(
                    model_uri, "PopularityRecommenderModel")

                # Set champion alias
                client = MlflowClient()
                client.set_registered_model_alias(
                    "PopularityRecommenderModel", "champion", result.version)

            print(
                f"Training completed - {len(self.popularity_scores)} products")
            print(
                f"Top product score: {self.popularity_scores['popularity_score'].iloc[0]:.4f}")


class PopularityModelClient:
    def __init__(self, model_name="PopularityRecommenderModel", alias="champion"):
        """Initialize client to load registered model"""
        self.model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}@{alias}")

    def get_popular_products(self, top_n=10):
        """Get top N popular products"""
        return self.model._model_impl.python_model.get_popular_products(top_n)


if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Train and register model
    recommender = PopularityRecommender(category="fashion")
    recommender.train_and_register()

    # Test loading model from registry
    client = PopularityModelClient()
    top_products = client.get_popular_products()
    print(top_products)
