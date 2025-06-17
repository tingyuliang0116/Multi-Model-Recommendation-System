import pandas as pd
import io
import os
import tempfile
from utility.minio.minio_utility import MinIOUtility
from config.environment_config import Config
import mlflow
import pickle
minio_utility = MinIOUtility()
config = Config()


class PopularityRecommender:
    def __init__(self, category='fashion'):
        self.popularity_scores = None
        self.product_stats = None
        self.category = category
        self.minio_client = minio_utility.get_minio_client()
        MINIO_HOST = config.MINIO_BIND["endpoint"]
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
        os.environ['AWS_ACCESS_KEY_ID'] = config.MINIO_BIND["access_key"]
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.MINIO_BIND["secret_key"]

    def load_data(self):
        reviews = self.minio_client.get_object(
            self.category, f"processed_{self.category}_reviews.parquet")
        reviews_data = reviews.read()
        reviews.close()
        buffer = io.BytesIO(reviews_data)
        reviews_df = pd.read_parquet(buffer)

        products = self.minio_client.get_object(
            self.category, f"processed_{self.category}_products.parquet")
        products_data = products.read()
        products.close()
        buffer = io.BytesIO(products_data)
        products_df = pd.read_parquet(buffer)

        return reviews_df, products_df

    def calculate_weighted_rating(self, reviews: pd.DataFrame):
        """Calculate weighted rating using IMDB formula"""
        # Calculate stats per product
        product_stats = reviews.groupby('product_id').agg({
            'rating': ['mean', 'count'],
            'helpful_votes': 'sum'
        }).round(4)

        # Flatten column names
        product_stats.columns = ['avg_rating',
                                 'review_count', 'total_helpful_votes']
        product_stats = product_stats.reset_index()

        # Filter products with minimum reviews
        product_stats = product_stats[product_stats['review_count']
                                      >= 5]

        # Calculate weighted rating (IMDB formula)
        # WR = (v/(v+m)) * R + (m/(v+m)) * C
        # Where: v = votes, m = minimum votes, R = average rating, C = mean rating across all products

        # Mean rating across all products
        C = product_stats['avg_rating'].mean()
        m = product_stats['review_count'].quantile(
            0.8)  # 80th percentile of review counts

        product_stats['weighted_rating'] = (
            (product_stats['review_count'] / (product_stats['review_count'] + m)) * product_stats['avg_rating'] +
            (m / (product_stats['review_count'] + m)) * C
        )

        return product_stats

    def prepare_data(self, reviews: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """Prepare and calculate popularity scores"""
        # Calculate basic statistics
        product_stats = self.calculate_weighted_rating(reviews)

        # Add product information
        product_info = products[['product_id', 'title',
                                 'categories', 'brand', 'price']].copy()
        product_stats = product_stats.merge(
            product_info, on='product_id', how='left')

        return product_stats

    def train(self):
        """Train popularity model"""
        with mlflow.start_run(run_name="popularity_recommender_training"):

            mlflow.log_param("recommendation_type", "popularity")
            reviews, products = self.load_data()

            # Prepare data
            self.product_stats = self.prepare_data(reviews, products)

            self.popularity_scores = self.product_stats[[
                'product_id', 'weighted_rating']].copy()
            self.popularity_scores.columns = ['product_id', 'popularity_score']

            # Sort by popularity score
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

            with tempfile.TemporaryDirectory() as tmp_dir:
                for name, df in {
                    "popularity_scores.pkl": self.popularity_scores,
                    "product_stats.pkl": self.product_stats
                }.items():
                    tmp_file_path = os.path.join(tmp_dir, name)
                    with open(tmp_file_path, "wb") as f:
                        pickle.dump(df, f)

                mlflow.log_artifacts(tmp_dir, artifact_path="artifacts")

            config.POPULARITY_MODEL_ARTIFACT = mlflow.get_artifact_uri("artifacts")
            local_path = mlflow.artifacts.download_artifacts(config.POPULARITY_MODEL_ARTIFACT)
            print(local_path)

            print(popularity_scores.head())

            print(
                f"Training completed - {len(self.popularity_scores)} products")
            print(
                f"Top product score: {self.popularity_scores['popularity_score'].iloc[0]:.4f}")

    def get_popular_products(self) -> pd.DataFrame:
        """Get popular products with details"""

        recommendations = self.popularity_scores.copy()

        # Return top N recommendations
        top_recommendations = recommendations.head(10)
        recommendations = [(row['product_id'], row['popularity_score'])
                           for _, row in top_recommendations.iterrows()]

        # Get product details
        product_ids = [rec[0] for rec in recommendations]
        product_details = self.product_stats[
            self.product_stats['product_id'].isin(product_ids)
        ].copy()

        # Merge with popularity scores to maintain order
        result = pd.DataFrame(recommendations, columns=[
                              'product_id', 'popularity_score'])
        result = result.merge(product_details, on='product_id', how='left')

        return result[['product_id', 'title', 'brand', 'categories', 'avg_rating',
                      'review_count', 'popularity_score']]


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mode = PopularityRecommender()
    mode.train()
