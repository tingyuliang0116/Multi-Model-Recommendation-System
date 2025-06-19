import pandas as pd
import io
import os
import tempfile
import pickle
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utility.minio.minio_utility import MinIOUtility
from config.environment_config import Config

minio_utility = MinIOUtility()
config = Config()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# Default to localhost for local Docker
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")


class ContentBasedModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load the trained model artifacts"""
        pop_path = context.artifacts.get("content_based_model")

        with open(pop_path, "rb") as f:
            model_data = pickle.load(f)

        self.product_embeddings = model_data['product_embeddings']
        self.similarity_matrix = model_data['similarity_matrix']
        self.product_to_idx = model_data['product_to_idx']
        self.idx_to_product = model_data['idx_to_product']
        self.product_stats = model_data['product_stats']

    def predict(self, context, model_input=None):
        """Returns top-N popular products with full metadata"""
        top_n = 10
        if model_input is not None and 'top_n' in model_input.columns:
            top_n = int(model_input['top_n'].iloc[0])

        if model_input is not None and 'product_id' in model_input.columns:
            product_id = model_input['product_id'].iloc[0]

        if product_id not in self.product_to_idx:
            return []

        idx = self.product_to_idx[product_id]
        similarities = self.similarity_matrix[idx]

        # Get top similar products (excluding the product itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        results = []
        for sim_idx in similar_indices:
            sim_product_id = self.idx_to_product[sim_idx]
            similarity_score = similarities[sim_idx]
            results.append({
                "product_id": sim_product_id,
                "similarity_score": similarity_score
            })

        # Convert to DataFrame before merging
        results_df = pd.DataFrame(results)

        # Merge with product stats
        result = results_df.merge(
            self.product_stats,
            on='product_id',
            how='left'
        )

        return result[[
            'product_id', 'title', 'brand', 'categories', 'similarity_score'
        ]]

    def get_similar_products(self, product_id: str, top_n: int = 10):
        """Get similar products based on content"""
        if product_id not in self.product_to_idx:
            return []

        idx = self.product_to_idx[product_id]
        similarities = self.similarity_matrix[idx]

        # Get top similar products (excluding the product itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        results = []
        for sim_idx in similar_indices:
            sim_product_id = self.idx_to_product[sim_idx]
            similarity_score = similarities[sim_idx]
            results.append({
                "product_id": sim_product_id,
                "similarity_score": similarity_score
            })

        # Convert to DataFrame before merging
        results_df = pd.DataFrame(results)

        # Merge with product stats
        result = results_df.merge(
            self.product_stats,
            on='product_id',
            how='left'
        )

        return result[[
            'product_id', 'title', 'brand', 'categories', 'similarity_score'
        ]]


class ContentBasedRecommender:
    def __init__(self, category='fashion', model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.sentence_model = SentenceTransformer(model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english')

        self.product_embeddings = None
        self.product_texts = None
        self.similarity_matrix = None
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

    def prepare_product_text(self, reviews: pd.DataFrame, products: pd.DataFrame):
        review_texts = reviews.groupby('product_id')['review_text'].apply(
            lambda x: ' '.join([str(text) for text in x if pd.notna(text)])
        ).to_dict()

        def create_text_representation(row):
            texts = []

            # Title
            if pd.notna(row['title']):
                texts.append(str(row['title']))

            # Brand
            if pd.notna(row['brand']):
                texts.append(f"Brand: {row['brand']}")

            # Categories
            if isinstance(row['categories'], list) and len(row['categories']) > 0:
                texts.append("Categories: " +
                             " ".join([str(cat) for cat in row['categories']]))

            # Description
            if pd.notna(row['description']) and len(str(row['description'])) > 0:
                texts.append(str(row['description']))

            # Sample of reviews (first 500 chars)
            product_id = row['product_id']
            if product_id in review_texts:
                review_sample = review_texts[product_id][:500]
                texts.append(f"Reviews: {review_sample}")

            return " ".join(texts)

        products['combined_text'] = products.apply(
            create_text_representation, axis=1)

        return products

    def train_and_register(self):
        """Train content-based model"""
        with mlflow.start_run(run_name="content_based_recommender"):
            mlflow.log_param("recommendation_type", "content_based")
            mlflow.log_param("embedding_model", self.model_name)
            mlflow.log_param("category", self.category)

            # Prepare text data
            reviews, products = self.load_data()
            product_texts = self.prepare_product_text(reviews, products)
            self.product_stats = products
            self.product_texts = product_texts

            # Generate embeddings
            texts = product_texts['combined_text'].tolist()

            print("Generating sentence embeddings...")
            embeddings = self.sentence_model.encode(
                texts, show_progress_bar=True)
            self.product_embeddings = embeddings

            # Compute similarity matrix
            print("Computing similarity matrix...")
            self.similarity_matrix = cosine_similarity(embeddings)

            # Create product ID to index mapping
            self.product_to_idx = {
                pid: idx for idx, pid in enumerate(product_texts['product_id'])
            }
            self.idx_to_product = {
                idx: pid for pid, idx in self.product_to_idx.items()
            }

            mlflow.log_metric("num_products", len(product_texts))
            mlflow.log_metric("embedding_dim", embeddings.shape[1])

            model_data = {
                'product_embeddings': self.product_embeddings,
                'similarity_matrix': self.similarity_matrix,
                'product_to_idx': self.product_to_idx,
                'idx_to_product': self.idx_to_product,
                'product_stats': self.product_stats
            }

            with tempfile.TemporaryDirectory() as tmp_dir:
                artifacts_path = os.path.join(
                    tmp_dir, "content_based_model.pkl")
                with open(artifacts_path, "wb") as f:
                    pickle.dump(model_data, f)

                artifacts = {
                    "content_based_model": artifacts_path
                }

                mlflow.pyfunc.log_model(
                    artifact_path="content_based_model",
                    python_model=ContentBasedModelWrapper(),
                    artifacts=artifacts,
                )
                # Register model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/content_based_model"
                result = mlflow.register_model(
                    model_uri, "ContentBasedModel")
                # Set champion aliasf
                client = MlflowClient()
                client.set_registered_model_alias(
                    "ContentBasedModel", "champion", result.version)
                assert os.path.exists(tmp_dir)
            print("After block:", os.path.exists(tmp_dir)) 
            print(
                f"Content-based model trained on {len(product_texts)} products")


class ContentModelClient:
    """Client for interacting with the registered content-based model."""

    def __init__(self, model_name="ContentBasedModel", alias="champion"):
        self.model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}@{alias}")

    def get_recommendations(self, product_id, top_n=10):
        return self.model._model_impl.python_model.get_similar_products(product_id, top_n)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    recommender = ContentBasedRecommender(category="fashion")
    recommender.train_and_register()

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
    os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

    print("\n--- Testing client with registered model ---")
    try:
        client = ContentModelClient()
        test_product_id = 'B08P3X2BZ6'
        recommendations = client.get_recommendations(test_product_id, top_n=5)

        print(f"Recommendations for product '{test_product_id}':")
        print(recommendations)
    except Exception as e:
        print(
            f"Could not load model to test client. Ensure MLflow server is running. Error: {e}")
