import pandas as pd
import io
import os
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import mlflow
import mlflow.artifacts
import mlflow.sklearn
from typing import List, Tuple
from utility.minio.minio_utility import MinIOUtility
from config.environment_config import Config
import pickle
import tempfile
minio_utility = MinIOUtility()
config = Config()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# Default to localhost for local Docker
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")


class CollaborativeModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pop_path = context.artifacts.get("collaborative_filtering_model")
        with open(pop_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["svd_model"]
        self.trainset = model_data["trainset"]
        self.product_stats = model_data['product_stats']

    def predict(self, context, model_input):
        if model_input is not None and 'user_id' in model_input.columns:
            user_id = model_input['user_id'].iloc[0]

        if model_input is not None and 'top_n' in model_input.columns:
            top_n = model_input['top_n'].iloc[0]

        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            user_exists = True
        except ValueError:
            print(
                f"Warning: User {user_id} not found in training set. Using global mean predictions.")
            user_exists = False
        
        all_items = set([self.trainset.to_raw_iid(inner_iid)
                        for inner_iid in self.trainset.all_items()])

        # Get items the user has already rated (if user exists)
        if user_exists:
            user_items = set([self.trainset.to_raw_iid(inner_iid)
                              for (inner_uid, inner_iid, _) in self.trainset.all_ratings()
                              if inner_uid == self.trainset.to_inner_uid(user_id)])
        else:
            user_items = set()

        candidate_items = all_items - user_items

        predictions = []
        for item_id in candidate_items:
            try:
                pred = self.model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))
            except Exception as e:
                print(f"Error predicting for item {item_id}: {e}")
                continue

        predictions.sort(key=lambda x: x[1], reverse=True)

        results = predictions[:top_n]
        results_df = pd.DataFrame(results, columns=['product_id', 'pred_rating'])

        result = results_df.merge(
            self.product_stats,
            on='product_id',
            how='left'
        )

        return result[[
            'product_id', 'title', 'brand', 'categories', 'pred_rating'
        ]]

    def recommend(self, user_id: str, top_n: int = 10):
        if self.model is None or self.trainset is None:
            raise ValueError("Model not trained yet")

        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            user_exists = True
        except ValueError:
            print(
                f"Warning: User {user_id} not found in training set. Using global mean predictions.")
            user_exists = False

        # Get all items from training set
        all_items = set([self.trainset.to_raw_iid(inner_iid)
                        for inner_iid in self.trainset.all_items()])

        # Get items the user has already rated (if user exists)
        if user_exists:
            user_items = set([self.trainset.to_raw_iid(inner_iid)
                              for (inner_uid, inner_iid, _) in self.trainset.all_ratings()
                              if inner_uid == self.trainset.to_inner_uid(user_id)])
        else:
            user_items = set()

        candidate_items = all_items - user_items

        predictions = []
        for item_id in candidate_items:
            try:
                pred = self.model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))
            except Exception as e:
                print(f"Error predicting for item {item_id}: {e}")
                continue

        predictions.sort(key=lambda x: x[1], reverse=True)

        results = predictions[:top_n]
        results_df = pd.DataFrame(results, columns=['product_id', 'pred_rating'])

        result = results_df.merge(
            self.product_stats,
            on='product_id',
            how='left'
        )
        return result[[
            'product_id', 'title', 'brand', 'categories', 'pred_rating'
        ]]


class CollaborativeFilteringRecommender:
    def __init__(self, category='fashion', n_factors=100, n_epochs=30, validation_split=0.2):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.model = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs)
        self.trainset = None
        self.product_stats = None
        self.category = category
        self.validation_split = validation_split
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

    def prepare_data(self, reviews: pd.DataFrame) -> Dataset:
        """Prepare data for Surprise library"""
        # Surprise expects (user, item, rating) format
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            reviews[['user_id', 'product_id', 'rating']],
            reader
        )
        return dataset

    def train_and_register(self):
        """Train collaborative filtering model"""
        with mlflow.start_run(run_name="collaborative_filtering_recommender"):
            mlflow.log_param("recommendation_type", "collaborative_filtering")
            mlflow.log_param("n_factors", self.n_factors)
            mlflow.log_param("n_epochs", self.n_epochs)
            mlflow.log_param("algorithm", 'SVD')
            mlflow.log_param("category", self.category)

            # Load and prepare data
            reviews, products = self.load_data()
            self.product_stats = products
            dataset = self.prepare_data(reviews)
            trainset, testset = train_test_split(
                dataset, test_size=self.validation_split)

            # Train model
            self.model.fit(trainset)
            self.trainset = trainset

            # Evaluate model
            predictions = self.model.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            model_data = {
                'svd_model': self.model,
                'trainset': self.trainset,
                'product_stats': self.product_stats
            }

            with tempfile.TemporaryDirectory() as tmp_dir:
                artifacts_path = os.path.join(
                    tmp_dir, "collaborative_filtering_model")
                with open(artifacts_path, "wb") as f:
                    pickle.dump(model_data, f)

                artifacts = {
                    "collaborative_filtering_model": artifacts_path
                }

                mlflow.pyfunc.log_model(
                    artifact_path="collaborative_filtering_model",
                    python_model=CollaborativeModelWrapper(),
                    artifacts=artifacts,
                )

                # Register model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/collaborative_filtering_model"
                result = mlflow.register_model(
                    model_uri, "CollaborativeFilteringdModel")

                # Set champion alias
                client = MlflowClient()
                client.set_registered_model_alias(
                    "CollaborativeFilteringdModel", "champion", result.version)

            print(f"Training completed - RMSE: {rmse:.4f}, MAE: {mae:.4f}")


class CollaborativeFilteringModelClient:
    """Client for interacting with the registered collaborative filtering model."""

    def __init__(self, model_name="CollaborativeFilteringdModel", alias="champion"):
        self.model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}@{alias}")

    def get_recommendations(self, user_id, top_n=10):
        return self.model._model_impl.python_model.recommend(user_id, top_n)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    recommender = CollaborativeFilteringRecommender(category="fashion")
    recommender.train_and_register()

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
    os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

    print("\n--- Testing client with registered model ---")
    try:
        client = CollaborativeFilteringModelClient()
        test_id = 'AFSKPY37N3C43SOI5IEXEK5JSIYA'
        recommendations = client.get_recommendations(test_id, top_n=10)
        print(recommendations)

    except Exception as e:
        print(f"Could not load model to test client. Error: {e}")
        import traceback
        traceback.print_exc()
