import os
import io
import pandas as pd
from utility.minio.minio_utility import MinIOUtility
from config.environment_config import Config
import logging
import json

config = Config()
minio_utility = MinIOUtility()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, category='fashion'):
        self.category = category
        self.minio_client = minio_utility.get_minio_client()

    def load_data(self, type='review'):
        if type == 'review':
            response_reviews = self.minio_client.get_object(
                self.category, f"{self.category}.jsonl")
            data = []
            for line in response_reviews:
                decoded_line = line.decode('utf-8').strip()
                if not decoded_line:
                    continue
                review = json.loads(decoded_line)
                data.append({
                    'user_id': review.get('user_id'),
                    'product_id': review.get('parent_asin'),
                    'rating': float(review.get('rating', 0)),
                    'review_text': review.get('text', ''),
                    'timestamp': review.get('timestamp'),
                    'helpful_votes': review.get('helpful_vote', 0)
                })
            data_df = pd.DataFrame(data)
            data_df['timestamp'] = pd.to_datetime(
                data_df['timestamp'], unit='ms', errors='coerce')
        elif type == 'product':
            response_products = self.minio_client.get_object(
                self.category, f"meta_{self.category}.jsonl")
            data = []
            for line in response_products:
                decoded_line = line.decode('utf-8').strip()
                if not decoded_line:
                    continue
                product = json.loads(decoded_line)
                data.append({
                    'product_id': product.get('parent_asin'),
                    'title': product.get('title', ''),
                    'categories': product.get('categories', []),
                    'brand': product.get('store', ''),
                    'price': product.get('price', ''),
                    'description': ' '.join(product.get('description', [])),
                    'features': product.get('feature', [])
                })
            data_df = pd.DataFrame(data)
        logger.info(f"Loaded {len(data_df)}")
        return data_df

    def clean_data(self, reviews: pd.DataFrame, products: pd.DataFrame):
        reviews = reviews.dropna(subset=['user_id', 'product_id', 'rating'])

        user_counts = reviews['user_id'].value_counts()
        product_counts = reviews['product_id'].value_counts()

        valid_users = user_counts[user_counts >= 5].index
        valid_products = product_counts[product_counts >= 5].index
        reviews_filtered = reviews[
            (reviews['user_id'].isin(valid_users)) &
            (reviews['product_id'].isin(valid_products))
        ]
        products_filtered = products[products['product_id'].isin(
            reviews_filtered['product_id'])]

        logger.info(
            f"After filtering: {len(reviews_filtered)} reviews, {len(products_filtered)} products")

        return reviews_filtered, products_filtered

    def save_processed_data(self, reviews: pd.DataFrame, products: pd.DataFrame):
        dataframes = {
            f"processed_{self.category}_reviews.parquet": reviews,
            f"processed_{self.category}_products.parquet": products
        }
        for filename, df in dataframes.items():
            buffer = io.BytesIO()
            df.to_parquet(buffer, engine='pyarrow', compression='snappy')
            buffer.seek(0)

            self.minio_client.put_object(
                bucket_name=self.category,
                object_name=filename,
                data=buffer,
                length=len(buffer.getvalue()),
                content_type='application/octet-stream'
            )

            logger.info(f"Saved {filename}: {df.shape}")


if __name__ == "__main__":
    processor = DataProcessor()
    reviews = processor.load_data(type='review')
    products = processor.load_data(type='product')
    reviews_clean, products_clean = processor.clean_data(reviews, products)
    processor.save_processed_data(reviews_clean, products_clean)
