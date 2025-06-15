from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
from io import BytesIO, StringIO
from minio import Minio
from minio.error import S3Error
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_minio_client_for_airflow_task():
    """Initializes and returns a MinIO client using Airflow task environment variables."""
    minio_client = Minio(
        os.getenv("MINIO_HOST"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )
    return minio_client

def _preprocess_amazon_data_task(
    input_bucket: str, input_file_key: str,
    output_bucket: str, output_file_key: str,
    data_type: str # 'reviews' or 'products'
):
    """
    Airflow task function to read Amazon data from MinIO, preprocess, and write back.
    Handles .jsonl format for both review and product metadata.
    """
    minio_client = get_minio_client_for_airflow_task()
    logger.info(f"Airflow Task: Starting preprocessing for {data_type} data from {input_file_key} in {input_bucket}")

    try:
        logger.info(f"Downloading {input_file_key} from bucket {input_bucket}")
        response = minio_client.get_object(input_bucket, input_file_key)
        
        data_list = []
        for line in response.stream(chunk_size=1024):
            try:
                decoded_line = line.decode('utf-8').strip()
                if not decoded_line: # Skip empty lines
                    continue
                
                record = json.loads(decoded_line)
                
                if data_type == 'reviews':
                    # Extract review data fields
                    data_list.append({
                        'user_id': record.get('user_id'),
                        'product_id': record.get('parent_asin'), # 'parent_asin' as product_id
                        'rating': float(record.get('rating', 0)),
                        'review_text': record.get('text', ''), # 'text' as review_text
                        'timestamp': record.get('timestamp'),
                        'helpful_votes': record.get('helpful_vote', 0) # 'helpful_vote' as helpful_votes
                    })
                elif data_type == 'products':
                    # Extract product metadata fields
                    data_list.append({
                        'product_id': record.get('parent_asin'), # 'parent_asin' as product_id
                        'title': record.get('title', ''),
                        'categories': record.get('categories', []),
                        'brand': record.get('store', ''), # 'store' as brand
                        'price': record.get('price', ''),
                        'description': ' '.join(record.get('description', [])),
                        'features': record.get('feature', [])
                    })
                else:
                    raise ValueError(f"Unknown data_type: {data_type}. Must be 'reviews' or 'products'.")

            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON line in {input_file_key}: {decoded_line[:100]}... Error: {e}")
            except Exception as e:
                logger.warning(f"Error processing line in {input_file_key}: {decoded_line[:100]}... Error: {e}")

        df = pd.DataFrame(data_list)
        
        response.close()
        response.release_conn()
        
        logger.info(f"Raw {data_type} data loaded. Shape: {df.shape}")

        # 2. Basic Preprocessing (common for both, or specific if needed)
        # Handle missing values: drop rows where critical IDs are missing
        if data_type == 'reviews':
            df.dropna(subset=['user_id', 'product_id', 'rating'], inplace=True)
            df['user_id'] = df['user_id'].astype(str)
            df['product_id'] = df['product_id'].astype(str)
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            if 'review_text' in df.columns:
                df['review_text'] = df['review_text'].fillna('').astype(str).str.lower().str.strip()
                df['review_text'] = df['review_text'].str.replace(r'[^\w\s]', '', regex=True)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce') # Assuming milliseconds
        elif data_type == 'products':
            df.dropna(subset=['product_id', 'title'], inplace=True)
            df['product_id'] = df['product_id'].astype(str)
            if 'categories' in df.columns:
                # Convert list of categories to a string for easier storage if needed
                df['categories'] = df['categories'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            # Other product specific cleaning could go here
            
        logger.info(f"{data_type} data preprocessed. Shape: {df.shape}")

        # 3. Write processed data back to MinIO (as CSV)
        processed_data_buffer = StringIO()
        df.to_csv(processed_data_buffer, index=False)
        processed_data_bytes = processed_data_buffer.getvalue().encode('utf-8')
        
        if not minio_client.bucket_exists(output_bucket):
            minio_client.make_bucket(output_bucket)
            logger.info(f"Created MinIO bucket: {output_bucket}")
            
        minio_client.put_object(
            output_bucket,
            output_file_key,
            data=BytesIO(processed_data_bytes),
            length=len(processed_data_bytes),
            content_type='text/csv'
        )
        logger.info(f"Processed {data_type} data saved to s3://{output_bucket}/{output_file_key}")

    except S3Error as e:
        logger.error(f"MinIO S3 Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during preprocessing for {data_type} data: {e}")
        raise

with DAG(
    dag_id='amazon_data_preprocessing_pipeline', # Renamed DAG ID
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'data_pipeline', 'reviews', 'products', 'minio'],
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        'executor_config': {
            "pod_override": {
                "spec": {
                    "containers": [
                        {
                            "name": "base",
                            "env": [
                                {
                                    "name": "MINIO_HOST",
                                    "value": "minio-service.minio:9000"
                                },
                                {
                                    "name": "MINIO_ACCESS_KEY",
                                    "value": "minioadmin"
                                },
                                {
                                    "name": "MINIO_SECRET_KEY",
                                    "value": "minioadmin"
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }
) as dag:
    
    # Task to preprocess review data
    preprocess_reviews_task = PythonOperator(
        task_id='preprocess_amazon_reviews',
        python_callable=_preprocess_amazon_data_task,
        op_kwargs={
            'input_bucket': 'raw-amazon-reviews',
            'input_file_key': 'Amazon_Fashion.jsonl', # <--- IMPORTANT: Update with your review .jsonl filename
            'output_bucket': 'processed-amazon-data', # Unified output bucket
            'output_file_key': 'processed_amazon_reviews.csv',
            'data_type': 'reviews', # Specifies the data type for processing logic
        }
    )

    # Task to preprocess product metadata
    preprocess_products_task = PythonOperator(
        task_id='preprocess_amazon_products',
        python_callable=_preprocess_amazon_data_task,
        op_kwargs={
            'input_bucket': 'raw-amazon-products', # <--- IMPORTANT: Create this bucket and upload product data
            'input_file_key': 'meta_Amazon_Fashion.jsonl', # <--- IMPORTANT: Update with your product .jsonl filename
            'output_bucket': 'processed-amazon-data', # Unified output bucket
            'output_file_key': 'processed_amazon_products.csv',
            'data_type': 'products', # Specifies the data type for processing logic
        }
    )

    # Define dependencies (optional for this, but good practice if one needs to finish before another)
    # For now, they can run independently or in parallel.
    # preprocess_reviews_task >> some_next_task
    # preprocess_products_task >> some_other_next_task