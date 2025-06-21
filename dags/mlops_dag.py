from airflow import DAG
from pipelines.data_processor import DataProcessor
from pipelines.model_training.collaborative_filter_model import CollaborativeFilteringRecommender
from pipelines.model_training.content_based_model import ContentBasedRecommender
from pipelines.model_training.popularity_model import PopularityRecommender
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_processor():
    return DataProcessor()

def get_collaborative_filtering_recommender():
    return CollaborativeFilteringRecommender()

def get_content_based_recommender():
    return ContentBasedRecommender()

def get_popularity_recommender():
    return PopularityRecommender()

with DAG(
    dag_id='data_preprocessing_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule=None,
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
    
    load_reviews = PythonOperator(
        task_id='load_reviews',
        python_callable=lambda: get_processor().load_data('review'),
        dag=dag
    )

    load_products = PythonOperator(
        task_id='load_products',
        python_callable=lambda: get_processor().load_data('product'),
        dag=dag
    )

    clean_data = PythonOperator(
        task_id='clean_data',
        python_callable=lambda **context: get_processor().clean_data(
            reviews=context['task_instance'].xcom_pull(task_ids='load_reviews'),
            products=context['task_instance'].xcom_pull(task_ids='load_products')
        ),
        provide_context=True,
        dag=dag
    )
    
    save_data = PythonOperator(
        task_id='save_processed_data',
        python_callable=lambda **context: get_processor().save_processed_data(
            reviews=context['task_instance'].xcom_pull(task_ids='clean_data')[0],
            products=context['task_instance'].xcom_pull(task_ids='clean_data')[1]
        ),
        provide_context=True,
        dag=dag
    )

    train_collaborative_model = PythonOperator(
        tasak_id='train_collaborative_model',
        python_callable = get_collaborative_filtering_recommender.train_and_register(),
        provide_context=True,
        dag=dag
    )

    train_content_model = PythonOperator(
        tasak_id='train_content_model',
        python_callable = get_content_based_recommender.train_and_register(),
        provide_context=True,
        dag=dag
    )

    train_popularity_model = PythonOperator(
        tasak_id='train_popularity_model',
        python_callable = get_popularity_recommender.train_and_register(),
        provide_context=True,
        dag=dag
    )

    [load_reviews, load_products] >> clean_data >> save_data >> [train_collaborative_model, train_content_model, train_popularity_model]