from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def print_hello():
    print("✅ DAG is working via gitSync in Kubernetes!")

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='test_gitsync_dag',
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,  # Trigger manually
    catchup=False,
    tags=['test', 'gitsync'],
) as dag:
    
    hello_task = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello,
    )