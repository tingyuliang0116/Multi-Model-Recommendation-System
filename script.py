# test_mlflow_minio_connection.py

import mlflow
import os
import tempfile
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MLflow and MinIO Configuration (from Environment Variables) ---
# Make sure these match your port-forwarded services and MinIO credentials.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000") # Use localhost, as you're port-forwarding to it
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

def run_mlflow_minio_test():
    """
    Performs a simple MLflow run and logs an artifact to test MinIO connection.
    """
    logger.info(f"Setting MLflow Tracking URI to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Configure MLflow's S3 client to point to MinIO
    logger.info(f"Setting MLFLOW_S3_ENDPOINT_URL to: http://{MINIO_HOST}")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{MINIO_HOST}"
    os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

    experiment_name = "MinIO_Connection_Test"
    run_name = f"Test_Run_{int(time.time())}"

    logger.info(f"Creating/setting experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log a simple parameter
        mlflow.log_param("test_param", "hello_minio")

        # Create a dummy artifact file
        artifact_content = "This is a test artifact logged to MinIO via MLflow."
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(artifact_content)
            temp_artifact_path = f.name
        
        logger.info(f"Logging artifact from: {temp_artifact_path}")
        mlflow.log_artifact(temp_artifact_path, artifact_path="test_artifacts")
        logger.info(f"Artifact logged successfully as 'test_artifacts/test_file.txt' (or similar name).")
        
        # Clean up the temporary file
        os.remove(temp_artifact_path)
        
        logger.info("Test completed. Check MLflow UI and MinIO UI.")

if __name__ == "__main__":
    # You can set these environment variables directly here for convenience if not using bash export
    # os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    # os.environ["MINIO_HOST"] = "localhost:9000"
    # os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
    # os.environ["MINIO_SECRET_KEY"] = "minioadmin"

    run_mlflow_minio_test()