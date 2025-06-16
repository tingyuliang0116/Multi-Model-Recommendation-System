from minio import Minio
from config.environment_config import Config

config = Config()


class MinIOUtility:
    def __init__(self):
        pass
    
    def get_minio_client(self):
        minioClient = Minio(
                endpoint=config.MINIO_BIND["endpoint"],
                access_key=config.MINIO_BIND["access_key"],
                secret_key=config.MINIO_BIND["secret_key"],
                secure=config.MINIO_BIND["secure"],
        )
        return minioClient