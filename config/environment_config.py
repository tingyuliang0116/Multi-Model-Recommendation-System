import os


class Config(object):

    MINIO_BIND = {
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "secure": False,
    }

