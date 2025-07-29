import pika
import os
import time
import json

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "30006")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")

QUEUE_NAME = 'recommendation_requests_queue'


def send_message(message_body: dict):
    connection = None
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=int(RABBITMQ_PORT),
                credentials=credentials,
                heartbeat=600
            )
        )
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        message_json = json.dumps(message_body)
        channel.basic_publish(
            exchange='',
            routing_key=QUEUE_NAME,
            body=message_json.encode('utf-8'),
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent
            )
        )
        print(f" [x] Sent '{message_json}' to '{QUEUE_NAME}'")
    except pika.exceptions.AMQPConnectionError as e:
        print(f" [!] Failed to connect to RabbitMQ: {e}")
    except Exception as e:
        print(f" [!] An error occurred: {e}")
    finally:
        if connection and not connection.is_closed:
            connection.close()


if __name__ == "__main__":
    print("--- RabbitMQ Producer ---")
    sample_messages = [
        {"recommendation_type": "popularity", "user_id": "A1B2C3D4E5", "top_n": 5},
        {"recommendation_type": "collaborative-filtering",
            "user_id": "A1ACV0K62X5L4Y", "top_n": 3},
        {"recommendation_type": "content-based",
            "product_id": "B00000J3O8", "limit": 2},
        {"recommendation_type": "popularity", "limit": 10}
    ]
    for i, msg in enumerate(sample_messages):
        time.sleep(1)
        send_message(msg)
