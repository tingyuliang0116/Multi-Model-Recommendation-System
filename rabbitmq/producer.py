# rabbitmq_producer.py
import pika
import os
import time
import json # Import json for creating JSON messages

# RabbitMQ connection details
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "30006") # NodePort for AMQP
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password") # Your configured password

QUEUE_NAME = 'recommendation_requests_queue'

def send_message(message_body: dict): # Expects a dict now
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

        message_json = json.dumps(message_body) # Convert dict to JSON string
        
        channel.basic_publish(
            exchange='',
            routing_key=QUEUE_NAME,
            body=message_json.encode('utf-8'), # Messages should be bytes
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
    # --- IMPORTANT: Set these Environment Variables before running ---
    # export RABBITMQ_HOST=$(minikube ip)
    # export RABBITMQ_PORT=30006 # Your AMQP NodePort
    # export RABBITMQ_USER=user
    # export RABBITMQ_PASS=password # Your secure password

    print("--- RabbitMQ Producer ---")
    
    # These messages directly map to your UnifiedRecommendationRequest Pydantic model
    sample_messages = [
        # Popularity Request
        {"recommendation_type": "popularity", "user_id": "A1B2C3D4E5", "top_n": 5},
        # Collaborative Filtering Request (user_id is required)
        {"recommendation_type": "collaborative-filtering", "user_id": "A1ACV0K62X5L4Y", "top_n": 3},
        # Content-Based Request (product_id is required)
        {"recommendation_type": "content-based", "product_id": "B00000J3O8", "limit": 2},
        # Example with optional fields omitted
        {"recommendation_type": "popularity", "limit": 10}
    ]

    for i, msg in enumerate(sample_messages):
        time.sleep(1) # Simulate delay
        send_message(msg)