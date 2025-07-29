import pika
import os
import json
import httpx
import asyncio

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "30006")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")

API_GATEWAY_HOST = os.getenv("API_GATEWAY_HOST", "localhost")
API_GATEWAY_PORT = os.getenv("API_GATEWAY_PORT", "8080")

QUEUE_NAME = 'recommendation_requests_queue'


async def process_recommendation_request(request_payload: dict):
    recommendation_type = request_payload.get("recommendation_type")
    user_id = request_payload.get("user_id", None)
    product_id = request_payload.get("product_id", None)
    top_n = request_payload.get("top_n", 10)

    gateway_base_url = f"http://{API_GATEWAY_HOST}:{API_GATEWAY_PORT}"

    payload = {
        "recommendation_type": recommendation_type,
        "user_id": user_id,
        "product_id": product_id,
        "top_n": top_n
    }

    print(
        f" [*] Calling API Gateway for type: {recommendation_type} with payload: {payload}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{gateway_base_url}/recommend/{recommendation_type}", json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            print(f" [x] API Gateway responded for {recommendation_type}:")
            print(json.dumps(result, indent=2))

        except httpx.HTTPStatusError as e:
            print(
                f" [!] API Gateway returned error {e.response.status_code} for {recommendation_type}: {e.response.text}")
        except httpx.RequestError as e:
            print(
                f" [!] Could not connect to API Gateway for {recommendation_type}: {e}")
        except Exception as e:
            print(
                f" [!] An unexpected error occurred during API call for {recommendation_type}: {e}")


def callback(ch, method, properties, body):
    message_data = body.decode('utf-8')
    print(f" [x] Received raw message: {message_data}")

    try:
        request_payload = json.loads(message_data)
        asyncio.run(process_recommendation_request(request_payload))
    except json.JSONDecodeError:
        print(f" [!] Error: Received malformed JSON: {message_data}")
    except Exception as e:
        print(f" [!] Error processing message: {e}")
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(" [x] Message acknowledged.")


def start_consuming():
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
        print(' [*] Waiting for messages. To exit press CTRL+C')
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
        channel.start_consuming()

    except pika.exceptions.AMQPConnectionError as e:
        print(f" [!] Failed to connect to RabbitMQ: {e}.")
    except KeyboardInterrupt:
        print(' [!] Consumer interrupted. Exiting.')
    except Exception as e:
        print(f" [!] An error occurred: {e}")
    finally:
        if connection and not connection.is_closed:
            connection.close()


if __name__ == "__main__":
    print("--- RabbitMQ Consumer with API Gateway Interaction ---")
    start_consuming()
