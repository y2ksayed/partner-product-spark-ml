"""Simulates partner product events and publishes them to Kafka.

This is a stand-in for the real partner feeds (SFTP drops, webhooks, API
polls). It generates realistic-looking events -- product views, clicks,
purchases, and inventory updates -- so you can run the whole pipeline
end to end without any partner credentials.

Usage:
    python src/producer.py --rate 20 --duration 120
"""

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer

from config import Settings

# A small fake catalog: (product_id, partner_id, category, base_price).
# In production this would come from the partner catalog sync.
CATALOG = [
    ("P1001", "partner_acme", "electronics", 249.99),
    ("P1002", "partner_acme", "electronics", 79.50),
    ("P1003", "partner_beta", "home", 45.00),
    ("P1004", "partner_beta", "home", 129.99),
    ("P1005", "partner_gamma", "sports", 59.95),
    ("P1006", "partner_gamma", "sports", 199.00),
    ("P1007", "partner_acme", "books", 24.99),
    ("P1008", "partner_delta", "toys", 34.50),
    ("P1009", "partner_delta", "toys", 89.99),
    ("P1010", "partner_beta", "electronics", 349.00),
]

EVENT_TYPES = ["view", "click", "purchase", "inventory_update"]
# Purchases are rare compared to views; weights reflect that.
EVENT_WEIGHTS = [0.55, 0.25, 0.08, 0.12]


def random_event() -> dict:
    """Build one random partner product event."""
    product_id, partner_id, category, base_price = random.choice(CATALOG)
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]

    # Add a little price jitter so the stream isn't perfectly uniform.
    price = round(base_price * random.uniform(0.9, 1.1), 2)

    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "product_id": product_id,
        "partner_id": partner_id,
        "category": category,
        "price": price,
        "user_id": f"user_{random.randint(1, 500)}",
        "event_time": datetime.now(timezone.utc).isoformat(),
    }

    if event_type == "inventory_update":
        # Inventory snapshots carry stock instead of a user action.
        event["inventory"] = random.randint(0, 200)
        event.pop("user_id")
    elif event_type == "purchase":
        event["quantity"] = random.randint(1, 3)
        event["order_id"] = str(uuid.uuid4())

    return event


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish simulated partner product events to Kafka."
    )
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Events per second (default: 10)")
    parser.add_argument("--duration", type=int, default=60,
                        help="How long to run, in seconds (default: 60)")
    parser.add_argument("--topic", type=str, default=None,
                        help="Kafka topic (default from config)")
    parser.add_argument("--bootstrap", type=str, default=None,
                        help="Kafka bootstrap servers (default from config)")
    args = parser.parse_args()

    settings = Settings()
    topic = args.topic or settings.events_topic
    bootstrap = args.bootstrap or settings.kafka_bootstrap

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=3,
    )

    print(f"Producing to topic '{topic}' at ~{args.rate} events/sec "
          f"for {args.duration}s (Ctrl+C to stop early)")
    interval = 1.0 / args.rate
    deadline = time.time() + args.duration
    sent = 0
    try:
        while time.time() < deadline:
            event = random_event()
            # Key by product so one partition keeps a product's events ordered.
            producer.send(topic, key=event["product_id"], value=event)
            sent += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
    finally:
        producer.flush()
        producer.close()
        print(f"Done. Sent {sent} events.")


if __name__ == "__main__":
    main()
