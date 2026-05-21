"""
tests/e2e_latency_test.py
=========================
Automated benchmark script to measure the End-to-End (E2E) visibility latency 
of the C-MAPSS Lambda Architecture.

Methodology:
1. Generates a synthetic "canary" telemetry reading (unit_id=9999).
2. Starts a high-resolution timer and publishes to Kafka.
3. Rapidly polls Elasticsearch until the canary document is indexed and searchable.
4. Calculates the exact wall-clock latency across the entire pipeline.

Usage:
    conda run -n pytorch-gpu python -m tests.e2e_latency_test
"""

import os
import json
import time
import argparse
import logging
from typing import Final

from kafka import KafkaProducer  # type: ignore[import]
from elasticsearch import Elasticsearch  # type: ignore[import]

from src.core.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Force logging to UTC (GMT) to ensure consistency across the distributed system
logging.Formatter.converter = time.gmtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [E2E_LATENCY_TEST] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAFKA_TOPIC: Final[str] = "cmapss_telemetry"
DEFAULT_ES_INDEX: Final[str] = "cmapss_predictions_v1"
DEFAULT_KAFKA_BROKER: Final[str] = "localhost:9092"
DEFAULT_ES_HOST: Final[str] = "http://localhost:9200"

CANARY_UNIT_ID: Final[int] = 9999
CANARY_CYCLE: Final[int] = 1
POLL_INTERVAL_SECONDS: Final[float] = 0.05
TIMEOUT_SECONDS: Final[int] = 30


def generate_canary_payload() -> dict:
    """Generates a dummy telemetry payload mimicking the C-MAPSS schema."""
    payload = {
        "unit_id": CANARY_UNIT_ID,
        "time_cycles": CANARY_CYCLE,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 0.0,
    }
    # Add dummy sensors 1-21
    for i in range(1, 22):
        payload[f"sensor_{i}"] = 0.0
        
    return payload


def run_e2e_latency_benchmark(
    es_index: str,
    kafka_broker: str,
    es_host: str,
    kafka_topic: str = KAFKA_TOPIC,
) -> None:
    """Executes the E2E latency test using a black-box polling strategy."""
    
    # 1. Initialize Clients
    logger.info("Initializing clients...")
    try:
        es_client = Elasticsearch(es_host)
        if not es_client.ping():
            raise RuntimeError("Elasticsearch host is unreachable.")
    except Exception as exc:
        raise RuntimeError(f"Could not communicate with Elasticsearch: {exc}") from exc

    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            compression_type="gzip",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Kafka at {kafka_broker}: {exc}") from exc

    # 2. Cleanup Previous Canaries (if any)
    logger.info("Cleaning up prior canary records in Elasticsearch...")
    if es_client.indices.exists(index=es_index):
        es_client.delete_by_query(
            index=es_index,
            query={"term": {"unit_id": CANARY_UNIT_ID}},
            ignore_unavailable=True,
            conflicts="proceed"
        )
        # Force a refresh so the deletion is immediately visible
        es_client.indices.refresh(index=es_index)
    else:
        logger.info(f"Index '{es_index}' does not exist yet. Skipping cleanup.")

    # 3. Prepare Payload
    payload = generate_canary_payload()
    partition_key = str(CANARY_UNIT_ID).encode("utf-8")
    
    # Pre-warm the search query structure
    search_query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"unit_id": CANARY_UNIT_ID}},
                    {"term": {"time_cycles": CANARY_CYCLE}}
                ]
            }
        }
    }

    logger.info(f"Injecting Canary (unit_id={CANARY_UNIT_ID}, cycle={CANARY_CYCLE}) into Kafka topic '{kafka_topic}'...")

    # 4. The Measurement Window
    t0 = time.perf_counter()
    
    # Send and force flush to ensure it leaves the network buffer
    producer.send(topic=kafka_topic, key=partition_key, value=payload)
    producer.flush()
    
    logger.info("Canary dispatched. Polling Elasticsearch every 50ms...")
    
    t1 = None
    attempts = 0
    while (time.perf_counter() - t0) < TIMEOUT_SECONDS:
        attempts += 1
        try:
            res = es_client.search(index=es_index, query=search_query["query"], size=1)
            if res["hits"]["total"]["value"] > 0:
                t1 = time.perf_counter()
                break
        except Exception as e:
            logger.warning(f"Poll attempt failed: {e}")
            
        time.sleep(POLL_INTERVAL_SECONDS)

    # 5. Results Processing
    if t1 is None:
        raise RuntimeError(f"Canary did not appear in Elasticsearch within {TIMEOUT_SECONDS} seconds.")

    e2e_latency = t1 - t0
    logger.info(f"SUCCESS! Canary detected in Elasticsearch.")
    logger.info(f"Total Polling Attempts: {attempts}")
    logger.info(f"E2E Visibility Latency: {e2e_latency:.4f} seconds")

    # 6. Save Report
    timestamp_folder = time.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(RESULTS_DIR, timestamp_folder)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "e2e_latency_report.md")

    report_content = f"""
## Test Name: End-to-End (E2E) Visibility Latency Benchmark
- Result: PASS
- E2E Latency: {e2e_latency:.4f} seconds
- Total Polls: {attempts}
- Polling Interval: {POLL_INTERVAL_SECONDS * 1000} ms
- Timestamp (UTC): {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Benchmark report saved to '{report_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS E2E Latency Benchmark")
    parser.add_argument("--kafka-topic", default=KAFKA_TOPIC)
    parser.add_argument("--es-index", default=DEFAULT_ES_INDEX)
    parser.add_argument("--kafka-broker", default=DEFAULT_KAFKA_BROKER)
    parser.add_argument("--es-host", default=DEFAULT_ES_HOST)
    args = parser.parse_args()

    try:
        run_e2e_latency_benchmark(
            es_index=args.es_index,
            kafka_broker=args.kafka_broker,
            es_host=args.es_host,
            kafka_topic=args.kafka_topic,
        )
    except Exception as exc:
        logger.exception("E2E benchmark failed:")
        exit(1)


if __name__ == "__main__":
    main()