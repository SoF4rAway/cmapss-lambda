"""
tests/throughput_test.py
========================
Automated benchmark script to measure the maximum throughput (Events Per Second)
of the C-MAPSS Lambda Architecture.

Methodology:
Floods Kafka with a large volume of distributed telemetry data, then times how 
long the Spark Speed Layer takes to process and index the entire batch into Elasticsearch,
isolating processing time from batch trigger delays.

Usage:
    conda run -n pytorch-gpu python -m tests.throughput_test --total-records 50000
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

# Force logging to UTC
logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [THROUGHPUT_TEST] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

KAFKA_TOPIC: Final[str] = "cmapss_telemetry"
DEFAULT_ES_INDEX: Final[str] = "cmapss_predictions_v1"
DEFAULT_KAFKA_BROKER: Final[str] = "localhost:9092"
DEFAULT_ES_HOST: Final[str] = "http://localhost:9200"

def generate_flood_data(total_records: int, max_units: int = 500) -> list:
    """
    Generates a large batch of synthetic telemetry data.
    Distributes records across multiple unit_ids to ensure Spark parallelism.
    """
    logger.info(f"Generating {total_records} synthetic records across {max_units} units in memory...")
    records = []
    
    # Calculate cycles per unit to reach total_records
    cycles_per_unit = total_records // max_units
    remainder = total_records % max_units

    for unit in range(1, max_units + 1):
        # Add extra cycle to some units to handle remainders
        cycles = cycles_per_unit + (1 if unit <= remainder else 0)
        
        for cycle in range(1, cycles + 1):
            payload = {
                "unit_id": unit + 9000,  # Offset to avoid colliding with real data
                "time_cycles": cycle,
                "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 0.0,
                "kafka_ingest_timestamp_ms": int(time.time() * 1000)
            }
            for i in range(1, 22):
                payload[f"sensor_{i}"] = 0.0
                
            records.append((str(payload["unit_id"]).encode("utf-8"), payload))
            
    return records


def run_throughput_benchmark(
    es_index: str,
    kafka_broker: str,
    es_host: str,
    total_records: int,
    kafka_topic: str = KAFKA_TOPIC,
) -> None:
    
    es_client = Elasticsearch(es_host)
    producer = KafkaProducer(
        bootstrap_servers=kafka_broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1, # Use acks=1 for faster flooding during stress tests
        compression_type="lz4", # LZ4 is faster for high-throughput batching
        linger_ms=10,
        batch_size=65536
    )

    # 1. Cleanup
    logger.info("Purging test index to ensure a clean slate...")
    if es_client.indices.exists(index=es_index):
        # We delete and recreate the index for a true zero-state start
        es_client.indices.delete(index=es_index)
        time.sleep(2)
        
    es_client.indices.create(index=es_index)

    # 2. Prepare Data
    records = generate_flood_data(total_records)
    
    # 3. Flood Kafka
    logger.info(f"Flooding Kafka topic '{kafka_topic}' with {total_records} records...")
    flood_start = time.perf_counter()
    for key, payload in records:
        producer.send(topic=kafka_topic, key=key, value=payload)
    producer.flush()
    flood_end = time.perf_counter()
    logger.info(f"Flood complete in {flood_end - flood_start:.2f}s. Awaiting Spark processing...")
    print(f"Flood complete in {flood_end - flood_start:.2f}s. Awaiting Spark processing...")

    # 4. Await First Output (Start Timer)
    t_start = None
    logger.info("Polling Elasticsearch for the first processed document...")
    while True:
        es_client.indices.refresh(index=es_index)
        count = es_client.count(index=es_index)["count"]
        if count > 0:
            t_start = time.perf_counter()
            logger.info(f"First document detected! Start processing timer. (Count: {count})")
            break
        time.sleep(0.5)

    # 5. Await Full Batch (End Timer)
    t_end = None
    last_count = 0
    logger.info(f"Polling until document count reaches {total_records}...")
    while True:
        es_client.indices.refresh(index=es_index)
        count = es_client.count(index=es_index)["count"]
        
        if count != last_count:
            logger.info(f"Progress: {count} / {total_records} records indexed...")
            last_count = count
            
        if count >= total_records:
            t_end = time.perf_counter()
            break
            
        time.sleep(1.0)

    # 6. Calculate Metrics
    processing_time = t_end - t_start
    eps = total_records / processing_time
    
    logger.info(f"SUCCESS! All {total_records} records processed.")
    logger.info(f"Total Processing Time: {processing_time:.2f} seconds")
    logger.info(f"Pipeline Throughput: {eps:.2f} Events Per Second (EPS)")

    # 7. Save Report
    timestamp_folder = time.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(RESULTS_DIR, timestamp_folder)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "throughput_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""
## Test Name: Pipeline Throughput Benchmark (Stress Test)
- Result: PASS
- Total Records Injected: {total_records}
- Total Processing Time: {processing_time:.2f} seconds
- **Calculated Throughput**: **{eps:.2f} EPS**
- Timestamp (UTC): {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
""")
    logger.info(f"Throughput report saved to '{report_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS Throughput Stress Test")
    parser.add_argument("--kafka-topic", default=KAFKA_TOPIC)
    parser.add_argument("--es-index", default=DEFAULT_ES_INDEX)
    parser.add_argument("--kafka-broker", default=DEFAULT_KAFKA_BROKER)
    parser.add_argument("--es-host", default=DEFAULT_ES_HOST)
    parser.add_argument("--total-records", type=int, default=50000, 
                        help="Number of records to flood into the pipeline.")
    args = parser.parse_args()

    try:
        run_throughput_benchmark(
            es_index=args.es_index,
            kafka_broker=args.kafka_broker,
            es_host=args.es_host,
            total_records=args.total_records,
            kafka_topic=args.kafka_topic,
        )
    except Exception as exc:
        logger.exception("Throughput benchmark failed:")
        exit(1)


if __name__ == "__main__":
    main()