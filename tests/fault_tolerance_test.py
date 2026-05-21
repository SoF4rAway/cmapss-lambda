"""
tests/fault_tolerance_test.py
=============================
Automated test script to validate the system's fault tolerance against worker node failures.
Kills the Spark worker using the Docker SDK and verifies zero data loss and idempotent
writes in Elasticsearch post-recovery.

Usage:
    conda run -n pytorch-gpu python -m tests.fault_tolerance_test
"""

import os
import time
import argparse
import logging
from typing import Final, Dict, Any, List

import docker  # type: ignore[import]
from elasticsearch import Elasticsearch  # type: ignore[import]
from kafka import KafkaConsumer  # type: ignore[import]

from src.core.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FAULT_TOLERANCE_TEST] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAFKA_TOPIC: Final[str] = "cmapss_telemetry"
DEFAULT_ES_INDEX: Final[str] = "cmapss_predictions_v1"
DEFAULT_KAFKA_BROKER: Final[str] = "localhost:9092"
DEFAULT_ES_HOST: Final[str] = "http://localhost:9200"
SPARK_WORKER_CONTAINER_NAME: Final[str] = "spark-worker"
RECOVERY_TIMEOUT_SECONDS: Final[int] = 120


def verify_kafka_ingestion(broker: str, topic: str) -> None:
    """
    Pre-condition check: Verifies that the Kafka topic is receiving data.
    If no messages are read within 5 seconds, raises a RuntimeError.
    
    Args:
        broker: Address of the Kafka broker.
        topic: Kafka topic to read from.
        
    Raises:
        RuntimeError: If connection fails or no messages are active in the topic.
    """
    logger.info("Performing pre-condition check: Verifying Kafka data ingestion flow...")
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[broker],
            auto_offset_reset="latest",
            consumer_timeout_ms=5000,
            enable_auto_commit=False
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Kafka broker at {broker}: {exc}") from exc

    try:
        msg = next(consumer, None)
        if msg is None:
            raise RuntimeError(
                f"No messages received on Kafka topic '{topic}' within timeout. "
                "Ensure that the Kafka producer is running (e.g. `python -m src.ingestion.kafka_producer`)."
            )
        logger.info("Kafka data flow check: PASS (Successfully read message from topic).")
    finally:
        consumer.close()


def verify_elasticsearch_connectivity(es_client: Elasticsearch, index_name: str) -> None:
    """
    Pre-condition check: Verifies Elasticsearch connectivity and index readiness.
    
    Args:
        es_client: Instantiated Elasticsearch client.
        index_name: Elasticsearch index to check.
        
    Raises:
        RuntimeError: If Elasticsearch is unreachable or the index is missing/empty.
    """
    logger.info("Performing pre-condition check: Verifying Elasticsearch index status...")
    try:
        if not es_client.ping():
            raise RuntimeError("Elasticsearch host is unreachable.")
    except Exception as exc:
        raise RuntimeError(f"Could not communicate with Elasticsearch: {exc}") from exc

    if not es_client.indices.exists(index=index_name):
        raise RuntimeError(
            f"Elasticsearch index '{index_name}' does not exist. "
            "Ensure the Spark speed layer streaming job is active and has written initial predictions."
        )

    # Check for non-zero prediction records
    res = es_client.count(index=index_name)
    count = res.get("count", 0)
    if count == 0:
        raise RuntimeError(
            f"Elasticsearch index '{index_name}' contains 0 records. "
            "Wait for the Speed Layer to write initial predictions."
        )
    logger.info(f"Elasticsearch check: PASS (Index '{index_name}' contains {count} predictions).")


def get_max_processed_cycle(es_client: Elasticsearch, index_name: str) -> int:
    """
    Queries Elasticsearch to find the current maximum processed telemetry cycle.
    
    Args:
        es_client: Instantiated Elasticsearch client.
        index_name: Elasticsearch index to query.
        
    Returns:
        The highest time_cycles value found in predictions, or 0 if none.
    """
    try:
        res = es_client.search(
            index=index_name,
            size=0,
            aggs={"max_cycle": {"max": {"field": "time_cycles"}}}
        )
        val = res["aggregations"]["max_cycle"]["value"]
        return int(val) if val is not None else 0
    except Exception as exc:
        logger.error("Failed to query max processed cycle from Elasticsearch: %s", exc)
        return 0


def run_fault_tolerance_audit(
    es_index: str,
    kafka_broker: str,
    es_host: str,
    worker_name: str,
    timeout: int,
    kafka_topic: str = KAFKA_TOPIC,
) -> None:
    """
    Runs the full fault tolerance validation cycle:
    1. Pre-checks (Kafka, Elasticsearch)
    2. Failure Injection (SIGKILL on spark-worker container)
    3. Monitoring (re-start verification, ES write resumption)
    4. Data Consistency verification
    5. Appends results to results/{timestamp}/report.md
    
    Args:
        es_index: Target Elasticsearch index.
        kafka_broker: Kafka broker address.
        es_host: Elasticsearch REST host address.
        worker_name: Docker container name for the Spark worker.
        timeout: Recovery timeout in seconds.
    """
    # 2. Client Initializations & Pre-checks
    es_client = Elasticsearch(es_host)
    verify_kafka_ingestion(kafka_broker, kafka_topic)
    verify_elasticsearch_connectivity(es_client, es_index)

    try:
        docker_client = docker.from_env()
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Docker daemon: {exc}") from exc

    # Identify spark-worker container
    containers = docker_client.containers.list(all=True, filters={"name": worker_name})
    if not containers:
        raise RuntimeError(f"Could not find Docker container with name '{worker_name}'")
    worker_container = containers[0]

    # Establish baseline watermark before crash
    pre_crash_max_cycle = get_max_processed_cycle(es_client, es_index)
    logger.info("Baseline established. Pre-crash max cycle count: %d", pre_crash_max_cycle)

    # 3. Failure Injection (SIGKILL)
    kill_timestamp = time.time()
    logger.info("Injecting failure: Sending SIGKILL to '%s' container...", worker_name)
    try:
        # Jalankan perintah kill -9 pada PID 1 (proses Spark utama) di dalam kontainer
        exec_log = worker_container.exec_run("pkill -9 -f java")
        logger.info("Executed internal kill command inside container.")
    except Exception as exc:
        raise RuntimeError(f"Failed to inject internal failure inside '{worker_name}': {exc}") from exc

    logger.info("SIGKILL injected successfully at epoch time: %.2f", kill_timestamp)

    # 4. Monitoring & Recovery Phase
    # Wait for the container to return to "running" status (restored by docker-compose).
    #
    # After a SIGKILL, Docker cycles the container through:
    #   exited  ->  restarting  ->  running
    #
    # The poll loop must wait for "running" specifically — "restarting" means Docker
    # accepted the restart but the process is not yet alive. We log every status
    # transition so the operator can see progress without noise.
    #
    # Timeout is 120 s (not 60 s): on WSL2/Docker Desktop a Spark JVM worker
    # can take 60-90 s to restart and register with the master before its web UI
    # responds (which is what the healthcheck probes), keeping it in "restarting"
    # for longer than the old 60 s limit.
    CONTAINER_RESTART_TIMEOUT = 120
    logger.info(
        "Waiting for spark-worker container to reach 'running' status "
        "(timeout: %ds)...", CONTAINER_RESTART_TIMEOUT
    )
    restarted = False
    last_logged_status: str = ""
    docker_poll_start = time.time()
    while time.time() - docker_poll_start < CONTAINER_RESTART_TIMEOUT:
        try:
            # Re-fetch by name each iteration — the post-kill container object is
            # stale and may reference the dead instance rather than the newly
            # restarted one.
            fresh_container = docker_client.containers.get(worker_name)
            current_status = fresh_container.status

            # Log only on status transitions to avoid log spam
            if current_status != last_logged_status:
                logger.info("spark-worker container status: '%s'", current_status)
                last_logged_status = current_status

            if current_status == "running":
                restarted = True
                logger.info("spark-worker container is running.")
                break
        except docker.errors.NotFound:
            if last_logged_status != "not_found":
                logger.info("spark-worker container not yet visible (being recreated)...")
                last_logged_status = "not_found"
        time.sleep(2)

    if not restarted:
        raise RuntimeError(
            f"spark-worker container failed to reach 'running' status within "
            f"{CONTAINER_RESTART_TIMEOUT} seconds. "
            f"Last observed status: '{last_logged_status}'. "
            "Check 'docker logs spark-worker' for startup errors."
        )

    # Poll Elasticsearch to verify Spark resumed processing
    logger.info("Polling Elasticsearch for post-recovery predictions (waiting for cycles > %d)...", pre_crash_max_cycle)
    resumed = False
    es_poll_start = time.time()
    recovery_timestamp = 0.0
    post_crash_max_cycle = pre_crash_max_cycle

    while time.time() - es_poll_start < timeout:
        current_max = get_max_processed_cycle(es_client, es_index)
        if current_max > pre_crash_max_cycle:
            recovery_timestamp = time.time()
            post_crash_max_cycle = current_max
            resumed = True
            logger.info("Spark job resumed processing! Post-crash max cycle: %d", post_crash_max_cycle)
            break
        time.sleep(2)

    if not resumed:
        raise RuntimeError(f"Spark speed layer failed to write new predictions to Elasticsearch within {timeout}s.")

    recovery_latency = recovery_timestamp - kill_timestamp
    logger.info("Recovery Latency: %.2f seconds.", recovery_latency)

    # Give ES a brief window to ensure all in-flight writes are indexed
    logger.info("Sleeping 3 seconds for index synchronization...")
    time.sleep(3)

    # 5. Data Integrity & Consistency Check
    logger.info("Analyzing data integrity for cycles within range [%d, %d]...", max(0, pre_crash_max_cycle - 5), post_crash_max_cycle)
    
    # Query ES for documents in the crash-affected window
    query_body = {
        "range": {
            "time_cycles": {
                "gte": max(0, int(pre_crash_max_cycle - 5)),
                "lte": int(post_crash_max_cycle)
            }
        }
    }
    
    try:
        # Paginate with search_after to avoid silent truncation at size=1000.
        # Sort by (time_cycles, _id) so the cursor is deterministic.
        hits: List[Dict[str, Any]] = []
        last_sort: Any = None
        while True:
            body: Dict[str, Any] = {
                "query": query_body,
                "size": 1000,
                "sort": [{"time_cycles": "asc"}, {"_id": "asc"}],
            }
            if last_sort is not None:
                body["search_after"] = last_sort
            page = es_client.search(index=es_index, **body)
            page_hits = page["hits"]["hits"]
            if not page_hits:
                break
            hits.extend(page_hits)
            last_sort = page_hits[-1]["sort"]
    except Exception as exc:
        raise RuntimeError(f"Failed to query verification records from Elasticsearch: {exc}") from exc

    logger.info("Retrieved %d prediction documents in target window.", len(hits))

    # Check for duplicate (unit_id, time_cycles) composite keys.
    # Elasticsearch guarantees unique _id per index, so checking _id for duplicates
    # always passes — the real idempotency signal is a repeated business key.
    seen_keys: set = set()
    duplicate_keys: List[tuple] = []
    unit_cycles: Dict[int, List[int]] = {}

    for hit in hits:
        src = hit["_source"]
        u_id = int(src["unit_id"])
        cycle = int(src["time_cycles"])
        key = (u_id, cycle)

        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)

        if u_id not in unit_cycles:
            unit_cycles[u_id] = []
        unit_cycles[u_id].append(cycle)

    duplicates = duplicate_keys
    
    # Check for sequence gaps per unit
    gaps_detected = False
    for u_id, cycles in unit_cycles.items():
        sorted_cycles = sorted(list(set(cycles)))
        if len(sorted_cycles) > 1:
            min_c = min(sorted_cycles)
            max_c = max(sorted_cycles)
            expected_cycles = set(range(min_c, max_c + 1))
            actual_cycles = set(sorted_cycles)
            missing = expected_cycles - actual_cycles
            if missing:
                logger.error("Data loss detected! Unit %d is missing cycles: %s", u_id, sorted(list(missing)))
                gaps_detected = True

    # Assess overall validation status
    test_passed = True
    consistency_result = "Confirmed No Duplicates/Loss"

    if duplicates:
        test_passed = False
        consistency_result = f"FAIL (Duplicate (unit_id, time_cycles) keys found: {len(duplicates)})"
        logger.error("Data consistency check failed: duplicate business keys detected: %s", duplicates[:10])
    elif gaps_detected:
        test_passed = False
        consistency_result = "FAIL (Sequence gaps detected / data loss)"
        logger.error("Data consistency check failed: missing cycles in time range.")
    else:
        logger.info("Data consistency check: PASS (Confirmed No Duplicates/Loss).")

    # 6. Append Results to Report
    timestamp_folder = time.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(RESULTS_DIR, timestamp_folder)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "report.md")

    report_content = f"""
## Test Name: Fault Tolerance / Worker Node Failure
- Result: {"PASS" if test_passed else "FAIL"}
- Recovery Latency: {recovery_latency:.2f} seconds
- Data Consistency: {consistency_result}
"""

    logger.info("Writing test results to report at %s...", report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Fault Tolerance Audit complete.")


def main() -> None:
    """Wire up CLI arguments and execute the test."""
    parser = argparse.ArgumentParser(description="CMAPSS Speed Layer Fault Tolerance Test")
    parser.add_argument(
        "--kafka-topic",
        default=KAFKA_TOPIC,
        help=f"Kafka topic to consume from (default: {KAFKA_TOPIC})",
    )
    parser.add_argument(
        "--es-index",
        default=DEFAULT_ES_INDEX,
        help=f"Target Elasticsearch index (default: {DEFAULT_ES_INDEX})",
    )
    parser.add_argument(
        "--kafka-broker",
        default=DEFAULT_KAFKA_BROKER,
        help=f"Kafka broker address (default: {DEFAULT_KAFKA_BROKER})",
    )
    parser.add_argument(
        "--es-host",
        default=DEFAULT_ES_HOST,
        help=f"Elasticsearch REST host address (default: {DEFAULT_ES_HOST})",
    )
    parser.add_argument(
        "--spark-worker-name",
        default=SPARK_WORKER_CONTAINER_NAME,
        help=f"Spark worker container name (default: {SPARK_WORKER_CONTAINER_NAME})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=RECOVERY_TIMEOUT_SECONDS,
        help=f"Recovery timeout in seconds (default: {RECOVERY_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    try:
        run_fault_tolerance_audit(
            es_index=args.es_index,
            kafka_broker=args.kafka_broker,
            es_host=args.es_host,
            worker_name=args.spark_worker_name,
            timeout=args.timeout,
            kafka_topic=args.kafka_topic,
        )
    except Exception as exc:
        logger.exception("Fault tolerance test failed with an error:")
        # Write failure to a new timestamped report if possible
        try:
            timestamp_folder = time.strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(RESULTS_DIR, timestamp_folder)
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"""
## Test Name: Fault Tolerance / Worker Node Failure
- Result: FAIL
- Recovery Latency: N/A
- Data Consistency: FAIL (Error: {exc})
""")
            logger.info("Saved failure report to %s", report_path)
        except Exception as report_exc:
            logger.error("Failed to write error report file: %s", report_exc)
        exit(1)


if __name__ == "__main__":
    main()