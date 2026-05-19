"""
src/ingestion/kafka_producer.py
================================
Real-time telemetry simulator for the C-MAPSS Lambda Architecture.

Reads raw NASA C-MAPSS test data line-by-line and streams each engine
reading as a typed JSON payload to the `cmapss_telemetry` Kafka topic.

Key design decisions:
- `unit_id` is used as the Kafka message *key* to guarantee partition-local
  stateful processing for each engine unit in the downstream Spark Speed Layer.
- `value_serializer` is set at producer construction time, keeping the hot-path
  `send()` call free of manual encoding logic.
- `producer.flush()` is called every FLUSH_INTERVAL messages (not after each
  send). Flushing after every single message disables Kafka's internal record
  batching, serializing all I/O and reducing throughput from ~50k msg/s to
  single-digit msg/s. Batch flushing also allows the `linger_ms` window to
  fill naturally before a forced drain.
- Paths are resolved relative to the project root, making the script
  execution-environment independent (run as module or direct script).

Usage:
    python -m src.ingestion.kafka_producer
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Final

from kafka import KafkaProducer  # type: ignore[import]
from kafka.errors import NoBrokersAvailable  # type: ignore[import]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Absolute path to the project root (two levels up from this file)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_FILE_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "CMAPSSData" / "test_FD001.txt"
)

KAFKA_BOOTSTRAP_SERVERS: Final[str] = "localhost:9092"
KAFKA_TOPIC: Final[str] = "cmapss_telemetry"

# Inter-message delay in seconds → ~20 messages per second
STREAM_INTERVAL_SECONDS: Final[float] = 0.05

# Flush to broker every N messages instead of after every send.
# This restores Kafka's internal record-batching pipeline and keeps
# delivery latency well within the 50 ms sleep window.
FLUSH_INTERVAL: Final[int] = 10

# Connection retry configuration
MAX_RETRIES: Final[int] = 5
RETRY_DELAY_SECONDS: Final[float] = 5.0

# Canonical ordered column names matching the 26-column C-MAPSS FD001 schema.
# Column layout: unit_id, time_cycles, op_setting_1..3, sensor_1..21
COLUMN_NAMES: Final[list[str]] = [
    "unit_id",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

# The first two columns are integer identifiers; the rest are floats.
_INT_COLUMNS: Final[frozenset[str]] = frozenset({"unit_id", "time_cycles"})


# ---------------------------------------------------------------------------
# Data Parsing
# ---------------------------------------------------------------------------


def parse_line(line: str) -> dict[str, int | float] | None:
    """Parse a single space-delimited row into a typed telemetry dictionary.

    Args:
        line: A raw text line from `test_FD001.txt`.

    Returns:
        A dict mapping column names to typed values (int or float), or
        ``None`` if the line is malformed (wrong column count or non-numeric
        values).
    """
    tokens = line.strip().split()

    # The raw file has a trailing whitespace column; strip it by checking
    # the minimum expected column count.
    if len(tokens) < len(COLUMN_NAMES):
        logger.warning(
            "Skipping malformed line (expected %d columns, got %d): %r",
            len(COLUMN_NAMES),
            len(tokens),
            line[:60],
        )
        return None

    payload: dict[str, int | float] = {}
    try:
        for col, token in zip(COLUMN_NAMES, tokens):
            payload[col] = int(float(token)) if col in _INT_COLUMNS else float(token)
    except ValueError as exc:
        logger.warning("Skipping line with non-numeric value: %s", exc)
        return None

    return payload


# ---------------------------------------------------------------------------
# Kafka Producer
# ---------------------------------------------------------------------------


def create_producer(
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> KafkaProducer:
    """Instantiate a ``KafkaProducer`` with retry logic.

    Retries up to ``max_retries`` times with ``retry_delay`` seconds between
    attempts to tolerate Kafka broker cold-start delays.

    Args:
        bootstrap_servers: Kafka broker address (e.g. ``"localhost:9092"``).
        max_retries: Maximum number of connection attempts.
        retry_delay: Seconds to wait between consecutive retry attempts.

    Returns:
        A configured and connected ``KafkaProducer`` instance.

    Raises:
        RuntimeError: If the broker is unreachable after all retry attempts.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Connecting to Kafka at %s (attempt %d/%d)...",
                bootstrap_servers,
                attempt,
                max_retries,
            )
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Ensure the broker received and committed the message before ack.
                acks="all",
                # Compress payloads to reduce network overhead.
                compression_type="gzip",
                # Allow up to 5 ms of natural batching before the producer
                # sends. Combined with FLUSH_INTERVAL this keeps end-to-end
                # latency well under 55 ms while restoring batching efficiency.
                linger_ms=5,
                # Retry at the transport level for transient send failures.
                retries=3,
            )
            logger.info("Successfully connected to Kafka at %s.", bootstrap_servers)
            return producer
        except NoBrokersAvailable:
            if attempt < max_retries:
                logger.warning(
                    "Broker unavailable. Retrying in %.0f seconds...", retry_delay
                )
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Failed to connect to Kafka at {bootstrap_servers} "
                    f"after {max_retries} attempts. "
                    "Ensure the Docker stack is running: "
                    "`docker compose up -d zookeeper kafka kafka-setup`"
                )


# ---------------------------------------------------------------------------
# Streaming Logic
# ---------------------------------------------------------------------------


def stream_telemetry(
    producer: KafkaProducer,
    data_path: Path = DATA_FILE_PATH,
    topic: str = KAFKA_TOPIC,
    sleep_interval: float = STREAM_INTERVAL_SECONDS,
) -> None:
    """Stream telemetry rows from ``data_path`` to the Kafka ``topic``.

    Reads the source file line-by-line to maintain a constant memory footprint
    regardless of file size. Each valid row is serialized as JSON and published
    with ``unit_id`` as the partition key.

    Args:
        producer: An active ``KafkaProducer`` instance.
        data_path: Absolute ``Path`` to the C-MAPSS ``.txt`` source file.
        topic: Target Kafka topic name.
        sleep_interval: Seconds to sleep between consecutive messages.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Verify the path relative to the project root."
        )

    logger.info("Starting stream from: %s → topic: %s", data_path.name, topic)
    logger.info(
        "Rate: 1 message every %.3fs (~%.0f msg/s)", sleep_interval, 1 / sleep_interval
    )

    message_count: int = 0
    skipped_count: int = 0

    with data_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            payload = parse_line(raw_line)

            if payload is None:
                skipped_count += 1
                continue

            partition_key: bytes = str(payload["unit_id"]).encode("utf-8")

            producer.send(
                topic=topic,
                key=partition_key,
                value=payload,
            )

            message_count += 1

            # Flush every FLUSH_INTERVAL messages rather than after each send.
            # Per-message flush forces synchronous round-trips to the broker,
            # negating all batching and inflating latency by 10-100×.
            if message_count % FLUSH_INTERVAL == 0:
                producer.flush()

            if message_count % 100 == 0:
                msg_bytes = len(json.dumps(payload).encode("utf-8"))
                logger.info(
                    "unit_id=%-3s | cycle=%-4s | msg_size=%d bytes | total_sent=%d",
                    payload["unit_id"],
                    payload["time_cycles"],
                    msg_bytes,
                    message_count,
                )

            time.sleep(sleep_interval)

    # Final flush to drain any remaining buffered messages after EOF.
    producer.flush()

    logger.info(
        "Stream complete. Published %d messages, skipped %d malformed lines.",
        message_count,
        skipped_count,
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Wire up and run the telemetry producer pipeline."""
    parser = argparse.ArgumentParser(
        description="Real-time telemetry simulator for the C-MAPSS Lambda Architecture."
    )
    parser.add_argument(
        "-d",
        "--data-file",
        type=str,
        default=str(DATA_FILE_PATH.relative_to(PROJECT_ROOT)),
        help="Path to the C-MAPSS data file, relative to the project root.",
    )
    args = parser.parse_args()

    # Resolve relative to project root. Handles absolute paths natively.
    data_path = (PROJECT_ROOT / args.data_file).resolve()

    producer: KafkaProducer | None = None

    try:
        # Pre-flight check: raise FileNotFoundError before connecting to Kafka broker
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                "Verify the path relative to the project root."
            )

        producer = create_producer()
        stream_telemetry(producer, data_path=data_path)
    except FileNotFoundError as exc:
        logger.error("Data error: %s", exc)
    except RuntimeError as exc:
        logger.error("Connection error: %s", exc)
    except KeyboardInterrupt:
        logger.info("Interrupt received. Shutting down gracefully...")
    finally:
        if producer is not None:
            logger.info("Flushing remaining messages and closing Kafka connection...")
            producer.flush()
            producer.close()
            logger.info("Kafka producer closed. Goodbye.")


if __name__ == "__main__":
    main()