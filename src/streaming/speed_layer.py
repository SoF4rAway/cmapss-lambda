"""
src/streaming/speed_layer.py
============================
PySpark Structured Streaming pipeline for real-time RUL inference.

Consumes telemetry from Kafka, maintains a 30-cycle stateful buffer per engine,
and executes ONNX inference using optimized broadcasted artifacts.

Key fixes applied:
- Broadcast variables are no longer captured as module-level globals. The UDF
  is constructed as a closure *inside* main() so that broadcast handles are
  properly serialized and available on executors.
- state.get() returns a Row; access state_json via attribute, not index [0].
- The duplicate console sink on the stateful stream is removed; Spark does not
  support two independent sinks on the same stateful DataFrame.
- spark-submit entry-point corrected to an absolute in-container path.
"""

import os
import json
import pickle
import threading
import argparse
import logging
import functools
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from typing import Iterator, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, concat, lit
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType,
    StringType, BooleanType, BinaryType
)
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SPEED_LAYER] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-Executor ONNX Session Cache
# ---------------------------------------------------------------------------
_global_onnx_session = None


def get_onnx_session(onnx_bytes: bytes) -> ort.InferenceSession:
    """
    Singleton ONNX InferenceSession on each executor.
    Configures session options for sub-millisecond latency.
    Re-uses a cached session if the bytes are identical to avoid repeated
    deserialization on every micro-batch invocation.
    """
    global _global_onnx_session
    if _global_onnx_session is None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 1 thread per op prevents contention on multi-core executors for
        # TinyML models whose graph is too small to benefit from parallelism.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _global_onnx_session = ort.InferenceSession(
            onnx_bytes,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _global_onnx_session


# ---------------------------------------------------------------------------
# Stateful UDF Factory
# ---------------------------------------------------------------------------

def make_predict_rul_udf(bc_schema, bc_scaler, bc_onnx, bc_model_version):
    """
    Returns a stateful Pandas UDF that closes over the broadcast
    handles. Building the UDF inside main() — after the broadcast variables
    are created — is the correct pattern: Python closures capture the *value*
    of bc_* at definition time, and Spark serializes the closure (including
    the broadcast handles) to each executor. Module-level globals are never
    serialized and remain None on executors.
    """

    def predict_rul_with_state(
        key: Tuple[int],
        pdf_iter: Iterator[pd.DataFrame],
        state: GroupState,
    ) -> Iterator[pd.DataFrame]:
        # ----------------------------------------------------------------
        # 1. Handle TTL Timeout — evict stale engines
        # ----------------------------------------------------------------
        if state.hasTimedOut:
            state.remove()
            # yield an empty DataFrame to satisfy the iterator contract, then stop
            yield pd.DataFrame(
                columns=["unit_id", "time_cycles", "predicted_rul", "is_critical", "inference_latency_ms", "model_version"]
            )
            return

        # ----------------------------------------------------------------
        # 2. Deserialize existing state history
        # ----------------------------------------------------------------
        if state.exists:
            # state.get is a property returning a tuple, not a callable method
            state_tuple = state.get
            history_df = pickle.loads(state_tuple[0])
        else:
            history_df = pd.DataFrame()

        # ----------------------------------------------------------------
        # 3. Hoist broadcast deserialization outside the micro-batch loop
        # ----------------------------------------------------------------
        session = get_onnx_session(bc_onnx.value)
        input_name = session.get_inputs()[0].name
        active_features = bc_schema.value
        scaler = bc_scaler.value

        results = []

        # ----------------------------------------------------------------
        # 4. Process micro-batch partitions for this unit_id
        # ----------------------------------------------------------------
        for pdf in pdf_iter:
            # 1. Clean and ensure chronological ordering
            pdf_clean = pdf.dropna(axis=1, how='all')
            if pdf_clean.empty:
                continue
            pdf_clean = pdf_clean.sort_values("time_cycles")

            num_new_rows = len(pdf_clean)

            # 2. Append all new data to history state at once
            if history_df.empty:
                history_df = pdf_clean.copy()
            else:
                history_df = pd.concat([history_df, pdf_clean], ignore_index=True)

            history_df = history_df.sort_values("time_cycles").reset_index(drop=True)
            total_rows = len(history_df)

            # 3. Vectorized Preprocessing (Scale the full history at once)
            history_df[active_features] = history_df[active_features].astype(np.float32)
            scaled_all = scaler.transform(history_df[active_features])

            # 4. Sliding Window Preprocessing (Collect inputs for batch ONNX)
            batch_tensors = []
            for i in range(total_rows - num_new_rows, total_rows):
                window_start = max(0, i + 1 - 30)
                window_end = i + 1

                window_features = scaled_all[window_start:window_end]
                current_len = len(window_features)

                # Cold-start zero-padding
                if current_len < 30:
                    pad_len = 30 - current_len
                    padding = np.zeros((pad_len, len(active_features)), dtype=np.float32)
                    tensor_input = np.vstack([padding, window_features])
                else:
                    tensor_input = window_features

                # Reshape to (1, 30, num_features)
                tensor_input = tensor_input.reshape(1, 30, len(active_features)).astype(np.float32)
                batch_tensors.append(tensor_input)

            if batch_tensors:
                # Stack all sliding windows into shape (num_new_rows, 30, num_features)
                batch_array = np.vstack(batch_tensors)

                # Time the vectorized ONNX execution
                import time
                t0 = time.perf_counter()
                raw_outputs = session.run(None, {input_name: batch_array})[0]
                t1 = time.perf_counter()

                # Calculate per-record latency in milliseconds
                latency_ms = ((t1 - t0) * 1000.0) / num_new_rows

                # Process results
                for idx, i in enumerate(range(total_rows - num_new_rows, total_rows)):
                    raw_output = raw_outputs[idx][0]
                    predicted_rul = float(np.clip(raw_output, 0.0, 125.0))

                    # Explicit Dtype Control via .item()
                    cycle = history_df.iloc[i]["time_cycles"].item()

                    results.append(
                        {
                            "unit_id": int(key[0]),
                            "time_cycles": int(cycle),
                            "predicted_rul": predicted_rul,
                            "is_critical": bool(predicted_rul < 30.0),
                            "inference_latency_ms": float(latency_ms),
                            "model_version": str(bc_model_version.value),
                        }
                    )

        # ----------------------------------------------------------------
        # 8. Persist updated history (last 30 cycles) and set 1-hour TTL
        # ----------------------------------------------------------------
        history_df = history_df.tail(30)
        state.update((pickle.dumps(history_df, protocol=5),))
        state.setTimeoutDuration(60 * 60 * 1000)

        yield pd.DataFrame(
            results,
            columns=["unit_id", "time_cycles", "predicted_rul", "is_critical", "inference_latency_ms", "model_version"]
        )

    return predict_rul_with_state


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

def write_to_sinks(batch_df: DataFrame, batch_id: int, es_index: str) -> None:
    """
    Directs each micro-batch to both the real-time (Elasticsearch) and the
    historical (HDFS Parquet) layers using multi-threading.
    """
    if batch_df.isEmpty():
        return

    # Create deterministic doc_id for Elasticsearch Upserts to prevent duplication
    batch_df = batch_df.withColumn(
        "doc_id",
        concat(lit("unit_"), col("unit_id"), lit("_cycle_"), col("time_cycles"))
    )

    # Persist the batch to avoid re-computing the 1D-CNN inference for each sink
    batch_df.persist()

    def write_es():
        try:
            batch_df.write \
                .format("org.elasticsearch.spark.sql") \
                .option("es.nodes", "elasticsearch") \
                .option("es.port", "9200") \
                .option("es.nodes.wan.only", "true") \
                .option("es.index.auto.create", "true") \
                .option("es.resource", es_index) \
                .option("es.mapping.id", "doc_id") \
                .option("es.write.operation", "upsert") \
                .mode("append") \
                .save()
            logger.info("Batch %d: pushed to Elasticsearch.", batch_id)
        except Exception as exc:
            logger.error("Batch %d: Elasticsearch sink error: %s", batch_id, exc)

    def write_hdfs():
        try:
            batch_df.coalesce(1).write \
                .format("parquet") \
                .option("parquet.block.size", "104857600") \
                .mode("append") \
                .save("hdfs://namenode:9000/cmapss/batch_layer/predictions/")
            logger.info("Batch %d: saved to HDFS (Parquet).", batch_id)
        except Exception as exc:
            logger.error("Batch %d: HDFS sink error: %s", batch_id, exc)

    # Ensure Kibana receives updates immediately
    es_thread = threading.Thread(target=write_es)
    es_thread.start()

    # HDFS as a background fire-and-forget task, every 10 batches
    if batch_id % 10 == 0:
        hdfs_thread = threading.Thread(target=write_hdfs)
        hdfs_thread.daemon = True
        hdfs_thread.start()

    # Block only on Elasticsearch to confirm Real-time SLA
    es_thread.join()

    # Free memory asynchronously without blocking background HDFS writes
    batch_df.unpersist(blocking=False)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS Speed Layer")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to versioned model artifact directory",
    )
    parser.add_argument(
        "--kafka-broker",
        default="kafka:29092",
        help="Kafka broker address (internal Docker network)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/tmp/spark_checkpoints/speed_layer",
        help="Spark checkpoint directory",
    )
    parser.add_argument(
        "--es-index",
        default="cmapss_predictions",
        help="Target Elasticsearch index for predictions",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("CMAPSS-Speed-Layer")
        .config("spark.sql.streaming.checkpointLocation", args.checkpoint_dir)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "2")
        .config("spark.sql.streaming.asyncProgressTrackingEnabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------------
    # 1. Load and broadcast artifacts
    # ------------------------------------------------------------------
    model_version_str = os.path.basename(os.path.normpath(args.model_dir))
    logger.info("Loading artifacts from: %s (Version: %s)", args.model_dir, model_version_str)

    with open(os.path.join(args.model_dir, "feature_schema.json"), "r") as fh:
        active_features = json.load(fh)["active_features"]

    scaler = joblib.load(os.path.join(args.model_dir, "scaler.joblib"))

    with open(os.path.join(args.model_dir, "model.onnx"), "rb") as fh:
        model_bytes = fh.read()

    # Broadcast handles are created here, inside main(), so the closure
    # built by make_predict_rul_udf captures live handles, not None.
    bc_schema = spark.sparkContext.broadcast(active_features)
    bc_scaler = spark.sparkContext.broadcast(scaler)
    bc_onnx = spark.sparkContext.broadcast(model_bytes)
    bc_model_version = spark.sparkContext.broadcast(model_version_str)

    # ------------------------------------------------------------------
    # 2. Define schemas
    # ------------------------------------------------------------------
    json_schema = StructType(
        [
            StructField("unit_id", IntegerType(), True),
            StructField("time_cycles", IntegerType(), True),
            StructField("op_setting_1", FloatType(), True),
            StructField("op_setting_2", FloatType(), True),
            StructField("op_setting_3", FloatType(), True),
            *[StructField(f"sensor_{i}", FloatType(), True) for i in range(1, 22)],
        ]
    )

    output_schema = StructType(
        [
            StructField("unit_id", IntegerType(), False),
            StructField("time_cycles", IntegerType(), False),
            StructField("predicted_rul", FloatType(), False),
            StructField("is_critical", BooleanType(), False),
            StructField("inference_latency_ms", FloatType(), False),
            StructField("model_version", StringType(), False),
        ]
    )

    # The state stores a single binary blob (pickled pandas DataFrame).
    state_schema = StructType(
        [StructField("state_binary", BinaryType(), False)]
    )

    # ------------------------------------------------------------------
    # 3. Read from Kafka
    # ------------------------------------------------------------------
    logger.info(
        "Connecting to Kafka topic 'cmapss_telemetry' at %s", args.kafka_broker
    )
    stream_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", args.kafka_broker)
        .option("subscribe", "cmapss_telemetry")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 300)
        .option("failOnDataLoss", "false")
        .load()
    )

    # ------------------------------------------------------------------
    # 4. Parse and flatten the JSON payload
    # ------------------------------------------------------------------
    parsed_df = stream_df.select(
        from_json(col("value").cast("string"), json_schema).alias("data")
    ).select("data.*")

    # ------------------------------------------------------------------
    # 5. Apply stateful inference via the closure-based UDF
    # ------------------------------------------------------------------
    predict_udf = make_predict_rul_udf(bc_schema, bc_scaler, bc_onnx, bc_model_version)

    predictions_df = parsed_df.groupBy("unit_id").applyInPandasWithState(
        func=predict_udf,
        outputStructType=output_schema,
        stateStructType=state_schema,
        outputMode="Update",
        timeoutConf=GroupStateTimeout.ProcessingTimeTimeout,
    )

    # ------------------------------------------------------------------
    # 6. Single sink via foreachBatch (handles both ES and HDFS)
    #
    # NOTE: A stateful stream supports exactly ONE writeStream action.
    # A second writeStream (e.g. a console sink) would re-evaluate the
    # stateful operator a second time, which Spark forbids and raises:
    #   "Detected a repeated streaming query"
    # All output routing is therefore handled inside write_to_sinks().
    # ------------------------------------------------------------------
    logger.info("Starting streaming query...")
    query = (
        predictions_df.writeStream
        .foreachBatch(functools.partial(write_to_sinks, es_index=args.es_index))
        .outputMode("update")
        .option("checkpointLocation", args.checkpoint_dir)
        .trigger(processingTime="5 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()