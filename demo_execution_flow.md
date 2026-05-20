# Demo Execution Flow: Lambda Architecture Lifecycle

This document outlines the step-by-step execution flow to validate the real-time inference latency tracking, batch retraining pipeline, and alias swap mechanisms.

## Phase 1: Environment Cleanup & Initialization

### 1. Reset Kafka Topic
Delete the existing topic and recreate it with **4 partitions** to align with the default Spark parallelism and avoid partition bottlenecks:
```bash
# Delete the topic (using the internal broker listener)
docker exec -it kafka kafka-topics --bootstrap-server kafka:29092 --delete --topic cmapss_telemetry

# Re-create the topic with 4 partitions
docker exec -it kafka kafka-topics --bootstrap-server kafka:29092 --create --topic cmapss_telemetry --partitions 4 --replication-factor 1
```

### 2. Reset HDFS Cold Storage
Remove historical telemetry data stored on HDFS:
```bash
docker exec -it namenode hdfs dfs -rm -r -f /cmapss
```

### 3. Clear Spark Checkpoints
Clean local and container checkpoints to prevent state serialization conflicts:
```bash
rm -rf /tmp/spark_checkpoints/speed_layer/*
docker exec spark-master rm -rf /tmp/spark_checkpoints/speed_layer
docker exec spark-worker rm -rf /tmp/spark_checkpoints/speed_layer
```

### 4. Reset Elasticsearch Indices & Alias
Drop existing indices to allow the Spark Elasticsearch connector to dynamically auto-create the correct mappings (including `inference_latency_ms` as a `float` field):
```bash
curl -X DELETE "localhost:9200/cmapss_predictions_v1"
curl -X DELETE "localhost:9200/cmapss_predictions_v2"
curl -X DELETE "localhost:9200/cmapss_predictions"
```

Initialize the baseline alias mapping `cmapss_predictions_current -> cmapss_predictions_v1`:
```bash
chmod +x scripts/*.sh
./scripts/init_alias.sh
```

---

## Phase 2: Running Pass 1 (Drift Ingestion & Storage)

### 1. Start Streaming Job 1 (v1 Model)
Run the streaming speed layer pointing to the baseline model and the `v1` index. 
> [!NOTE]
> Keep this terminal open as it runs in the foreground. Open a **new terminal** to run subsequent commands.
```bash
docker exec -it spark-master spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.1 \
  /app/src/streaming/speed_layer.py \
  --model-dir /app/models/20260517_145257 \
  --es-index cmapss_predictions_v1 \
  --checkpoint-dir /tmp/spark_checkpoints/speed_layer
```

### 2. Ingest Drifted Data
In your second terminal, generate the drifted validation dataset and stream it into the Kafka topic:
```bash
# Generate the drifted telemetry file
PYTHONPATH=. python -m src.ingestion.generate_drift_data

# Stream drifted data (this takes 1-2 minutes to complete)
PYTHONPATH=. python -m src.ingestion.kafka_producer --data-file data/CMAPSSData/test_FD001_drifted.txt
```

### 3. Stop Streaming Job 1
Wait about 10-15 seconds after the producer finishes to ensure Spark has completed processing and written the final batches to HDFS.
Go back to the first terminal and stop the streaming job using `Ctrl+C`.

---

## Phase 3: Batch Retraining & Model Upgrade

### 1. Clear Checkpoints for Pass 2
To prevent schema or state recovery issues, clear the checkpoint directory before starting the updated job:
```bash
rm -rf /tmp/spark_checkpoints/speed_layer/*
docker exec spark-master rm -rf /tmp/spark_checkpoints/speed_layer
docker exec spark-worker rm -rf /tmp/spark_checkpoints/speed_layer
```

### 2. Run Batch Retraining
Fine-tune the model on the drifted data retrieved from the HDFS cold path:
```bash
conda run -n pytorch-gpu env PYTHONPATH=. python -m src.batch.batch_retrain
```
*Look at the training log output to find the versioned directory path of the new model (e.g., `models/v2_drift_corrected/20260520_104615`). Replace `<TIMESTAMP_FOLDER>` in the next command with this timestamp.*

---

## Phase 4: Running Pass 2 (Corrected Model & Alias Swap)

### 1. Start Streaming Job 2 (v2 Model)
Start the speed layer using the fine-tuned model and point it to the `v2` index:
> [!NOTE]
> Run this in the first terminal. Keep it open.
```bash
docker exec -it spark-master spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.1 \
  /app/src/streaming/speed_layer.py \
  --model-dir /app/models/v2_drift_corrected/<TIMESTAMP_FOLDER> \
  --es-index cmapss_predictions_v2 \
  --checkpoint-dir /tmp/spark_checkpoints/speed_layer
```

### 2. Stream Clean Test Data and Swap Alias
In the second terminal, begin streaming the original clean test data, and immediately trigger the alias swap script to redirect dashboard traffic atomically from `v1` to `v2`:
```bash
# Start ingestion
PYTHONPATH=. python -m src.ingestion.kafka_producer --data-file data/CMAPSSData/test_FD001.txt

# Run in a separate shell immediately after the ingestion starts
./scripts/swap_alias.sh
```

---

## Phase 5: Evaluation & Latency Reporting

Once the clean data ingestion completes, run the evaluation script to generate the performance report, fetching the newly indexed production latencies directly from Elasticsearch:
```bash
PYTHONPATH=. python evaluate.py
```
