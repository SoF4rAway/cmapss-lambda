# C-MAPSS Predictive Maintenance: Lambda Architecture

A high-performance Big Data system for real-time Remaining Useful Life (RUL) prediction of NASA C-MAPSS turbofan engines. This project implements a full Lambda Architecture utilizing Spark Structured Streaming and TinyML for sub-millisecond inference.

## 🚀 Key Features

*   **Sub-Millisecond Inference**: Cache-optimized 1D-CNN (MobileNet-inspired, <500KB) achieving ~0.04ms inference.
*   **Zero Data Loss Streaming**: Vectorized sliding-window Pandas UDFs in Spark ensure every telemetry tick is processed and predicted.
*   **Tiered Multi-Sink**: Decoupled Hot Path (Synchronous Elasticsearch Upserts) and Cold Path (Asynchronous Daemon-thread HDFS writes).
*   **Production State Management**: State maintained via high-performance binary **Pickle (Protocol 5)** serialization.
*   **Robust Infrastructure**: Dockerized stack with Zookeeper HTTP Admin healthchecks and strict initialization sequencing.

## 🏗 System Architecture

1.  **Ingestion**: Python-based simulator streams raw telemetry to **Kafka** at a controlled rate (~20 msg/s).
2.  **Speed Layer**: **Spark Structured Streaming** consumes from Kafka, maintains a 30-cycle stateful buffer, and executes **ONNX** inference.
3.  **Batch Layer**: Telemetry and predictions are archived to **HDFS (Parquet)** for historical analysis and retraining.
4.  **Serving Layer**: **Elasticsearch + Kibana** provides real-time health monitoring and critical RUL alerts.

## 📊 Performance Benchmarks

| Metric | Target | Current Baseline |
| :--- | :--- | :--- |
| **RMSE** | < 20.0 | **15.3680** |
| **NASA Score** | < 500 | **308.66** |
| **Inference Latency** | < 0.1 ms | **0.0381 ms** |
| **Model Size** | < 500 KB | **22.14 KB** |

## 📂 Project Structure

```text
cmapss-lambda/
├── src/                       # Modular source package
│   ├── core/                  # Global config and reproducibility
│   ├── data/                  # Preprocessing and DataLoaders
│   ├── models/                # Architecture definitions (1D-CNN, SE-Blocks)
│   ├── training/              # Tuning (Optuna) and training pipelines
│   ├── evaluation/            # Benchmarking and reporting
│   ├── ingestion/             # Kafka producer and drift simulation
│   └── streaming/             # Spark Structured Streaming (Speed Layer)
├── models/                    # Versioned Artifact Bundles
├── results/                   # Timestamped evaluation reports
├── data/                      # Raw C-MAPSS dataset (FD001, etc.)
├── preprocess.py              # Entry: Data Pipeline
├── tune.py                    # Entry: Training Pipeline (NAS)
└── evaluate.py                # Entry: Evaluation Pipeline
```

## 🛠 Getting Started

### 1. Start Infrastructure
Launch the Dockerized Big Data stack:
```bash
docker compose up -d
```

### 2. Run Speed Layer
Submit the Spark streaming job (inside the Spark Master container):
```bash
docker exec -it spark-master spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.1 \
  /opt/spark-apps/src/streaming/speed_layer.py \
  --model-dir /opt/spark-apps/models/{timestamp}
```

### 3. Start Telemetry Stream
Run the Kafka producer simulator:
```bash
python -m src.ingestion.kafka_producer
```

### 4. Optional: Test Resilience (Drift Injection)
Inject progressive noise and scalar drift to test system robustness:
```bash
python -m src.ingestion.generate_drift_data
# Then update kafka_producer.py to stream test_FD001_drifted.txt
```

---
*For detailed technical documentation, feature pruning logic, and scoring functions, see [ARCHITECTURE.md](ARCHITECTURE.md).*
