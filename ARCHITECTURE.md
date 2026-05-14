# Project Architecture: Antigravity Agent
---
scope: PROJECT_ARCHITECTURE
priority: 1 (Absolute Source of Truth for Implementation)
---

## 1. Project Objective
Build a high-performance, safety-critical Big Data system to ingest, process, and analyze NASA C-MAPSS turbofan telemetry. The system utilizes a "TinyML" approach to predict Remaining Useful Life (RUL) with sub-millisecond latency.

## 2. Infrastructure Stack (Dockerized)
*   **Ingestion:** Apache Kafka (Topic: `cmapss_telemetry`) + Zookeeper (Configured with HTTP Admin API healthchecks for robust startup).
*   **Stream Processing:** Spark Structured Streaming (Python/PySpark) with strict container initialization dependencies.
*   **Inference Engine:** ONNX Runtime (Optimized for x64 L1/L2 Cache residency).
*   **Storage (Cold):** HDFS / Parquet (Snappy Compression).
*   **Observability (Hot):** Elasticsearch + Kibana (Real-time RUL Dashboards).

## 3. Data Strategy & Preprocessing
### Feature Selection
*   **Variance Pruning:** Automatic removal of constant features via a **Variance Threshold (1e-5)**.
*   **Monotonicity Selection:** Calculates **Absolute Spearman Correlation** per engine and averages across the fleet. Prunes the bottom 5 least monotonic sensors or those with correlation < 0.2.
*   **FD001 Specifics:** Typically drops ~15 features (Op_Settings 1-3, Sensors 1, 5, 6, 10, 16, 18, 19 + lowest monotonic).

### Temporal Transformation
*   **Sliding Window:** 2D telemetry is reshaped into 3D tensors `(batch, channels, window_size)` for 1D-CNN temporal feature extraction.
*   **Target Labeling:** Piecewise Linear RUL capped at **125 cycles** to reduce early-life noise.

### Operational Modes (Production Ready)
*   **Fit Mode:** Calculates monotonicity, fits the scaler, and determines the final `active_features` list.
*   **Transform Mode:** Completely skips selection/fitting. Loads `feature_schema.json` and `scaler.joblib` from a versioned artifact directory for deterministic execution.

## 4. Lambda Pipeline Phases

### Phase 1: Ingestion (The Producer)
*   **Mechanism**: Python-based simulator reading `.txt` source files.
*   **Serialization**: Sensor arrays are serialized into **JSON payloads** keyed by `unit_id` to guarantee partition-local stateful processing in Spark.
*   **Throughput Control**: Streams at a controlled rate (~20 msgs/sec) with an optimized flush strategy (every 10 messages). This preserves Kafka's internal record-batching efficiency while keeping end-to-end latency strictly bounded.

### Phase 2: Speed Layer (Real-time Inference)
*   **Mechanism:** Spark Structured Streaming consuming from Kafka, configured with a 2-second processing trigger and `maxOffsetsPerTrigger` to manage backpressure.
*   **State Management:** Maintains a 30-cycle engine history buffer using `applyInPandasWithState`. State is serialized as binary **Pickle (Protocol 5)** to maximize throughput.
*   **Zero Data Loss UDF:** Uses a **Vectorized Sliding Window** approach inside the Pandas UDF. This ensures that every single telemetry message results in a prediction, even when messages are clumped together in a single micro-batch, by sliding a 30-cycle window over the entire incoming data partition.
*   **Performance Tuning:** 
    *   **Shuffle Optimization:** `spark.sql.shuffle.partitions` is set to 2 (matching Kafka partitions) to eliminate the default 200-task scheduling overhead.
    *   **Vectorization:** Features are scaled in a single batch before the inference loop to minimize compute cycles.
*   **Inference:** Vectorized execution via **Pandas UDFs** loading the ONNX model. Broadcast variables (model, schema, scaler) are explicitly bound via closures to guarantee correct executor serialization.
*   **Multi-Sink Strategy:** Uses a tiered approach within `foreachBatch` to decouple output streams:
    *   **Hot Path (Sync):** Writes to Elasticsearch in the foreground (blocking) to guarantee sub-second dashboard updates.
    *   **Cold Path (Async):** Offloads HDFS Parquet writes to a non-blocking daemon thread every 10 micro-batches. Uses `.coalesce(1)` and 100MB block sizes to stabilize the Datanode buffer.
*   **Target Latency:** < 0.1ms per inference (Current benchmark: 0.05ms).

### Phase 3: Batch Layer (Historical & Retraining)
*   **Persistence:** Raw telemetry and predictions are synced to HDFS in Parquet format via asynchronous daemon threads managed by the Speed Layer.
*   **Retraining Loop:** Periodic batch jobs trigger PyTorch retraining if statistical drift (Bias > 10.0) is detected.
*   **Redeploy:** Automated export to **ONNX (Opset 14)** for hot-swapping into the Speed Layer.

### Phase 4: Serving (ELK Stack)
*   **Sink:** Predictions are pushed to Elasticsearch via the `spark-elasticsearch` connector. We enforce deterministic document IDs (`unit_X_cycle_Y`) and `upsert` operations to eliminate duplicate records.
*   **Visualization:** Kibana dashboards tracking:
    *   Individual Engine Health (RUL countdown).
    *   System-wide Error Distribution (Residuals).
    *   Safety Alerts (Critical: `RUL < 30`).

## 5. Performance Benchmarks & Metrics
The system is evaluated against the NASA C-MAPSS FD001 dataset:

| Metric | Target | Current Baseline |
| :--- | :--- | :--- |
| **RMSE** | < 20.0 | **16.2307** |
| **NASA Score** | < 500 | **356.19** |
| **Inference Latency** | < 0.1 ms | **0.0515 ms** |
| **Model Size** | < 500 KB | **27.64 KB** |

### NASA Asymmetric Scoring Function
Where $d_{i} = \hat{y}_{i} - y_{i}$:
$$S = \sum_{i=1}^{n} \left( e^{-\frac{d_i}{13}} - 1 \right) \text{ for } d_i < 0$$
$$S = \sum_{i=1}^{n} \left( e^{\frac{d_i}{10}} - 1 \right) \text{ for } d_i \ge 0$$

## 6. Ensuring Reproducibility & Maintainability
* **Global Seed Control**: Centralized `set_seed(42)` in `src/core/config.py` governing Python, NumPy, PyTorch, and CuDNN.
* **Deterministic DataLoaders**: Multi-threaded loaders use `seed_worker` and `torch.Generator` for absolute data shuffling consistency.
* **Artifact Bundling**: Every training run produces a synchronized bundle in `models/{timestamp}/` containing:
    *   `model.onnx`: The optimized inference graph.
    *   `scaler.joblib`: The fitted `StandardScaler`.
    *   `feature_schema.json`: The canonical list of active sensors.
* **Evaluation Versioning**: Every evaluation run generates a versioned package in `results/{timestamp}/` containing the performance plot (`.png`) and academic report (`.md`).

## 7. Project Directory Structure
```text
cmapss-lambda/
├── src/                       # Modular source package
│   ├── core/                  # Global config and reproducibility
│   ├── data/                  # Preprocessing and DataLoaders
│   ├── models/                # Architecture definitions (1D-CNN, SE-Blocks)
│   ├── training/              # Tuning (Optuna) and training pipelines
│   ├── evaluation/            # Benchmarking and reporting
│   ├── ingestion/             # Kafka producer and data simulation
│   └── streaming/             # Spark Structured Streaming (Speed Layer)
├── models/                    # Versioned Artifact Bundles
│   └── {timestamp}/           # model.onnx, scaler.joblib, feature_schema.json
├── results/                   # Timestamped evaluation reports
│   └── {timestamp}/           # report.md, performance_results.png
├── data/                      # Raw C-MAPSS dataset (FD001, etc.)
├── preprocess.py              # Entry: Data Pipeline
├── tune.py                    # Entry: Training Pipeline (NAS)
└── evaluate.py                # Entry: Evaluation Pipeline
```
