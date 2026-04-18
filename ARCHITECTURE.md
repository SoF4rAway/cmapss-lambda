# Project Architecture: Antigravity Agent
---
scope: PROJECT_ARCHITECTURE
priority: 1 (Absolute Source of Truth for Implementation)
---

## 1. Project Objective
Build a high-performance, safety-critical Big Data system to ingest, process, and analyze NASA C-MAPSS turbofan telemetry. The system utilizes a "TinyML" approach to predict Remaining Useful Life (RUL) with sub-millisecond latency.

## 2. Infrastructure Stack (Dockerized)
*   **Ingestion:** Apache Kafka (Topic: `cmapss_telemetry`) + Zookeeper.
*   **Stream Processing:** Spark Structured Streaming (Python/PySpark).
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

## 4. Lambda Pipeline Phases

### Phase 1: Ingestion (The Producer)
*   Python-based simulator reading `.txt` source files.
*   Serializes sensor arrays into **JSON payloads** keyed by `unit_id` to ensure partition-local stateful processing in Spark.

### Phase 2: Speed Layer (Real-time Inference)
*   **Mechanism:** Spark Structured Streaming consuming from Kafka.
*   **Inference:** Vectorized execution via **Pandas UDFs** loading the ONNX model.
*   **Optimization:** Model size is kept **<1MB** (current: 378KB) to maximize CPU cache hits and minimize memory bus contention.
*   **Target Latency:** < 0.1ms per inference (Current benchmark: 0.07ms).

### Phase 3: Batch Layer (Historical & Retraining)
*   **Persistence:** Raw telemetry and predictions are synced to HDFS in Parquet format.
*   **Retraining Loop:** Periodic batch jobs trigger PyTorch retraining if statistical drift (Bias > 10.0) is detected.
*   **Redeploy:** Automated export to **ONNX (Opset 14)** for hot-swapping into the Speed Layer.

### Phase 4: Serving (ELK Stack)
*   **Sink:** Predictions pushed to Elasticsearch via `spark-elasticsearch` connector.
*   **Visualization:** Kibana dashboards tracking:
    *   Individual Engine Health (RUL countdown).
    *   System-wide Error Distribution (Residuals).
    *   Safety Alerts (Critical: `RUL < 30`).

## 5. Performance Benchmarks & Metrics
The system is evaluated against the NASA C-MAPSS FD001 dataset:

| Metric | Target | Current Baseline |
| :--- | :--- | :--- |
| **RMSE** | < 20.0 | **19.15** |
| **NASA Score** | < 1100 | **1023.09** |
| **Inference Latency** | < 0.1 ms | **0.07 ms** |
| **Model Size** | < 500 KB | **378.23 KB** |

### NASA Asymmetric Scoring Function
Where $d_{i} = \hat{y}_{i} - y_{i}$:
$$S = \sum_{i=1}^{n} \left( e^{-\frac{d_i}{13}} - 1 \right) \text{ for } d_i < 0$$
$$S = \sum_{i=1}^{n} \left( e^{\frac{d_i}{10}} - 1 \right) \text{ for } d_i \ge 0$$

## 6. Ensuring Reproducibility & Maintainability
* **Global Seed Control**: Centralized `set_seed(42)` in `src/core/config.py` governing Python, NumPy, PyTorch, and CuDNN.
* **Deterministic DataLoaders**: Multi-threaded loaders use `seed_worker` and `torch.Generator` for absolute data shuffling consistency.
* **Schema Versioning**: Feature selection results are exported to `feature_schema.json` to synchronize the Spark Streaming pipeline with the trained model's input requirements.
* **Model Versioning**: All models are stored in `models/` with timestamped filenames (`YYYYMMDD_HHMMSS_rul_model.onnx`).
* **Evaluation Versioning**: Every evaluation run generates a versioned package in `results/{timestamp}/` containing the performance plot (`.png`) and academic report (`.md`).

## 7. Project Directory Structure
```text
cmapss-lambda/
├── src/                       # Modular source package
│   ├── core/                  # Global config and reproducibility
│   ├── data/                  # Preprocessing and DataLoaders
│   ├── models/                # Architecture definitions
│   ├── training/              # Tuning and training pipelines
│   └── evaluation/            # Benchmarking and reporting
├── models/                    # Persistent storage for ONNX models
├── results/                   # Timestamped evaluation reports
├── data/                      # Raw C-MAPSS dataset
├── preprocess.py              # Entry: Data Pipeline
├── tune.py                    # Entry: Training Pipeline
└── evaluate.py                # Entry: Evaluation Pipeline
```
