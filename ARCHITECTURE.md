---
scope: PROJECT_ARCHITECTURE
priority: 1 (Absolute Source of Truth for Implementation)
---

# Predictive Maintenance: Lambda Architecture

## 1. Project Objective
Build an end-to-end Big Data system to ingest, process, and analyze NASA C-MAPSS turbofan sensor data to predict Remaining Useful Life (RUL) and detect anomalies.

## 2. Infrastructure Stack (Docker Compose)
* **Ingestion:** Apache Kafka (Topic: `cmapss_telemetry`, JSON format) + Zookeeper.
* **Compute/Batch:** Apache Spark + Hadoop Distributed File System (HDFS).
* **Serving/Analytics:** Elasticsearch (Time-series index) + Kibana (Dashboards/Alerts).

## 3. Data Schema (C-MAPSS)
Incoming JSON payloads will map to 26 features:
* `Unit Number`: Engine ID
* `Time/Cycles`: Operational cycle
* `Operational Settings (1-3)`
* `Sensor Measurements (1-21)`

## 4. Pipeline Phases
1.  **Ingestion:** Python Kafka Producer reading raw `.txt` simulating continuous high-speed JSON stream.
2.  **Batch Layer:** Persist data in HDFS/Parquet. Feature engineering via PySpark (sliding windows). Train PyTorch model (LSTM/1D-CNN) locally to predict RUL.
3.  **Speed Layer:** Spark Structured Streaming consuming Kafka. Applies Pandas UDFs to load the TorchScript model for sub-second, distributed inference.
4.  **Serving:** Sink predictions to Elasticsearch. Trigger Kibana alerts if `RUL < 30`.

## 5. Evaluation Metrics
Models must be evaluated using:
1.  **RMSE**
2.  **Asymmetric Scoring Function:** Penalizes late predictions (estimating longer life than reality) heavier than early predictions. 
    Where $d_{i} = \hat{y}_{i} - y_{i}$:
    $$S = \sum_{i=1}^{n} \left( e^{-\frac{d_i}{13}} - 1 \right) \text{ for } d_i < 0$$
    $$S = \sum_{i=1}^{n} \left( e^{\frac{d_i}{10}} - 1 \right) \text{ for } d_i \ge 0$$