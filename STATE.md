---
scope: PROJECT_STATE
priority: 2 (Dynamic Context)
---

# Workspace State Tracker
*The AI must read this file before generating any implementation plan to understand its current operational context.*

## 1. Operational Context
* **Host OS:** Ubuntu WSL2 (Windows 11 Host)
* **Path Style:** POSIX (`~/cmapss-predictive-maintenance`)
* **Active Environment Manager:** Docker (Infrastructure) / Conda (Local PyTorch Training)
* **Target Environment Name:** `pytorch-gpu` (for local model authoring)

## 2. Network Mapping
* **Kafka Broker:** `localhost:9092` (via Docker WSL bridge)
* **Spark Master UI:** `http://localhost:8080`
* **Elasticsearch:** `http://localhost:9200`
* **Kibana UI:** `http://localhost:5601`

## 3. Current State
*   **Last Shell Command:** N/A (code authoring session)
*   **Last Completed Task:** Built the PySpark Structured Streaming Speed Layer (`src/streaming/speed_layer.py`) with stateful ONNX inference, dual Elasticsearch/HDFS sinks, and safety alerts.
*   **Current Active Task:** None. Speed Layer implementation complete.
*   **Known Blockers/Issues:** `kafka-python` must be installed in the `pytorch-gpu` conda env before running (`pip install kafka-python`). Docker stack must be up: `docker compose up -d zookeeper kafka kafka-setup`.
*   **Next Planned Step:** End-to-end integration testing and Kibana dashboard verification.
