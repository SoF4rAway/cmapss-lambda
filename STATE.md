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
*   **Last Shell Command:** `docker exec -it spark-master spark-submit --packages ... src/streaming/speed_layer.py ...`
*   **Last Completed Task:** Implemented Tiered Sink Strategy (sync Elasticsearch, daemon HDFS every 10 batches), resolved `FutureWarning` in Pandas UDFs, and optimized Elasticsearch `refresh_interval` to 100ms for sub-second Kibana updates.
*   **Current Active Task:** Spark Speed Layer is optimized and ready for low-latency telemetry processing.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Start the Kafka producer and verify sub-second end-to-end latency in the Kibana dashboard.
